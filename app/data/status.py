"""Game status classification based on play-by-play data"""
import pandas as pd
from typing import Literal


GameStatus = Literal["Not Started", "Early 1H", "First Half", "Second Half", "Halftime", "Complete"]


def classify_game_status_pbp(df: pd.DataFrame) -> GameStatus:
    """Classify game status based on play-by-play data.
    
    Definitions:
    - Early 1H: period 1 with more than 600 seconds (10 minutes) remaining
    - First Half: period 1 with more than 60 seconds remaining (but not Early 1H)
    - Halftime: period 1 with 60 seconds or less remaining, OR period 1 ended and period 2 not yet started
    - Second Half: period 2 with clock running
    - Complete: period 2 ended or period > 2
    
    Args:
        df: Play-by-play DataFrame
        
    Returns:
        Game status string
    """
    if df is None or len(df) == 0:
        return "Not Started"
    
    # Check if we have period information
    if "period_number" not in df.columns:
        return "Not Started"
    
    # Filter out rows with missing period_number
    df = df[df["period_number"].notna()].copy()
    
    if len(df) == 0:
        return "Not Started"
    
    periods = df["period_number"].unique()
    
    if len(periods) == 0:
        return "Not Started"
    
    max_period = int(periods.max())
    
    # Check if game has clock information
    if "clock_value" not in df.columns:
        # Fallback: if no clock info, use period number only
        if max_period == 1:
            return "Halftime"  # Assume halftime if period 1 and no clock
        elif max_period == 2:
            return "Second Half"
        else:
            return "Complete"
    
    # Sort by period and clock (descending) to get most recent plays first
    # Clock counts down, so lower values = more recent
    df_sorted = df.sort_values(
        by=["period_number", "clock_value", "sequence_number"],
        ascending=[True, True, True]  # Period asc, clock asc (lowest = most recent), sequence asc
    )
    
    # Period 1 logic
    if max_period == 1:
        # Get period 1 data and find the most recent clock value (minimum)
        period1_data = df_sorted[df_sorted["period_number"] == 1]
        if len(period1_data) == 0:
            return "Not Started"
        
        # Get the minimum clock value (most recent play in period 1)
        # Filter out None/NaN values
        period1_clocks = period1_data["clock_value"].dropna()
        if len(period1_clocks) == 0:
            return "Not Started"
        
        last_clock = float(period1_clocks.min())
        
        # Early 1H: more than 600 seconds (10 minutes) remaining
        if last_clock > 600:
            return "Early 1H"
        # First Half: more than 60 seconds remaining (but not Early 1H)
        elif last_clock > 60:
            return "First Half"
        # Halftime: 60 seconds or less remaining, or clock at 0
        else:
            return "Halftime"
    
    # Period 2 logic
    elif max_period == 2:
        # Check if period 2 has started (has any plays in period 2)
        period2_data = df_sorted[df_sorted["period_number"] == 2]
        if len(period2_data) == 0:
            # Period 1 ended but period 2 not started = Halftime
            return "Halftime"
        
        # Get the minimum clock value from period 2 (most recent play)
        period2_clocks = period2_data["clock_value"].dropna()
        if len(period2_clocks) == 0:
            # No valid clock data, assume second half if period 2 exists
            return "Second Half"
        
        last_clock = float(period2_clocks.min())
        
        if last_clock > 0:
            return "Second Half"
        else:
            return "Complete"
    
    # Period > 2 = Complete
    else:
        return "Complete"

