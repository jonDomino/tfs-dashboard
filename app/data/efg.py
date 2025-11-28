"""Effective Field Goal Percentage (eFG%) calculation from PBP data"""
import pandas as pd
from typing import Optional, Tuple


def calculate_efg(pbp_df: pd.DataFrame, period: Optional[int] = None) -> Optional[float]:
    """Calculate effective field goal percentage from PBP data.
    
    eFG% = (FGM + 0.5 * 3PM) / FGA
    
    Where:
    - FGA = shooting_play = True AND score_value > 1
    - FGM = scoring_play = True AND score_value > 1
    - 3PM = scoring_play = True AND score_value = 3
    
    Args:
        pbp_df: Play-by-play DataFrame
        period: Period number (1 for first half, 2 for second half). 
                If None, calculates for all periods combined.
    
    Returns:
        eFG% as a float (0-1 scale), or None if no field goal attempts
    """
    if pbp_df.empty:
        return None
    
    # Filter by period if specified
    df = pbp_df.copy()
    if period is not None:
        df = df[df["period_number"] == period].copy()
    
    if df.empty:
        return None
    
    # Ensure score_value is numeric
    if "score_value" not in df.columns:
        return None
    
    df["score_value"] = pd.to_numeric(df["score_value"], errors="coerce").fillna(0)
    
    # Field Goal Attempts (FGA): shooting_play = True AND score_value > 1
    fga = df[(df["shooting_play"] == True) & (df["score_value"] > 1)]
    
    if len(fga) == 0:
        return None
    
    # Field Goals Made (FGM): scoring_play = True AND score_value > 1
    fgm = df[(df["scoring_play"] == True) & (df["score_value"] > 1)]
    
    # Three Pointers Made (3PM): scoring_play = True AND score_value = 3
    threes = df[(df["scoring_play"] == True) & (df["score_value"] == 3)]
    
    # Calculate eFG%
    fga_count = len(fga)
    fgm_count = len(fgm)
    threes_count = len(threes)
    
    if fga_count == 0:
        return None
    
    efg = (fgm_count + 0.5 * threes_count) / fga_count
    
    return efg


def calculate_efg_by_half(pbp_df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    """Calculate eFG% for first half and second half separately.
    
    Args:
        pbp_df: Play-by-play DataFrame
    
    Returns:
        Tuple of (first_half_efg, second_half_efg)
    """
    first_half_efg = calculate_efg(pbp_df, period=1)
    second_half_efg = calculate_efg(pbp_df, period=2)
    
    return first_half_efg, second_half_efg

