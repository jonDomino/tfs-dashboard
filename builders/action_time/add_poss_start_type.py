"""
add_poss_start_type.py

Adds possession start type field that tracks how each possession started:
- rebound: Previous possession ended with defensive/dead ball rebound
- turnover: Previous possession contained a turnover
- oppo_made_shot: Previous possession ended with made shot (FG or final FT)
- period_start: First possession of period/game
- None: Edge cases
"""

import pandas as pd


def add_poss_start_type(df: pd.DataFrame) -> pd.DataFrame:
    """Add poss_start_type field to track how each possession started.
    
    For each possession, checks the LAST action of the PREVIOUS possession
    to determine the start type. Applies the same value to all rows within
    a possession.
    
    Args:
        df: DataFrame with possession_id, type_text, scoring_play, final_ft columns
            Must be sorted by [game_id, period_number, possession_id, clock_value, sequence_number]
    
    Returns:
        DataFrame with poss_start_type column added
    """
    df = df.copy()
    
    # Initialize poss_start_type column
    df["poss_start_type"] = None
    
    # Group by game and period to handle period starts
    for (game_id, period), period_df in df.groupby(["game_id", "period_number"], group_keys=False):
        period_indices = period_df.index
        
        # Get unique possession IDs in this period
        poss_ids = period_df["possession_id"].unique()
        
        for i, poss_id in enumerate(poss_ids):
            poss_mask = (df.index.isin(period_indices)) & (df["possession_id"] == poss_id)
            poss_rows = df[poss_mask]
            
            if poss_rows.empty:
                continue
            
            # Check if this is the first possession of the period
            if i == 0:
                # First possession of period = period_start
                df.loc[poss_mask, "poss_start_type"] = "period_start"
                continue
            
            # Get previous possession
            prev_poss_id = poss_ids[i - 1]
            prev_poss_mask = (df.index.isin(period_indices)) & (df["possession_id"] == prev_poss_id)
            prev_poss_rows = df[prev_poss_mask]
            
            if prev_poss_rows.empty:
                df.loc[poss_mask, "poss_start_type"] = None
                continue
            
            # Get the LAST action of the previous possession
            last_action = prev_poss_rows.iloc[-1]
            
            # Check in order (mutually exclusive):
            # 1. Rebound: Previous ended with "Defensive Rebound" or "Dead Ball Rebound"
            type_text = str(last_action.get("type_text", "")).lower()
            if "defensive rebound" in type_text or "dead ball rebound" in type_text:
                df.loc[poss_mask, "poss_start_type"] = "rebound"
                continue
            
            # 2. Turnover: Previous possession contains ANY action with "turnover" in type_text
            prev_poss_has_turnover = prev_poss_rows["type_text"].astype(str).str.lower().str.contains("turnover", na=False).any()
            if prev_poss_has_turnover:
                df.loc[poss_mask, "poss_start_type"] = "turnover"
                continue
            
            # 3. Opponent Made Shot: Previous ended with made shot
            # Check if last action was a scoring play (field goal made)
            if last_action.get("scoring_play", False) and last_action.get("score_value", 0) > 1:
                df.loc[poss_mask, "poss_start_type"] = "oppo_made_shot"
                continue
            
            # Check if last action was final free throw that transfers possession
            # (final_ft flag indicates this is the last FT in a sequence that transfers possession)
            # Note: final_ft alone doesn't guarantee possession transfer, but if it's scoring, it does
            if last_action.get("final_ft", False):
                # Check if this FT transfers possession (typically the last FT in a sequence)
                # We'll treat final_ft as a made shot that transfers possession
                df.loc[poss_mask, "poss_start_type"] = "oppo_made_shot"
                continue
            
            # 4. None: Edge cases
            df.loc[poss_mask, "poss_start_type"] = None
    
    return df

