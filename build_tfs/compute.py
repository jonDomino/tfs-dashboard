"""TFS computation and filtering"""
import pandas as pd
from builders.action_time.build_tfs_detailed import build_tfs_detailed


def compute_tfs(df: pd.DataFrame) -> pd.DataFrame:
    """Compute TFS (Time to First Shot) from preprocessed data.
    
    Args:
        df: Preprocessed play-by-play DataFrame
        
    Returns:
        DataFrame with TFS data, filtered to 3-40 seconds range
        Only includes actual field goal attempts (excludes free throws, rebounds, fouls)
    """
    # Build detailed TFS
    df = build_tfs_detailed(df)
    
    # Filter to first FIELD GOAL ATTEMPTS (not free throws) in possession
    # Fix for quadruple counting: Only include actual field goal attempts, exclude:
    # - Free throws (MadeFreeThrow)
    # - Rebounds (Offensive/Defensive Rebound)
    # - Fouls (PersonalFoul, etc.)
    
    # Identify field goal attempts using regex (exclude free throws)
    is_field_goal_attempt = (
        df["type_text"].str.contains(
            "jumper|three|layup|dunk|shot", 
            case=False, na=False
        ) & 
        ~df["type_text"].str.contains("freethrow", case=False, na=False)
    )
    
    # Also exclude rebounds and fouls explicitly
    is_field_goal_attempt = (
        is_field_goal_attempt &
        ~df["type_text"].str.contains("rebound|foul", case=False, na=False)
    )
    
    # Filter to first shots that are field goal attempts
    tfs_df = df[
        (df["shot_count_in_poss"] == 1) & 
        is_field_goal_attempt
    ].copy()
    
    # Filter to valid action time range (3-40 seconds)
    tfs_df = tfs_df[
        (tfs_df["action_time"] >= 3) & (tfs_df["action_time"] <= 40)
    ]
    
    if tfs_df.empty:
        raise ValueError("No valid TFS rows after filtering 3–40 seconds.")
    
    # Sort and add chronological index
    tfs_df = tfs_df.sort_values(["period_number", "possession_id"])
    tfs_df["chrono_index"] = range(1, len(tfs_df) + 1)
    
    return tfs_df

