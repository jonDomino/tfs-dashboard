"""TFS computation and filtering"""
import pandas as pd
from builders.action_time.build_tfs_detailed import build_tfs_detailed
from app.tfs.preprocess import preprocess_pbp


def compute_tfs(df: pd.DataFrame) -> pd.DataFrame:
    """Compute TFS (Time to First Shot) from preprocessed data.
    
    Args:
        df: Preprocessed play-by-play DataFrame
        
    Returns:
        DataFrame with TFS data, filtered to 3-40 seconds range
    """
    # Build detailed TFS
    df = build_tfs_detailed(df)
    
    # Filter to first shots in possession
    tfs_df = df[df["shot_count_in_poss"] == 1].copy()
    
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

