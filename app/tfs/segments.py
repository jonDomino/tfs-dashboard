"""Game segment markers for visualization"""
import pandas as pd
from typing import List, Optional, Tuple


def get_segment_x(tfs_df: pd.DataFrame, period: int, clock_threshold: float) -> Optional[float]:
    """Get chronological index for a specific clock time in a period.
    
    Args:
        tfs_df: TFS DataFrame with chrono_index
        period: Period number
        clock_threshold: Clock value to find
        
    Returns:
        Chronological index or None if not found
    """
    sub = tfs_df[tfs_df["period_number"] == period]
    if sub.empty:
        return None
    
    if "clock_value" not in sub.columns:
        return None
    
    diffs = (sub["clock_value"] - clock_threshold).abs()
    idx = diffs.idxmin()
    return float(sub.loc[idx, "chrono_index"])


def get_segment_lines(tfs_df: pd.DataFrame) -> List[Tuple[float, str, dict]]:
    """Get all segment line positions and styles.
    
    Args:
        tfs_df: TFS DataFrame
        
    Returns:
        List of tuples: (x_position, label, style_dict)
    """
    segments = []
    
    # First half thirds: 800s (6:40), 400s (3:20)
    for thr in [800, 400]:
        x = get_segment_x(tfs_df, 1, thr)
        if x is not None:
            segments.append((
                x,
                "First Half Segment",
                {"color": "gray", "linewidth": 1.5, "linestyle": "--", "alpha": 0.6}
            ))
    
    # Halftime line
    if (tfs_df["period_number"] == 2).any():
        x = float(tfs_df[tfs_df["period_number"] == 2]["chrono_index"].iloc[0])
        segments.append((
            x,
            "Halftime",
            {"color": "red", "linewidth": 3, "linestyle": "--", "alpha": 0.9}
        ))
    
    # Second-half markers: 16:00 (960s), 6:00 (360s), 2:00 (120s) remaining
    for thr in [960, 360, 120]:
        x = get_segment_x(tfs_df, 2, thr)
        if x is not None:
            segments.append((
                x,
                "Second Half Segment",
                {"color": "gray", "linewidth": 1.5, "linestyle": "--", "alpha": 0.6}
            ))
    
    return segments

