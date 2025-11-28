"""TFS prediction and smoothing"""
import numpy as np
import pandas as pd
from typing import Tuple


def predict_tfs_next(tfs_df: pd.DataFrame, n_ahead: int = 12) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict TFS for next N possessions using exponential smoothing.
    
    Args:
        tfs_df: TFS DataFrame with chrono_index and action_time
        n_ahead: Number of possessions to predict ahead
        
    Returns:
        Tuple of (pred_curve, conf_hi, conf_lo) arrays
    """
    y = tfs_df["action_time"].values.astype(float)
    
    if len(y) < 3:
        # Not enough data, return flat predictions
        pred = np.full(n_ahead, np.mean(y) if len(y) > 0 else 20.0)
        conf_hi = pred + 5.0
        conf_lo = pred - 5.0
        return pred, conf_hi, conf_lo
    
    # Exponential smoothing (Holt-Winters-like)
    alpha = 0.3  # Smoothing parameter
    last_value = y[-1]
    trend = np.mean(np.diff(y[-min(5, len(y)):])) if len(y) > 1 else 0.0
    
    # Generate predictions
    pred = []
    current = last_value
    
    for _ in range(n_ahead):
        current = alpha * current + (1 - alpha) * last_value + trend
        pred.append(current)
        last_value = current
    
    pred = np.array(pred)
    
    # Simple confidence bands based on recent variance
    recent_std = np.std(y[-min(10, len(y)):])
    conf_hi = pred + 1.96 * recent_std  # 95% confidence
    conf_lo = pred - 1.96 * recent_std
    
    return pred, conf_hi, conf_lo

