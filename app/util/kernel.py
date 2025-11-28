"""Kernel smoothing utilities"""
import numpy as np
from typing import Tuple, Optional


def gaussian_kernel_smoother(
    x: np.ndarray,
    y: np.ndarray,
    bandwidth: float,
    grid: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply Gaussian kernel smoothing.
    
    Args:
        x: Input x values
        y: Input y values
        bandwidth: Kernel bandwidth
        grid: Optional grid points (defaults to x)
        
    Returns:
        Tuple of (grid, smoothed_y)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    if grid is None:
        grid = x
    grid = np.asarray(grid, dtype=float)
    
    smooth = np.zeros_like(grid, dtype=float)
    
    for i, g in enumerate(grid):
        w = np.exp(-0.5 * ((g - x) / bandwidth) ** 2)
        if w.sum() == 0:
            smooth[i] = np.nan
        else:
            smooth[i] = np.sum(w * y * w) / np.sum(w * w)
    
    return grid, smooth

