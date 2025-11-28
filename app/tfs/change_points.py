"""CUSUM change-point detection for TFS analysis"""
import numpy as np
from typing import List


def detect_cusum(y: np.ndarray, threshold: float, drift: float) -> List[int]:
    """Detect change points using CUSUM algorithm.
    
    Args:
        y: Input signal array
        threshold: Detection threshold
        drift: Drift parameter
        
    Returns:
        List of change point indices
    """
    y = np.asarray(y, dtype=float)
    gpos = np.zeros(len(y))
    gneg = np.zeros(len(y))
    cps = []
    
    for i in range(1, len(y)):
        gpos[i] = max(0, gpos[i - 1] + (y[i] - drift))
        gneg[i] = min(0, gneg[i - 1] + (y[i] - drift))
        
        if gpos[i] > threshold:
            cps.append(i)
            gpos[i] = 0
            gneg[i] = 0
        
        if gneg[i] < -threshold:
            cps.append(i)
            gpos[i] = 0
            gneg[i] = 0
    
    return sorted(set(cps))


def find_change_points(y: np.ndarray, min_change_magnitude: float = 4.0) -> List[int]:
    """Find change points in signal using adaptive threshold.
    
    More restrictive: only detects material changes in TFS.
    
    Args:
        y: Input signal array
        min_change_magnitude: Minimum TFS change (seconds) to consider material
        
    Returns:
        List of change point indices (empty if none found or insufficient data)
    """
    y = np.asarray(y, dtype=float)
    
    if len(y) < 10:
        return []
    
    std = float(np.std(y))
    if std == 0:
        return []
    
    # More restrictive threshold multipliers (higher = fewer detections)
    # Start at 5.5 and go up to 9.0 for more restrictive detection
    for mult in np.linspace(5.5, 9.0, 15):
        th = std * mult
        cps = detect_cusum(y, threshold=th, drift=float(np.mean(y)))
        
        # More restrictive: only accept 1-3 change points (was 2-6)
        if 1 <= len(cps) <= 3:
            # Filter by minimum change magnitude
            filtered_cps = filter_by_magnitude(y, cps, min_change_magnitude)
            if len(filtered_cps) > 0:
                return filtered_cps
    
    return []


def filter_by_magnitude(y: np.ndarray, cps: List[int], min_magnitude: float) -> List[int]:
    """Filter change points by minimum magnitude of change.
    
    Only keeps change points where the actual TFS change is material.
    
    Args:
        y: Input signal array
        cps: List of change point indices
        min_magnitude: Minimum change magnitude in seconds
        
    Returns:
        Filtered list of change point indices
    """
    if len(cps) == 0:
        return []
    
    filtered = []
    
    for cp in cps:
        if cp < 1 or cp >= len(y):
            continue
        
        # Calculate change magnitude: compare before/after segments
        # Use a window around the change point
        window = min(5, cp, len(y) - cp - 1)
        
        if window < 2:
            continue
        
        before_mean = np.mean(y[max(0, cp - window):cp])
        after_mean = np.mean(y[cp:min(len(y), cp + window)])
        
        change_magnitude = abs(after_mean - before_mean)
        
        # Only keep if change is material
        if change_magnitude >= min_magnitude:
            filtered.append(cp)
    
    return filtered

