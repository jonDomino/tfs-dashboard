"""Play-by-play data loading with caching and throttling"""
import pandas as pd
import streamlit as st
from .get_pbp import get_pbp


@st.cache_data(ttl=60)  # Cache for 60 seconds
def load_pbp_cached(game_id: str) -> pd.DataFrame:
    """Load play-by-play data with caching.
    
    Args:
        game_id: Game identifier as string
        
    Returns:
        DataFrame with play-by-play data
        
    Raises:
        ValueError: If no play data found
    """
    raw = get_pbp(int(game_id))
    if raw is None or len(raw) == 0:
        raise ValueError("No play data found.")
    return raw


def load_pbp(game_id: str, use_cache: bool = True) -> pd.DataFrame:
    """Load play-by-play data.
    
    Args:
        game_id: Game identifier as string
        use_cache: Whether to use cached data (default: True)
        
    Returns:
        DataFrame with play-by-play data
    """
    if use_cache:
        return load_pbp_cached(game_id)
    else:
        raw = get_pbp(int(game_id))
        if raw is None or len(raw) == 0:
            raise ValueError("No play data found.")
        return raw

