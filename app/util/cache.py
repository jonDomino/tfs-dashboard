"""Caching utilities"""
import streamlit as st
from functools import wraps
from typing import Callable, Any


def cached_data(ttl: int = 60):
    """Decorator for caching data with TTL.
    
    Args:
        ttl: Time to live in seconds
    """
    def decorator(func: Callable) -> Callable:
        @st.cache_data(ttl=ttl)
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator

