"""Configuration constants for the TFS Kernel Dashboard"""
from dataclasses import dataclass


@dataclass
class Config:
    """Centralized configuration for the dashboard"""
    
    # Cache TTLs (seconds)
    CACHE_TTL_STATUS: int = 30
    CACHE_TTL_PBP: int = 60
    CACHE_TTL_CLOSING_TOTALS: int = 3600
    
    # Time restrictions (BigQuery query window)
    BQ_QUERY_START_HOUR: int = 8
    BQ_QUERY_END_HOUR: int = 22
    
    # Performance
    MAX_WORKERS: int = 5
    REFRESH_INTERVAL: int = 30
    NOT_STARTED_THROTTLE_MINUTES: int = 10
    
    # Visualization
    COLS_PER_ROW: int = 2


# Global config instance
config = Config()

