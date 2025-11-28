"""Schedule data loading module"""
import pandas as pd
from .get_sched import get_sched


def load_schedule() -> pd.DataFrame:
    """Load and return schedule data.
    
    Returns:
        DataFrame with schedule information
    """
    return get_sched()

