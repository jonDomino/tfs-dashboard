"""TFS preprocessing pipeline"""
import pandas as pd
from builders.action_time.clean_pbp import load_and_sort_game_csv
from builders.action_time.flag_ft import add_ft_flags
from builders.action_time.assign_poss_teams import add_pre_post_possession
from builders.action_time.build_poss_context import add_possession_context
from builders.action_time.add_poss_start_type import add_poss_start_type
from builders.action_time.build_action_context import add_action_context


def preprocess_pbp(raw_pbp: pd.DataFrame) -> pd.DataFrame:
    """Run full preprocessing pipeline on play-by-play data.
    
    Args:
        raw_pbp: Raw play-by-play DataFrame
        
    Returns:
        Preprocessed DataFrame
    """
    df = load_and_sort_game_csv(raw_pbp)
    df = add_ft_flags(df)
    df = add_pre_post_possession(df)
    df = add_possession_context(df)
    df = add_poss_start_type(df)
    df = add_action_context(df)
    return df

