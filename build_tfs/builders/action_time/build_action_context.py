"""
add_action_context.py

Adds possession-level action context:
- shot_count_in_poss  (for shooting plays only)
- action_time          (for all plays)
"""

import pandas as pd

def add_action_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds two contextual fields to play-by-play data:
      - shot_count_in_poss: counts shooting plays within each possession
      - action_time: time elapsed since possession start (for all plays)
    """

    # ------------------------------------------------------------------
    # Global sort for consistent grouping (faithful to Python order)
    # ------------------------------------------------------------------
    df = df.sort_values(
        ["game_id", "period_number", "possession_id",
         "clock_value", "sequence_number"],
        ascending=[True, True, True, False, True]
    ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 1. Compute action_time for all plays (vectorized)
    # ------------------------------------------------------------------
    df["action_time"] = (
        df["poss_start_time"]
        .groupby([df["game_id"], df["period_number"], df["possession_id"]])
        .transform("first")
        - df["clock_value"]
    )

    # ------------------------------------------------------------------
    # 2. Compute shot_count_in_poss (for shooting plays only)
    # ------------------------------------------------------------------
    shot_counter = (
        df["shooting_play"]
        .astype(int)
        .groupby([df["game_id"], df["period_number"], df["possession_id"]])
        .cumsum()
    )

    # Zero out non-shooting rows
    df["shot_count_in_poss"] = shot_counter.where(df["shooting_play"], 0)

    return df

