"""
build_poss_context.py

Builds possession context:
- possession_id (resets per game)
- poss_start_time
- poss_end_time
"""

import pandas as pd


def add_possession_context(df: pd.DataFrame) -> pd.DataFrame:

    # ============================================================
    # STEP 1 — Detect possession change boundaries
    # ============================================================
    change_mask = (
        (df["has_ball_pre_play"] != df["has_ball_pre_play"].shift(1))
        | (df["period_number"] != df["period_number"].shift(1))
        | (df["game_id"] != df["game_id"].shift(1))
    )

    # ============================================================
    # STEP 2 — Vectorized per-game cumulative possession_id
    # ============================================================
    # We group by game_id and cumulatively sum change_mask within each game
    df["possession_id"] = (
        df.groupby("game_id")["game_id"]
        .transform(lambda g: change_mask.loc[g.index].cumsum())
    )

    # ============================================================
    # STEP 3 — Compute start and end times (Python faithful)
    # ============================================================
    start_times = pd.Series(index=df.index, dtype="float64")
    end_times = pd.Series(index=df.index, dtype="float64")

    for (gid, per, pid), g in df.groupby(["game_id", "period_number", "possession_id"], group_keys=False):
        poss_end = g.iloc[-1].clock_value
        prev_idx = g.index[0] - 1
        if prev_idx >= 0 and df.loc[prev_idx, "game_id"] == gid and df.loc[prev_idx, "period_number"] == per:
            poss_start = df.loc[prev_idx, "clock_value"]
        else:
            poss_start = 1200
        start_times.loc[g.index] = poss_start
        end_times.loc[g.index] = poss_end

    df["poss_start_time"] = start_times
    df["poss_end_time"] = end_times

    return df

