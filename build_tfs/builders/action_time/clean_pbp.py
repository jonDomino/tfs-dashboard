"""
clean_pbp.py

Responsibilities:
1. Clean known-bad or irrelevant ESPN events (timeouts, subs, reviews)
2. Remove artificial "team rebounds" between free throws
3. Produce a strictly chronological, fully reindexed PBP table so that
   downstream pipeline stages (possession boundaries, action_time)
   can safely assume:  row[i-1] → row[i] is true basketball time.

Authoritative Sort Order (within each game):
    1. period_number ASC
    2. clock_value DESC        -- time remaining in period
    3. sequence_number ASC     -- ESPN tie-breaker
    4. wallclock ASC           -- unreliable but safe final resolver
"""

import pandas as pd


def load_and_sort_game_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and chronologically sort ESPN play-by-play."""

    # ------------------------------------------------------------
    # STEP 1 — Drop incomplete / malformed rows
    # These are unusable for possession modeling.
    # ------------------------------------------------------------
    df = df.dropna(subset=["type_text", "team_id", "period_number"]).copy()

    # Add wallclock if missing (for compatibility)
    if "wallclock" not in df.columns:
        df["wallclock"] = pd.NaT

    # ------------------------------------------------------------
    # STEP 2 — Authoritative chronological sort *BEFORE* cleaning
    # This ensures adjacency logic (FT rebound sandwich) is reliable.
    # ------------------------------------------------------------
    sort_cols = ["game_id", "period_number", "clock_value", "sequence_number"]
    if "wallclock" in df.columns:
        sort_cols.append("wallclock")
    
    df = (
        df.sort_values(
            sort_cols,
            ascending=[True, True, False, True, True] if "wallclock" in sort_cols else [True, True, False, True]
        )
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------
    # STEP 3 — Remove ESPN noise rows
    # These rows do not change possession or shot timing.
    # ------------------------------------------------------------
    noise_mask = df["type_text"].str.contains(
        "timeout|substitution|review", case=False, na=False
    )
    df = df[~noise_mask].reset_index(drop=True)

    # ------------------------------------------------------------
    # STEP 4 — Drop "FT-sandwiched" artificial rebounds
    # ESPN often inserts a fake rebound between consecutive FTs.
    # Pattern: FT → REBOUND → FT
    # ------------------------------------------------------------
    def is_ft(x):
        return "freethrow" in str(x).lower()

    def is_reb(x):
        return "rebound" in str(x).lower()

    to_drop = []
    for i in range(1, len(df) - 1):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]
        nxt = df.iloc[i + 1]

        if (
            prev.game_id == curr.game_id == nxt.game_id
            and prev.period_number == curr.period_number == nxt.period_number
            and is_ft(prev.type_text)
            and is_ft(nxt.type_text)
            and is_reb(curr.type_text)
        ):
            to_drop.append(i)

    if to_drop:
        df = df.drop(df.index[to_drop]).reset_index(drop=True)

    # ------------------------------------------------------------
    # STEP 5 — Final authoritative chronological sort
    # Cleaning operations may have disrupted ordering.
    # ------------------------------------------------------------
    df = (
        df.sort_values(
            sort_cols,
            ascending=[True, True, False, True, True] if "wallclock" in sort_cols else [True, True, False, True]
        )
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------
    # STEP 6 — Return pristine, chronological PBP
    # Downstream code can rely on row adjacency.
    # ------------------------------------------------------------
    return df

