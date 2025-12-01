import pandas as pd

def build_tfs_detailed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds *detailed* TFS information on a per-play basis.
    This mirrors the audit pipeline logic that produced
    action_time, shot_count_in_poss, and the possession-level
    ordering needed for kernel tempo curves.
    """

    # --------------------------------------------------------
    # Ensure correct chronological ordering before grouping
    # --------------------------------------------------------
    df = df.sort_values(
        ["period_number", "possession_id", "clock_value"],
        ascending=[True, True, False]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Identify shot attempts
    # --------------------------------------------------------
    shot_mask = df["type_text"].str.contains(
        "jumper|three|layup|dunk|shot", case=False, na=False
    )
    df["is_shot"] = shot_mask.astype(int)

    # --------------------------------------------------------
    # Count shots within each possession
    # --------------------------------------------------------
    df["shot_count_in_poss"] = (
        df.groupby(["game_id", "period_number", "possession_id"])["is_shot"]
          .cumsum()
    )

    # --------------------------------------------------------
    # Compute TFS: action_time already exists from add_action_context
    # --------------------------------------------------------
    # Valid only for first-shot plays
    df["tfs"] = df["action_time"].where(df["shot_count_in_poss"] == 1)

    return df

