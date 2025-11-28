"""
flag_ft.py (Optimized)

Identifies contiguous free-throw sequences.
Adds:
- first_ft
- final_ft
- ft_group_id
"""

import pandas as pd
import numpy as np


def add_ft_flags(df: pd.DataFrame) -> pd.DataFrame:

    # Initialize output columns
    df["first_ft"] = False
    df["final_ft"] = False
    df["ft_group_id"] = pd.NA

    # Boolean mask for free throws
    is_ft = df["type_text"] == "MadeFreeThrow"

    # Precompute index as array for adjacency detection
    idx = np.arange(len(df))

    # ============================================================
    # Vectorized contiguous cluster detection per game_id
    # ============================================================
    group_ids = np.full(len(df), np.nan)
    first_flags = np.zeros(len(df), dtype=bool)
    final_flags = np.zeros(len(df), dtype=bool)

    for gid, g in df.groupby("game_id", group_keys=False):
        ft_mask = is_ft.loc[g.index].to_numpy()
        if not ft_mask.any():
            continue

        # positions within this game's subframe
        game_idx = g.index.to_numpy()[ft_mask]
        # detect cluster boundaries (non-contiguous gaps)
        new_cluster = np.empty(len(game_idx), dtype=bool)
        new_cluster[0] = True
        new_cluster[1:] = (game_idx[1:] != game_idx[:-1] + 1)

        # cumulative sum of cluster starts → cluster numbering within game
        cluster_ids = np.cumsum(new_cluster)

        # mark first/last
        first_in_cluster = np.zeros_like(ft_mask)
        last_in_cluster = np.zeros_like(ft_mask)
        first_in_cluster[np.where(ft_mask)[0][new_cluster]] = True
        last_in_cluster[np.where(ft_mask)[0][np.r_[np.nonzero(new_cluster)[0][1:] - 1, len(game_idx) - 1]]] = True

        # assign back
        group_ids[g.index[ft_mask]] = cluster_ids
        first_flags[g.index[first_in_cluster]] = True
        final_flags[g.index[last_in_cluster]] = True

    # assign to df
    df["ft_group_id"] = pd.Series(group_ids).astype("Int64")
    df["first_ft"] = first_flags
    df["final_ft"] = final_flags

    return df

