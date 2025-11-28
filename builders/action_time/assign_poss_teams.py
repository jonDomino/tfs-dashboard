"""
assign_poss_teams.py

Determines possession before and after each play using only row-level info.
Adds:
- has_ball_pre_play
- has_ball_post_play
"""

import pandas as pd


def _determine_pre_post(row):
    """Determine pre- and post-possession teams based on the play context."""
    team = row.team_name
    home, away = row.home_team_name, row.away_team_name
    opponent = away if team == home else home
    t = str(row.type_text)
    scoring = bool(row.get("scoring_play", False))
    shooting = bool(row.get("shooting_play", False))
    first_ft = bool(row.get("first_ft", False))
    final_ft = bool(row.get("final_ft", False))

    # -----------------------------------------------------------
    # Pre-possession team logic
    # -----------------------------------------------------------
    if shooting and t != "MadeFreeThrow":
        pre = team
    elif t == "MadeFreeThrow":
        pre = team
    elif t == "Lost Ball Turnover":
        pre = team
    elif t in ["Defensive Rebound", "Dead Ball Rebound"]:
        pre = opponent
    elif t == "Offensive Rebound":
        pre = team
    elif t == "Steal":
        pre = opponent
    elif t in ["Block Shot", "PersonalFoul", "Technical Foul"]:
        pre = opponent
    else:
        pre = team

    # -----------------------------------------------------------
    # Post-possession team logic
    # -----------------------------------------------------------
    if shooting and t != "MadeFreeThrow":
        post = opponent if scoring else team
    elif t == "MadeFreeThrow":
        if not final_ft:
            post = team
        elif scoring:
            post = opponent
        else:
            post = team
    elif t == "Lost Ball Turnover":
        post = opponent
    elif t in ["Defensive Rebound", "Dead Ball Rebound"]:
        post = team
    elif t == "Offensive Rebound":
        post = team
    elif t == "Steal":
        post = team
    elif t == "Block Shot":
        post = opponent
    elif t in ["PersonalFoul", "Technical Foul"]:
        post = opponent
    else:
        post = team

    return pre, post


def add_pre_post_possession(df: pd.DataFrame) -> pd.DataFrame:
    """Adds has_ball_pre_play and has_ball_post_play columns to the PBP DataFrame."""
    pre_list, post_list = [], []

    for _, row in df.iterrows():
        pre, post = _determine_pre_post(row)
        pre_list.append(pre)
        post_list.append(post)

    df["has_ball_pre_play"] = pre_list
    df["has_ball_post_play"] = post_list
    return df

