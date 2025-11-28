"""
get_pbp.py

Fetches and flattens ESPN Core API play-by-play JSON
for a given college basketball game into a pandas DataFrame.
Includes key fields, unpacks team info (using team_location as team_name),
normalizes boolean dtypes, and infers home/away teams.
"""

import requests
import pandas as pd
import os


def get_json(url: str):
    """Helper to safely get JSON from a given ESPN Core API $ref URL."""
    r = requests.get(url)
    r.raise_for_status()
    return r.json()


def get_pbp(game_id: int) -> pd.DataFrame:
    """
    Fetch play-by-play data from ESPN Core API and flatten to a DataFrame.
    Includes only essential columns, expands team info (location as name),
    enforces boolean dtypes, and infers home/away teams.
    """
    url = (
        f"https://sports.core.api.espn.com/v2/sports/basketball/"
        f"leagues/mens-college-basketball/events/{game_id}/competitions/{game_id}/plays?limit=500"
    )
    data = get_json(url)

    plays = data.get("items", [])
    if not plays:
        print("No play data found.")
        return pd.DataFrame()

    rows = []
    team_cache = {}

    for play in plays:
        row = {
            "id": play.get("id"),
            "sequence_number": play.get("sequenceNumber"),
            "type_id": play.get("type", {}).get("id"),
            "type_text": play.get("type", {}).get("text"),
            "text": play.get("text"),
            "away_score": play.get("awayScore"),
            "home_score": play.get("homeScore"),
            "period_number": play.get("period", {}).get("number"),
            "clock_value": play.get("clock", {}).get("value"),
            "scoring_play": play.get("scoringPlay"),
            "shooting_play": play.get("shootingPlay"),
            "score_value": play.get("scoreValue"),
            "valid": play.get("valid"),
            "priority": play.get("priority"),
            "modified": play.get("modified"),
        }

        # Fetch and unpack team info
        team_ref = play.get("team", {}).get("$ref")
        if team_ref:
            if team_ref not in team_cache:
                try:
                    team_data = get_json(team_ref)
                    team_cache[team_ref] = {
                        "team_id": team_data.get("id"),
                        # Use location instead of displayName
                        "team_name": team_data.get("location"),
                        "team_abbrev": team_data.get("abbreviation"),
                    }
                except Exception as e:
                    print(f"Warning: could not fetch team info for {team_ref}: {e}")
                    team_cache[team_ref] = {
                        "team_id": None,
                        "team_name": None,
                        "team_abbrev": None,
                    }
            row.update(team_cache[team_ref])
        else:
            row.update({
                "team_id": None,
                "team_name": None,
                "team_abbrev": None,
            })

        rows.append(row)

    df = pd.DataFrame(rows)

    # Ensure boolean fields remain native bools (fill nulls with False)
    bool_cols = ["scoring_play", "shooting_play"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)

    # ------------------------------------------------------------
    # Assert dtype integrity for pipeline-critical fields
    # ------------------------------------------------------------
    for critical_col in ["scoring_play", "shooting_play"]:
        assert df[critical_col].dtype == bool, (
            f"{critical_col} must be bool dtype, found {df[critical_col].dtype}"
        )

    # Add game_id
    df.insert(0, "game_id", game_id)

    # ------------------------------------------------------------
    # Infer home/away team names and IDs
    # ------------------------------------------------------------
    home_team_id = away_team_id = home_team_name = away_team_name = None

    # Sort by period ascending, clock descending (chronological order)
    df_sorted = df.sort_values(["period_number", "clock_value"], ascending=[True, False])

    # Find first scoring play
    first_score = df_sorted[df_sorted["scoring_play"]].head(1)
    if not first_score.empty:
        scorer = first_score.iloc[0]
        scorer_team_id = scorer["team_id"]
        scorer_team_name = scorer["team_name"]

        # Determine away/home based on score values
        if scorer["away_score"] > scorer["home_score"]:
            away_team_id, away_team_name = scorer_team_id, scorer_team_name
        elif scorer["home_score"] > scorer["away_score"]:
            home_team_id, home_team_name = scorer_team_id, scorer_team_name

        # Identify the other team in the dataset
        other_teams = df["team_id"].dropna().unique().tolist()
        if len(other_teams) > 1:
            for tid in other_teams:
                if tid != scorer_team_id:
                    other_team_name = df.loc[df["team_id"] == tid, "team_name"].iloc[0]
                    if away_team_id:
                        home_team_id, home_team_name = tid, other_team_name
                    elif home_team_id:
                        away_team_id, away_team_name = tid, other_team_name
    else:
        print("No scoring plays yet — home/away inference skipped.")

    # Attach inferred values (same for all rows)
    df["away_team_id"] = away_team_id
    df["away_team_name"] = away_team_name
    df["home_team_id"] = home_team_id
    df["home_team_name"] = home_team_name

    # Final column ordering
    df = df[[
        "game_id",
        "id",
        "sequence_number",
        "type_id",
        "type_text",
        "text",
        "away_score",
        "home_score",
        "period_number",
        "clock_value",
        "scoring_play",
        "shooting_play",
        "score_value",
        "valid",
        "priority",
        "modified",
        "team_id",
        "team_name",       # based on team_location
        "team_abbrev",
        "away_team_id",
        "away_team_name",
        "home_team_id",
        "home_team_name",
    ]]

    return df


if __name__ == "__main__":
    game_id = 401700174
    df = get_pbp(game_id)
    print(df.head(10))
    print(f"\nTotal plays parsed: {len(df)}")

    os.makedirs("data", exist_ok=True)
    out_path = f"data/pbp_{game_id}.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved sample to {out_path}")

