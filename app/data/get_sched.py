"""
get_sched.py
-------------
Fetch Men's College Basketball schedule data from ESPN's hidden API.

Outputs:
    game_id, game_date, game_date_time,
    away_team_id, away_team,
    home_team_id, home_team

Notes:
    - Uses the team 'location' field (e.g., "Duke") instead of 'displayName' ("Duke Blue Devils")
    - Automatically pulls data for date range: (today - 2 days) → (today + 3 days)
    - Uses PST (UTC-8) for date calculations
"""

import requests
import pandas as pd
from datetime import datetime, timedelta, timezone


def get_sched():
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"
    rows = []

    # === Dynamic date window ===
    # Use PST (UTC-8) for date calculations
    pst = timezone(timedelta(hours=-8))
    today = datetime.now(pst).date()
    start = today - timedelta(days=2)
    end = today + timedelta(days=3)

    while start <= end:
        datestr = start.strftime("%Y%m%d")
        params = {"dates": datestr, "groups": "50", "limit": "500"}
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"Failed to fetch {datestr}: {e}")
            start += timedelta(days=1)
            continue

        for e in data.get("events", []):
            game_id = e.get("id")
            game_date_time = e.get("date")
            
            # Convert game_date_time to PST and extract date
            if game_date_time:
                # Parse the datetime string (ESPN provides ISO format, typically UTC)
                try:
                    # Parse as UTC first (ESPN dates are typically in UTC)
                    dt_utc = pd.to_datetime(game_date_time, utc=True)
                    # Convert to PST (UTC-8)
                    dt_pst = dt_utc.tz_convert(pst)
                    # Extract date in PST
                    game_date = dt_pst.date()
                except Exception:
                    # Fallback: just use the date part if parsing fails
                    game_date = game_date_time.split("T")[0] if game_date_time else None
            else:
                game_date = None

            away_team = home_team = None
            away_team_id = home_team_id = None

            competitions = e.get("competitions", [])
            if competitions:
                comps = competitions[0].get("competitors", [])
                for c in comps:
                    side = c.get("homeAway")
                    team = c.get("team", {})
                    team_name = team.get("location")   # use location field
                    team_id = team.get("id")

                    if side == "home":
                        home_team = team_name
                        home_team_id = team_id
                    elif side == "away":
                        away_team = team_name
                        away_team_id = team_id

            rows.append({
                "game_id": game_id,
                "game_date": game_date,
                "game_date_time": game_date_time,
                "away_team_id": away_team_id,
                "home_team_id": home_team_id,
                "away_team": away_team,
                "home_team": home_team,
            })

        start += timedelta(days=1)

    # === Build DataFrame ===
    df = pd.DataFrame(rows).drop_duplicates(subset=["game_id"])
    
    # Convert game_date to datetime for sorting
    # game_date is already a date object (PST date), convert to datetime for DataFrame operations
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    
    df = df.sort_values("game_date").reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = get_sched()
    print(df)

