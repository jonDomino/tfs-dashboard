"""BigQuery data loading for closing totals"""
import json
import os
import pandas as pd
import streamlit as st
from datetime import datetime, time
from google.cloud import bigquery
from typing import Dict, Optional, Tuple
from app.config import config

try:
    from streamlit.errors import StreamlitSecretNotFoundError
except ImportError:
    # Fallback for older Streamlit versions
    StreamlitSecretNotFoundError = Exception


def should_run_query() -> bool:
    """Determine if we should run the BigQuery query.
    
    Rules:
    - Don't run between configured end hour and start hour
    - Otherwise, allow query (caching will handle once-per-hour)
    
    Returns:
        True if query should run, False otherwise
    """
    now = datetime.now()
    current_time = now.time()
    
    # Don't run between end hour and start hour
    if time(config.BQ_QUERY_END_HOUR, 0) <= current_time or current_time < time(config.BQ_QUERY_START_HOUR, 0):
        return False
    
    return True


@st.cache_data(ttl=config.CACHE_TTL_CLOSING_TOTALS)  # Cache for configured duration (closing totals don't change often)
def _get_closing_totals_internal(game_ids: list) -> Dict[str, Tuple[float, str, Optional[int], Optional[float], Optional[float], Optional[float], Optional[str]]]:
    """Internal function to fetch closing totals from BigQuery.
    
    This is cached for 1 hour. The wrapper function handles time restrictions.
    
    Args:
        game_ids: List of game ID strings
        
    Returns:
        Dictionary mapping game_id to (closing_total, board, rotation_number, closing_1h_total, lookahead_2h_total, closing_spread_home, home_team_name)
    """
    
    # Load credentials - try multiple methods
    creds_path = None
    client = None
    
    # Method 1: Check Streamlit secrets (for Streamlit Cloud)
    if hasattr(st, 'secrets'):
        try:
            # Try to access secrets - this will raise StreamlitSecretNotFoundError if no secrets file exists
            if 'bq_credentials' in st.secrets:
                # Create temporary JSON file from secrets
                import tempfile
                import json as json_lib
                
                creds_dict = dict(st.secrets.bq_credentials)
                # Handle JSON string if stored that way
                if 'json_content' in creds_dict:
                    creds_dict = json_lib.loads(creds_dict['json_content'])
                
                # Write to temp file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json_lib.dump(creds_dict, f)
                    creds_path = f.name
                
                client = bigquery.Client.from_service_account_json(creds_path)
        except StreamlitSecretNotFoundError:
            # No secrets.toml file exists locally - fall through to next method
            pass
        except (KeyError, AttributeError):
            # bq_credentials not in secrets - fall through to next method
            pass
        except Exception:
            # Other errors accessing secrets - fall through to next method
            pass
    
    # Method 2: Check environment variable or default location (if Method 1 didn't work)
    if client is None:
        # Method 2: Check environment variable
        env_creds_path = os.getenv("BIGQUERY_CREDENTIALS_PATH", None)
        if env_creds_path and os.path.exists(env_creds_path):
            creds_path = env_creds_path
        
        # Method 3: Check default location
        if not creds_path:
            if os.path.exists("meatloaf.json"):
                creds_path = "meatloaf.json"
        
        # Initialize BigQuery client
        try:
            if creds_path:
                # Use service account JSON file
                client = bigquery.Client.from_service_account_json(creds_path)
            else:
                # Try using Application Default Credentials (if gcloud is configured)
                client = bigquery.Client()
        except Exception as e:
            # Log credential loading error for debugging
            import traceback
            print(f"ERROR: Failed to load BigQuery credentials: {e}")
            print(traceback.format_exc())
            return {}
    
    if client is None:
        print("ERROR: BigQuery client is None - credentials not loaded")
        return {}
    
    try:
        
        # Convert game_ids to string for SQL
        game_ids_str = ",".join([f"'{gid}'" for gid in game_ids])
        
        query = f"""
        WITH last_two_days_pg AS (
          SELECT *
          FROM `meatloaf-427522.markets.pregame`
          WHERE book IN ('Unabated', 'Bookmaker')
            AND DATE(modifiedOn) >= DATE_SUB(CURRENT_DATE(), INTERVAL 2 DAY)
        ),
        
        last_two_days_1h AS (
          SELECT *
          FROM `meatloaf-427522.markets.first_half`
          WHERE book IN ('Unabated', 'Bookmaker')
            AND DATE(modifiedOn) >= DATE_SUB(CURRENT_DATE(), INTERVAL 2 DAY)
        ),
        
        -- ============================================================
        -- FULL GAME TOTALS (UNABADED FIRST, THEN BOOKMAKER)
        -- ============================================================
        full_totals_ranked AS (
          SELECT
            eventId,
            book,
            points AS closing_total,
            modifiedOn,
            ROW_NUMBER() OVER (
              PARTITION BY eventId, book
              ORDER BY modifiedOn DESC
            ) AS rn
          FROM last_two_days_pg
          WHERE betType = 3
            AND market = 'total'
        ),
        
        full_totals AS (
          SELECT
            eventId,
            -- Prefer Unabated; fallback to Bookmaker
            COALESCE(
              MAX(IF(book = 'Unabated'  AND rn = 1, closing_total, NULL)),
              MAX(IF(book = 'Bookmaker' AND rn = 1, closing_total, NULL))
            ) AS closing_total
          FROM full_totals_ranked
          GROUP BY eventId
        ),
        
        -- ============================================================
        -- FIRST HALF TOTALS (UNABADED FIRST, THEN BOOKMAKER)
        -- ============================================================
        first_half_totals_ranked AS (
          SELECT
            eventId,
            book,
            points AS closing_1h_total,
            modifiedOn,
            ROW_NUMBER() OVER (
              PARTITION BY eventId, book
              ORDER BY modifiedOn DESC
            ) AS rn
          FROM last_two_days_1h
          WHERE betType = 3
            AND market = 'total'
        ),
        
        first_half_totals AS (
          SELECT
            eventId,
            -- Prefer Unabated; fallback to Bookmaker
            COALESCE(
              MAX(IF(book = 'Unabated'  AND rn = 1, closing_1h_total, NULL)),
              MAX(IF(book = 'Bookmaker' AND rn = 1, closing_1h_total, NULL))
            ) AS closing_1h_total
          FROM first_half_totals_ranked
          GROUP BY eventId
        ),
        
        -- ============================================================
        -- SPREADS (HOME POV, UNABADED FIRST, THEN BOOKMAKER)
        -- ============================================================
        closing_home_spread_ranked AS (
          SELECT
            eventId,
            book,
            points AS closing_spread_home,
            modifiedOn,
            ROW_NUMBER() OVER (
              PARTITION BY eventId, book
              ORDER BY modifiedOn DESC
            ) AS rn
          FROM last_two_days_pg
          WHERE betType = 2
            AND market = 'spread'
            AND side = 'si1'     -- home POV
        ),
        
        closing_home_spread AS (
          SELECT
            eventId,
            -- Prefer Unabated; fallback to Bookmaker
            COALESCE(
              MAX(IF(book = 'Unabated'  AND rn = 1, closing_spread_home, NULL)),
              MAX(IF(book = 'Bookmaker' AND rn = 1, closing_spread_home, NULL))
            ) AS closing_spread_home
          FROM closing_home_spread_ranked
          GROUP BY eventId
        ),
        
        -- ============================================================
        -- EVENT META (from Unabated only)
        -- ============================================================
        event_meta AS (
          SELECT
            eventId,
            eventStart,
            awayTeamName,
            homeTeamName,
            away_rotationNumber,
            home_rotationNumber
          FROM last_two_days_pg
          WHERE book = 'Unabated'
          GROUP BY 1,2,3,4,5,6
        ),
        
        -- ============================================================
        -- FINAL OUTPUT
        -- ============================================================
        SELECT
          x.game_id,
          m.eventId,
          m.eventStart,
          m.awayTeamName,
          m.homeTeamName,
          m.away_rotationNumber,
          m.home_rotationNumber,
        
          CASE WHEN COALESCE(m.away_rotationNumber, 9999) < 1000 THEN 'main' ELSE 'extra' END AS board,
        
          ft.closing_total,
        
          -- ============================================
          -- CLOSING 1H TOTAL WITH FALLBACK RULE
          -- 1. Unabated
          -- 2. Bookmaker
          -- 3. Forced value = 47.2% of closing total (rounded to nearest half)
          -- ============================================
          CASE 
            WHEN ft.closing_total IS NOT NULL THEN
              COALESCE(
                fht.closing_1h_total,
                ROUND(ft.closing_total * 0.472 * 2) / 2
              )
            ELSE NULL
          END AS closing_1h_total,
        
          sp.closing_spread_home,
        
          -- lookahead 2H uses the same forced value (only calculate if closing_total exists)
          CASE 
            WHEN ft.closing_total IS NOT NULL THEN
              (ft.closing_total -
               COALESCE(
                 fht.closing_1h_total,
                 ROUND(ft.closing_total * 0.472 * 2) / 2
               )
              )
            ELSE NULL
          END AS lookahead_2h_total
        
        FROM `meatloaf-427522.cbb_2025.xref_games` x
        LEFT JOIN event_meta m
          ON x.event_id = m.eventId
        LEFT JOIN full_totals ft
          ON x.event_id = ft.eventId
        LEFT JOIN first_half_totals fht
          ON x.event_id = fht.eventId
        LEFT JOIN closing_home_spread sp
          ON x.event_id = sp.eventId
        WHERE x.game_id IN ({game_ids_str})
        ORDER BY COALESCE(m.eventStart, TIMESTAMP('1900-01-01')) DESC
        """
        
        # Execute query
        query_job = client.query(query)
        results = query_job.result()
        
        # Build dictionary with (closing_total, board, rotation_number, closing_1h_total, lookahead_2h_total, closing_spread_home, home_team_name) tuples
        closing_totals = {}
        row_count = 0
        skipped_count = 0
        for row in results:
            row_count += 1
            if not row.game_id:
                skipped_count += 1
                continue
            if row.closing_total is None:
                # Log but don't skip - we want to include games even without closing totals
                # They just won't have market data
                skipped_count += 1
                continue
            
            try:
                # Ensure closing_total is a float
                closing_total = float(row.closing_total)
                board = str(row.board) if row.board else 'main'
                rotation_number = int(row.away_rotationNumber) if row.away_rotationNumber is not None else None
                closing_1h_total = float(row.closing_1h_total) if row.closing_1h_total is not None else None
                lookahead_2h_total = float(row.lookahead_2h_total) if row.lookahead_2h_total is not None else None
                closing_spread_home = float(row.closing_spread_home) if row.closing_spread_home is not None else None
                home_team_name = str(row.homeTeamName) if row.homeTeamName else None
                closing_totals[str(row.game_id)] = (closing_total, board, rotation_number, closing_1h_total, lookahead_2h_total, closing_spread_home, home_team_name)
            except Exception as e:
                print(f"ERROR processing row for game_id {row.game_id}: {e}")
                skipped_count += 1
                continue
        
        print(f"BigQuery results: {row_count} rows processed, {len(closing_totals)} added to dict, {skipped_count} skipped")
        return closing_totals
        
    except Exception as e:
        # Log query error for debugging
        import traceback
        print(f"ERROR: BigQuery query failed: {e}")
        print(traceback.format_exc())
        # Return empty dict on error (fail gracefully)
        return {}


def get_closing_totals(game_ids: list) -> Dict[str, Tuple[float, str, Optional[int], Optional[float], Optional[float], Optional[float], Optional[str]]]:
    """Get closing totals, board info, rotation numbers, first half totals, lookahead 2H totals, home spreads, and home team names for a list of game IDs from BigQuery.
    
    Only runs query once per hour (via cache) and skips between 10pm-8am.
    During off-hours, returns cached data if available, otherwise empty dict.
    
    Args:
        game_ids: List of game ID strings
        
    Returns:
        Dictionary mapping game_id to (closing_total, board, rotation_number, closing_1h_total, lookahead_2h_total, closing_spread_home, home_team_name)
    """
    # Check if we should run the query based on time
    if not should_run_query():
        # During off-hours, try to use cached data
        # The cache key includes game_ids, so we need to call with the same signature
        # If cache is available, it will return it; otherwise returns empty
        try:
            return _get_closing_totals_internal(game_ids)
        except:
            return {}
    
    # During allowed hours, run the query (will use cache if available)
    return _get_closing_totals_internal(game_ids)


def calculate_expected_tfs(
    closing_total: float, 
    poss_start_type: Optional[str] = None,
    period_number: Optional[int] = None,
    score_diff: Optional[float] = None
) -> float:
    """Calculate expected TFS from closing total, possession start type, period, and score differential.
    
    Uses period-specific formulas:
    
    Period 1 formulas:
    - turnover: TFS = 23.4283 + -0.068865 * closing_total
    - rebound: TFS = 23.2206 + -0.070364 * closing_total
    - oppo_made_shot: TFS = 35.8503 + -0.105015 * closing_total
    - oppo_made_ft: TFS = 28.1118 + -0.065201 * closing_total
    
    Period 2 formulas (require score_diff):
    - turnover: TFS = 22.0475 + -0.057148 * closing_total + -0.061952 * score_diff
    - rebound: TFS = 24.2071 + -0.072452 * closing_total + -0.045162 * score_diff
    - oppo_made_shot: TFS = 35.0632 + -0.097778 * closing_total + -0.034749 * score_diff
    - oppo_made_ft: TFS = 29.7614 + -0.073256 * closing_total + -0.030282 * score_diff
    
    If period_number is not provided or is 1, uses Period 1 formulas.
    If period_number is 2 or greater and score_diff is provided, uses Period 2 formulas.
    Otherwise falls back to Period 1 formulas.
    
    Args:
        closing_total: Closing total from betting market (will be converted to float)
        poss_start_type: Possession start type (optional)
        period_number: Period number (1, 2, etc.) - optional, defaults to Period 1
        score_diff: Score differential at end of Period 1 (abs(away_score - home_score)) - optional, required for Period 2
        
    Returns:
        Expected TFS value
    """
    # Ensure closing_total is a float
    closing_total = float(closing_total)
    
    # Determine which period formulas to use
    # Default to Period 1 if period_number is None or 1
    use_period_2 = (period_number is not None and period_number >= 2 and score_diff is not None)
    
    # Use possession-specific formulas if poss_start_type is provided
    if poss_start_type:
        poss_start_type = str(poss_start_type).lower()
        
        if use_period_2:
            # Period 2 formulas (with score_diff)
            score_diff = float(score_diff)
            
            if poss_start_type == "turnover":
                return 22.0475 + (-0.057148 * closing_total) + (-0.061952 * score_diff)
            elif poss_start_type == "rebound":
                return 24.2071 + (-0.072452 * closing_total) + (-0.045162 * score_diff)
            elif poss_start_type == "oppo_made_shot":
                return 35.0632 + (-0.097778 * closing_total) + (-0.034749 * score_diff)
            elif poss_start_type == "oppo_made_ft":
                return 29.7614 + (-0.073256 * closing_total) + (-0.030282 * score_diff)
            else:
                # Unknown type, use rebound formula as default
                return 24.2071 + (-0.072452 * closing_total) + (-0.045162 * score_diff)
        else:
            # Period 1 formulas
            if poss_start_type == "turnover":
                return 23.4283 + (-0.068865 * closing_total)
            elif poss_start_type == "rebound":
                return 23.2206 + (-0.070364 * closing_total)
            elif poss_start_type == "oppo_made_shot":
                return 35.8503 + (-0.105015 * closing_total)
            elif poss_start_type == "oppo_made_ft":
                return 28.1118 + (-0.065201 * closing_total)
            else:
                # Unknown type, use rebound formula as default
                return 23.2206 + (-0.070364 * closing_total)
    
    # Fallback to old game-level formula if no poss_start_type
    return 27.65 - 0.08 * closing_total

