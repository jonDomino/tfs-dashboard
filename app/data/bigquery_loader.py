"""BigQuery data loading for closing totals"""
import json
import os
import pandas as pd
import streamlit as st
from datetime import datetime, time
from google.cloud import bigquery
from typing import Dict, Optional, Tuple
from app.config import config


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
def _get_closing_totals_internal(game_ids: list) -> Dict[str, Tuple[float, str, Optional[int]]]:
    """Internal function to fetch closing totals from BigQuery.
    
    This is cached for 1 hour. The wrapper function handles time restrictions.
    
    Args:
        game_ids: List of game ID strings
        
    Returns:
        Dictionary mapping game_id to (closing_total, board, rotation_number)
    """
    
    # Load credentials - try multiple methods
    creds_path = None
    client = None
    
    # Method 1: Check Streamlit secrets (for Streamlit Cloud)
    try:
        if hasattr(st, 'secrets') and 'bq_credentials' in st.secrets:
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
        else:
            raise AttributeError("No Streamlit secrets")
    except (AttributeError, KeyError):
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
            return {}
    
    try:
        
        # Convert game_ids to string for SQL
        game_ids_str = ",".join([f"'{gid}'" for gid in game_ids])
        
        query = f"""
        WITH last_two_days AS (
          SELECT *
          FROM `meatloaf-427522.markets.pregame`
          WHERE book = 'Unabated'
            AND betType = 3               -- totals
            AND market = 'total'
            AND DATE(modifiedOn) >= DATE_SUB(CURRENT_DATE(), INTERVAL 2 DAY)
        ),
        
        ranked AS (
          SELECT
            p.eventId,
            p.eventStart,
            p.awayTeamName,
            p.homeTeamName,
            p.away_rotationNumber,
            p.points AS closing_total,
            p.modifiedOn,
            ROW_NUMBER() OVER (
              PARTITION BY p.eventId
              ORDER BY p.modifiedOn DESC
            ) AS rn
          FROM last_two_days p
        )
        
        SELECT
          x.game_id,
          r.eventId,
          r.eventStart,
          r.awayTeamName,
          r.homeTeamName,
          r.away_rotationNumber,
          CASE 
            WHEN r.away_rotationNumber < 1000 THEN 'main'
            ELSE 'extra'
          END AS board,
          r.closing_total,
          r.modifiedOn AS closing_timestamp
        FROM ranked r
        LEFT JOIN `meatloaf-427522.cbb_2025.xref_games` x
          ON r.eventId = x.event_id
        WHERE r.rn = 1
          AND x.game_id IN ({game_ids_str})
        ORDER BY r.modifiedOn DESC
        """
        
        # Execute query
        query_job = client.query(query)
        results = query_job.result()
        
        # Build dictionary with (closing_total, board, rotation_number) tuples
        closing_totals = {}
        for row in results:
            if row.game_id and row.closing_total is not None:
                # Ensure closing_total is a float
                closing_total = float(row.closing_total)
                board = str(row.board) if row.board else 'main'
                rotation_number = int(row.away_rotationNumber) if row.away_rotationNumber is not None else None
                closing_totals[str(row.game_id)] = (closing_total, board, rotation_number)
        
        return closing_totals
        
    except Exception as e:
        # Return empty dict on error (fail gracefully)
        return {}


def get_closing_totals(game_ids: list) -> Dict[str, Tuple[float, str, Optional[int]]]:
    """Get closing totals, board info, and rotation numbers for a list of game IDs from BigQuery.
    
    Only runs query once per hour (via cache) and skips between 10pm-8am.
    During off-hours, returns cached data if available, otherwise empty dict.
    
    Args:
        game_ids: List of game ID strings
        
    Returns:
        Dictionary mapping game_id to (closing_total, board, rotation_number)
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


def calculate_expected_tfs(closing_total: float, poss_start_type: Optional[str] = None) -> float:
    """Calculate expected TFS from closing total and possession start type.
    
    If poss_start_type is provided, uses possession-specific formulas:
    - OPPO_MADE_SHOT: 36.4397 + -0.1025 * closing_total
    - REBOUND: 24.2977 + -0.0692 * closing_total
    - TURNOVER: 23.9754 + -0.0619 * closing_total
    
    Otherwise, uses the old game-level formula (for backward compatibility):
    - exp_tfs = 27.65 - 0.08 * closing_total
    
    Args:
        closing_total: Closing total from betting market (will be converted to float)
        poss_start_type: Possession start type (optional)
        
    Returns:
        Expected TFS value
    """
    # Ensure closing_total is a float
    closing_total = float(closing_total)
    
    # Use possession-specific formulas if poss_start_type is provided
    if poss_start_type:
        poss_start_type = str(poss_start_type).lower()
        
        if poss_start_type == "oppo_made_shot":
            return 36.4397 + (-0.1025 * closing_total)
        elif poss_start_type == "rebound":
            return 24.2977 + (-0.0692 * closing_total)
        elif poss_start_type == "turnover":
            return 23.9754 + (-0.0619 * closing_total)
        # For period_start or other types, use rebound formula as default
        elif poss_start_type == "period_start":
            return 24.2977 + (-0.0692 * closing_total)
        else:
            # Unknown type, use rebound as default
            return 24.2977 + (-0.0692 * closing_total)
    
    # Fallback to old game-level formula if no poss_start_type
    return 27.65 - 0.08 * closing_total

