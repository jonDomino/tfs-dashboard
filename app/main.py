"""Main Streamlit application entry point"""
import streamlit as st
from datetime import datetime, date, timedelta
from typing import List, Dict, Set, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.data.schedule_loader import load_schedule
from app.data.pbp_loader import load_pbp
from app.data.status import classify_game_status_pbp, GameStatus
from app.data.bigquery_loader import get_closing_totals
from app.data.efg import calculate_efg_by_half
from app.tfs.preprocess import preprocess_pbp
from app.tfs.compute import compute_tfs
from app.ui.selectors import date_selector, game_selector, status_filter, board_filter
from app.ui.renderer import render_chart, render_error, render_warning, render_info, render_badge
from app.ui.layout import render_game_grid, create_game_grid
from app.plots.tempo import build_tempo_figure
from app.util.time import setup_refresh_timer
from app.config import config


def should_scan_game(game_id: str) -> bool:
    """Determine if we should scan a game for new PBP data.
    
    Rules:
    - Completed games: Never scan again
    - Not Started games: Only scan if 10 minutes have passed since last check
    
    Args:
        game_id: Game identifier
        
    Returns:
        True if we should scan, False otherwise
    """
    # Initialize session state tracking if needed
    if 'completed_games' not in st.session_state:
        st.session_state.completed_games: Set[str] = set()
    
    if 'not_started_last_check' not in st.session_state:
        st.session_state.not_started_last_check: Dict[str, datetime] = {}
    
    # Never scan completed games
    if game_id in st.session_state.completed_games:
        return False
    
    # Check if this game was previously "Not Started"
    if game_id in st.session_state.not_started_last_check:
        last_check = st.session_state.not_started_last_check[game_id]
        time_since_check = datetime.now() - last_check
        
        # Only scan if configured throttle time has passed
        if time_since_check < timedelta(minutes=config.NOT_STARTED_THROTTLE_MINUTES):
            return False
    
    return True


def _scan_single_game(game_id: str, now: datetime) -> Tuple[str, str, bool]:
    """Scan a single game and return its status.
    
    Helper function for parallel processing.
    Returns status and tracking info, but doesn't update session_state
    (that's done in the main thread for thread safety).
    
    Uses direct get_pbp call to avoid Streamlit context warnings in threads.
    
    Args:
        game_id: Game identifier
        now: Current datetime for throttling
        
    Returns:
        Tuple of (game_id, status, is_not_started)
    """
    try:
        # Import here to avoid circular imports
        from app.data.get_pbp import get_pbp
        
        # Call get_pbp directly instead of load_pbp to avoid Streamlit context issues
        # The caching will still work at the get_pbp level if it has its own caching
        raw_pbp = get_pbp(int(game_id))
        if raw_pbp is None or len(raw_pbp) == 0:
            return game_id, "Not Started", True
        
        status = classify_game_status_pbp(raw_pbp)
        is_not_started = (status == "Not Started")
        return game_id, status, is_not_started
    except Exception as e:
        # If we can't load PBP, assume "Not Started" and throttle
        return game_id, "Not Started", True


@st.cache_data(ttl=config.CACHE_TTL_STATUS)  # Cache status for configured duration
def get_game_statuses(game_ids: List[str]) -> Dict[str, str]:
    """Get game statuses for a list of game IDs using parallel processing.
    
    Efficient scanning:
    - Skips completed games (never scans again)
    - Throttles "Not Started" games (only checks every 10 minutes)
    - Uses parallel processing for games that need scanning
    
    Args:
        game_ids: List of game ID strings
        
    Returns:
        Dictionary mapping game_id to status
    """
    # Initialize session state if needed
    if 'completed_games' not in st.session_state:
        st.session_state.completed_games: Set[str] = set()
    
    if 'not_started_last_check' not in st.session_state:
        st.session_state.not_started_last_check: Dict[str, datetime] = {}
    
    statuses = {}
    now = datetime.now()
    
    # Separate games that need scanning from those that don't
    games_to_scan = []
    skipped_games = []
    
    for game_id in game_ids:
        if not should_scan_game(game_id):
            # Use cached/last known status without making API call
            if game_id in st.session_state.completed_games:
                statuses[game_id] = "Complete"
            elif game_id in st.session_state.not_started_last_check:
                statuses[game_id] = "Not Started"
            else:
                statuses[game_id] = "Not Started"
        else:
            games_to_scan.append(game_id)
    
    # Process games that need scanning in parallel
    if games_to_scan:
        # Use ThreadPoolExecutor for I/O-bound operations (API calls)
        # Max workers: configured to avoid overwhelming the API
        with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
            # Submit all tasks
            future_to_game = {
                executor.submit(_scan_single_game, game_id, now): game_id
                for game_id in games_to_scan
            }
            
            # Collect results as they complete
            completed_games_to_add = set()
            not_started_to_track = {}
            
            for future in as_completed(future_to_game):
                try:
                    game_id, status, is_not_started = future.result()
                    statuses[game_id] = status
                    
                    # Collect tracking info (update session_state in main thread)
                    if status == "Complete":
                        completed_games_to_add.add(game_id)
                    elif is_not_started:
                        not_started_to_track[game_id] = now
                except Exception as e:
                    # Handle any errors from parallel execution
                    game_id = future_to_game[future]
                    statuses[game_id] = "Not Started"
                    not_started_to_track[game_id] = now
            
            # Update session_state in main thread (thread-safe)
            if completed_games_to_add:
                st.session_state.completed_games.update(completed_games_to_add)
            
            if not_started_to_track:
                st.session_state.not_started_last_check.update(not_started_to_track)
            
            # Remove games that started from throttling
            for game_id in games_to_scan:
                if game_id in st.session_state.not_started_last_check and statuses.get(game_id) != "Not Started":
                    st.session_state.not_started_last_check.pop(game_id, None)
    
    return statuses


def filter_games_by_status(game_ids: List[str], selected_statuses: List[str]) -> List[str]:
    """Filter game IDs based on selected statuses.
    
    Args:
        game_ids: List of game ID strings
        selected_statuses: List of selected status filter strings
        
    Returns:
        Filtered list of game IDs
    """
    if not selected_statuses:
        return game_ids
    
    # Handle "Live Only" special case
    if "Live Only" in selected_statuses:
        # "Live Only" means First Half or Second Half
        if "First Half" not in selected_statuses:
            selected_statuses.append("First Half")
        if "Second Half" not in selected_statuses:
            selected_statuses.append("Second Half")
        # Remove "Live Only" from the list since we've expanded it
        selected_statuses = [s for s in selected_statuses if s != "Live Only"]
    
    # Get statuses for all games
    statuses = get_game_statuses(game_ids)
    
    # Filter games that match selected statuses
    filtered_ids = [
        game_id for game_id in game_ids
        if statuses.get(game_id, "Not Started") in selected_statuses
    ]
    
    return filtered_ids


def process_game(game_id: str):
    """Process a single game and return TFS DataFrame and raw PBP.
    
    Args:
        game_id: Game identifier
        
    Returns:
        Tuple of (TFS DataFrame, raw PBP DataFrame)
    """
    raw_pbp = load_pbp(game_id)
    df = preprocess_pbp(raw_pbp)
    tfs_df = compute_tfs(df)
    return tfs_df, raw_pbp


@st.cache_data(ttl=config.CACHE_TTL_STATUS)
def get_game_data(game_id: str):
    """Get processed game data with caching.
    
    Efficient scanning:
    - Skips completed games (uses cached data only)
    - Throttles "Not Started" games
    
    Args:
        game_id: Game identifier
        
    Returns:
        Tuple of (tfs_df, raw_pbp, status, efg_first_half, efg_second_half) or None if error
    """
    # Check if we should scan this game
    if not should_scan_game(game_id):
        # For completed games, try to use cached data
        if game_id in st.session_state.get('completed_games', set()):
            # Try to load from cache (won't make new API call)
            try:
                raw_pbp = load_pbp(game_id, use_cache=True)
                df = preprocess_pbp(raw_pbp)
                tfs_df = compute_tfs(df)
                status = "Complete"
                # Calculate eFG%
                efg_1h, efg_2h = calculate_efg_by_half(raw_pbp)
                return tfs_df, raw_pbp, status, efg_1h, efg_2h
            except:
                # If cache miss, return None (game is complete, no new data)
                return None, None, None, None, None
    
    try:
        raw_pbp = load_pbp(game_id)
        df = preprocess_pbp(raw_pbp)
        tfs_df = compute_tfs(df)
        status = classify_game_status_pbp(raw_pbp)
        
        # Calculate eFG% for both halves
        efg_1h, efg_2h = calculate_efg_by_half(raw_pbp)
        
        # Track completed games
        if status == "Complete":
            if 'completed_games' not in st.session_state:
                st.session_state.completed_games: Set[str] = set()
            st.session_state.completed_games.add(game_id)
        
        return tfs_df, raw_pbp, status, efg_1h, efg_2h
    except Exception as e:
        # Log error for debugging but don't crash the app
        import traceback
        if 'error_log' not in st.session_state:
            st.session_state.error_log = []
        st.session_state.error_log.append(f"Game {game_id}: {str(e)}")
        # Keep only last 10 errors
        if len(st.session_state.error_log) > 10:
            st.session_state.error_log = st.session_state.error_log[-10:]
        return None, None, None, None, None


def render_game(
    game_id: str, 
    closing_totals: Dict[str, float] = None, 
    rotation_number: Optional[int] = None,
    lookahead_2h_total: Optional[float] = None,
    closing_spread_home: Optional[float] = None,
    home_team_name: Optional[str] = None,
    opening_2h_total: Optional[float] = None,
    closing_2h_total: Optional[float] = None,
    opening_2h_spread: Optional[float] = None,
    closing_2h_spread: Optional[float] = None
):
    """Render a single game's visualization.
    
    Args:
        game_id: Game identifier
        closing_totals: Dictionary mapping game_id to closing_total
        rotation_number: Away team rotation number (optional)
        lookahead_2h_total: Lookahead 2H total (optional)
        closing_spread_home: Closing spread from home team's perspective (optional)
        home_team_name: Home team name (optional)
        opening_2h_total: Opening 2H total (optional)
        closing_2h_total: Closing 2H total (optional)
        opening_2h_spread: Opening 2H spread (optional)
        closing_2h_spread: Closing 2H spread (optional)
    """
    st.markdown(f"**Game {game_id}**")
    
    # Get cached game data (this is fast due to caching)
    result = get_game_data(game_id)
    if result[0] is None:
        render_error(f"Error processing game {game_id}")
        return
    
    tfs_df, raw_pbp, status, efg_1h, efg_2h = result
    
    # Render status badge
    status_colors = {
        "Not Started": "gray",
        "Early 1H": "lightblue",
        "First Half": "blue",
        "Second Half": "green",
        "Halftime": "orange",
        "Complete": "red"
    }
    render_badge(status, status_colors.get(status, "blue"))
    
    # Get closing total for possession-level expected TFS calculation
    closing_total = None
    if closing_totals and game_id in closing_totals:
        closing_total = closing_totals[game_id]
    
    # Get lookahead 2H total and spread for this game
    lookahead_2h = lookahead_2h_total if lookahead_2h_total is not None else None
    spread_home = closing_spread_home if closing_spread_home is not None else None
    home_name = home_team_name if home_team_name is not None else None
    
    # Build and render figure with status label, expected TFS trend, and eFG%
    fig = build_tempo_figure(
        tfs_df, 
        game_id, 
        show_predictions=False, 
        game_status=status,
        closing_total=closing_total,
        efg_first_half=efg_1h,
        efg_second_half=efg_2h,
        rotation_number=rotation_number,
        lookahead_2h_total=lookahead_2h,
        closing_spread_home=spread_home,
        home_team_name=home_name
    )
    render_chart(fig)


def render():
    """Main render function - flicker-free pattern."""
    try:
        st.set_page_config(
            page_title="Live TFS Kernel Dashboard", 
            layout="wide",
            page_icon="🏀"  # Optional: add basketball emoji as icon
        )
    except Exception:
        # set_page_config can only be called once, ignore if already set
        pass
    
    # Always show title to prevent blank screen
    st.title("Live TFS Kernel Dashboard")
    
    try:
        _render_content()
    except Exception as e:
        # Catch any unhandled errors to prevent blank screen
        import traceback
        st.error("⚠️ An error occurred while rendering the dashboard")
        with st.expander("Error Details", expanded=True):
            st.exception(e)
            st.code(traceback.format_exc())
        
        # Show error log if available
        if 'error_log' in st.session_state and st.session_state.error_log:
            with st.expander("Recent Errors"):
                for err in st.session_state.error_log[-5:]:
                    st.text(err)
        
        # Add a refresh button
        if st.button("🔄 Refresh Dashboard"):
            st.rerun()


def _render_content():
    """Internal render function with actual content."""
    # Load schedule
    try:
        sched = load_schedule()
    except Exception as e:
        render_error(f"Error loading schedule: {e}")
        return
    
    if sched.empty:
        render_error("No schedule data available.")
        return
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Add possession start type legend in sidebar
    st.sidebar.markdown("### Possession Start Types")
    from app.util.style import get_poss_start_color
    
    poss_start_info = [
        ("Rebound", "rebound", "#d62728"),
        ("Turnover", "turnover", "#1f77b4"),
        ("Opp Made Shot", "oppo_made_shot", "#2ca02c"),
        ("Opp Made FT", "oppo_made_ft", "#ff7f0e"),
    ]
    
    for label, poss_type, color in poss_start_info:
        st.sidebar.markdown(
            f'<span style="display: inline-block; width: 20px; height: 20px; background-color: {color}; border-radius: 3px; margin-right: 8px; vertical-align: middle;"></span>'
            f'<span style="vertical-align: middle;">{label}</span>',
            unsafe_allow_html=True
        )
    
    st.sidebar.markdown("---")
    
    selected_date = date_selector()
    selected_boards = board_filter()
    
    # Get all games for the selected date
    game_ids = game_selector(sched, selected_date, auto_select_all=True)
    
    if not game_ids:
        render_warning(f"No games selected or available for {selected_date}")
        return
    
    # Get statuses for all games (cached)
    with st.spinner("Loading game statuses..."):
        statuses = get_game_statuses(game_ids)
    
    # Fetch closing totals for all games (cached, runs once)
    closing_totals_raw = {}
    if game_ids:
        try:
            closing_totals_raw = get_closing_totals(game_ids)
            print(f"DEBUG: get_closing_totals returned {len(closing_totals_raw)} games for {len(game_ids)} requested")
        except Exception as e:
            # Fail gracefully if BigQuery fails
            import traceback
            print(f"ERROR: get_closing_totals failed: {e}")
            print(traceback.format_exc())
            closing_totals_raw = {}
    
    # Filter by board and build closing_totals dict (just closing_total values) and rotation_numbers dict
    closing_totals = {}
    rotation_numbers = {}
    lookahead_2h_totals = {}
    closing_spread_home = {}
    home_team_names = {}  # Store home team names for spread display
    opening_2h_totals = {}
    closing_2h_totals = {}
    opening_2h_spreads = {}
    closing_2h_spreads = {}
    if closing_totals_raw:
        for gid in game_ids:
            if gid in closing_totals_raw:
                try:
                    closing_total, board, rotation_number, closing_1h_total, lookahead_2h_total, spread_home, home_team_name, opening_2h_total, closing_2h_total, opening_2h_spread, closing_2h_spread = closing_totals_raw[gid]
                    # Ensure closing_total is a float
                    closing_total = float(closing_total)
                    # Only include if board matches filter
                    if board in selected_boards:
                        closing_totals[gid] = closing_total
                        if rotation_number is not None:
                            rotation_numbers[gid] = rotation_number
                        if lookahead_2h_total is not None:
                            lookahead_2h_totals[gid] = float(lookahead_2h_total)
                        if spread_home is not None:
                            closing_spread_home[gid] = float(spread_home)
                        if home_team_name:
                            home_team_names[gid] = home_team_name
                        if opening_2h_total is not None:
                            opening_2h_totals[gid] = float(opening_2h_total)
                        if closing_2h_total is not None:
                            closing_2h_totals[gid] = float(closing_2h_total)
                        if opening_2h_spread is not None:
                            opening_2h_spreads[gid] = float(opening_2h_spread)
                        if closing_2h_spread is not None:
                            closing_2h_spreads[gid] = float(closing_2h_spread)
                except Exception as e:
                    print(f"ERROR unpacking data for game {gid}: {e}")
                    import traceback
                    print(traceback.format_exc())
    
    print(f"DEBUG: After filtering - closing_totals: {len(closing_totals)}, selected_boards: {selected_boards}")
    
    # Group games by status
    games_by_status: Dict[str, List[str]] = {
        "Early 1H": [],
        "First Half": [],
        "Halftime": [],
        "Second Half": [],
        "Complete": [],
        "Not Started": []
    }
    
    for gid in game_ids:
        # Only include games with closing totals if board filter is active
        if selected_boards and gid not in closing_totals:
            continue
        
        status = statuses.get(gid, "Not Started")
        if status in games_by_status:
            games_by_status[status].append(gid)
    
    # Create tabs for each status
    tab_names = []
    tab_games = []
    
    # Order for display: Early 1H, First Half, Halftime, Second Half, Complete, Not Started
    status_order = ["Early 1H", "First Half", "Halftime", "Second Half", "Complete", "Not Started"]
    
    # Build all tabs first
    all_tabs_data = []
    for status in status_order:
        games = games_by_status[status]
        if games:  # Only create tab if there are games
            # Sort games by rotation number descending (higher rotation numbers first)
            # Games without rotation numbers go to the end
            games_sorted = sorted(
                games,
                key=lambda gid: (rotation_numbers.get(gid) is None, rotation_numbers.get(gid) or 0),
                reverse=True
            )
            all_tabs_data.append({
                'status': status,
                'name': f"{status} ({len(games)})",
                'games': games_sorted
            })
    
    if not all_tabs_data:
        render_info("No games available for the selected date and board filter.")
        return
    
    # Priority order for default tab selection (first available gets selected)
    default_priority = ["Halftime", "First Half", "Early 1H", "Second Half", "Complete"]
    
    # Find the index of the highest priority available tab
    default_tab_idx = 0
    for priority_status in default_priority:
        for idx, tab_data in enumerate(all_tabs_data):
            if tab_data['status'] == priority_status:
                default_tab_idx = idx
                break
        else:
            continue
        break  # Found the priority tab, stop searching
    
    # Reorder tabs so the default tab is first (Streamlit selects first tab by default)
    if default_tab_idx > 0:
        all_tabs_data = [all_tabs_data[default_tab_idx]] + \
                       [tab for idx, tab in enumerate(all_tabs_data) if idx != default_tab_idx]
    
    # Build final tab lists
    for tab_data in all_tabs_data:
        tab_names.append(tab_data['name'])
        tab_games.append((tab_data['status'], tab_data['games']))
    
    # Create tabs
    tabs = st.tabs(tab_names)
    
    # Initialize plot containers if needed
    if 'plot_containers' not in st.session_state:
        st.session_state.plot_containers = {}
    
    # Render games in each tab
    for tab_idx, (tab, (status, game_list)) in enumerate(zip(tabs, tab_games)):
        with tab:
            if not game_list:
                st.info(f"No {status} games available.")
                continue
            
            # Render games in grid
            rows = create_game_grid(game_list, cols_per_row=config.COLS_PER_ROW)
            
            for row_idx, row in enumerate(rows):
                columns = st.columns(len(row))
                for col_idx, gid in enumerate(row):
                    with columns[col_idx]:
                        # Create unique key for this game's container
                        container_key = f"{selected_date}_{status}_{gid}_{row_idx}_{col_idx}"
                        
                        # Get or create empty container (persists across refreshes)
                        if container_key not in st.session_state.plot_containers:
                            st.session_state.plot_containers[container_key] = st.empty()
                        
                        # Render directly into the container (atomic update - no gray out)
                        try:
                            with st.session_state.plot_containers[container_key]:
                                rotation_number = rotation_numbers.get(gid)
                                lookahead_2h = lookahead_2h_totals.get(gid)
                                spread_home = closing_spread_home.get(gid)
                                home_name = home_team_names.get(gid)
                                opening_2h_t = opening_2h_totals.get(gid)
                                closing_2h_t = closing_2h_totals.get(gid)
                                opening_2h_s = opening_2h_spreads.get(gid)
                                closing_2h_s = closing_2h_spreads.get(gid)
                                render_game(
                                    gid, 
                                    closing_totals=closing_totals, 
                                    rotation_number=rotation_number,
                                    lookahead_2h_total=lookahead_2h,
                                    closing_spread_home=spread_home,
                                    home_team_name=home_name,
                                    opening_2h_total=opening_2h_t,
                                    closing_2h_total=closing_2h_t,
                                    opening_2h_spread=opening_2h_s,
                                    closing_2h_spread=closing_2h_s
                                )
                        except Exception as e:
                            # Log error but don't crash the whole dashboard
                            if 'error_log' not in st.session_state:
                                st.session_state.error_log = []
                            st.session_state.error_log.append(f"Game {gid} render error: {str(e)}")
                            with st.session_state.plot_containers[container_key]:
                                st.error(f"Error rendering game {gid}")


if __name__ == "__main__":
    render()
    setup_refresh_timer(config.REFRESH_INTERVAL)

