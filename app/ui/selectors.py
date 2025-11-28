"""UI selector components"""
import pandas as pd
import streamlit as st
from datetime import datetime, date, timezone, timedelta
from typing import List, Optional
from app.data.status import GameStatus


def date_selector(default_date: Optional[date] = None) -> date:
    """Create date selector widget.
    
    Args:
        default_date: Default date (defaults to today in PST)
        
    Returns:
        Selected date
    """
    if default_date is None:
        # Get current time in PST (UTC-8)
        pst = timezone(timedelta(hours=-8))
        default_date = datetime.now(pst).date()
    
    return st.sidebar.date_input("Game date", value=default_date)


def status_filter() -> List[GameStatus]:
    """Create status filter widget.
    
    Returns:
        List of selected statuses
    """
    options = ["Not Started", "Early 1H", "First Half", "Second Half", "Halftime", "Complete", "Live Only"]
    selected = st.sidebar.multiselect(
        "Game Status",
        options=options,
        default=["First Half", "Halftime"]  # Default to First Half and Halftime
    )
    return selected


def board_filter() -> List[str]:
    """Create board filter widget using checkboxes.
    
    Returns:
        List of selected boards (e.g., ["main", "extra"])
    """
    st.sidebar.markdown("**Board Filter**")
    main_checked = st.sidebar.checkbox("Main", value=True)
    extra_checked = st.sidebar.checkbox("Extra", value=False)
    
    selected = []
    if main_checked:
        selected.append("main")
    if extra_checked:
        selected.append("extra")
    
    return selected


def game_selector(sched: pd.DataFrame, selected_date: date, auto_select_all: bool = False) -> List[str]:
    """Create game selector widget.
    
    Args:
        sched: Schedule DataFrame
        selected_date: Selected date
        auto_select_all: If True, automatically select all games (for status filtering)
        
    Returns:
        List of selected game IDs
    """
    # Normalize dates
    sched["game_date"] = pd.to_datetime(sched["game_date"], errors="coerce")
    sched["game_date_only"] = sched["game_date"].dt.date
    
    # Filter by date
    day_games = sched[sched["game_date_only"] == selected_date].copy()
    
    if day_games.empty:
        return []
    
    # Create labels
    day_games["label"] = (
        day_games["away_team"] + " @ " + 
        day_games["home_team"] + " (" + 
        day_games["game_id"].astype(str) + ")"
    )
    
    labels = day_games["label"].tolist()
    
    if auto_select_all:
        # Auto-select all games when status filtering is active
        default = labels
    else:
        default = labels[:min(4, len(labels))]
    
    selected_labels = st.sidebar.multiselect(
        "Select games",
        options=labels,
        default=default,
    )
    
    if not selected_labels:
        return []
    
    # Map back to game IDs
    label_to_gid = dict(zip(day_games["label"], day_games["game_id"].astype(str)))
    game_ids = [label_to_gid[label] for label in selected_labels]
    
    return game_ids

