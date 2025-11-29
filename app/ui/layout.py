"""Layout utilities"""
import streamlit as st
from typing import List, Callable, Any
from app.config import config


def create_game_grid(game_ids: List[str], cols_per_row: int = config.COLS_PER_ROW) -> List[List[str]]:
    """Create grid layout for games.
    
    Args:
        game_ids: List of game IDs
        cols_per_row: Number of columns per row
        
    Returns:
        List of rows, each containing game IDs
    """
    return [
        game_ids[i:i + cols_per_row]
        for i in range(0, len(game_ids), cols_per_row)
    ]


def render_game_grid(
    game_ids: List[str],
    render_func: Callable[[str], Any],
    cols_per_row: int = config.COLS_PER_ROW
):
    """Render games in a grid layout.
    
    Args:
        game_ids: List of game IDs
        render_func: Function that takes game_id and renders content
        cols_per_row: Number of columns per row
    """
    rows = create_game_grid(game_ids, cols_per_row)
    
    for row in rows:
        columns = st.columns(len(row))
        for gid, col in zip(row, columns):
            with col:
                render_func(gid)

