"""Score differential visualization"""
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional
from app.util.style import get_plot_style, get_color


def build_score_diff_figure(
    tfs_df: pd.DataFrame,
    game_id: str
) -> Optional[plt.Figure]:
    """Build score differential bar chart.
    
    Args:
        tfs_df: TFS DataFrame (should have score columns)
        game_id: Game identifier
        
    Returns:
        Matplotlib figure or None if score data unavailable
    """
    # Check if we have score information
    score_cols = ["away_score", "home_score", "score_differential"]
    if not any(col in tfs_df.columns for col in score_cols):
        return None
    
    # Try to compute score differential if not present
    if "score_differential" not in tfs_df.columns:
        if "away_score" in tfs_df.columns and "home_score" in tfs_df.columns:
            tfs_df = tfs_df.copy()
            tfs_df["score_differential"] = tfs_df["away_score"] - tfs_df["home_score"]
        else:
            return None
    
    x = tfs_df["chrono_index"].values
    y = tfs_df["score_differential"].values
    
    style = get_plot_style()
    
    fig, ax = plt.subplots(figsize=(style["figsize"][0], style["figsize"][1] * 0.6))
    
    # Color bars by sign
    colors = [get_color("success") if val > 0 else get_color("danger") if val < 0 else get_color("info") for val in y]
    
    ax.bar(x, y, color=colors, alpha=0.6, width=0.8)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    
    ax.set_title(f"Game {game_id}\nScore Differential", fontsize=style["fontsize_title"])
    ax.set_xlabel("Possession")
    ax.set_ylabel("Score Differential (Away - Home)")
    
    fig.tight_layout()
    return fig

