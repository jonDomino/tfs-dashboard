"""Combined visualization plots"""
import matplotlib.pyplot as plt
import pandas as pd
from app.plots.tempo import build_tempo_figure
from app.plots.score_diff import build_score_diff_figure
from typing import Optional


def build_combined_figure(
    tfs_df: pd.DataFrame,
    game_id: str,
    show_score_diff: bool = True,
    show_predictions: bool = False
) -> plt.Figure:
    """Build combined tempo and score differential figure.
    
    Args:
        tfs_df: TFS DataFrame
        game_id: Game identifier
        show_score_diff: Whether to include score differential subplot
        show_predictions: Whether to show predicted TFS
        
    Returns:
        Matplotlib figure with subplots
    """
    if show_score_diff:
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        ax_tempo, ax_score = axes
    else:
        fig, ax_tempo = plt.subplots(figsize=(8, 4))
        ax_score = None
    
    # Build tempo plot (we'll need to extract the logic or call it differently)
    tempo_fig = build_tempo_figure(tfs_df, game_id, show_predictions=show_predictions)
    
    # For now, just return the tempo figure
    # TODO: Implement proper subplot combination
    return tempo_fig

