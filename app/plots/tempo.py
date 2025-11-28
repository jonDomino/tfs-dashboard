"""Tempo visualization plot"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from app.tfs.change_points import find_change_points
from app.tfs.segments import get_segment_lines
from app.util.kernel import gaussian_kernel_smoother
from app.util.style import get_plot_style, get_color, get_poss_start_color


def get_team_names(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """Extract team names from DataFrame.
    
    Args:
        df: DataFrame with team information
        
    Returns:
        Tuple of (away_team, home_team)
    """
    def try_get(fields):
        for f in fields:
            if f in df.columns and pd.notna(df[f].iloc[0]):
                return str(df[f].iloc[0])
        return None
    
    away = try_get(["away_team_name", "away_team"])
    home = try_get(["home_team_name", "home_team"])
    
    return away, home


def build_tempo_figure(
    tfs_df: pd.DataFrame,
    game_id: str,
    show_predictions: bool = False,
    game_status: Optional[str] = None,
    expected_tfs: Optional[float] = None,
    closing_total: Optional[float] = None,
    efg_first_half: Optional[float] = None,
    efg_second_half: Optional[float] = None
) -> plt.Figure:
    """Build tempo visualization figure.
    
    Args:
        tfs_df: TFS DataFrame
        game_id: Game identifier
        show_predictions: Whether to show predicted TFS
        game_status: Game status string
        expected_tfs: Expected TFS value (deprecated, use closing_total instead)
        closing_total: Closing total for calculating possession-level expected TFS
        efg_first_half: First half eFG%
        efg_second_half: Second half eFG%
        
    Returns:
        Matplotlib figure
    """
    away, home = get_team_names(tfs_df)
    header = f"{away} @ {home}" if (away and home) else f"Game {game_id}"
    
    # Add status to header if provided
    if game_status:
        header = f"{header} [{game_status}]"
    
    x = tfs_df["chrono_index"].values.astype(float)
    y = tfs_df["action_time"].values.astype(float)
    
    # Create smooth grid
    grid = np.linspace(x.min(), x.max(), 200)
    gx, gy = gaussian_kernel_smoother(x, y, bandwidth=5, grid=grid)
    
    # Find change points
    cps = find_change_points(y)
    
    # Get segment lines
    segments = get_segment_lines(tfs_df)
    
    # Get style
    style = get_plot_style()
    
    # Create figure
    fig, ax = plt.subplots(figsize=style["figsize"])
    
    # Plot raw data with color-coding by poss_start_type
    if "poss_start_type" in tfs_df.columns:
        # Group by poss_start_type and plot each group with different color
        # Don't add labels to main legend - they'll be in separate legend
        poss_start_types = ["rebound", "turnover", "oppo_made_shot", "period_start", None]
        
        for poss_type in poss_start_types:
            mask = tfs_df["poss_start_type"] == poss_type
            if mask.any():
                ax.scatter(
                    x[mask],
                    y[mask],
                    alpha=style["alpha_scatter"],
                    s=style["scatter_size"],
                    color=get_poss_start_color(poss_type),
                    label=None  # No label in main legend
                )
    else:
        # Fallback if poss_start_type not available
        ax.scatter(
            x, y,
            alpha=style["alpha_scatter"],
            s=style["scatter_size"],
            label="Raw TFS"
        )
    
    # Plot smoothed line
    ax.plot(
        gx, gy,
        linewidth=style["linewidth"],
        label="Average Tempo (TFS)",
        color=get_color("primary")
    )
    
    # Add segment lines
    for seg_x, label, seg_style in segments:
        ax.axvline(x=seg_x, **seg_style)
    
    # Add change points
    for i, cp in enumerate(cps):
        if 0 <= cp < len(x):
            ax.axvline(
                x[cp],
                color=get_color("change_point"),
                linestyle=":",
                linewidth=2,
                alpha=0.9,
                label="Change Point" if i == 0 else None
            )
    
    # Add predictions if requested
    if show_predictions:
        from app.tfs.predict import predict_tfs_next
        pred, conf_hi, conf_lo = predict_tfs_next(tfs_df, n_ahead=12)
        pred_x = np.arange(len(x), len(x) + len(pred)) + 1
        
        ax.plot(pred_x, pred, '--', color=get_color("secondary"), linewidth=2, label="Predicted TFS")
        ax.fill_between(pred_x, conf_lo, conf_hi, alpha=0.2, color=get_color("secondary"))
    
    # Calculate game-level expected TFS (flat line for reference)
    game_level_exp_tfs = None
    if closing_total is not None:
        from app.data.bigquery_loader import calculate_expected_tfs
        # Calculate game-level expected TFS (no poss_start_type)
        game_level_exp_tfs = calculate_expected_tfs(float(closing_total), None)
        
        # Draw the game-level expected TFS line (solid, thin, dark gray, dashed)
        ax.axhline(
            y=game_level_exp_tfs,
            color="darkgray",
            linestyle="--",
            linewidth=1,
            alpha=0.6,
            label=f"Game-Level Expected TFS ({game_level_exp_tfs:.1f}s)"
        )
    
    # Calculate and plot possession-level expected TFS trend
    exp_gx, exp_gy = None, None
    if closing_total is not None and len(tfs_df) > 0:
        try:
            # Calculate expected TFS for each possession based on its poss_start_type
            # Make sure we iterate in the same order as the DataFrame (which matches chrono_index)
            exp_tfs_values = []
            for idx in range(len(tfs_df)):
                poss_type = None
                if "poss_start_type" in tfs_df.columns:
                    poss_type_val = tfs_df.iloc[idx]["poss_start_type"]
                    # Handle NaN/None values
                    if pd.notna(poss_type_val) and poss_type_val is not None:
                        poss_type = str(poss_type_val).lower()
                exp_tfs = calculate_expected_tfs(float(closing_total), poss_type)
                exp_tfs_values.append(exp_tfs)
            
            exp_tfs_array = np.array(exp_tfs_values)
            
            # Smooth the expected TFS trend using the same kernel smoother
            # Use the same grid points as the kernel curve
            if len(exp_tfs_array) > 0:
                exp_gx, exp_gy = gaussian_kernel_smoother(x, exp_tfs_array, bandwidth=5, grid=grid)
        except Exception as e:
            # If there's an error, fall back to old behavior
            import traceback
            print(f"Error calculating possession-level expected TFS: {e}")
            print(traceback.format_exc())
            exp_gx, exp_gy = None, None
    
    # Plot possession-level expected TFS trend and apply shading
    if exp_gx is not None and exp_gy is not None:
        # Plot the expected TFS trend line
        ax.plot(
            exp_gx, exp_gy,
            color="darkgray",
            linestyle="-",
            linewidth=1.5,
            alpha=0.8,
            label="Expected TFS (by Poss. Type)"
        )
        
        # Add colored shading based on kernel curve vs expected TFS trend
        # Red shading: area between exp_tfs trend and curve when curve > exp_tfs (tempo slower than expected)
        # Green shading: area between curve and exp_tfs trend when curve < exp_tfs (tempo faster than expected)
        
        # Use where parameter to only fill where condition is true (prevents connecting disconnected regions)
        # Red shading: where actual curve is above expected trend (slower tempo)
        above_mask = gy > exp_gy
        if above_mask.any():
            ax.fill_between(
                gx,
                exp_gy,  # lower bound (expected trend)
                gy,      # upper bound (actual curve)
                where=above_mask,  # only fill where condition is true
                color="red",
                alpha=0.15,
                interpolate=False,  # Don't interpolate across gaps
                label="Slower than Expected" if not (gy < exp_gy).any() else None
            )
        
        # Green shading: where actual curve is below expected trend (faster tempo)
        below_mask = gy < exp_gy
        if below_mask.any():
            ax.fill_between(
                gx,
                gy,      # lower bound (actual curve)
                exp_gy,  # upper bound (expected trend)
                where=below_mask,  # only fill where condition is true
                color="green",
                alpha=0.15,
                interpolate=False,  # Don't interpolate across gaps
                label="Faster than Expected" if not (gy > exp_gy).any() else None
            )
    
    # Fallback: if closing_total not available but expected_tfs is (for backward compatibility)
    elif expected_tfs is not None:
        # Draw the expected TFS line (solid, thin, dark gray)
        ax.axhline(
            y=expected_tfs,
            color="darkgray",
            linestyle="--",
            linewidth=1,
            alpha=0.6,
            label=f"Expected TFS ({expected_tfs:.1f}s)"
        )
    
    # Add eFG% annotations if available
    efg_text = []
    if efg_first_half is not None:
        efg_text.append(f"1H eFG%: {efg_first_half:.1%}")
    if efg_second_half is not None:
        efg_text.append(f"2H eFG%: {efg_second_half:.1%}")
    
    if efg_text:
        # Add text box in upper left corner
        efg_str = " | ".join(efg_text)
        ax.text(
            0.02, 0.98, efg_str,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontweight='bold'
        )
    
    # Calculate residuals and statistics if we have expected TFS
    residual_stats_text = None
    if exp_gx is not None and exp_gy is not None and len(tfs_df) > 0:
        try:
            from app.data.bigquery_loader import calculate_expected_tfs
            
            # Calculate residuals for each possession
            residuals = []
            residuals_by_type = {"rebound": [], "turnover": [], "oppo_made_shot": [], "period_start": [], "other": []}
            above_exp_count = 0
            below_exp_count = 0
            
            for idx in range(len(tfs_df)):
                actual_tfs = float(tfs_df.iloc[idx]["action_time"])
                poss_type = None
                if "poss_start_type" in tfs_df.columns:
                    poss_type_val = tfs_df.iloc[idx]["poss_start_type"]
                    if pd.notna(poss_type_val) and poss_type_val is not None:
                        poss_type = str(poss_type_val).lower()
                
                expected_tfs = calculate_expected_tfs(float(closing_total), poss_type)
                residual = actual_tfs - expected_tfs
                residuals.append(residual)
                
                # Track by type
                if poss_type in residuals_by_type:
                    residuals_by_type[poss_type].append(residual)
                else:
                    residuals_by_type["other"].append(residual)
                
                # Count above/below
                if residual > 0:
                    above_exp_count += 1
                else:
                    below_exp_count += 1
            
            # Calculate statistics
            avg_residual = np.mean(residuals) if residuals else 0.0
            total_poss = len(residuals)
            pct_above = (above_exp_count / total_poss * 100) if total_poss > 0 else 0.0
            pct_below = (below_exp_count / total_poss * 100) if total_poss > 0 else 0.0
            
            # Calculate average residual by type
            avg_by_type = {}
            type_labels = {
                "rebound": "Reb",
                "turnover": "TO",
                "oppo_made_shot": "Made",
                "period_start": "Start",
                "other": "Other"
            }
            
            for poss_type, res_list in residuals_by_type.items():
                if res_list:
                    avg_by_type[poss_type] = np.mean(res_list)
            
            # Build statistics text
            stats_parts = [f"Avg Residual: {avg_residual:+.1f}s"]
            
            # Add by-type residuals
            type_parts = []
            for poss_type in ["rebound", "turnover", "oppo_made_shot", "period_start"]:
                if poss_type in avg_by_type:
                    type_parts.append(f"{type_labels[poss_type]}: {avg_by_type[poss_type]:+.1f}s")
            if type_parts:
                stats_parts.append(" | ".join(type_parts))
            
            # Add percentages
            stats_parts.append(f"Above: {pct_above:.1f}% | Below: {pct_below:.1f}%")
            
            residual_stats_text = " | ".join(stats_parts)
        except Exception as e:
            # If calculation fails, just skip the stats
            import traceback
            print(f"Error calculating residual statistics: {e}")
            print(traceback.format_exc())
    
    # Styling
    ax.set_title(f"{header}\nAverage Tempo (TFS)", fontsize=style["fontsize_title"])
    ax.set_xlabel("Possession")
    ax.set_ylabel("Time to First Shot (seconds)")
    
    # Create main legend (for lines, change points, etc.)
    ax.legend(loc="upper right", fontsize=style["fontsize_legend"])
    
    # Add residual statistics at the bottom
    if residual_stats_text:
        ax.text(
            0.5, -0.12, residual_stats_text,
            transform=ax.transAxes,
            fontsize=9,
            horizontalalignment='center',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7),
            family='monospace'
        )
    
    fig.tight_layout()
    return fig

