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
    efg_second_half: Optional[float] = None,
    rotation_number: Optional[int] = None,
    lookahead_2h_total: Optional[float] = None,
    closing_spread_home: Optional[float] = None,
    home_team_name: Optional[str] = None,
    opening_2h_total: Optional[float] = None,
    closing_2h_total: Optional[float] = None,
    opening_2h_spread: Optional[float] = None,
    closing_2h_spread: Optional[float] = None
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
        rotation_number: Away team rotation number (optional)
        lookahead_2h_total: Lookahead 2H total (optional)
        closing_spread_home: Closing spread from home team's perspective (optional)
        home_team_name: Home team name (optional)
        opening_2h_total: Opening 2H total (optional)
        closing_2h_total: Closing 2H total (optional)
        opening_2h_spread: Opening 2H spread (optional)
        closing_2h_spread: Closing 2H spread (optional)
        
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
    
    # Calculate score_diff from Period 1 scores (if available)
    score_diff = None
    if "away_score" in tfs_df.columns and "home_score" in tfs_df.columns and "period_number" in tfs_df.columns:
        period_1_data = tfs_df[tfs_df["period_number"] == 1]
        if len(period_1_data) > 0:
            # Get max scores from period 1
            max_away_score = period_1_data["away_score"].max()
            max_home_score = period_1_data["home_score"].max()
            if pd.notna(max_away_score) and pd.notna(max_home_score):
                score_diff = abs(float(max_away_score) - float(max_home_score))
    
    # Calculate residuals early if we have closing_total (needed for subplot)
    residual_data = None
    if closing_total is not None and len(tfs_df) > 0:
        try:
            from app.data.bigquery_loader import calculate_expected_tfs
            
            # Calculate residuals for each possession, tracking by period and type
            residuals = []
            residuals_p1 = []
            residuals_p2 = []
            
            # Track by type and period
            residuals_by_type_p1 = {"rebound": [], "turnover": [], "oppo_made_shot": [], "oppo_made_ft": [], "other": []}
            residuals_by_type_p2 = {"rebound": [], "turnover": [], "oppo_made_shot": [], "oppo_made_ft": [], "other": []}
            
            above_exp_count_by_type_p1 = {"rebound": 0, "turnover": 0, "oppo_made_shot": 0, "oppo_made_ft": 0, "other": 0}
            above_exp_count_by_type_p2 = {"rebound": 0, "turnover": 0, "oppo_made_shot": 0, "oppo_made_ft": 0, "other": 0}
            total_count_by_type_p1 = {"rebound": 0, "turnover": 0, "oppo_made_shot": 0, "oppo_made_ft": 0, "other": 0}
            total_count_by_type_p2 = {"rebound": 0, "turnover": 0, "oppo_made_shot": 0, "oppo_made_ft": 0, "other": 0}
            
            above_exp_count = 0
            above_exp_count_p1 = 0
            above_exp_count_p2 = 0
            
            for idx in range(len(tfs_df)):
                actual_tfs = float(tfs_df.iloc[idx]["action_time"])
                poss_type = None
                period_num = None
                if "poss_start_type" in tfs_df.columns:
                    poss_type_val = tfs_df.iloc[idx]["poss_start_type"]
                    if pd.notna(poss_type_val) and poss_type_val is not None:
                        poss_type = str(poss_type_val).lower()
                if "period_number" in tfs_df.columns:
                    period_num_val = tfs_df.iloc[idx]["period_number"]
                    if pd.notna(period_num_val):
                        period_num = int(period_num_val)
                
                expected_tfs = calculate_expected_tfs(float(closing_total), poss_type, period_num, score_diff)
                residual = actual_tfs - expected_tfs
                residuals.append(residual)
                
                # Track by period
                if period_num == 1:
                    residuals_p1.append(residual)
                    if residual > 0:
                        above_exp_count_p1 += 1
                elif period_num and period_num >= 2:
                    residuals_p2.append(residual)
                    if residual > 0:
                        above_exp_count_p2 += 1
                
                # Track by type and period
                if poss_type in residuals_by_type_p1:
                    if period_num == 1:
                        residuals_by_type_p1[poss_type].append(residual)
                        total_count_by_type_p1[poss_type] += 1
                        if residual > 0:
                            above_exp_count_by_type_p1[poss_type] += 1
                    elif period_num and period_num >= 2:
                        residuals_by_type_p2[poss_type].append(residual)
                        total_count_by_type_p2[poss_type] += 1
                        if residual > 0:
                            above_exp_count_by_type_p2[poss_type] += 1
                else:
                    if period_num == 1:
                        residuals_by_type_p1["other"].append(residual)
                        total_count_by_type_p1["other"] += 1
                        if residual > 0:
                            above_exp_count_by_type_p1["other"] += 1
                    elif period_num and period_num >= 2:
                        residuals_by_type_p2["other"].append(residual)
                        total_count_by_type_p2["other"] += 1
                        if residual > 0:
                            above_exp_count_by_type_p2["other"] += 1
                
                # Count above/below (overall)
                if residual > 0:
                    above_exp_count += 1
            
            # Calculate overall statistics
            avg_residual = np.mean(residuals) if residuals else 0.0
            median_residual = np.median(residuals) if residuals else 0.0
            total_poss = len(residuals)
            pct_above = (above_exp_count / total_poss * 100) if total_poss > 0 else 0.0
            
            # Calculate Period 1 statistics
            avg_residual_p1 = np.mean(residuals_p1) if residuals_p1 else 0.0
            median_residual_p1 = np.median(residuals_p1) if residuals_p1 else 0.0
            total_poss_p1 = len(residuals_p1)
            pct_above_p1 = (above_exp_count_p1 / total_poss_p1 * 100) if total_poss_p1 > 0 else 0.0
            
            # Calculate Period 2 statistics
            avg_residual_p2 = np.mean(residuals_p2) if residuals_p2 else 0.0
            median_residual_p2 = np.median(residuals_p2) if residuals_p2 else 0.0
            total_poss_p2 = len(residuals_p2)
            pct_above_p2 = (above_exp_count_p2 / total_poss_p2 * 100) if total_poss_p2 > 0 else 0.0
            
            # Calculate statistics by type for Period 1
            avg_by_type_p1 = {}
            median_by_type_p1 = {}
            pct_above_by_type_p1 = {}
            count_by_type_p1 = {}
            for poss_type, res_list in residuals_by_type_p1.items():
                if res_list:
                    avg_by_type_p1[poss_type] = np.mean(res_list)
                    median_by_type_p1[poss_type] = np.median(res_list)
                    pct_above_by_type_p1[poss_type] = (above_exp_count_by_type_p1[poss_type] / total_count_by_type_p1[poss_type] * 100) if total_count_by_type_p1[poss_type] > 0 else 0.0
                    count_by_type_p1[poss_type] = total_count_by_type_p1[poss_type]
            
            # Calculate statistics by type for Period 2
            avg_by_type_p2 = {}
            median_by_type_p2 = {}
            pct_above_by_type_p2 = {}
            count_by_type_p2 = {}
            for poss_type, res_list in residuals_by_type_p2.items():
                if res_list:
                    avg_by_type_p2[poss_type] = np.mean(res_list)
                    median_by_type_p2[poss_type] = np.median(res_list)
                    pct_above_by_type_p2[poss_type] = (above_exp_count_by_type_p2[poss_type] / total_count_by_type_p2[poss_type] * 100) if total_count_by_type_p2[poss_type] > 0 else 0.0
                    count_by_type_p2[poss_type] = total_count_by_type_p2[poss_type]
            
            # Calculate overall statistics by type (for backward compatibility)
            avg_by_type = {}
            median_by_type = {}
            pct_above_by_type = {}
            count_by_type = {}
            for poss_type in ["rebound", "turnover", "oppo_made_shot", "oppo_made_ft", "other"]:
                all_res = residuals_by_type_p1.get(poss_type, []) + residuals_by_type_p2.get(poss_type, [])
                if all_res:
                    avg_by_type[poss_type] = np.mean(all_res)
                    median_by_type[poss_type] = np.median(all_res)
                    total_count = total_count_by_type_p1.get(poss_type, 0) + total_count_by_type_p2.get(poss_type, 0)
                    above_count = above_exp_count_by_type_p1.get(poss_type, 0) + above_exp_count_by_type_p2.get(poss_type, 0)
                    pct_above_by_type[poss_type] = (above_count / total_count * 100) if total_count > 0 else 0.0
                    count_by_type[poss_type] = total_count
            
            residual_data = {
                # Overall (Game)
                "avg_residual": avg_residual,
                "median_residual": median_residual,
                "pct_above": pct_above,
                "total_poss": total_poss,
                # Period 1
                "avg_residual_p1": avg_residual_p1,
                "median_residual_p1": median_residual_p1,
                "pct_above_p1": pct_above_p1,
                "total_poss_p1": total_poss_p1,
                # Period 2
                "avg_residual_p2": avg_residual_p2,
                "median_residual_p2": median_residual_p2,
                "pct_above_p2": pct_above_p2,
                "total_poss_p2": total_poss_p2,
                # By type (overall)
                "avg_by_type": avg_by_type,
                "median_by_type": median_by_type,
                "pct_above_by_type": pct_above_by_type,
                "count_by_type": count_by_type,
                # By type Period 1
                "avg_by_type_p1": avg_by_type_p1,
                "median_by_type_p1": median_by_type_p1,
                "pct_above_by_type_p1": pct_above_by_type_p1,
                "count_by_type_p1": count_by_type_p1,
                # By type Period 2
                "avg_by_type_p2": avg_by_type_p2,
                "median_by_type_p2": median_by_type_p2,
                "pct_above_by_type_p2": pct_above_by_type_p2,
                "count_by_type_p2": count_by_type_p2,
            }
        except Exception as e:
            # If calculation fails, skip residual chart
            import traceback
            print(f"Error calculating residual statistics: {e}")
            print(traceback.format_exc())
    
    # Create figure with subplots if we have residual data, otherwise single plot
    if residual_data:
        fig, axes = plt.subplots(2, 1, figsize=(style["figsize"][0], style["figsize"][1] * 1.3), 
                                 height_ratios=[3, 1], sharex=False)
        ax = axes[0]
        ax_residual = axes[1]
    else:
        fig, ax = plt.subplots(figsize=style["figsize"])
        ax_residual = None
    
    # Plot raw data with color-coding by poss_start_type
    if "poss_start_type" in tfs_df.columns:
        # Group by poss_start_type and plot each group with different color
        # Don't add labels to main legend - they'll be in separate legend
        poss_start_types = ["rebound", "turnover", "oppo_made_shot", "oppo_made_ft", None]
        
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
            # Calculate expected TFS for each possession based on its poss_start_type and period
            # Make sure we iterate in the same order as the DataFrame (which matches chrono_index)
            exp_tfs_values = []
            for idx in range(len(tfs_df)):
                poss_type = None
                period_num = None
                if "poss_start_type" in tfs_df.columns:
                    poss_type_val = tfs_df.iloc[idx]["poss_start_type"]
                    # Handle NaN/None values
                    if pd.notna(poss_type_val) and poss_type_val is not None:
                        poss_type = str(poss_type_val).lower()
                if "period_number" in tfs_df.columns:
                    period_num_val = tfs_df.iloc[idx]["period_number"]
                    if pd.notna(period_num_val):
                        period_num = int(period_num_val)
                exp_tfs = calculate_expected_tfs(float(closing_total), poss_type, period_num, score_diff)
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
    
    # Add score overlay (compact, transparent, top left, below eFG)
    # Check if eFG will be displayed to position score correctly
    has_efg = efg_first_half is not None or efg_second_half is not None
    
    if "away_score" in tfs_df.columns and "home_score" in tfs_df.columns and "period_number" in tfs_df.columns:
        try:
            # Extract scores by period - using max score for each period (score at end of period)
            # Score definition: max(away_score) and max(home_score) for each period_number
            # This gives us the score at the end of each period
            away_team_name = tfs_df["away_team_name"].iloc[0] if "away_team_name" in tfs_df.columns else "Away"
            home_team_name = tfs_df["home_team_name"].iloc[0] if "home_team_name" in tfs_df.columns else "Home"
            
            # Get max scores for each period (represents score at end of period)
            periods = sorted(tfs_df["period_number"].unique())
            score_data = []
            
            for period in periods:
                period_data = tfs_df[tfs_df["period_number"] == period]
                if len(period_data) > 0:
                    away_score = period_data["away_score"].max()
                    home_score = period_data["home_score"].max()
                    if pd.notna(away_score) and pd.notna(home_score):
                        score_data.append({
                            "period": period,
                            "away_score": int(away_score),
                            "home_score": int(home_score)
                        })
            
            if score_data:
                # Build compact score string
                period_labels = []
                away_scores = []
                home_scores = []
                
                for score_row in score_data:
                    period = score_row["period"]
                    if period == 1:
                        period_labels.append("H1")
                    elif period == 2:
                        period_labels.append("H2")
                    elif period > 2:
                        period_labels.append(f"OT{period - 2}")
                    else:
                        period_labels.append(f"P{period}")
                    away_scores.append(str(score_row["away_score"]))
                    home_scores.append(str(score_row["home_score"]))
                
                # Final score (last period's score)
                final_away = away_scores[-1]
                final_home = home_scores[-1]
                
                # Build compact text with aligned columns
                # Find max team name length for alignment
                max_team_len = max(len(away_team_name), len(home_team_name))
                
                # Build score strings with aligned columns
                # Format: "Team Name: H1 30 H2 45 FNL 75" with fixed-width team name
                away_score_str = " ".join([f"{label} {score:>3}" for label, score in zip(period_labels, away_scores)]) + f" FNL {final_away:>3}"
                home_score_str = " ".join([f"{label} {score:>3}" for label, score in zip(period_labels, home_scores)]) + f" FNL {final_home:>3}"
                
                # Format with aligned team names
                score_parts = []
                score_parts.append(f"{away_team_name:<{max_team_len}}: {away_score_str}")
                score_parts.append(f"{home_team_name:<{max_team_len}}: {home_score_str}")
                score_text = "\n".join(score_parts)
                
                # Position below eFG if eFG is present, otherwise at top
                score_y_pos = 0.90 if has_efg else 0.95
                
                # Add as text box with transparent background
                ax.text(
                    0.02, score_y_pos, score_text,
                    transform=ax.transAxes,
                    fontsize=8,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='gray', linewidth=0.5),
                    family='monospace'  # Monospace for alignment
                )
        except Exception as e:
            import traceback
            print(f"Error creating score overlay: {e}")
            print(traceback.format_exc())
    
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
    
    # Build residual statistics text for display (if needed)
    residual_stats_text = None
    if residual_data:
        type_labels = {
            "rebound": "Reb",
            "turnover": "TO",
            "oppo_made_shot": "Made",
            "oppo_made_ft": "FT",
            "other": "Other"
        }
        
        stats_parts = [f"Avg Residual: {residual_data['avg_residual']:+.1f}s"]
        
        # Add by-type residuals
        type_parts = []
        for poss_type in ["rebound", "turnover", "oppo_made_shot", "oppo_made_ft"]:
            if poss_type in residual_data['avg_by_type']:
                type_parts.append(f"{type_labels[poss_type]}: {residual_data['avg_by_type'][poss_type]:+.1f}s")
        if type_parts:
            stats_parts.append(" | ".join(type_parts))
        
        # Add percentages
        pct_below = 100 - residual_data['pct_above']
        stats_parts.append(f"Above: {residual_data['pct_above']:.1f}% | Below: {pct_below:.1f}%")
        
        residual_stats_text = " | ".join(stats_parts)
    
    # Styling
    ax.set_title(f"{header}\nAverage Tempo (TFS)", fontsize=style["fontsize_title"])
    ax.set_xlabel("Possession")
    ax.set_ylabel("Time to First Shot (seconds)")
    
    # Create main legend (for lines, change points, etc.)
    ax.legend(loc="upper right", fontsize=style["fontsize_legend"])
    
    # Add residual statistics table below if we have residual data
    if ax_residual is not None and residual_data:
        # Prepare data for table with columns: Metric, P1 Count, P2 Count, Gm Count, P1 Mean, P2 Mean, Gm Mean, P1 Median, P2 Median, Gm Median, P1 % Slower, P2 % Slower, Gm % Slower
        type_labels_display = {
            "oppo_made_shot": "Made Shot",
            "oppo_made_ft": "Made FT",
            "rebound": "Rebound",
            "turnover": "Turnover"
        }
        
        # Build table data: rows are Overall, Made Shot, Made FT, Rebound, Turnover
        table_data = []
        
        # Helper function to format value or return "-" if not available
        def fmt_val(val, default="-"):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return default
            return val
        
        # Overall row - reordered: Metric, P1 (Cnt, Mean, Med, Slow%), P2 (Cnt, Mean, Med, Slow%), Gm (Cnt, Mean, Med, Slow%)
        table_data.append([
            "Overall",
            str(residual_data.get('total_poss_p1', 0)),
            f"{residual_data.get('avg_residual_p1', 0):+.1f}s" if residual_data.get('total_poss_p1', 0) > 0 else "-",
            f"{residual_data.get('median_residual_p1', 0):+.1f}s" if residual_data.get('total_poss_p1', 0) > 0 else "-",
            f"{residual_data.get('pct_above_p1', 0):.1f}%" if residual_data.get('total_poss_p1', 0) > 0 else "-",
            str(residual_data.get('total_poss_p2', 0)),
            f"{residual_data.get('avg_residual_p2', 0):+.1f}s" if residual_data.get('total_poss_p2', 0) > 0 else "-",
            f"{residual_data.get('median_residual_p2', 0):+.1f}s" if residual_data.get('total_poss_p2', 0) > 0 else "-",
            f"{residual_data.get('pct_above_p2', 0):.1f}%" if residual_data.get('total_poss_p2', 0) > 0 else "-",
            str(residual_data.get('total_poss', 0)),
            f"{residual_data.get('avg_residual', 0):+.1f}s",
            f"{residual_data.get('median_residual', 0):+.1f}s",
            f"{residual_data.get('pct_above', 0):.1f}%"
        ])
        
        # Add possession type rows (Made Shot, Made FT, Rebound, Turnover)
        for poss_type in ["oppo_made_shot", "oppo_made_ft", "rebound", "turnover"]:
            count_p1 = residual_data['count_by_type_p1'].get(poss_type, 0)
            count_p2 = residual_data['count_by_type_p2'].get(poss_type, 0)
            count_gm = residual_data['count_by_type'].get(poss_type, 0)
            
            if count_gm > 0:  # Only add row if there's data
                avg_p1 = residual_data['avg_by_type_p1'].get(poss_type, 0.0) if count_p1 > 0 else None
                avg_p2 = residual_data['avg_by_type_p2'].get(poss_type, 0.0) if count_p2 > 0 else None
                avg_gm = residual_data['avg_by_type'].get(poss_type, 0.0)
                
                median_p1 = residual_data['median_by_type_p1'].get(poss_type, 0.0) if count_p1 > 0 else None
                median_p2 = residual_data['median_by_type_p2'].get(poss_type, 0.0) if count_p2 > 0 else None
                median_gm = residual_data['median_by_type'].get(poss_type, 0.0)
                
                pct_p1 = residual_data['pct_above_by_type_p1'].get(poss_type, 0.0) if count_p1 > 0 else None
                pct_p2 = residual_data['pct_above_by_type_p2'].get(poss_type, 0.0) if count_p2 > 0 else None
                pct_gm = residual_data['pct_above_by_type'].get(poss_type, 0.0)
                
                table_data.append([
                    type_labels_display[poss_type],
                    str(count_p1),
                    f"{avg_p1:+.1f}s" if avg_p1 is not None else "-",
                    f"{median_p1:+.1f}s" if median_p1 is not None else "-",
                    f"{pct_p1:.1f}%" if pct_p1 is not None else "-",
                    str(count_p2),
                    f"{avg_p2:+.1f}s" if avg_p2 is not None else "-",
                    f"{median_p2:+.1f}s" if median_p2 is not None else "-",
                    f"{pct_p2:.1f}%" if pct_p2 is not None else "-",
                    str(count_gm),
                    f"{avg_gm:+.1f}s",
                    f"{median_gm:+.1f}s",
                    f"{pct_gm:.1f}%"
                ])
        
        # Create table with 13 columns - ordered: Metric, then all P1, then all P2, then all Gm
        col_labels = ["Metric", "P1 Cnt", "P1 Mean", "P1 Med", "P1 Slow%", 
                     "P2 Cnt", "P2 Mean", "P2 Med", "P2 Slow%",
                     "Gm Cnt", "Gm Mean", "Gm Med", "Gm Slow%"]
        table = ax_residual.table(
            cellText=table_data,
            colLabels=col_labels,
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(7)  # Smaller font to fit more columns
        table.scale(1, 2)
        
        # Style header row (row 0 in matplotlib table)
        for j in range(len(col_labels)):
            table[(0, j)].set_facecolor('#4472C4')
            table[(0, j)].set_text_props(weight='bold', color='white')
        
        # Color code data cells
        for i, row in enumerate(table_data):
            row_idx = i + 1  # Data rows start at index 1 (after header)
            # Color code Mean columns (indices 2, 6, 10) - P1 Mean, P2 Mean, Gm Mean
            for col_idx in [2, 6, 10]:
                if col_idx < len(row) and row[col_idx] != "-":
                    try:
                        val = float(row[col_idx].replace("s", ""))
                        if val > 0:
                            table[(row_idx, col_idx)].set_facecolor('#ffcccc')  # Light red
                        else:
                            table[(row_idx, col_idx)].set_facecolor('#ccffcc')  # Light green
                    except:
                        pass
            
            # Color code Median columns (indices 3, 7, 11) - P1 Med, P2 Med, Gm Med
            for col_idx in [3, 7, 11]:
                if col_idx < len(row) and row[col_idx] != "-":
                    try:
                        val = float(row[col_idx].replace("s", ""))
                        if val > 0:
                            table[(row_idx, col_idx)].set_facecolor('#ffcccc')  # Light red
                        else:
                            table[(row_idx, col_idx)].set_facecolor('#ccffcc')  # Light green
                    except:
                        pass
            
            # Color code % Slower columns (indices 4, 8, 12) - P1 Slow%, P2 Slow%, Gm Slow%
            for col_idx in [4, 8, 12]:
                if col_idx < len(row) and row[col_idx] != "-":
                    try:
                        val = float(row[col_idx].replace("%", ""))
                        if val > 50:
                            table[(row_idx, col_idx)].set_facecolor('#ffcccc')  # Light red
                        else:
                            table[(row_idx, col_idx)].set_facecolor('#ccffcc')  # Light green
                    except:
                        pass
            
            # Column 0: Metric column (light gray)
            table[(row_idx, 0)].set_facecolor('#F0F0F0')
            
            # Count columns (1, 5, 9) - P1 Cnt, P2 Cnt, Gm Cnt - white background
            for col_idx in [1, 5, 9]:
                table[(row_idx, col_idx)].set_facecolor('#FFFFFF')
        
        # Remove axes for table
        ax_residual.axis('off')
        ax_residual.set_title(f'Residual Statistics', fontsize=10, fontweight='bold', pad=10)
        
        # Set x-axis label on main plot
        ax.set_xlabel("Possession", fontsize=10)
    
    # Add residual statistics text at the bottom of main plot (if no subplot)
    elif residual_stats_text:
        ax.text(
            0.5, -0.12, residual_stats_text,
            transform=ax.transAxes,
            fontsize=9,
            horizontalalignment='center',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7),
            family='monospace'
        )
        ax.set_xlabel("Possession")
    else:
        ax.set_xlabel("Possession")
    
    # Add rotation number text in top-left margin (outside plot area)
    if rotation_number is not None:
        fig.text(
            0.02, 0.98,  # Top-left corner in figure coordinates
            f"Roto: {rotation_number}",
            fontsize=9,
            color='#0a0a0a',  # Very dark, almost black
            horizontalalignment='left',
            verticalalignment='top',
            transform=fig.transFigure  # Use figure coordinates
        )
    
    # Create table in top-right above plot
    if closing_total is not None:
        # Prepare table data
        table_data = []
        table_cols = ['', '2H Open', '2H Close']
        
        # Row 1: Total
        total_row = [f"Total: {closing_total:.1f}", '', '']
        table_data.append(total_row)
        
        # Row 2: 2H Looka
        if lookahead_2h_total is not None:
            looka_row = [f"2H Looka: {lookahead_2h_total:.1f}", '', '']
            if opening_2h_total is not None:
                looka_row[1] = f"{opening_2h_total:.1f}"
            if closing_2h_total is not None:
                looka_row[2] = f"{closing_2h_total:.1f}"
            table_data.append(looka_row)
        
        # Row 3: Home team spread
        if closing_spread_home is not None and home_team_name:
            spread_str = f"{closing_spread_home:.1f}"
            # Remove trailing .0 if it's a whole number
            if spread_str.endswith('.0'):
                spread_str = spread_str[:-2]
            spread_row = [f"{home_team_name}: {spread_str}", '', '']
            if opening_2h_spread is not None:
                spread_val = f"{opening_2h_spread:.1f}"
                if spread_val.endswith('.0'):
                    spread_val = spread_val[:-2]
                spread_row[1] = spread_val
            if closing_2h_spread is not None:
                spread_val = f"{closing_2h_spread:.1f}"
                if spread_val.endswith('.0'):
                    spread_val = spread_val[:-2]
                spread_row[2] = spread_val
            table_data.append(spread_row)
        
        # Create table using matplotlib
        if table_data:
            # Position table in top-right
            table_ax = fig.add_axes([0.70, 0.92, 0.28, 0.08])  # [left, bottom, width, height] in figure coordinates
            table_ax.axis('off')
            
            # Create table
            table = table_ax.table(
                cellText=table_data,
                colLabels=table_cols,
                cellLoc='left',
                loc='upper right',
                bbox=[0, 0, 1, 1]
            )
            
            # Style the table
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.5)
            
            # Style header row
            for i in range(len(table_cols)):
                cell = table[(0, i)]
                cell.set_facecolor('#f0f0f0')
                cell.set_text_props(weight='bold', fontsize=8)
            
            # Style data cells - left align first column, right align others
            for i, row in enumerate(table_data):
                for j in range(len(table_cols)):
                    cell = table[(i + 1, j)]
                    if j == 0:
                        cell.set_text_props(ha='left')
                    else:
                        cell.set_text_props(ha='right')
                    cell.set_edgecolor('#d0d0d0')
                    cell.set_linewidth(0.5)
    
    fig.tight_layout()
    return fig

