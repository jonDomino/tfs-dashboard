"""Style and theme utilities"""
from typing import Dict, Any


# Color scheme
COLORS = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "success": "#2ca02c",
    "danger": "#d62728",
    "warning": "#ffbb78",
    "info": "#17becf",
    "change_point": "orange",
    "halftime": "red",
    "segment": "gray",
}

# Possession start type colors
POSS_START_COLORS = {
    "rebound": "#d62728",        # Red
    "turnover": "#1f77b4",       # Blue
    "oppo_made_shot": "#2ca02c", # Green
    "period_start": "#9467bd",   # Purple
    None: "#7f7f7f",             # Gray for edge cases
}


# Plot styling
PLOT_STYLE = {
    "figsize": (8, 4),
    "fontsize_title": 12,
    "fontsize_label": 10,
    "fontsize_legend": 8,
    "alpha_scatter": 0.35,
    "scatter_size": 20,
    "linewidth": 2.5,
}


def get_plot_style() -> Dict[str, Any]:
    """Get default plot styling configuration.
    
    Returns:
        Dictionary of style parameters
    """
    return PLOT_STYLE.copy()


def get_color(name: str) -> str:
    """Get color by name.
    
    Args:
        name: Color name
        
    Returns:
        Hex color string
    """
    return COLORS.get(name, "#000000")


def get_poss_start_color(poss_start_type) -> str:
    """Get color for possession start type.
    
    Args:
        poss_start_type: Possession start type (rebound, turnover, oppo_made_shot, period_start, None)
        
    Returns:
        Hex color string
    """
    return POSS_START_COLORS.get(poss_start_type, "#7f7f7f")

