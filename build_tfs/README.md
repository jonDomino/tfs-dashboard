# TFS Processing Module

Standalone module for fetching play-by-play data and computing Time to First Shot (TFS) metrics.

## Overview

This module provides a complete pipeline to:
1. Fetch play-by-play data from ESPN API for a given game ID
2. Preprocess and clean the data
3. Compute TFS (Time to First Shot) metrics

## Usage

### Basic Usage

```python
from process_game import process_game

# Process a game and get TFS DataFrame
tfs_df = process_game(game_id=401700174)

# The DataFrame contains:
# - game_id, period_number, possession_id
# - action_time (TFS in seconds)
# - type_text (play type)
# - poss_start_type (how possession started)
# - chrono_index (chronological order)
# - And other contextual fields
```

### Step-by-Step Usage

```python
from get_pbp import get_pbp
from preprocess import preprocess_pbp
from compute import compute_tfs

# 1. Fetch raw PBP from ESPN
raw_pbp = get_pbp(game_id=401700174)

# 2. Preprocess (clean, add possession context, etc.)
df = preprocess_pbp(raw_pbp)

# 3. Compute TFS (filters to first field goal attempts, 3-40 seconds)
tfs_df = compute_tfs(df)
```

## Output

The final TFS DataFrame contains:
- **Filtered to first field goal attempts only** (excludes free throws, rebounds, fouls)
- **Filtered to 3-40 seconds** action time range
- **Sorted chronologically** with `chrono_index` column
- **Includes possession context**: `possession_id`, `poss_start_type`, `action_time`

## Dependencies

- pandas
- numpy
- requests

Install with:
```bash
pip install pandas numpy requests
```

## File Structure

```
tfs/
├── get_pbp.py              # Fetch PBP from ESPN API
├── preprocess.py           # Preprocessing pipeline
├── compute.py             # TFS computation and filtering
├── process_game.py         # Main entry point
├── builders/
│   └── action_time/
│       ├── clean_pbp.py           # Clean and sort PBP
│       ├── flag_ft.py             # Flag free throw sequences
│       ├── assign_poss_teams.py   # Determine possession teams
│       ├── build_poss_context.py  # Build possession context
│       ├── add_poss_start_type.py # Add possession start types
│       ├── build_action_context.py # Add action time and shot counts
│       └── build_tfs_detailed.py  # Build detailed TFS metrics
└── README.md
```

## Notes

- The module filters out non-shooting plays (free throws, rebounds, fouls) to avoid "quadruple counting"
- Only first field goal attempts in each possession are included
- Action times are filtered to 3-40 seconds range
- All processing is done in-memory with pandas DataFrames

