# Dependencies Added to Dashboard

This document lists all the files and dependencies that were added to make the modularized dashboard work.

## Files Added

### 1. Root Level Files
- **`get_sched.py`** - Fetches schedule data from ESPN API
  - Location: `dashboard/get_sched.py`
  - Dependencies: `requests`, `pandas`
  
- **`get_pbp.py`** - Fetches play-by-play data from ESPN API
  - Location: `dashboard/get_pbp.py`
  - Dependencies: `requests`, `pandas`

### 2. Builders Module (`builders/action_time/`)
All files needed for the TFS preprocessing pipeline:

- **`builders/__init__.py`** - Package init
- **`builders/action_time/__init__.py`** - Package init
- **`builders/action_time/clean_pbp.py`** - Cleans and sorts PBP data
- **`builders/action_time/flag_ft.py`** - Flags free throw sequences
- **`builders/action_time/assign_poss_teams.py`** - Assigns possession teams
- **`builders/action_time/build_poss_context.py`** - Builds possession context
- **`builders/action_time/build_action_context.py`** - Builds action context (action_time, shot_count_in_poss)
- **`builders/action_time/build_tfs_detailed.py`** - Builds detailed TFS information

## Updated Files

### `requirements.txt`
Added:
- `requests>=2.28.0` - Required for ESPN API calls

## Import Structure

The app now imports from:
- `get_sched` → `get_sched.py` (root)
- `get_pbp` → `get_pbp.py` (root)
- `builders.action_time.*` → `builders/action_time/*.py`

## Testing

To verify everything works:
```bash
streamlit run streamlit_app.py
```

## Notes

- All builder modules are self-contained and don't require external dependencies beyond pandas/numpy
- The `clean_pbp.py` handles missing `wallclock` column gracefully
- All modules follow the same structure as the original audit pipeline

