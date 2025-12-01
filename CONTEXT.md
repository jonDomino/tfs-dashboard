# TFS Kernel Dashboard - Context for New Agent

## Project Overview

This is a **Streamlit dashboard** for real-time monitoring of **Time to First Shot (TFS)** tempo in college basketball games. The dashboard visualizes game tempo using kernel smoothing, shows expected TFS based on closing totals and possession start types, and provides real-time updates every 30 seconds.

## Current State (Latest Session)

### Recent Work Completed

#### Latest Features (Most Recent)
1. ✅ **Period-Specific Expected TFS**: Split expected TFS calculations into Period 1 and Period 2 formulas
   - Period 2 formulas include score differential (abs(away_score - home_score) at end of Period 1)
   - Automatically calculates score_diff from Period 1 data
2. ✅ **New Possession Start Type**: Added `oppo_made_ft` (opponent made free throw) as separate type from `oppo_made_shot`
   - Different expected TFS formulas due to clock stopping behavior
   - Fully integrated in plots, legends, and residual calculations
3. ✅ **Pipeline Refactoring**: Switched from `builders/` to `build_tfs/builders/` directory
   - Fixed ESPN PBP API pagination to handle games with >500 plays
   - Removed old `builders/` directory
4. ✅ **Removed Period Start**: Dropped `period_start` from all visualizations and legends
   - No longer shown in plots or residual statistics table

#### Previous Improvements
1. ✅ **Rotation Numbers Display**: Added away team rotation numbers to plots (displayed in top-left margin, dark gray text)
2. ✅ **Rotation-Based Sorting**: Games sorted by rotation number descending (higher rotation numbers appear first)
3. ✅ **Default Tab Selection**: Dashboard automatically opens to highest priority available tab:
   - Priority order: Halftime > First Half > Early 1H > Second Half > Complete > alphabetical
4. ✅ **Possession-level Expected TFS**: Implemented dynamic expected TFS calculation based on `poss_start_type` with different formulas for each type
5. ✅ **Dual Expected TFS Lines**: Shows both game-level (flat dashed line) and possession-level (smooth trend line) expected TFS
6. ✅ **Shading Logic**: Red/green shading compares actual kernel curve vs possession-level expected kernel (not game-level)
7. ✅ **Enhanced Residual Statistics Table**: Comprehensive data table below main tempo plot with:
   - Columns: Metric, Count, Mean Res, Median Res, % Slower
   - Rows: Overall, Made Shot, Made FT, Rebound, Turnover
   - Color-coded cells:
     - Mean Res & Median Res: Positive (red) = slower than expected, Negative (green) = faster
     - % Slower: >50% (red), <=50% (green)
8. ✅ **File Cleanup**: Removed duplicate files, obsolete documentation, and git debugging files
9. ✅ **Git History Cleanup**: Removed credentials from git history using orphan branch approach
10. ✅ **Timezone Fix**: Converted all game dates to PST at pipeline entry to fix filtering issues
11. ✅ **Possession Start Legend**: Moved from individual plots to sidebar as single legend
12. ✅ **Code Review**: Created comprehensive `CODE_REVIEW.md` documenting requirements and refactoring recommendations

### Current Status

**Status:**
- ✅ Repository has clean history (credentials removed)
- ✅ All obsolete files cleaned up (duplicates, git debugging files removed)
- ✅ Enhanced residuals table with Count, Mean Res, Median Res columns
- ✅ Streamlit Cloud deployment working correctly
- ✅ Credentials stored in Streamlit Cloud settings (not in repo)

## Key Files & Their Purpose

### Core Application
- `streamlit_app.py` - Entry point (thin wrapper)
- `app/main.py` - Main application logic (500 lines - needs refactoring)
- `app/data/` - Data loading modules
  - `schedule_loader.py` - Loads schedule from ESPN
  - `pbp_loader.py` - Loads play-by-play data (cached)
  - `status.py` - Classifies game status from PBP
  - `bigquery_loader.py` - Fetches closing totals, rotation numbers, board info, calculates expected TFS
  - `efg.py` - Calculates effective field goal percentage
  - `get_sched.py` - ESPN API wrapper for schedule
  - `get_pbp.py` - ESPN API wrapper for PBP

### TFS Processing
- `app/tfs/preprocess.py` - Preprocessing pipeline (imports from build_tfs)
- `app/tfs/compute.py` - TFS computation (imports from build_tfs)
- `app/tfs/change_points.py` - CUSUM change-point detection
- `build_tfs/` - Standalone TFS processing module
  - `get_pbp.py` - ESPN API wrapper with pagination support (handles >500 plays)
  - `preprocess.py` - Preprocessing orchestrator
  - `compute.py` - TFS computation and filtering
  - `process_game.py` - Main entry point for processing games
  - `builders/action_time/` - Action time processing pipeline modules

### Visualization
- `app/plots/tempo.py` - Main tempo plot (526 lines - needs refactoring)
  - Shows kernel-smoothed tempo curve
  - Game-level expected TFS (dashed line)
  - Possession-level expected TFS (trend line)
  - Red/green shading (actual vs possession-level expected)
  - Rotation number display (top-left margin, dark gray text)
  - Enhanced residual statistics table (subplot below main plot):
    - Columns: Metric, Count, Mean Res, Median Res, % Slower
    - Rows: Overall, Made Shot, Made FT, Rebound, Turnover
    - Color-coded: Positive residuals (red), Negative residuals (green)
    - % Slower: >50% (red), <=50% (green)

### UI Components
- `app/ui/selectors.py` - Date, game, board selectors
- `app/ui/renderer.py` - Chart/error rendering
- `app/ui/layout.py` - Grid layout

### Configuration
- `requirements.txt` - Dependencies
- `meatloaf.json` - BigQuery credentials (NOT in git)
- `.gitignore` - Excludes credentials

## Key Features & Requirements

### Expected TFS Formulas

**Game-level**: `27.65 - 0.08 * closing_total` (reference line, backward compatibility)

**Period 1 Formulas**:
- **Turnover**: `TFS = 23.4283 + -0.068865 * closing_total`
- **Rebound**: `TFS = 23.2206 + -0.070364 * closing_total`
- **Oppo Made Shot**: `TFS = 35.8503 + -0.105015 * closing_total`
- **Oppo Made FT**: `TFS = 28.1118 + -0.065201 * closing_total`

**Period 2 Formulas** (include score differential):
- **Turnover**: `TFS = 22.0475 + -0.057148 * closing_total + -0.061952 * score_diff`
- **Rebound**: `TFS = 24.2071 + -0.072452 * closing_total + -0.045162 * score_diff`
- **Oppo Made Shot**: `TFS = 35.0632 + -0.097778 * closing_total + -0.034749 * score_diff`
- **Oppo Made FT**: `TFS = 29.7614 + -0.073256 * closing_total + -0.030282 * score_diff`

**Note**: `score_diff` is calculated as `abs(max(away_score) - max(home_score))` from Period 1 data. Formulas automatically switch based on `period_number` and availability of `score_diff`.

### Game Status Classification
- **Early 1H**: Period 1, >600 seconds remaining
- **First Half**: Period 1, 60-600 seconds remaining
- **Halftime**: Period 1, ≤60 seconds OR period 1 ended, period 2 not started
- **Second Half**: Period 2, clock running
- **Complete**: Period 2 ended or period > 2
- **Not Started**: No PBP data

### Caching Strategy
- Game statuses: 30 seconds TTL
- PBP data: 60 seconds TTL
- Closing totals: 1 hour TTL (only query 8am-10pm PST)

### Performance Optimizations
- Parallel status checking (ThreadPoolExecutor, max 5 workers)
- Skip completed games (never re-scan)
- Throttle "Not Started" games (10 min intervals)
- Flicker-free updates using `st.empty()` containers

## Technical Debt & Known Issues

1. **`app/main.py` too large** (500 lines) - needs splitting
2. **`app/plots/tempo.py` too complex** (505 lines) - needs refactoring (includes enhanced residuals table logic)
3. **Duplicate code** - Expected TFS calculation repeated
4. **Magic numbers** - Cache TTLs, time restrictions hardcoded
5. **Timezone** - Uses fixed UTC-8 offset (doesn't handle DST)
6. **No tests** - Zero test coverage

See `CODE_REVIEW.md` for detailed refactoring recommendations.

## Recent Git History

```
6aae134 - Fix circular import: remove unused preprocess_pbp import from compute.py
021390a - Update Period 2 expected TFS formulas with new coefficients
af7ee1a - Update expected TFS formulas: split into Period 1 and Period 2 with score_diff
ae4826b - Refactor: Switch pipeline to build_tfs, add oppo_made_ft support, remove period_start
```

**Note**: Repository has clean history. Previous credentials have been removed.

## Next Steps (Priority Order)

1. **HIGH**: Consider refactoring based on `CODE_REVIEW.md` (especially `app/main.py` and `app/plots/tempo.py`)
2. **MEDIUM**: Add proper timezone support (handle DST automatically)
3. **MEDIUM**: Extract configuration constants to centralized config module
4. **LOW**: Add tests, improve documentation

## Important Notes

- **Credentials**: Never commit `meatloaf.json` or actual credentials
- **Streamlit Cloud**: Uses secrets from `st.secrets.bq_credentials`
- **Timezone**: All dates converted to PST at pipeline entry
- **BigQuery**: Only queries 8am-10pm PST, uses 1-hour cache
- **Refresh**: Auto-refresh every 30 seconds, manual refresh available

## Code Structure

```
dashboard/
├── app/
│   ├── data/          # Data loading (schedule, PBP, BigQuery, status, eFG)
│   ├── tfs/           # TFS computation (preprocess, compute, change_points, segments)
│   ├── plots/         # Visualization (tempo, score_diff, combined)
│   ├── ui/            # UI components (selectors, renderer, layout)
│   ├── util/          # Utilities (cache, kernel, style, time)
│   └── main.py        # Main application (needs refactoring)
├── build_tfs/         # Standalone TFS processing module
│   ├── get_pbp.py     # ESPN API with pagination (handles >500 plays)
│   ├── preprocess.py  # Preprocessing orchestrator
│   ├── compute.py     # TFS computation
│   ├── process_game.py # Main entry point
│   └── builders/      # Action time processing pipeline
│       └── action_time/
├── streamlit_app.py   # Entry point
└── CODE_REVIEW.md     # Comprehensive code review & refactoring plan
```

## Key Dependencies

- `streamlit>=1.28.0`
- `pandas>=1.5.0`
- `numpy>=1.23.0`
- `matplotlib>=3.6.0`
- `requests>=2.28.0`
- `google-cloud-bigquery>=3.11.0`

## Contact Points for Questions

- **Expected TFS formulas**: `app/data/bigquery_loader.py::calculate_expected_tfs()`
- **Status classification**: `app/data/status.py::classify_game_status_pbp()`
- **Plot rendering**: `app/plots/tempo.py::build_tempo_figure()`
- **Main logic**: `app/main.py::_render_content()`

---

**Last Updated**: Current session
**Status**: Functional, clean git history, ready for refactoring to reduce technical debt

