# TFS Kernel Dashboard - Context for New Agent

## Project Overview

This is a **Streamlit dashboard** for real-time monitoring of **Time to First Shot (TFS)** tempo in college basketball games. The dashboard visualizes game tempo using kernel smoothing, shows expected TFS based on closing totals and possession start types, and provides real-time updates every 30 seconds.

## Current State (Latest Session)

### Recent Work Completed
1. ✅ **Possession-level Expected TFS**: Implemented dynamic expected TFS calculation based on `poss_start_type` (rebound, turnover, oppo_made_shot) with different formulas for each type
2. ✅ **Dual Expected TFS Lines**: Shows both game-level (flat dashed line) and possession-level (smooth trend line) expected TFS
3. ✅ **Shading Logic**: Red/green shading compares actual kernel curve vs possession-level expected kernel (not game-level)
4. ✅ **Residual Statistics**: Added bottom legend showing average residual, residuals by possession type, and % above/below expected
5. ✅ **Timezone Fix**: Converted all game dates to PST at pipeline entry to fix filtering issues
6. ✅ **Possession Start Legend**: Moved from individual plots to sidebar as single legend
7. ✅ **Code Review**: Created comprehensive `CODE_REVIEW.md` documenting requirements and refactoring recommendations

### Current Issue (BLOCKING)

**GitHub Push Protection Error**: Cannot push to repository because `STREAMLIT_SECRETS.md` contains actual Google Cloud Service Account credentials in commit history.

**Error Details:**
- Commit: `2428f4cf03a50d07127b3d2c54618a4ff5cae35e`
- File: `STREAMLIT_SECRETS.md:59`
- Issue: Contains private key, client_email, and other credentials

**Status:**
- ✅ Fixed: Replaced actual credentials with placeholders in `STREAMLIT_SECRETS.md`
- ✅ Fixed: Added `STREAMLIT_SECRETS.md` to `.gitignore`
- ⚠️ **TODO**: Need to remove the file from git history (commit `2428f4c`)

**Solution Needed:**
```bash
# Remove file from commit history
git rm --cached STREAMLIT_SECRETS.md
git commit --amend --no-edit
# Or if already pushed, need to rewrite history:
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch STREAMLIT_SECRETS.md" \
  --prune-empty --tag-name-filter cat -- --all
```

## Key Files & Their Purpose

### Core Application
- `streamlit_app.py` - Entry point (thin wrapper)
- `app/main.py` - Main application logic (502 lines - needs refactoring)
- `app/data/` - Data loading modules
  - `schedule_loader.py` - Loads schedule from ESPN
  - `pbp_loader.py` - Loads play-by-play data (cached)
  - `status.py` - Classifies game status from PBP
  - `bigquery_loader.py` - Fetches closing totals, calculates expected TFS
  - `efg.py` - Calculates effective field goal percentage
  - `get_sched.py` - ESPN API wrapper for schedule
  - `get_pbp.py` - ESPN API wrapper for PBP

### TFS Processing
- `app/tfs/preprocess.py` - Preprocessing pipeline
- `app/tfs/compute.py` - TFS computation
- `app/tfs/change_points.py` - CUSUM change-point detection
- `builders/action_time/` - Action time processing pipeline

### Visualization
- `app/plots/tempo.py` - Main tempo plot (366 lines - needs refactoring)
  - Shows kernel-smoothed tempo curve
  - Game-level expected TFS (dashed line)
  - Possession-level expected TFS (trend line)
  - Red/green shading (actual vs possession-level expected)
  - Residual statistics at bottom

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
- **Game-level**: `27.65 - 0.08 * closing_total` (reference line)
- **Possession-level**:
  - Oppo Made Shot: `36.4397 + (-0.1025 * closing_total)`
  - Rebound: `24.2977 + (-0.0692 * closing_total)`
  - Turnover: `23.9754 + (-0.0619 * closing_total)`
  - Period Start: Uses rebound formula

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

1. **`app/main.py` too large** (502 lines) - needs splitting
2. **`app/plots/tempo.py` too complex** (366 lines) - needs refactoring
3. **Duplicate code** - Expected TFS calculation repeated
4. **Magic numbers** - Cache TTLs, time restrictions hardcoded
5. **Timezone** - Uses fixed UTC-8 offset (doesn't handle DST)
6. **No tests** - Zero test coverage

See `CODE_REVIEW.md` for detailed refactoring recommendations.

## Recent Git History

```
a3686a6 - Fix shading: use where parameter in fill_between to prevent connecting disconnected regions
c59cb91 - Fix shading: use full grid with NaN masking to prevent connecting disconnected regions
39d56c6 - Keep game-level exp TFS for reference, shade based on poss-level exp kernel, move poss start legend to sidebar
c1de602 - Update expected TFS to possession-level based on poss_start_type and closing_total, show as trend line with updated shading
74b7317 - Convert game dates to PST at pipeline entry to fix timezone filtering issues
274c7bb - Fix timezone: use PST instead of UTC for date selection and schedule loading
```

## Next Steps (Priority Order)

1. **URGENT**: Fix git push issue - remove credentials from commit history
2. **HIGH**: Complete residual statistics feature (was in progress)
3. **MEDIUM**: Consider refactoring based on `CODE_REVIEW.md`
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
├── builders/          # Action time processing pipeline
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
**Status**: Functional, but has blocking git issue and technical debt

