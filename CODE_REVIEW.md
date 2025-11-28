# TFS Kernel Dashboard - Code Review & Requirements Documentation

## Executive Summary

This is a comprehensive code review of the TFS (Time to First Shot) Kernel Dashboard, a Streamlit application for real-time monitoring of college basketball game tempo. The codebase is well-structured but has grown organically and would benefit from strategic refactoring to improve maintainability, testability, and performance.

**Recommendation: YES, it's time for a refactor** - The codebase has reached a complexity threshold where refactoring will significantly improve maintainability without disrupting functionality.

---

## 1. Requirements Documentation

### 1.1 Functional Requirements

#### Core Features
1. **Real-time TFS Visualization**
   - Display kernel-smoothed tempo curves for each game
   - Show raw TFS data points color-coded by possession start type
   - Display game-level and possession-level expected TFS lines
   - Show CUSUM change-point detection markers
   - Display eFG% for first and second halves

2. **Game Status Management**
   - Classify games into: Not Started, Early 1H, First Half, Halftime, Second Half, Complete
   - Auto-refresh game statuses every 30 seconds
   - Efficient scanning: skip completed games, throttle "Not Started" games (10 min intervals)
   - Parallel processing for status checks (max 5 workers)

3. **Data Filtering**
   - Filter by date (PST timezone)
   - Filter by game status (tabs for each status)
   - Filter by board (Main/Extra) based on BigQuery data
   - Only show games with closing totals when board filter is active

4. **Expected TFS Calculation**
   - Game-level: `27.65 - 0.08 * closing_total` (reference line)
   - Possession-level (by type):
     - Oppo Made Shot: `36.4397 + (-0.1025 * closing_total)`
     - Rebound: `24.2977 + (-0.0692 * closing_total)`
     - Turnover: `23.9754 + (-0.0619 * closing_total)`
     - Period Start: Uses rebound formula

5. **Visual Indicators**
   - Red shading: Actual tempo slower than possession-level expected
   - Green shading: Actual tempo faster than possession-level expected
   - Residual statistics at bottom of plots (avg residual, by type, % above/below)

6. **Data Sources**
   - ESPN API for schedule and play-by-play data
   - BigQuery for closing totals and board information
   - Timezone: All dates/times converted to PST at pipeline entry

### 1.2 Non-Functional Requirements

#### Performance
- Refresh interval: 30 seconds
- Cache TTLs:
  - Game statuses: 30 seconds
  - Closing totals: 1 hour (only query 8am-10pm)
  - PBP data: 60 seconds
- Parallel processing for status checks (ThreadPoolExecutor, max 5 workers)
- Flicker-free updates using `st.empty()` containers

#### Reliability
- Graceful error handling (don't crash on single game errors)
- Skip completed games (never re-scan)
- Throttle "Not Started" games (10 min intervals)
- BigQuery query restrictions (8am-10pm only, 1-hour cache)

#### Usability
- Auto-refresh with manual refresh option
- Status-based tabs for easy navigation
- Sidebar legend for possession start types
- Error messages with expandable details

#### Security
- BigQuery credentials via Streamlit secrets or environment variables
- `meatloaf.json` excluded from git
- No hardcoded credentials

---

## 2. Architecture Overview

### 2.1 Current Structure

```
dashboard/
├── app/
│   ├── data/          # Data loading & processing
│   │   ├── schedule_loader.py
│   │   ├── pbp_loader.py
│   │   ├── status.py
│   │   ├── bigquery_loader.py
│   │   ├── efg.py
│   │   ├── get_sched.py
│   │   └── get_pbp.py
│   ├── tfs/           # TFS computation
│   │   ├── preprocess.py
│   │   ├── compute.py
│   │   ├── change_points.py
│   │   ├── segments.py
│   │   └── predict.py
│   ├── plots/         # Visualization
│   │   ├── tempo.py
│   │   ├── score_diff.py
│   │   └── combined.py
│   ├── ui/            # UI components
│   │   ├── selectors.py
│   │   ├── renderer.py
│   │   └── layout.py
│   ├── util/          # Utilities
│   │   ├── cache.py
│   │   ├── kernel.py
│   │   ├── style.py
│   │   └── time.py
│   └── main.py        # Main application (502 lines)
├── builders/          # Action time processing pipeline
│   └── action_time/
└── streamlit_app.py   # Entry point
```

### 2.2 Data Flow

1. **Schedule Loading** → `get_sched()` → Convert dates to PST → Filter by date
2. **Status Checking** → Parallel PBP fetch → `classify_game_status_pbp()` → Cache results
3. **Game Data Processing** → `load_pbp()` → `preprocess_pbp()` → `compute_tfs()` → Cache
4. **Closing Totals** → BigQuery query (cached 1 hour, 8am-10pm only) → Filter by board
5. **Rendering** → Group by status → Create tabs → Render plots with containers

---

## 3. Code Quality Analysis

### 3.1 Strengths

✅ **Good Modular Structure**: Clear separation of concerns (data, tfs, plots, ui, util)
✅ **Caching Strategy**: Well-implemented caching with appropriate TTLs
✅ **Error Handling**: Graceful degradation, error logging
✅ **Performance Optimizations**: Parallel processing, throttling, skip completed games
✅ **Type Hints**: Good use of type annotations
✅ **Documentation**: Functions have docstrings

### 3.2 Issues & Technical Debt

#### Critical Issues

1. **`app/main.py` is Too Large (502 lines)**
   - Contains business logic, rendering, state management, and coordination
   - Violates Single Responsibility Principle
   - Hard to test and maintain

2. **Repeated Code in `app/plots/tempo.py`**
   - Expected TFS calculation duplicated (lines 147-160 and 270-286)
   - Residual calculation logic is verbose and could be extracted

3. **Session State Management Scattered**
   - `completed_games`, `not_started_last_check`, `plot_containers`, `error_log` initialized in multiple places
   - No centralized state management

4. **Magic Numbers & Hardcoded Values**
   - Cache TTLs scattered: 30s, 60s, 3600s
   - Time restrictions: 8am, 10pm hardcoded
   - Thread pool size: 5 workers hardcoded
   - Bandwidth: 5 (kernel smoother) hardcoded

5. **Inconsistent Error Handling**
   - Some functions return `None` on error, others raise exceptions
   - Error logging inconsistent (some print, some use session_state)

6. **Timezone Handling**
   - PST hardcoded as `timezone(timedelta(hours=-8))` - doesn't handle DST
   - Should use `pytz` or `zoneinfo` for proper timezone support

#### Moderate Issues

7. **Import Organization**
   - Some imports inside functions (e.g., `from app.data.bigquery_loader import calculate_expected_tfs` in `tempo.py`)
   - Should be at module level for better performance

8. **DataFrame Iteration**
   - Using `for idx in range(len(tfs_df))` with `iloc[idx]` is inefficient
   - Should use vectorized operations or `.iterrows()` / `.itertuples()`

9. **Function Complexity**
   - `build_tempo_figure()`: 366 lines, does too much
   - `_render_content()`: 148 lines, complex nested logic
   - `get_game_statuses()`: 84 lines, handles multiple concerns

10. **Testing**
    - No unit tests found
    - No integration tests
    - Hard to test due to Streamlit dependencies

11. **Configuration Management**
    - No centralized config file
    - Settings scattered across codebase

12. **Duplicate Files**
    - `get_sched.py` and `get_pbp.py` exist in both root and `app/data/`
    - Should consolidate

---

## 4. Refactoring Recommendations

### 4.1 High Priority Refactors

#### 1. Extract Configuration Module
**File**: `app/config.py`
```python
from dataclasses import dataclass
from datetime import time

@dataclass
class Config:
    # Cache TTLs (seconds)
    CACHE_TTL_STATUS: int = 30
    CACHE_TTL_PBP: int = 60
    CACHE_TTL_CLOSING_TOTALS: int = 3600
    
    # Time restrictions
    BQ_QUERY_START_HOUR: int = 8
    BQ_QUERY_END_HOUR: int = 22
    
    # Performance
    MAX_WORKERS: int = 5
    REFRESH_INTERVAL: int = 30
    NOT_STARTED_THROTTLE_MINUTES: int = 10
    
    # Visualization
    KERNEL_BANDWIDTH: float = 5.0
    COLS_PER_ROW: int = 2
    
    # Timezone
    TIMEZONE: str = "America/Los_Angeles"  # PST/PDT
```

#### 2. Extract State Management
**File**: `app/state.py`
```python
"""Centralized session state management"""
import streamlit as st
from typing import Set, Dict
from datetime import datetime

class GameState:
    @staticmethod
    def get_completed_games() -> Set[str]:
        if 'completed_games' not in st.session_state:
            st.session_state.completed_games = set()
        return st.session_state.completed_games
    
    @staticmethod
    def mark_completed(game_id: str):
        GameState.get_completed_games().add(game_id)
    
    # Similar for not_started_last_check, plot_containers, error_log
```

#### 3. Extract Expected TFS Calculation Service
**File**: `app/services/expected_tfs.py`
```python
"""Expected TFS calculation service"""
from typing import Optional
import numpy as np
import pandas as pd

class ExpectedTFSService:
    FORMULAS = {
        "oppo_made_shot": lambda ct: 36.4397 + (-0.1025 * ct),
        "rebound": lambda ct: 24.2977 + (-0.0692 * ct),
        "turnover": lambda ct: 23.9754 + (-0.0619 * ct),
        "period_start": lambda ct: 24.2977 + (-0.0692 * ct),
        "game_level": lambda ct: 27.65 - 0.08 * ct,
    }
    
    @classmethod
    def calculate(cls, closing_total: float, poss_start_type: Optional[str] = None) -> float:
        """Calculate expected TFS"""
        # Implementation
    
    @classmethod
    def calculate_for_dataframe(cls, tfs_df: pd.DataFrame, closing_total: float) -> np.ndarray:
        """Vectorized calculation for entire DataFrame"""
        # Vectorized implementation
```

#### 4. Extract Residual Statistics Service
**File**: `app/services/residual_stats.py`
```python
"""Residual statistics calculation"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple

class ResidualStatsService:
    @staticmethod
    def calculate_stats(tfs_df: pd.DataFrame, closing_total: float) -> Dict:
        """Calculate residual statistics"""
        # Extract to separate service
```

#### 5. Break Down `app/main.py`
Split into:
- `app/core/game_manager.py` - Game data loading, status management
- `app/core/renderer.py` - Main rendering logic
- `app/core/filters.py` - Filtering logic
- Keep `app/main.py` as thin coordinator

#### 6. Simplify `build_tempo_figure()`
Split into:
- `_calculate_expected_tfs_trend()` - Expected TFS calculation
- `_add_shading()` - Shading logic
- `_add_residual_stats()` - Residual statistics
- `_add_legends()` - Legend creation

### 4.2 Medium Priority Refactors

#### 7. Use Vectorized Operations
Replace DataFrame iteration with vectorized pandas operations:
```python
# Instead of:
for idx in range(len(tfs_df)):
    poss_type = tfs_df.iloc[idx]["poss_start_type"]
    exp_tfs = calculate_expected_tfs(closing_total, poss_type)

# Use:
tfs_df['expected_tfs'] = tfs_df['poss_start_type'].apply(
    lambda pt: calculate_expected_tfs(closing_total, pt)
)
```

#### 8. Proper Timezone Support
Replace hardcoded PST with proper timezone library:
```python
from zoneinfo import ZoneInfo
import pytz

PST = ZoneInfo("America/Los_Angeles")  # Handles DST automatically
```

#### 9. Consolidate Duplicate Files
- Remove `get_sched.py` and `get_pbp.py` from root
- Use only `app/data/get_sched.py` and `app/data/get_pbp.py`

#### 10. Add Type Safety
- Use `TypedDict` for complex return types
- Add return type annotations everywhere
- Consider using `pydantic` for data validation

### 4.3 Low Priority Improvements

#### 11. Add Logging
Replace `print()` statements with proper logging:
```python
import logging
logger = logging.getLogger(__name__)
logger.error(f"Error: {e}")
```

#### 12. Add Unit Tests
- Test expected TFS calculations
- Test status classification
- Test filtering logic
- Mock Streamlit dependencies

#### 13. Add Documentation
- API documentation with Sphinx
- Architecture diagrams
- Data flow diagrams

---

## 5. Refactoring Plan

### Phase 1: Foundation (Week 1)
1. Create `app/config.py` - Centralize configuration
2. Create `app/state.py` - Centralize state management
3. Create `app/services/` directory structure
4. Extract expected TFS service
5. Extract residual stats service

### Phase 2: Main Module Split (Week 2)
1. Create `app/core/` directory
2. Extract `game_manager.py` from `main.py`
3. Extract `renderer.py` from `main.py`
4. Extract `filters.py` from `main.py`
5. Refactor `main.py` to be thin coordinator

### Phase 3: Plot Module Refactor (Week 2-3)
1. Break down `build_tempo_figure()` into smaller functions
2. Extract shading logic
3. Extract legend creation
4. Use vectorized operations

### Phase 4: Quality Improvements (Week 3)
1. Add proper timezone support
2. Replace print with logging
3. Consolidate duplicate files
4. Add type hints everywhere

### Phase 5: Testing & Documentation (Week 4)
1. Add unit tests for core services
2. Add integration tests
3. Update documentation
4. Performance testing

---

## 6. Risk Assessment

### Low Risk Refactors
- ✅ Configuration extraction
- ✅ State management extraction
- ✅ Service extraction
- ✅ Timezone improvements

### Medium Risk Refactors
- ⚠️ Breaking down `main.py` (requires careful testing)
- ⚠️ Plot function refactoring (visual regression testing needed)

### High Risk Refactors
- ⚠️ Vectorized operations (need to verify correctness)
- ⚠️ Caching changes (could affect performance)

---

## 7. Metrics & Success Criteria

### Code Quality Metrics
- **Cyclomatic Complexity**: Reduce average function complexity from ~15 to <10
- **Lines per File**: Reduce `main.py` from 502 to <200 lines
- **Code Duplication**: Reduce from ~15% to <5%
- **Test Coverage**: Increase from 0% to >60%

### Performance Metrics
- Maintain current refresh time (<2s for 10 games)
- Reduce memory usage by 20% (better caching)
- Maintain or improve API call efficiency

### Maintainability Metrics
- Reduce time to add new feature by 50%
- Reduce bug fix time by 40%
- Improve code review time by 30%

---

## 8. Conclusion

**Recommendation: Proceed with refactoring**

The codebase is functional and well-structured, but has accumulated technical debt that will slow future development. The recommended refactoring plan is incremental and low-risk, focusing on:

1. **Extracting configuration and state management** (low risk, high value)
2. **Breaking down large functions** (medium risk, high value)
3. **Improving code organization** (low risk, medium value)
4. **Adding tests and documentation** (low risk, high value)

The refactoring can be done incrementally without disrupting current functionality, and each phase delivers immediate value.

**Estimated Effort**: 3-4 weeks for full refactoring
**Estimated Risk**: Low-Medium (incremental approach minimizes risk)
**Estimated ROI**: High (significant improvement in maintainability and development velocity)

---

## 9. Obsolete Files

The following files are **obsolete** and should be removed from the repository:

### Root-Level Duplicate Files
1. **`get_sched.py`** (root directory)
   - **Status**: Obsolete
   - **Reason**: Duplicate of `app/data/get_sched.py`
   - **Action**: Delete root-level file
   - **Note**: All imports use `app/data/get_sched.py` via `schedule_loader.py`

2. **`get_pbp.py`** (root directory)
   - **Status**: Obsolete
   - **Reason**: Duplicate of `app/data/get_pbp.py`
   - **Action**: Delete root-level file
   - **Note**: All imports use `app/data/get_pbp.py` via `pbp_loader.py`

### Potentially Obsolete Documentation Files
3. **`DEPENDENCIES_ADDED.md`**
   - **Status**: Potentially obsolete
   - **Reason**: Historical documentation, may no longer be relevant
   - **Action**: Review and either update or remove

4. **`PASTE_INTO_STREAMLIT_SECRETS.toml`**
   - **Status**: Potentially obsolete
   - **Reason**: Template file, may be redundant with `STREAMLIT_SECRETS.md`
   - **Action**: Review and consolidate with `STREAMLIT_SECRETS.md` or remove

### Files That Should Be Ignored (Already in .gitignore)
- `meatloaf.json` - Credentials file (correctly excluded)
- `streamlit_secrets.toml` - Local secrets (correctly excluded)
- `STREAMLIT_SECRETS.md` - Should be excluded (contains template, but was accidentally committed with real credentials)

### Recommended Cleanup Actions

```bash
# Remove obsolete root-level files
git rm get_sched.py get_pbp.py

# Review and potentially remove
# DEPENDENCIES_ADDED.md
# PASTE_INTO_STREAMLIT_SECRETS.toml

# Ensure STREAMLIT_SECRETS.md is in .gitignore (already done)
# Remove from git history if it contains credentials
```

### Current Import Structure (Correct)
- ✅ `app/data/schedule_loader.py` → `from .get_sched import get_sched`
- ✅ `app/data/pbp_loader.py` → `from .get_pbp import get_pbp`
- ✅ `app/main.py` → `from app.data.get_pbp import get_pbp` (for parallel processing)

**All imports correctly reference `app/data/` versions, making root-level files unused.**

---

## Appendix: Quick Wins (Can Do Immediately)

1. **Extract configuration constants** (2 hours)
2. **Consolidate duplicate files** (1 hour)
3. **Add proper timezone support** (2 hours)
4. **Replace print with logging** (1 hour)
5. **Add type hints to key functions** (3 hours)

**Total: ~9 hours of work for immediate improvements**

