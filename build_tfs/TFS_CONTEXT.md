# TFS (Time to First Shot) - Technical Context Document

## Executive Summary

**TFS (Time to First Shot)** measures how quickly a team takes its first field goal attempt after gaining possession of the ball. This metric is fundamental to understanding basketball tempo and offensive strategy. This document provides a complete technical specification of how we compute TFS from raw ESPN play-by-play data.

**Key Goal**: Transform raw ESPN API play-by-play JSON into a clean DataFrame containing one TFS measurement per possession, filtered to valid field goal attempts within a 3-40 second range.

---

## 1. What is TFS and Why Does It Matter?

### Conceptual Definition

**Time to First Shot (TFS)** is the elapsed time (in seconds) from when a team gains possession until they attempt their first field goal (not including free throws).

### Basketball Context

- **Fast tempo teams**: Low TFS values (5-10 seconds) indicate aggressive, fast-break offenses
- **Slow tempo teams**: High TFS values (20-30 seconds) indicate deliberate, half-court offenses
- **Tempo changes**: Tracking TFS over time reveals when teams speed up or slow down during a game
- **Predictive value**: TFS patterns can predict game outcomes and betting market movements

### Why We Measure It

1. **Real-time tempo tracking**: Monitor how fast/slow a game is playing in real-time
2. **Expected vs. actual comparison**: Compare actual TFS to expected TFS based on closing totals (betting market data)
3. **Change-point detection**: Identify when teams change tempo mid-game
4. **Kernel smoothing**: Create smooth tempo curves for visualization

---

## 2. Data Pipeline Overview

The TFS computation pipeline consists of three main stages:

```
ESPN API (JSON) 
    ↓
[Stage 1: Fetch] get_pbp.py
    ↓
Raw PBP DataFrame
    ↓
[Stage 2: Preprocess] preprocess.py → builders/action_time/*
    ↓
Preprocessed DataFrame (with possession context)
    ↓
[Stage 3: Compute] compute.py
    ↓
Final TFS DataFrame (one row per possession, filtered)
```

### Stage 1: Fetch (`get_pbp.py`)
- Fetches play-by-play JSON from ESPN Core API
- Flattens nested JSON structure into pandas DataFrame
- Infers home/away teams from scoring plays
- Returns raw PBP with basic fields

### Stage 2: Preprocess (`preprocess.py` + `builders/action_time/*`)
- Cleans and sorts chronological data
- Identifies possession boundaries
- Computes possession context (start time, end time, start type)
- Calculates action time (elapsed time since possession start)

### Stage 3: Compute (`compute.py`)
- Filters to first field goal attempts only
- Applies 3-40 second range filter
- Adds chronological index
- Returns final TFS DataFrame

---

## 3. Detailed Pipeline Stages

### Stage 1: Fetching Raw PBP Data

**File**: `get_pbp.py`  
**Function**: `get_pbp(game_id: int) -> pd.DataFrame`

#### Process

1. **API Request**: 
   - URL: `https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/events/{game_id}/competitions/{game_id}/plays?limit=500`
   - Fetches all plays for the game

2. **JSON Flattening**:
   - Extracts play-level fields: `id`, `sequence_number`, `type_text`, `clock_value`, `period_number`, scores, etc.
   - Fetches team information via `$ref` URLs (cached to avoid duplicate requests)
   - Uses `team.location` (e.g., "Duke") instead of `team.displayName` (e.g., "Duke Blue Devils")

3. **Team Inference**:
   - Finds first scoring play to determine home/away teams
   - Compares `away_score` vs `home_score` to identify scoring team
   - Attaches `away_team_id`, `away_team_name`, `home_team_id`, `home_team_name` to all rows

4. **Data Type Enforcement**:
   - Ensures `scoring_play` and `shooting_play` are boolean (not nullable)
   - Critical for downstream filtering logic

#### Output Schema

```python
Columns:
- game_id (int)
- id, sequence_number, type_id, type_text, text
- away_score, home_score
- period_number, clock_value (float, seconds remaining in period)
- scoring_play (bool), shooting_play (bool)
- score_value, valid, priority, modified
- team_id, team_name, team_abbrev
- away_team_id, away_team_name, home_team_id, home_team_name
```

**Key Fields for TFS**:
- `type_text`: Play type (e.g., "MadeThreePointJumper", "MadeFreeThrow", "Defensive Rebound")
- `clock_value`: Time remaining in period (1200.0 = start of period, 0.0 = end)
- `shooting_play`: Boolean indicating if this is a shooting play
- `period_number`: Period (1, 2, 3, 4, etc.)

---

### Stage 2: Preprocessing Pipeline

**File**: `preprocess.py`  
**Function**: `preprocess_pbp(raw_pbp: pd.DataFrame) -> pd.DataFrame`

The preprocessing pipeline consists of 6 sequential transformations:

#### 2.1 Clean and Sort (`clean_pbp.py`)

**Function**: `load_and_sort_game_csv()`

**Purpose**: Remove noise and establish authoritative chronological ordering.

**Steps**:

1. **Drop incomplete rows**: Remove rows missing `type_text`, `team_id`, or `period_number`

2. **Initial chronological sort**:
   ```python
   Sort by: [game_id, period_number, clock_value DESC, sequence_number]
   ```
   - `clock_value DESC` because higher values = earlier in period (1200.0 = start, 0.0 = end)

3. **Remove noise events**:
   - Filters out: `timeout`, `substitution`, `review`
   - These don't affect possession or shot timing

4. **Remove "FT rebound sandwiches"**:
   - ESPN sometimes inserts fake rebounds between consecutive free throws
   - Pattern: `MadeFreeThrow → Rebound → MadeFreeThrow`
   - Detects and removes the middle rebound

5. **Final sort**: Re-sort after cleaning to ensure chronological order

**Output**: Clean, chronologically sorted PBP where row adjacency represents true basketball time.

#### 2.2 Flag Free Throws (`flag_ft.py`)

**Function**: `add_ft_flags()`

**Purpose**: Identify contiguous free throw sequences.

**Process**:
- Detects sequences of consecutive `MadeFreeThrow` plays
- Marks first and last free throw in each sequence
- Assigns `ft_group_id` to group related free throws

**Output Columns**:
- `first_ft` (bool): True if this is the first FT in a sequence
- `final_ft` (bool): True if this is the last FT in a sequence
- `ft_group_id` (int): Groups consecutive FTs together

**Why It Matters**: Final free throws can transfer possession, affecting possession boundaries.

#### 2.3 Assign Possession Teams (`assign_poss_teams.py`)

**Function**: `add_pre_post_possession()`

**Purpose**: Determine which team has the ball before and after each play.

**Logic** (simplified):

- **Pre-possession** (who has ball before play):
  - Shooting plays → team that shot
  - MadeFreeThrow → team shooting FTs
  - Defensive Rebound → opponent (defensive team gets rebound)
  - Offensive Rebound → same team (offensive team keeps ball)
  - Turnover → team that turned it over
  - Steal → opponent (stealing team gets ball)

- **Post-possession** (who has ball after play):
  - Made shot → opponent (possession transfers)
  - Missed shot → same team (possession stays until rebound determines it)
  - Final FT (scoring) → opponent (possession transfers)
  - Final FT (non-scoring/missed) → **same team** (treated same as missed shot)
    - Note: ESPN records all FTs as "MadeFreeThrow" regardless of make/miss
    - We identify missed FTs as `MadeFreeThrow` with `scoring_play = False`
    - For missed final FTs, possession stays with shooting team (same as missed shot)
    - The next play (rebound) will determine actual possession:
      - If next play is **Defensive Rebound** → defensive team gets possession
      - If next play is **Offensive Rebound** → offensive team keeps possession
  - Turnover → opponent
  - Defensive Rebound → team that got rebound
  - Offensive Rebound → same team

**Output Columns**:
- `has_ball_pre_play` (str): Team name with ball before play
- `has_ball_post_play` (str): Team name with ball after play

**Critical**: Changes in `has_ball_pre_play` between consecutive rows indicate possession changes.

#### 2.4 Build Possession Context (`build_poss_context.py`)

**Function**: `add_possession_context()`

**Purpose**: Identify possession boundaries and compute possession-level metadata.

**Process**:

1. **Detect possession changes**:
   ```python
   change_mask = (
       (df["has_ball_pre_play"] != df["has_ball_pre_play"].shift(1)) |
       (df["period_number"] != df["period_number"].shift(1)) |
       (df["game_id"] != df["game_id"].shift(1))
   )
   ```
   - Possession changes when: team with ball changes, period changes, or game changes

2. **Assign possession_id**:
   - Cumulative sum of `change_mask` within each game
   - Each possession gets unique `possession_id` (resets per game)

3. **Compute possession times**:
   - `poss_start_time`: Clock value when possession started
     - If previous play exists in same period → use previous play's `clock_value`
     - Otherwise → 1200.0 (start of period)
   - `poss_end_time`: Clock value when possession ended (last play's `clock_value`)

**Output Columns**:
- `possession_id` (int): Unique ID for each possession (1, 2, 3, ...)
- `poss_start_time` (float): Clock value at possession start
- `poss_end_time` (float): Clock value at possession end

#### 2.5 Add Possession Start Type (`add_poss_start_type.py`)

**Function**: `add_poss_start_type()`

**Purpose**: Classify how each possession started (important for expected TFS calculations).

**Process**:

For each possession, examine the **last action of the previous possession**:

1. **Rebound**: Previous ended with "Defensive Rebound" or "Dead Ball Rebound"
2. **Turnover**: Previous possession contained any action with "turnover" in `type_text`
3. **Opponent Made Free Throw**: Previous ended with made final free throw (`final_ft == True` and `scoring_play == True`)
   - **Important**: Clock stops after made FTs until ball is inbounded, affecting TFS timing
4. **Opponent Made Shot**: Previous ended with made field goal (scoring play with `score_value > 1`)
   - **Important**: Clock continues running after made shots while team is inbounding
5. **Period Start**: First possession of each period

**Output Column**:
- `poss_start_type` (str): One of `"rebound"`, `"turnover"`, `"oppo_made_ft"`, `"oppo_made_shot"`, `"period_start"`, or `None`

**Why It Matters**: Different possession start types have different expected TFS values (used in linear regression models). The distinction between `"oppo_made_ft"` and `"oppo_made_shot"` is critical because:
- **Made FTs**: Clock stops until ball is inbounded → typically shorter TFS
- **Made Shots**: Clock continues running during inbound → typically longer TFS

#### 2.6 Build Action Context (`build_action_context.py`)

**Function**: `add_action_context()`

**Purpose**: Compute time-based metrics for each play.

**Process**:

1. **Compute `action_time`** (elapsed time since possession start):
   ```python
   action_time = poss_start_time - clock_value
   ```
   - Example: If possession started at 1200.0 and play occurs at 1195.0, `action_time = 5.0` seconds
   - This is the **TFS value** for first shots

2. **Compute `shot_count_in_poss`** (shot attempt number within possession):
   ```python
   shot_counter = (
       df["shooting_play"]
       .astype(int)
       .groupby([game_id, period_number, possession_id])
       .cumsum()
   )
   df["shot_count_in_poss"] = shot_counter.where(df["shooting_play"], 0)
   ```
   - Counts shooting plays within each possession
   - Non-shooting plays get `shot_count_in_poss = 0`
   - First shot in possession has `shot_count_in_poss = 1`

**Output Columns**:
- `action_time` (float): Seconds elapsed since possession start (TFS for first shots)
- `shot_count_in_poss` (int): Shot attempt number (1 = first shot, 2 = second shot, etc.)

**Critical**: `action_time` where `shot_count_in_poss == 1` is the TFS value we want.

---

### Stage 3: TFS Computation and Filtering

**File**: `compute.py`  
**Function**: `compute_tfs(df: pd.DataFrame) -> pd.DataFrame`

**Purpose**: Filter preprocessed data to final TFS output (one row per possession, first field goal attempts only).

#### Step 1: Build Detailed TFS (`build_tfs_detailed.py`)

**Function**: `build_tfs_detailed()`

**Purpose**: Recalculate shot counts using regex matching (more reliable than ESPN's `shooting_play` flag).

**Process**:

1. **Identify shot attempts via regex**:
   ```python
   shot_mask = df["type_text"].str.contains(
       "jumper|three|layup|dunk|shot", case=False, na=False
   )
   ```
   - Matches: "MadeThreePointJumper", "MissedLayup", "Dunk", etc.

2. **Recalculate `shot_count_in_poss`**:
   - Uses regex-based shot detection instead of ESPN's `shooting_play` flag
   - More reliable for edge cases

3. **Add `tfs` column**:
   ```python
   df["tfs"] = df["action_time"].where(df["shot_count_in_poss"] == 1)
   ```
   - Only first shots get TFS values (others are NaN)

#### Step 2: Filter to First Field Goal Attempts

**Critical Fix**: We must exclude non-shooting plays that might have `shot_count_in_poss == 1`.

**Problem (Quadruple Counting Bug)**:
- Original code only checked `shot_count_in_poss == 1`
- This allowed free throws, rebounds, and fouls to pass through
- Result: Multiple TFS entries per possession (e.g., 92 entries for 49 possessions)

**Solution**:
```python
# Identify field goal attempts (exclude free throws, rebounds, fouls)
is_field_goal_attempt = (
    df["type_text"].str.contains("jumper|three|layup|dunk|shot", case=False, na=False) &
    ~df["type_text"].str.contains("freethrow", case=False, na=False) &
    ~df["type_text"].str.contains("rebound|foul", case=False, na=False)
)

# Filter to first field goal attempts only
tfs_df = df[
    (df["shot_count_in_poss"] == 1) &
    is_field_goal_attempt
].copy()
```

**What Gets Excluded**:
- ❌ Free throws (`MadeFreeThrow`)
- ❌ Rebounds (`Offensive Rebound`, `Defensive Rebound`)
- ❌ Fouls (`PersonalFoul`, `Technical Foul`)
- ❌ Any play that isn't a field goal attempt

**What Gets Included**:
- ✅ Made field goals (`MadeThreePointJumper`, `MadeLayup`, etc.)
- ✅ Missed field goals (`MissedJumper`, `MissedThreePointJumper`, etc.)
- ✅ Only the **first** field goal attempt in each possession

#### Step 3: Filter to Valid Time Range (3-40 seconds)

**Purpose**: Remove outliers and edge cases.

**Rationale**:
- **< 3 seconds**: Likely data errors, fast breaks that don't represent normal tempo
- **> 40 seconds**: Likely clock errors, end-of-period situations, or data quality issues

**Implementation**:
```python
tfs_df = tfs_df[
    (tfs_df["action_time"] >= 3) &
    (tfs_df["action_time"] <= 40)
]
```

#### Step 4: Add Chronological Index

**Purpose**: Enable sequential analysis and visualization.

**Process**:
```python
tfs_df = tfs_df.sort_values(["period_number", "possession_id"])
tfs_df["chrono_index"] = range(1, len(tfs_df) + 1)
```

**Output**: Sequential index (1, 2, 3, ...) representing order of possessions in the game.

---

## 4. Final Output Schema

### TFS DataFrame Columns

**Core TFS Metrics**:
- `action_time` (float): **TFS value in seconds** (primary output)
- `chrono_index` (int): Sequential possession number (1, 2, 3, ...)

**Game Context**:
- `game_id` (int): ESPN game ID
- `period_number` (int): Period (1, 2, 3, 4, ...)
- `possession_id` (int): Unique possession ID within game

**Possession Context**:
- `poss_start_type` (str): How possession started (`"rebound"`, `"turnover"`, `"oppo_made_ft"`, `"oppo_made_shot"`, `"period_start"`)
- `poss_start_time` (float): Clock value at possession start
- `poss_end_time` (float): Clock value at possession end

**Play Details**:
- `type_text` (str): Play type (e.g., "MadeThreePointJumper")
- `clock_value` (float): Time remaining in period when play occurred
- `team_name` (str): Team that took the shot
- `away_team_name`, `home_team_name` (str): Team names

**Metadata**:
- `shot_count_in_poss` (int): Always 1 (we filter to first shots only)
- `tfs` (float): Same as `action_time` (redundant, kept for compatibility)

### Expected Output Characteristics

1. **One row per possession**: Each `possession_id` appears exactly once
2. **Only field goal attempts**: No free throws, rebounds, or fouls
3. **Valid time range**: All `action_time` values between 3-40 seconds
4. **Chronological order**: `chrono_index` increases sequentially
5. **Complete possession context**: All possession metadata included

### Example Output

```
   game_id  period_number  possession_id  action_time  type_text              poss_start_type  chrono_index
0  401700174  1              1             12.5         MadeThreePointJumper   period_start    1
1  401700174  1              2             8.3         MissedJumper          rebound         2
2  401700174  1              3             15.7         MadeLayup              oppo_made_shot  3
...
```

---

## 5. Key Technical Decisions

### 5.1 Why We Use `clock_value` Instead of Absolute Time

**Decision**: Use `clock_value` (seconds remaining in period) instead of wall-clock time.

**Rationale**:
- ESPN provides `clock_value` directly (more reliable)
- Wall-clock time requires parsing timestamps (error-prone)
- `clock_value` is sufficient for computing elapsed time within possessions

**Calculation**:
```python
action_time = poss_start_time - clock_value
```

### 5.2 Why We Filter to First Shots Only

**Decision**: Only include first field goal attempt in each possession.

**Rationale**:
- TFS measures "time to **first** shot" (by definition)
- Second/third shots don't represent possession tempo
- Including multiple shots per possession would inflate sample size incorrectly

### 5.3 Why We Exclude Free Throws

**Decision**: Free throws are not "field goal attempts" for TFS purposes.

**Rationale**:
- Free throws are not part of normal offensive flow
- They occur after fouls (not possession starts)
- Including them would skew tempo metrics
- Free throws have different expected timing than field goals

### 5.4 Why We Use 3-40 Second Range

**Decision**: Filter to `action_time` between 3 and 40 seconds.

**Rationale**:
- **< 3 seconds**: Likely data errors or extremely rare fast breaks
- **> 40 seconds**: Likely clock errors, end-of-period situations, or data quality issues
- Most legitimate possessions fall in 5-30 second range
- This range captures 99%+ of valid possessions while filtering outliers

### 5.5 Why We Recalculate Shot Counts

**Decision**: Use regex-based shot detection instead of ESPN's `shooting_play` flag.

**Rationale**:
- ESPN's `shooting_play` flag can be inconsistent
- Regex matching on `type_text` is more reliable
- Allows explicit control over what counts as a "shot"

---

## 6. Edge Cases and Data Quality Issues

### 6.1 Multiple Plays at Same Clock Time

**Issue**: Multiple plays can occur at the exact same `clock_value` (e.g., shot + block + foul).

**Handling**:
- Sort by `sequence_number` as tie-breaker
- All plays at same clock time are processed in sequence
- Possession boundaries are detected correctly via `has_ball_pre_play` changes

### 6.2 Free Throw "Rebound Sandwiches"

**Issue**: ESPN inserts fake rebounds between consecutive free throws.

**Handling**:
- `clean_pbp.py` detects pattern: `MadeFreeThrow → Rebound → MadeFreeThrow`
- Removes the middle rebound before processing

### 6.3 Possession Boundary Detection

**Issue**: Determining when possession changes can be ambiguous (e.g., blocked shots, offensive rebounds).

**Handling**:
- `assign_poss_teams.py` uses explicit logic for each play type
- `build_poss_context.py` detects changes in `has_ball_pre_play`
- Period boundaries always create new possessions

### 6.4 Missing or Incomplete Data

**Issue**: ESPN API may return incomplete data (missing fields, null values).

**Handling**:
- `clean_pbp.py` drops rows missing critical fields
- Boolean fields are enforced (fillna + astype(bool))
- Empty DataFrames raise `ValueError` with descriptive messages

### 6.5 Quadruple Counting Bug (Fixed)

**Issue**: Original code allowed non-shooting plays to pass through filters.

**Root Cause**: Only checked `shot_count_in_poss == 1` without verifying play type.

**Fix**: Added explicit field goal attempt detection with exclusions for free throws, rebounds, and fouls.

**Impact**: Reduced TFS entries from ~92 per 49 possessions to ~49 (one per possession).

---

## 7. Usage Examples

### Basic Usage

```python
from process_game import process_game

# Process a game
tfs_df = process_game(game_id=401700174)

# Access TFS values
tfs_values = tfs_df["action_time"].values  # Array of TFS values in seconds

# Filter by possession start type
rebound_tfs = tfs_df[tfs_df["poss_start_type"] == "rebound"]["action_time"]
turnover_tfs = tfs_df[tfs_df["poss_start_type"] == "turnover"]["action_time"]
ft_tfs = tfs_df[tfs_df["poss_start_type"] == "oppo_made_ft"]["action_time"]
shot_tfs = tfs_df[tfs_df["poss_start_type"] == "oppo_made_shot"]["action_time"]
```

### Step-by-Step Usage

```python
from get_pbp import get_pbp
from preprocess import preprocess_pbp
from compute import compute_tfs

# Stage 1: Fetch
raw_pbp = get_pbp(game_id=401700174)

# Stage 2: Preprocess
df = preprocess_pbp(raw_pbp)

# Stage 3: Compute TFS
tfs_df = compute_tfs(df)

# Analyze
print(f"Average TFS: {tfs_df['action_time'].mean():.2f} seconds")
print(f"Total possessions: {len(tfs_df)}")
```

### Filtering and Analysis

```python
# Filter by period
first_half = tfs_df[tfs_df["period_number"] <= 2]

# Filter by possession start type
fast_break = tfs_df[tfs_df["poss_start_type"] == "turnover"]

# Compute statistics
stats = tfs_df.groupby("poss_start_type")["action_time"].agg(["mean", "median", "std"])
```

---

## 8. Dependencies and Requirements

### Python Packages

- `pandas >= 1.5.0`: DataFrame operations
- `numpy >= 1.20.0`: Numerical operations
- `requests >= 2.28.0`: HTTP requests to ESPN API

### External Dependencies

- **ESPN Core API**: Public API (no authentication required)
- **Internet connection**: Required for fetching PBP data

### No External Data Storage

- All processing is in-memory (pandas DataFrames)
- No database or file system dependencies
- No caching (can be added if needed)

---

## 9. Performance Characteristics

### Typical Processing Times

- **Fetch**: 1-3 seconds (depends on ESPN API response time)
- **Preprocess**: 0.5-2 seconds (depends on number of plays, typically 200-500 plays per game)
- **Compute**: < 0.1 seconds (filtering is fast)

**Total**: ~2-5 seconds per game

### Scalability

- **Single game**: Fast (seconds)
- **Batch processing**: Can process hundreds of games sequentially
- **Parallelization**: Can be parallelized (each game is independent)

### Memory Usage

- **Raw PBP**: ~1-5 MB per game (depends on number of plays)
- **Preprocessed**: ~2-10 MB per game (additional columns)
- **Final TFS**: ~50-200 KB per game (filtered to ~50-100 rows)

---

## 10. Testing and Validation

### Validation Checks

1. **One TFS entry per possession**: `len(tfs_df) == tfs_df["possession_id"].nunique()`
2. **Valid time range**: All `action_time` values between 3-40 seconds
3. **Only field goal attempts**: No free throws, rebounds, or fouls in `type_text`
4. **Chronological order**: `chrono_index` increases sequentially
5. **Complete data**: No missing values in critical columns

### Known Issues

- **ESPN API rate limits**: No official rate limit, but excessive requests may be throttled
- **Missing games**: Some games may not have PBP data available
- **Data quality**: ESPN data can have occasional errors (handled via filtering)

---

## 11. Future Enhancements

### Potential Improvements

1. **Caching**: Cache raw PBP data to avoid repeated API calls
2. **Batch processing**: Process multiple games in parallel
3. **Error handling**: More robust error handling for API failures
4. **Data validation**: Additional validation checks for data quality
5. **Alternative data sources**: Support for other PBP data sources (if needed)

### Not in Scope

- Real-time updates (this is a batch processing module)
- Database storage (all processing is in-memory)
- Visualization (separate module handles plotting)
- Expected TFS calculations (separate module handles linear regression)

---

## 12. Summary

This TFS processing module provides a complete, production-ready pipeline for computing Time to First Shot metrics from ESPN play-by-play data. The pipeline:

1. **Fetches** raw PBP data from ESPN API
2. **Preprocesses** data through 6 sequential transformations
3. **Computes** TFS by filtering to first field goal attempts (3-40 seconds)

**Key Features**:
- ✅ Handles edge cases (free throw rebounds, multiple plays at same time)
- ✅ Filters out non-shooting plays (fixes quadruple counting bug)
- ✅ Provides complete possession context (start type, times, etc.)
- ✅ Validates data quality (time range, completeness)
- ✅ Standalone and portable (no external dependencies beyond packages)

**Output**: Clean DataFrame with one TFS value per possession, ready for analysis, visualization, or further processing.

---

## Appendix: File Reference

### Core Files

- `get_pbp.py`: ESPN API fetching
- `preprocess.py`: Main preprocessing orchestrator
- `compute.py`: TFS computation and filtering
- `process_game.py`: Main entry point

### Builder Files (`builders/action_time/`)

- `clean_pbp.py`: Clean and sort PBP
- `flag_ft.py`: Flag free throw sequences
- `assign_poss_teams.py`: Determine possession teams
- `build_poss_context.py`: Build possession boundaries
- `add_poss_start_type.py`: Classify possession starts
- `build_action_context.py`: Compute action time and shot counts
- `build_tfs_detailed.py`: Build detailed TFS metrics

### Documentation

- `README.md`: Quick start guide
- `TFS_CONTEXT.md`: This document (detailed technical specification)

