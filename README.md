# TFS Kernel Dashboard

Live dashboard for monitoring Time to First Shot (TFS) tempo in college basketball games.

## Features

- Real-time TFS tempo visualization with kernel smoothing
- Game status filtering (Early 1H, First Half, Halftime, Second Half, Complete)
- Board filtering (Main/Extra)
- Expected TFS based on closing totals
- CUSUM change-point detection
- Parallel processing for efficient game scanning

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Credentials

Create a `meatloaf.json` file in the root directory with your BigQuery service account credentials. This file is excluded from git for security.

### 3. Run the Dashboard

```bash
streamlit run streamlit_app.py
```

## Project Structure

```
dashboard/
├── app/
│   ├── data/          # Data loading (schedule, PBP, BigQuery, status)
│   ├── tfs/           # TFS computation and analysis
│   ├── plots/         # Visualization modules
│   ├── ui/            # UI components
│   ├── util/           # Utilities (caching, time, style, kernel)
│   └── main.py         # Main application entry point
├── builders/           # Action time processing pipeline
├── streamlit_app.py    # Streamlit entry point
└── requirements.txt    # Python dependencies
```

## Configuration

- **Refresh Interval**: 30 seconds
- **BigQuery Query Window**: 8am - 10pm (uses cache outside these hours)
- **Cache TTL**: 
  - Game statuses: 30 seconds
  - Closing totals: 1 hour
  - PBP data: 60 seconds

## Notes

- The dashboard automatically filters to games with closing totals
- Completed games are never re-scanned
- "Not Started" games are checked every 10 minutes
- BigQuery queries are cached and only run during business hours (8am-10pm)

