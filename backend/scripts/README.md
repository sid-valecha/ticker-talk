# Demo Data Update Script

## Overview

`update_demo_data.py` refreshes demo CSVs from Alpha Vantage and clears cache entries for updated tickers.

Default mode is **rotating batch refresh** (5 tickers/day), which is safer for Alpha Vantage free-tier limits.

## Usage

```bash
cd backend
python3 scripts/update_demo_data.py
```

### Modes

```bash
# Default: rotate through demo tickers in daily batches
python3 scripts/update_demo_data.py --mode rotate --batch-size 5

# Refresh all demo tickers (slow; uses near full daily quota)
python3 scripts/update_demo_data.py --mode all

# Refresh one ticker
python3 scripts/update_demo_data.py --mode ticker --ticker AAPL
```

## Key behavior

- Uses `TIME_SERIES_DAILY` via the shared Alpha Vantage client.
- Writes normalized CSV columns: `date, open, high, low, close, adj_close, volume`.
- Clears only cache rows for tickers actually updated.
- Sleeps between calls (default `12s`) to respect 5 calls/minute.

## Scheduling recommendation

Run once daily:

```bash
0 5 * * * cd /path/to/ticker-talk/backend && /usr/bin/python3 scripts/update_demo_data.py --mode rotate --batch-size 5 >> logs/update.log 2>&1
```

This refreshes all 25 demo tickers over ~5 days while leaving API headroom for live requests.

## Requirements

- `ALPHA_VANTAGE_API_KEY` must be set.
- Demo CSV files must already exist in `backend/demo_data/`.
