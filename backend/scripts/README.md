# Demo Data Update Script

## Overview

The `update_demo_data.py` script keeps demo ticker CSVs up-to-date by fetching the latest data from Alpha Vantage.

## What It Does

1. Reads each demo CSV file (AAPL, MSFT, GOOGL, etc.)
2. Checks the latest date in each file
3. Fetches new data from Alpha Vantage API
4. Appends new rows to the CSV
5. Clears the cache so fresh data is loaded

## Usage

### Manual Run (Recommended)

```bash
cd backend
conda activate ttalk  # or source venv/bin/activate
python scripts/update_demo_data.py
```

Run this once per day to keep data fresh.

### Automated (Optional)

Add to your crontab to run daily at 5 AM:

```bash
# Edit crontab
crontab -e

# Add this line:
0 5 * * * cd /path/to/ticker-talk/backend && /path/to/conda/envs/ttalk/bin/python scripts/update_demo_data.py >> logs/update.log 2>&1
```

## Initial Setup

Before running the script, you need to download initial CSV files for the demo tickers.

### Download from NASDAQ

1. Go to https://www.nasdaq.com/market-activity/stocks/[TICKER]/historical
   - Replace `[TICKER]` with: aapl, msft, googl, amzn, meta, tsla, nvda, nflx, amd, intc

2. Select "MAX" timeframe (10 years)

3. Click "Download Data"

4. Rename downloaded file to `[TICKER].csv` (uppercase)

5. Move to `backend/demo_data/`

Repeat for all 10 tickers.

### CSV Format Requirements

The downloaded NASDAQ CSVs should have these columns:
- Date
- Open
- High
- Low
- Close
- Volume

**Note:** `Adj_Close` is not needed - the loader will use `Close` if `Adj_Close` is missing.

## What Happens Next

Once you have the initial CSV files:

1. The backend will preload them on startup (see `DEMO_TICKERS` in `app/data/demo_data.py`)
2. Users can analyze these tickers immediately (no API calls needed)
3. Run `update_demo_data.py` daily to append the latest day's data
4. The 25/day Alpha Vantage limit is conserved for updates only

## Troubleshooting

**"No API key configured"**
- Set `ALPHA_VANTAGE_API_KEY` in `backend/.env`

**"Rate limited"**
- You've hit the 5 calls/minute limit
- Wait 60 seconds and try again
- Consider running updates at off-peak hours

**"CSV doesn't exist yet"**
- Download the initial CSV files from NASDAQ (see above)

**"No data returned"**
- Check if the ticker symbol is correct
- Verify your API key is valid

## API Limits

Alpha Vantage free tier:
- **25 calls per day**
- **5 calls per minute**

With 10 demo tickers, one daily update uses 10 API calls, leaving 15 for live queries.

## Example Output

```
============================================================
Ticker Talk Demo Data Update - 2026-02-03 05:00:00
============================================================

AAPL:
  Latest date in AAPL.csv: 2026-02-02
  + Adding 1 new row(s) to AAPL.csv

MSFT:
  Latest date in MSFT.csv: 2026-02-02
  + Adding 1 new row(s) to MSFT.csv

...

============================================================
Updated 10/10 tickers
✓ Cache cleared
============================================================
```
