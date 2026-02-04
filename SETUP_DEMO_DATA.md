# Demo Data Setup Guide

## Quick Start

To get your ticker-talk app working with 10 popular stocks, follow these steps:

### Step 1: Download Historical Data from NASDAQ

For each ticker (AAPL, MSFT, GOOGL, AMZN, META, TSLA, NVDA, NFLX, AMD, INTC):

1. **Go to NASDAQ historical data page:**
   ```
   https://www.nasdaq.com/market-activity/stocks/[ticker]/historical
   ```
   Replace `[ticker]` with: `aapl`, `msft`, `googl`, `amzn`, `meta`, `tsla`, `nvda`, `nflx`, `amd`, `intc`

2. **Select timeframe:** Click "MAX" to get 10 years of data

3. **Download:** Click "Download Data" button

4. **Rename file:** Change downloaded filename to `[TICKER].csv` (uppercase)
   - Example: `HistoricalData_1234567890.csv` → `AAPL.csv`

5. **Move to demo_data folder:**
   ```bash
   mv AAPL.csv /Users/sidvalecha/Developer/ticker-talk/backend/demo_data/
   ```

### Step 2: Verify CSV Format

Open one of the CSVs to verify it has these columns:
```
Date, Close, Volume, Open, High, Low
```

**Note:** `Adj_Close` is optional - the app will use `Close` if `Adj_Close` is missing.

### Step 3: Restart Backend

```bash
cd backend
conda activate ttalk
uvicorn app.main:app --reload
```

You should see in the logs:
```
INFO: Application startup complete.
```

### Step 4: Test the App

1. Open frontend: http://localhost:3000
2. You should see a dropdown with all 10 tickers
3. Select AAPL → Click "Analyze"
4. Results should load from your local CSV (no API call needed!)

## Daily Updates

To keep data fresh, run the update script once per day:

```bash
cd backend
conda activate ttalk
python scripts/update_demo_data.py
```

This fetches the latest data from Alpha Vantage and appends it to your CSVs.

See `backend/scripts/README.md` for automation options (cron job).

## Tickers Included

1. **AAPL** - Apple Inc.
2. **MSFT** - Microsoft Corporation
3. **GOOGL** - Alphabet Inc. (Google)
4. **AMZN** - Amazon.com Inc.
5. **META** - Meta Platforms Inc. (Facebook)
6. **TSLA** - Tesla Inc.
7. **NVDA** - NVIDIA Corporation
8. **NFLX** - Netflix Inc.
9. **AMD** - Advanced Micro Devices Inc.
10. **INTC** - Intel Corporation

## Why This Approach?

- ✅ **Reliable:** Works 100% of the time (no API failures)
- ✅ **Fast:** No API latency, data loads instantly from cache
- ✅ **Full history:** 10 years of data for accurate forecasting
- ✅ **API efficient:** Saves Alpha Vantage calls for updates only
- ✅ **Resume-ready:** Shows production thinking (caching, fallbacks)

## Troubleshooting

**"Stock data unavailable" error:**
- CSV file doesn't exist in `backend/demo_data/`
- Download from NASDAQ and place in correct folder

**Dropdown is empty:**
- Backend not running
- Check CORS errors in browser console
- Verify `/api/tickers` endpoint works: `curl http://localhost:8000/api/tickers`

**Old data showing:**
- Run `python scripts/update_demo_data.py` to fetch latest
- Or manually download fresh CSV from NASDAQ

## Alternative: Use BMW as Template

If you want to start with just one ticker:

1. `backend/demo_data/BMW.csv` already exists with full history
2. Open it and see the format
3. Download similar CSVs for other tickers from NASDAQ
4. Follow the same column structure

## Next Steps

Once you have 5-10 tickers loaded:
- Add LLM explanations (see plan.md Phase 3)
- Deploy to Railway + Vercel
- Add to your resume with live demo link!
