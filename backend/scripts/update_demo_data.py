#!/usr/bin/env python
"""
Daily update script for demo data.

Usage:
    python scripts/update_demo_data.py

This script:
1. Reads the latest date from each demo data CSV
2. Fetches the latest data from Alpha Vantage API
3. Appends new rows to the CSV files
4. Clears the cache so fresh data is loaded

Run this once per day (e.g., via cron: 0 5 * * * cd /path/to/backend && python scripts/update_demo_data.py)
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

# Add parent directory to path so we can import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.data.cache import get_db_connection

DEMO_DATA_DIR = Path(__file__).parent.parent / "demo_data"
DEMO_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "NFLX", "AMD", "INTC"]
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"


def fetch_latest_data(ticker: str) -> pd.DataFrame | None:
    """Fetch latest data from Alpha Vantage using TIME_SERIES_DAILY."""
    if not settings.ALPHA_VANTAGE_API_KEY:
        print(f"  ⚠️  No API key configured, skipping {ticker}")
        return None

    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": ticker,
        "apikey": settings.ALPHA_VANTAGE_API_KEY,
        "outputsize": "full",
    }

    try:
        response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        # Check for errors
        if "Error Message" in data:
            print(f"  ✗ API error for {ticker}: {data['Error Message']}")
            return None

        if "Note" in data:  # Rate limited
            print(f"  ✗ Rate limited for {ticker}")
            return None

        if "Time Series (Daily)" not in data:
            print(f"  ✗ No data returned for {ticker}")
            return None

        # Parse time series
        ts = data["Time Series (Daily)"]
        records = []

        for date_str, values in ts.items():
            records.append({
                "date": pd.to_datetime(date_str),
                "open": float(values["1. open"]),
                "high": float(values["2. high"]),
                "low": float(values["3. low"]),
                "close": float(values["4. close"]),
                "volume": int(values["5. volume"]),
            })

        df = pd.DataFrame(records)
        df.sort_values("date", inplace=True)
        return df

    except requests.RequestException as e:
        print(f"  ✗ Network error for {ticker}: {e}")
        return None
    except Exception as e:
        print(f"  ✗ Error parsing data for {ticker}: {e}")
        return None


def update_csv(ticker: str) -> bool:
    """Update CSV file by appending new data."""
    csv_path = DEMO_DATA_DIR / f"{ticker}.csv"

    # Check if file exists
    if not csv_path.exists():
        print(f"  ℹ️  {ticker}.csv doesn't exist yet, skipping")
        return False

    # Read existing data
    try:
        existing_df = pd.read_csv(csv_path)
        existing_df.columns = [col.strip().lower() for col in existing_df.columns]
        existing_df["date"] = pd.to_datetime(existing_df["date"])
    except Exception as e:
        print(f"  ✗ Error reading {ticker}.csv: {e}")
        return False

    # Fetch new data
    new_df = fetch_latest_data(ticker)
    if new_df is None:
        return False

    # Find latest date in existing data
    latest_date = existing_df["date"].max()
    print(f"  Latest date in {ticker}.csv: {latest_date.strftime('%Y-%m-%d')}")

    # Filter to only new data
    new_rows = new_df[new_df["date"] > latest_date]

    if len(new_rows) == 0:
        print(f"  ✓ {ticker} is already up-to-date")
        return False

    print(f"  + Adding {len(new_rows)} new row(s) to {ticker}.csv")

    # Append to existing data
    combined_df = pd.concat([existing_df, new_rows], ignore_index=True)
    combined_df.sort_values("date", inplace=True)
    combined_df.drop_duplicates(subset=["date"], inplace=True)

    # Write back
    try:
        combined_df.to_csv(csv_path, index=False)
        return True
    except Exception as e:
        print(f"  ✗ Error writing {ticker}.csv: {e}")
        return False


def clear_cache() -> None:
    """Clear the cache so new data is loaded on next request."""
    try:
        conn = get_db_connection()
        conn.execute("DELETE FROM ticker_cache")
        conn.execute("DELETE FROM request_metrics")
        conn.commit()
        conn.close()
        print("✓ Cache cleared")
    except Exception as e:
        print(f"⚠️  Error clearing cache: {e}")


def main() -> None:
    """Update all demo data files."""
    print("=" * 60)
    print(f"Ticker Talk Demo Data Update - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    updated_count = 0

    for ticker in DEMO_TICKERS:
        print(f"\n{ticker}:")
        if update_csv(ticker):
            updated_count += 1

    print("\n" + "=" * 60)
    print(f"Updated {updated_count}/{len(DEMO_TICKERS)} tickers")

    if updated_count > 0:
        clear_cache()

    print("=" * 60)


if __name__ == "__main__":
    main()
