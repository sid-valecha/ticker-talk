#!/usr/bin/env python3
"""Update demo ticker CSVs using Alpha Vantage with quota-aware batching.

Usage examples:
    python scripts/update_demo_data.py
    python scripts/update_demo_data.py --mode all
    python scripts/update_demo_data.py --mode ticker --ticker AAPL

Default behavior uses rotating batches to stay within free-tier limits.
"""

from __future__ import annotations

import argparse
import math
import sqlite3
import sys
import time
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd

# Add backend root to import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.data.alpha_vantage import (
    AlphaVantageAPIError,
    AlphaVantageInvalidTicker,
    AlphaVantageRateLimit,
    fetch_daily,
)
from app.data.cache import get_db_path
from app.data.demo_data import DEMO_TICKERS, load_demo_data

DEMO_DATA_DIR = Path(__file__).resolve().parent.parent / "demo_data"


def _normalize_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().reset_index()
    if "index" in out.columns:
        out.rename(columns={"index": "date"}, inplace=True)

    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    cols = ["date", "open", "high", "low", "close", "adj_close", "volume"]
    return out[cols]


def _rotation_batch(tickers: list[str], batch_size: int, offset_days: int = 0) -> list[str]:
    if not tickers:
        return []

    size = max(1, batch_size)
    batches = math.ceil(len(tickers) / size)
    batch_index = (date.today().toordinal() + offset_days) % batches
    start = batch_index * size
    end = start + size
    return tickers[start:end]


def _select_tickers(args: argparse.Namespace) -> list[str]:
    all_tickers = sorted(set(t.upper() for t in DEMO_TICKERS))

    if args.mode == "ticker":
        if not args.ticker:
            raise ValueError("--ticker is required when mode=ticker")
        ticker = args.ticker.strip().upper()
        if ticker not in all_tickers:
            raise ValueError(f"Ticker {ticker} is not in DEMO_TICKERS")
        return [ticker]

    if args.mode == "all":
        return all_tickers

    return _rotation_batch(all_tickers, batch_size=args.batch_size, offset_days=args.offset_days)


def _append_new_rows(ticker: str) -> bool:
    csv_path = DEMO_DATA_DIR / f"{ticker}.csv"
    existing = load_demo_data(ticker)

    if existing is None:
        print(f"  ✗ Missing demo CSV for {ticker}: {csv_path}")
        return False

    try:
        api_df = fetch_daily(ticker)
    except AlphaVantageRateLimit as exc:
        print(f"  ✗ Rate limited for {ticker}: {exc}")
        return False
    except AlphaVantageInvalidTicker as exc:
        print(f"  ✗ Invalid ticker for {ticker}: {exc}")
        return False
    except AlphaVantageAPIError as exc:
        print(f"  ✗ API error for {ticker}: {exc}")
        return False

    latest_existing = existing.index.max()
    new_rows = api_df[api_df.index > latest_existing]

    if new_rows.empty:
        print(f"  ✓ {ticker} already up-to-date (max {latest_existing.strftime('%Y-%m-%d')})")
        return False

    combined = pd.concat([existing, new_rows]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]

    _normalize_for_csv(combined).to_csv(csv_path, index=False)
    print(
        f"  + Added {len(new_rows)} row(s) for {ticker} "
        f"({latest_existing.strftime('%Y-%m-%d')} → {combined.index.max().strftime('%Y-%m-%d')})"
    )
    return True


def _clear_cache_for_tickers(tickers: Iterable[str]) -> None:
    db_path = get_db_path()
    if not db_path.exists():
        return

    ticker_list = [t.upper() for t in tickers]
    placeholders = ",".join(["?"] * len(ticker_list))

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            f"DELETE FROM ticker_cache WHERE ticker IN ({placeholders})",
            ticker_list,
        )
        conn.commit()

    print(f"✓ Cleared cache for {len(ticker_list)} ticker(s)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update demo data CSV files")
    parser.add_argument(
        "--mode",
        choices=["rotate", "all", "ticker"],
        default="rotate",
        help="rotate (default): update a daily batch; all: update all demo tickers; ticker: one ticker",
    )
    parser.add_argument("--ticker", help="Ticker symbol when --mode=ticker")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Batch size for rotate mode (default: 5)",
    )
    parser.add_argument(
        "--offset-days",
        type=int,
        default=0,
        help="Rotation offset in days (default: 0)",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=12.0,
        help="Delay between API calls to respect 5/min limit (default: 12s)",
    )
    return parser.parse_args()


def main() -> int:
    if not settings.ALPHA_VANTAGE_API_KEY:
        print("ALPHA_VANTAGE_API_KEY is required")
        return 1

    args = parse_args()

    try:
        tickers = _select_tickers(args)
    except ValueError as exc:
        print(str(exc))
        return 1

    if not tickers:
        print("No tickers selected")
        return 0

    print("=" * 60)
    print(f"Demo Data Update ({args.mode})")
    print("Tickers:", ", ".join(tickers))
    print("=" * 60)

    updated: list[str] = []
    for idx, ticker in enumerate(tickers):
        print(f"\n{ticker}:")
        if _append_new_rows(ticker):
            updated.append(ticker)

        # Alpha Vantage free tier is 5 calls/minute.
        if idx < len(tickers) - 1:
            time.sleep(max(0.0, args.sleep_seconds))

    print("\n" + "=" * 60)
    print(f"Updated {len(updated)}/{len(tickers)} ticker(s)")

    if updated:
        _clear_cache_for_tickers(updated)

    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
