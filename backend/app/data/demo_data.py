from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd

from app.config import settings
from app.data.alpha_vantage import (
    AlphaVantageAPIError,
    AlphaVantageInvalidTicker,
    AlphaVantageRateLimit,
    fetch_daily_adjusted,
)
from app.data.cache import get_cached_data, store_data

DEMO_TICKERS = [
    "AAPL", "AMD", "AMZN", "BA", "BAC", "CAT", "CVX", "DIS", "GOOGL",
    "GS", "INTC", "JNJ", "JPM", "LMT", "META", "MSFT", "NEE", "NFLX", "NOC",
    "NVDA", "PFE", "TSLA", "UNH", "WMT", "XOM"
]
DEMO_DATA_DIR_ENV = "TICKER_TALK_DEMO_DATA_DIR"


def get_demo_data_dir() -> Path:
    env_path = os.getenv(DEMO_DATA_DIR_ENV)
    if env_path:
        return Path(env_path)
    return Path(__file__).resolve().parents[2] / "demo_data"


def load_demo_data(ticker: str) -> Optional[pd.DataFrame]:
    demo_dir = get_demo_data_dir()
    path = demo_dir / f"{ticker}.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path)

    # Normalize column names: lowercase and strip spaces
    df.columns = [col.strip().lower().replace("/", "_") for col in df.columns]

    # Handle NASDAQ format: "close/last" -> "close_last"
    column_mapping = {
        "close_last": "close",
        "date": "date",
    }
    df.rename(columns=column_mapping, inplace=True)

    # Clean ALL numeric columns (prices and volume)
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in df.columns:
            # Convert to string first (in case it's already partially numeric)
            df[col] = df[col].astype(str)
            # Remove dollar signs, commas, and whitespace
            df[col] = df[col].str.replace("$", "", regex=False).str.replace(",", "", regex=False).str.strip()
            # Convert to numeric
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Required columns
    required = ["date", "open", "high", "low", "close", "volume"]
    if not set(required).issubset(set(df.columns)):
        return None

    # Use adj_close if available, otherwise fall back to close
    if "adj_close" not in df.columns:
        df["adj_close"] = df["close"].copy()
    else:
        # Ensure adj_close is also numeric if it exists
        df["adj_close"] = df["adj_close"].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False).str.strip()
        df["adj_close"] = pd.to_numeric(df["adj_close"], errors="coerce")

    # Select columns and ensure all price columns are float
    df = df[required + ["adj_close"]].copy()
    for col in ["open", "high", "low", "close", "adj_close"]:
        df[col] = df[col].astype(float)

    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df.set_index("date", inplace=True)
    return df


def preload_demo_data_if_needed() -> None:
    for ticker in DEMO_TICKERS:
        cached = get_cached_data(ticker, ttl_hours=settings.CACHE_TTL_HOURS)
        if cached:
            continue

        demo_df = load_demo_data(ticker)
        if demo_df is not None:
            store_data(ticker, demo_df, source="demo")
            continue

        if settings.ALPHA_VANTAGE_API_KEY:
            try:
                api_df = fetch_daily_adjusted(ticker)
                store_data(ticker, api_df, source="alpha_vantage")
                continue
            except (AlphaVantageAPIError, AlphaVantageRateLimit, AlphaVantageInvalidTicker):
                pass

        raise RuntimeError(
            "Demo data missing for preloaded tickers and no Alpha Vantage API key available."
        )
