import os
import sqlite3
from datetime import datetime, timedelta

import pandas as pd

from app.config import settings
from app.data.cache import get_cached_data, get_db_path, init_db, is_cache_valid, store_data
from app.data.demo_data import DEMO_DATA_DIR_ENV, DEMO_TICKERS, preload_demo_data_if_needed


def _sample_df() -> pd.DataFrame:
    data = {
        "date": ["2026-01-30", "2026-01-31"],
        "open": [100.0, 101.0],
        "high": [102.0, 103.0],
        "low": [99.0, 100.0],
        "close": [101.0, 102.0],
        "adj_close": [101.0, 102.0],
        "volume": [1000000, 1200000],
    }
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df


def test_cache_store_and_retrieve(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("TICKER_TALK_DB_PATH", str(db_path))

    init_db()
    df = _sample_df()
    store_data("AAPL", df, source="demo")

    cached = get_cached_data("AAPL", ttl_hours=24)
    assert cached is not None
    assert cached["row_count"] == 2
    assert cached["min_date"] == "2026-01-30"
    assert cached["max_date"] == "2026-01-31"
    assert cached["source"] == "demo"


def test_cache_ttl_expiration(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("TICKER_TALK_DB_PATH", str(db_path))

    init_db()
    df = _sample_df()
    store_data("AAPL", df, source="demo")

    conn = sqlite3.connect(get_db_path())
    stale_time = (datetime.now() - timedelta(days=2)).isoformat(timespec="seconds")
    conn.execute("UPDATE ticker_cache SET fetched_at = ? WHERE ticker = ?", (stale_time, "AAPL"))
    conn.commit()
    conn.close()

    cached = get_cached_data("AAPL", ttl_hours=24)
    assert cached is None


def test_demo_preload_uses_csv(tmp_path, monkeypatch):
    demo_dir = tmp_path / "demo_data"
    demo_dir.mkdir()

    for ticker in DEMO_TICKERS:
        df = _sample_df().reset_index()
        df.to_csv(demo_dir / f"{ticker}.csv", index=False)

    monkeypatch.setenv(DEMO_DATA_DIR_ENV, str(demo_dir))
    monkeypatch.setenv("TICKER_TALK_DB_PATH", str(tmp_path / "test.db"))

    settings.ALPHA_VANTAGE_API_KEY = ""

    preload_demo_data_if_needed()
    cached = get_cached_data("AAPL", ttl_hours=24)
    assert cached is not None
    assert cached["source"] == "demo"
