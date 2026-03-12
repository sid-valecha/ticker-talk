import sqlite3
from datetime import datetime, timedelta

import pandas as pd
import pytest

import app.data.cache as cache_module
from app.config import settings
from app.data.cache import (
    get_cached_data,
    get_db_path,
    get_refresh_state,
    init_db,
    mark_refresh_failure,
    mark_refresh_success,
    refresh_is_blocked,
    store_data,
)
from app.data.demo_data import DEMO_DATA_DIR_ENV, DEMO_TICKERS, preload_demo_data_if_needed


@pytest.fixture(autouse=True)
def reset_db_initialized():
    """Reset the _db_initialized flag before each test."""
    cache_module._db_initialized = False
    yield
    cache_module._db_initialized = False


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


def _sample_df_with_dates(first: str, second: str) -> pd.DataFrame:
    data = {
        "date": [first, second],
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

    init_db()  # Must init DB before preload
    preload_demo_data_if_needed()
    cached = get_cached_data(DEMO_TICKERS[0], ttl_hours=24)
    assert cached is not None
    assert cached["source"] == "demo"


def test_demo_preload_does_not_overwrite_newer_cache(tmp_path, monkeypatch):
    demo_dir = tmp_path / "demo_data"
    demo_dir.mkdir()

    for ticker in DEMO_TICKERS:
        # Older demo snapshot
        df = _sample_df_with_dates("2026-01-30", "2026-01-31").reset_index()
        df.to_csv(demo_dir / f"{ticker}.csv", index=False)

    monkeypatch.setenv(DEMO_DATA_DIR_ENV, str(demo_dir))
    monkeypatch.setenv("TICKER_TALK_DB_PATH", str(tmp_path / "test.db"))

    settings.ALPHA_VANTAGE_API_KEY = ""

    init_db()
    # Newer cached data should be preserved
    store_data("AAPL", _sample_df_with_dates("2026-02-03", "2026-02-04"), source="alpha_vantage")

    preload_demo_data_if_needed()
    cached = get_cached_data("AAPL", ttl_hours=24, allow_stale=True)
    assert cached is not None
    assert cached["source"] == "alpha_vantage"
    assert cached["max_date"] == "2026-02-04"


def test_refresh_failure_sets_cooldown(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("TICKER_TALK_DB_PATH", str(db_path))

    init_db()
    state = mark_refresh_failure("AAPL", reason="rate_limited", cooldown_minutes=30)

    assert state["last_failure_reason"] == "rate_limited"
    assert refresh_is_blocked("AAPL") is not None


def test_refresh_success_clears_cooldown(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("TICKER_TALK_DB_PATH", str(db_path))

    init_db()
    mark_refresh_failure("AAPL", reason="rate_limited", cooldown_minutes=30)
    mark_refresh_success("AAPL")

    state = get_refresh_state("AAPL")
    assert state is not None
    assert state["last_failure_reason"] is None
    assert state["blocked_until"] is None
    assert refresh_is_blocked("AAPL") is None
