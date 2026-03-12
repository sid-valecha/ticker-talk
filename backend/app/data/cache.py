"""SQLite cache for stock data."""

import json
import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import pandas as pd

DEFAULT_DB_FILENAME = "ticker_talk.db"
DB_PATH_ENV = "TICKER_TALK_DB_PATH"

# Track if DB has been initialized this session
_db_initialized = False


def get_db_path() -> Path:
    env_path = os.getenv(DB_PATH_ENV)
    if env_path:
        return Path(env_path)

    base_dir = Path(__file__).resolve().parents[2]
    return base_dir / "data_cache" / DEFAULT_DB_FILENAME


def init_db() -> None:
    """Initialize database tables. Should be called once at startup."""
    global _db_initialized
    if _db_initialized:
        return

    db_path = get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ticker_cache (
                ticker TEXT PRIMARY KEY,
                data_json TEXT NOT NULL,
                fetched_at TEXT NOT NULL,
                row_count INTEGER,
                min_date TEXT,
                max_date TEXT,
                source TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS request_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                endpoint TEXT,
                ticker TEXT,
                cache_hit BOOLEAN,
                latency_ms INTEGER,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ticker_refresh_state (
                ticker TEXT PRIMARY KEY,
                last_attempted_at TEXT,
                last_succeeded_at TEXT,
                last_failed_at TEXT,
                last_failure_reason TEXT,
                blocked_until TEXT
            )
            """
        )
        conn.commit()

    _db_initialized = True


def is_cache_valid(fetched_at_str: str, ttl_hours: int = 24) -> bool:
    try:
        fetched_at = datetime.fromisoformat(fetched_at_str)
    except ValueError:
        return False

    age = datetime.now() - fetched_at
    return age.total_seconds() < (ttl_hours * 3600)


def serialize_dataframe(df: pd.DataFrame) -> tuple[list[dict[str, Any]], str, str, int]:
    df = df.copy()
    if "date" not in df.columns:
        df = df.reset_index()

    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)

    min_date = df["date"].min().strftime("%Y-%m-%d")
    max_date = df["date"].max().strftime("%Y-%m-%d")

    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    records = df.to_dict(orient="records")
    return records, min_date, max_date, len(df)


def store_data(ticker: str, df: pd.DataFrame, source: str) -> dict[str, Any]:
    db_path = get_db_path()

    records, min_date, max_date, row_count = serialize_dataframe(df)
    fetched_at = datetime.now().isoformat(timespec="seconds")

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO ticker_cache (
                ticker, data_json, fetched_at, row_count, min_date, max_date, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET
                data_json = excluded.data_json,
                fetched_at = excluded.fetched_at,
                row_count = excluded.row_count,
                min_date = excluded.min_date,
                max_date = excluded.max_date,
                source = excluded.source
            """,
            (
                ticker,
                json.dumps(records),
                fetched_at,
                row_count,
                min_date,
                max_date,
                source,
            ),
        )
        conn.commit()

    return {
        "fetched_at": fetched_at,
        "row_count": row_count,
        "min_date": min_date,
        "max_date": max_date,
        "source": source,
        "data": records,
    }


def get_cached_data(
    ticker: str,
    ttl_hours: int = 24,
    allow_stale: bool = False,
) -> Optional[dict[str, Any]]:
    db_path = get_db_path()

    if not db_path.exists():
        return None

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT data_json, fetched_at, row_count, min_date, max_date, source
            FROM ticker_cache
            WHERE ticker = ?
            """,
            (ticker,),
        )
        row = cursor.fetchone()

    if not row:
        return None

    data_json, fetched_at, row_count, min_date, max_date, source = row
    if not is_cache_valid(fetched_at, ttl_hours) and not allow_stale:
        return None

    try:
        data = json.loads(data_json)
    except json.JSONDecodeError:
        return None

    return {
        "data": data,
        "fetched_at": fetched_at,
        "row_count": row_count,
        "min_date": min_date,
        "max_date": max_date,
        "source": source,
    }


def log_request(endpoint: str, ticker: str, cache_hit: bool, latency_ms: int) -> None:
    try:
        db_path = get_db_path()
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                INSERT INTO request_metrics (endpoint, ticker, cache_hit, latency_ms)
                VALUES (?, ?, ?, ?)
                """,
                (endpoint, ticker, cache_hit, latency_ms),
            )
            conn.commit()
    except sqlite3.Error:
        # Metrics should never break the request path
        pass


def get_refresh_state(ticker: str) -> Optional[dict[str, Any]]:
    db_path = get_db_path()
    if not db_path.exists():
        return None

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT last_attempted_at, last_succeeded_at, last_failed_at,
                   last_failure_reason, blocked_until
            FROM ticker_refresh_state
            WHERE ticker = ?
            """,
            (ticker,),
        )
        row = cursor.fetchone()

    if not row:
        return None

    return {
        "last_attempted_at": row[0],
        "last_succeeded_at": row[1],
        "last_failed_at": row[2],
        "last_failure_reason": row[3],
        "blocked_until": row[4],
    }


def mark_refresh_success(ticker: str) -> None:
    db_path = get_db_path()
    timestamp = datetime.now().isoformat(timespec="seconds")

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO ticker_refresh_state (
                ticker, last_attempted_at, last_succeeded_at,
                last_failed_at, last_failure_reason, blocked_until
            ) VALUES (?, ?, ?, NULL, NULL, NULL)
            ON CONFLICT(ticker) DO UPDATE SET
                last_attempted_at = excluded.last_attempted_at,
                last_succeeded_at = excluded.last_succeeded_at,
                last_failed_at = NULL,
                last_failure_reason = NULL,
                blocked_until = NULL
            """,
            (ticker, timestamp, timestamp),
        )
        conn.commit()


def mark_refresh_failure(
    ticker: str,
    reason: str,
    cooldown_minutes: int,
) -> dict[str, str]:
    db_path = get_db_path()
    timestamp = datetime.now().isoformat(timespec="seconds")
    blocked_until = (
        datetime.now() + timedelta(minutes=max(cooldown_minutes, 0))
    ).isoformat(timespec="seconds")

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO ticker_refresh_state (
                ticker, last_attempted_at, last_succeeded_at,
                last_failed_at, last_failure_reason, blocked_until
            ) VALUES (?, ?, NULL, ?, ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET
                last_attempted_at = excluded.last_attempted_at,
                last_failed_at = excluded.last_failed_at,
                last_failure_reason = excluded.last_failure_reason,
                blocked_until = excluded.blocked_until
            """,
            (ticker, timestamp, timestamp, reason, blocked_until),
        )
        conn.commit()

    return {
        "last_attempted_at": timestamp,
        "last_failed_at": timestamp,
        "last_failure_reason": reason,
        "blocked_until": blocked_until,
    }


def refresh_is_blocked(ticker: str) -> Optional[dict[str, str]]:
    state = get_refresh_state(ticker)
    if not state:
        return None

    blocked_until = state.get("blocked_until")
    if not blocked_until:
        return None

    try:
        blocked_dt = datetime.fromisoformat(blocked_until)
    except ValueError:
        return None

    if blocked_dt <= datetime.now():
        return None

    return {
        "blocked_until": blocked_until,
        "last_failure_reason": state.get("last_failure_reason") or "",
    }
