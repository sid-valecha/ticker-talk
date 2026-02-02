import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

DEFAULT_DB_FILENAME = "ticker_talk.db"
DB_PATH_ENV = "TICKER_TALK_DB_PATH"


def get_db_path() -> Path:
    env_path = os.getenv(DB_PATH_ENV)
    if env_path:
        return Path(env_path)

    base_dir = Path(__file__).resolve().parents[2]
    return base_dir / "data_cache" / DEFAULT_DB_FILENAME


def init_db() -> None:
    db_path = get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
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
    conn.commit()
    conn.close()


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
    init_db()
    db_path = get_db_path()

    records, min_date, max_date, row_count = serialize_dataframe(df)
    fetched_at = datetime.now().isoformat(timespec="seconds")

    conn = sqlite3.connect(db_path)
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
    conn.close()

    return {
        "fetched_at": fetched_at,
        "row_count": row_count,
        "min_date": min_date,
        "max_date": max_date,
        "source": source,
        "data": records,
    }


def get_cached_data(ticker: str, ttl_hours: int = 24) -> Optional[dict[str, Any]]:
    init_db()
    db_path = get_db_path()

    conn = sqlite3.connect(db_path)
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
    conn.close()

    if not row:
        return None

    data_json, fetched_at, row_count, min_date, max_date, source = row
    if not is_cache_valid(fetched_at, ttl_hours):
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
        init_db()
        db_path = get_db_path()
        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            INSERT INTO request_metrics (endpoint, ticker, cache_hit, latency_ms)
            VALUES (?, ?, ?, ?)
            """,
            (endpoint, ticker, cache_hit, latency_ms),
        )
        conn.commit()
        conn.close()
    except sqlite3.Error:
        # Metrics should never break the request path
        return
