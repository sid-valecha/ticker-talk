from fastapi import APIRouter, HTTPException, Request
import sqlite3

from app.data.cache import get_db_path
from app.config import settings

router = APIRouter()


@router.get("/metrics")
def get_metrics(request: Request):
    """Return usage metrics from SQLite."""
    if not settings.METRICS_API_KEY:
        raise HTTPException(status_code=404, detail="Not found")

    provided_key = request.headers.get("x-metrics-key")
    if provided_key != settings.METRICS_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    db_path = get_db_path()

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        total = cursor.execute("SELECT COUNT(*) FROM request_metrics").fetchone()[0]

        cache_hits = cursor.execute(
            "SELECT COUNT(*) FROM request_metrics WHERE cache_hit = 1"
        ).fetchone()[0]
        cache_hit_rate = (cache_hits / total * 100) if total > 0 else 0

        avg_latency = cursor.execute(
            "SELECT AVG(latency_ms) FROM request_metrics"
        ).fetchone()[0] or 0

        top_tickers = cursor.execute(
            """
            SELECT ticker, COUNT(*) as count
            FROM request_metrics
            GROUP BY ticker
            ORDER BY count DESC
            LIMIT 10
            """
        ).fetchall()

    return {
        "total_requests": total,
        "cache_hit_rate": round(cache_hit_rate, 1),
        "avg_latency_ms": round(avg_latency, 1),
        "top_tickers": [{"ticker": t[0], "count": t[1]} for t in top_tickers],
    }
