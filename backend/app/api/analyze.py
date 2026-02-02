from datetime import date, timedelta
import re
import time

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.data.alpha_vantage import (
    AlphaVantageAPIError,
    AlphaVantageInvalidTicker,
    AlphaVantageRateLimit,
    fetch_daily_adjusted,
)
from app.data.cache import get_cached_data, log_request, store_data
from app.data.demo_data import load_demo_data
from app.models.schemas import AnalyzeRequest, AnalyzeResponse, Metadata

router = APIRouter()

TICKER_PATTERN = re.compile(r"^[A-Z]{1,5}$")


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze_stock(request: AnalyzeRequest) -> AnalyzeResponse:
    start_time = time.perf_counter()

    ticker = request.ticker.strip().upper()
    if not TICKER_PATTERN.match(ticker):
        raise HTTPException(status_code=400, detail="Ticker must be 1-5 uppercase letters")

    # Defaults for later steps; included here for forward compatibility
    _start_date = request.start_date or (date.today() - timedelta(days=365)).isoformat()
    _end_date = request.end_date or date.today().isoformat()

    cached = get_cached_data(ticker, ttl_hours=settings.CACHE_TTL_HOURS)
    if cached:
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        log_request("/api/analyze", ticker, True, latency_ms)
        return AnalyzeResponse(
            metadata=Metadata(
                ticker=ticker,
                cache_hit=True,
                data_last_updated=cached["fetched_at"],
                source=cached.get("source") or "alpha_vantage",
                row_count=cached["row_count"],
                min_date=cached["min_date"],
                max_date=cached["max_date"],
            )
        )

    source = "alpha_vantage"
    try:
        df = fetch_daily_adjusted(ticker)
    except AlphaVantageRateLimit:
        df = load_demo_data(ticker)
        source = "demo" if df is not None else source
    except AlphaVantageInvalidTicker as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except AlphaVantageAPIError:
        df = load_demo_data(ticker)
        source = "demo" if df is not None else source

    if df is None:
        raise HTTPException(
            status_code=503,
            detail="Stock data unavailable (API error or missing demo data).",
        )

    stored = store_data(ticker, df, source=source)
    latency_ms = int((time.perf_counter() - start_time) * 1000)
    log_request("/api/analyze", ticker, False, latency_ms)

    return AnalyzeResponse(
        metadata=Metadata(
            ticker=ticker,
            cache_hit=False,
            data_last_updated=stored["fetched_at"],
            source=stored["source"],
            row_count=stored["row_count"],
            min_date=stored["min_date"],
            max_date=stored["max_date"],
        )
    )
