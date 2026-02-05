"""Stock analysis API endpoint."""

from datetime import date, timedelta
import logging
import re
import time
from typing import Optional, Tuple

import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import settings

limiter = Limiter(key_func=get_remote_address)
from app.compute.backtest import walk_forward_backtest
from app.compute.forecast_arima import forecast_arima
from app.compute.indicators import compute_all_indicators, extract_indicator_summary
from app.data.alpha_vantage import (
    AlphaVantageAPIError,
    AlphaVantageInvalidTicker,
    AlphaVantageRateLimit,
    fetch_daily_adjusted,
)
from app.data.cache import get_cached_data, log_request, store_data
from app.data.demo_data import load_demo_data
from app.llm.explain import generate_explanation
from app.llm.intent import get_example_queries, parse_intent
from app.models.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    BacktestMetrics,
    ForecastResult,
    IndicatorSummary,
    Metadata,
    Plots,
)
from app.plots.charts import (
    plot_forecast,
    plot_price_and_ma,
    plot_returns_volatility,
    plot_rsi,
)

router = APIRouter()

TICKER_PATTERN = re.compile(r"^[A-Z]{1,5}$")
logger = logging.getLogger(__name__)


@router.post("/parse_intent")
@limiter.limit("20/minute")
def parse_natural_language_query(request: Request, body: dict) -> dict:
    """Parse natural language query into structured request.

    Example request:
    {
        "query": "forecast AAPL for 30 days"
    }

    Example responses:
    Success: {"ticker": "AAPL", "forecast_horizon": 30}
    Error: {"error": "Could not identify ticker symbol"}
    """
    query = body.get("query", "").strip()

    if not query:
        raise HTTPException(status_code=400, detail="Query field required")
    if len(query) > 500:
        raise HTTPException(status_code=400, detail="Query too long (max 500 characters)")

    result = parse_intent(query)
    return result


@router.get("/example_queries")
def get_examples() -> dict:
    """Return example queries for UI hints."""
    return {"examples": get_example_queries()}


def _cached_data_to_dataframe(cached_data: list) -> Optional[pd.DataFrame]:
    """Convert cached JSON data back to DataFrame with proper date index.

    Returns None if data is empty or invalid.
    """
    if not cached_data:
        return None

    df = pd.DataFrame(cached_data)
    if df.empty or "date" not in df.columns:
        return None

    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df.set_index("date", inplace=True)

    # Ensure numeric columns have correct dtypes (JSON round-trip loses types)
    numeric_cols = ["open", "high", "low", "close", "adj_close", "volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _generate_plots(
    df: pd.DataFrame,
    ticker: str,
    forecast_data: Optional[dict] = None,
) -> Plots:
    """Generate all charts as base64 PNG strings."""
    price_ma_plot = plot_price_and_ma(df, ticker=ticker)
    returns_vol_plot = plot_returns_volatility(df, ticker=ticker)
    rsi_plot = plot_rsi(df, ticker=ticker)

    forecast_plot = None
    if forecast_data is not None:
        forecast_plot = plot_forecast(df, forecast_data, ticker=ticker)

    return Plots(
        price_and_ma=price_ma_plot,
        returns_volatility=returns_vol_plot,
        rsi=rsi_plot,
        forecast=forecast_plot,
    )


def _run_forecast_and_backtest(
    df: pd.DataFrame, horizon: int
) -> Tuple[Optional[ForecastResult], Optional[BacktestMetrics], Optional[dict]]:
    """Run ARIMA forecast and walk-forward backtest.

    Returns (ForecastResult, BacktestMetrics, raw_forecast_data) or (None, None, None) if data insufficient.
    """
    series = df["adj_close"]

    if len(series) < 60:
        # Not enough data for forecasting
        return None, None, None

    try:
        # Run forecast
        forecast_data = forecast_arima(series, horizon=horizon)
        forecast_result = ForecastResult(
            horizon=horizon,
            forecast=forecast_data["forecast"],
            lower_ci=forecast_data["lower_ci"],
            upper_ci=forecast_data["upper_ci"],
            dates=forecast_data["dates"],
            trend=forecast_data["trend"],
        )

        # Run backtest
        backtest_data = walk_forward_backtest(series, horizon=horizon)
        backtest_result = BacktestMetrics(
            mae=backtest_data["mae"],
            rmse=backtest_data["rmse"],
            mape=backtest_data["mape"],
        )

        return forecast_result, backtest_result, forecast_data

    except Exception as exc:
        # Data insufficient, ARIMA convergence failure, or other model error
        import logging
        logging.getLogger(__name__).warning("Forecast/backtest failed: %s", exc)
        return None, None, None


@router.post("/analyze", response_model=AnalyzeResponse)
@limiter.limit("10/minute")
def analyze_stock(request: Request, body: AnalyzeRequest) -> AnalyzeResponse:
    start_time = time.perf_counter()

    ticker = body.ticker.strip().upper()
    if not TICKER_PATTERN.match(ticker):
        raise HTTPException(status_code=400, detail="Ticker must be 1-5 uppercase letters")

    # Check cache first
    cached = get_cached_data(ticker, ttl_hours=settings.CACHE_TTL_HOURS)

    if cached:
        # Reconstruct DataFrame from cached data
        df = _cached_data_to_dataframe(cached["data"])
        if df is not None:
            cache_hit = True
            source = cached.get("source") or "alpha_vantage"
            fetched_at = cached["fetched_at"]
            row_count = cached["row_count"]
            min_date = cached["min_date"]
            max_date = cached["max_date"]
        else:
            # Invalid cached data, treat as cache miss
            cached = None

    if not cached:
        # Fetch from API or demo data
        source = "alpha_vantage"
        df = None
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

        # Store in cache
        stored = store_data(ticker, df, source=source)
        cache_hit = False
        fetched_at = stored["fetched_at"]
        row_count = stored["row_count"]
        min_date = stored["min_date"]
        max_date = stored["max_date"]

    # Compute indicators
    df_with_indicators = compute_all_indicators(df)
    indicator_summary = extract_indicator_summary(df_with_indicators)

    # Run forecast and backtest if requested
    forecast_result = None
    backtest_result = None
    forecast_data_raw = None
    if body.forecast_horizon is not None:
        forecast_result, backtest_result, forecast_data_raw = _run_forecast_and_backtest(
            df, body.forecast_horizon
        )

    # Generate plots
    plots = _generate_plots(df_with_indicators, ticker, forecast_data=forecast_data_raw)

    # Generate LLM explanation
    explanation = None
    try:
        analysis_summary = {
            "ticker": ticker,
            "latest_price": indicator_summary["latest_price"],
            "latest_return": indicator_summary["latest_return"],
            "ma_20_latest": indicator_summary["ma_20_latest"],
            "volatility_20_latest": indicator_summary["volatility_20_latest"],
            "rsi_14_latest": indicator_summary["rsi_14_latest"],
            "avg_return_30d": indicator_summary["avg_return_30d"],
            "avg_volatility_30d": indicator_summary["avg_volatility_30d"],
        }

        if forecast_result is not None and backtest_result is not None:
            analysis_summary.update(
                {
                    "forecast_horizon": forecast_result.horizon,
                    "forecast_trend": forecast_result.trend,
                    "forecast_start_price": indicator_summary["latest_price"],
                    "forecast_end_price": forecast_result.forecast[-1],
                    "forecast_lower_ci": forecast_result.lower_ci[-1],
                    "forecast_upper_ci": forecast_result.upper_ci[-1],
                    "backtest_mae": backtest_result.mae,
                    "backtest_rmse": backtest_result.rmse,
                    "backtest_mape": backtest_result.mape,
                }
            )

        explanation = generate_explanation(analysis_summary)
    except Exception as exc:  # noqa: BLE001 - explanation should never fail request
        logger.warning("Explanation generation failed: %s", exc)
        explanation = None

    # Log request metrics
    latency_ms = int((time.perf_counter() - start_time) * 1000)
    log_request("/api/analyze", ticker, cache_hit, latency_ms)

    return AnalyzeResponse(
        metadata=Metadata(
            ticker=ticker,
            cache_hit=cache_hit,
            data_last_updated=fetched_at,
            source=source,
            row_count=row_count,
            min_date=min_date,
            max_date=max_date,
        ),
        indicators=IndicatorSummary(**indicator_summary),
        forecast=forecast_result,
        backtest=backtest_result,
        plots=plots,
        explanation=explanation,
    )
