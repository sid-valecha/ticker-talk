from __future__ import annotations

from typing import Any

import pandas as pd
import requests

from app.config import settings


class AlphaVantageRateLimit(Exception):
    """Raised when Alpha Vantage rate limit is hit."""

    def __init__(self, message: str, reason: str = "rate_limited"):
        super().__init__(message)
        self.reason = reason


class AlphaVantageInvalidTicker(Exception):
    """Raised when Alpha Vantage reports an invalid ticker."""


class AlphaVantageAPIError(Exception):
    """Raised for other Alpha Vantage API errors."""


def _parse_time_series(time_series: dict[str, Any]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    for date_str, values in time_series.items():
        close = float(values["4. close"])
        adjusted_close = float(values.get("5. adjusted close", close))
        volume_key = "6. volume" if "6. volume" in values else "5. volume"
        records.append(
            {
                "date": pd.to_datetime(date_str),
                "open": float(values["1. open"]),
                "high": float(values["2. high"]),
                "low": float(values["3. low"]),
                "close": close,
                "adj_close": adjusted_close,
                "volume": int(values[volume_key]),
            }
        )

    df = pd.DataFrame(records)
    if df.empty:
        raise AlphaVantageAPIError("No data returned from Alpha Vantage")

    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)
    return df


def fetch_daily(ticker: str) -> pd.DataFrame:
    if not settings.ALPHA_VANTAGE_API_KEY:
        raise AlphaVantageAPIError("Missing Alpha Vantage API key")

    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": ticker,
        "apikey": settings.ALPHA_VANTAGE_API_KEY,
        "outputsize": "compact",
    }

    try:
        response = requests.get(settings.ALPHA_VANTAGE_BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise AlphaVantageAPIError("Network error while calling Alpha Vantage") from exc
    except ValueError as exc:
        raise AlphaVantageAPIError("Invalid JSON from Alpha Vantage") from exc

    if "Note" in payload:
        raise AlphaVantageRateLimit(payload["Note"], reason="rate_limited")

    if "Information" in payload:
        info = payload["Information"]
        reason = "rate_limited"
        if "premium" in info.lower():
            reason = "premium_endpoint"
        raise AlphaVantageRateLimit(info, reason=reason)

    if "Error Message" in payload:
        raise AlphaVantageInvalidTicker(payload["Error Message"])

    time_series = payload.get("Time Series (Daily)")
    if not time_series:
        raise AlphaVantageAPIError("Missing Time Series data in Alpha Vantage response")

    return _parse_time_series(time_series)


def fetch_daily_adjusted(ticker: str) -> pd.DataFrame:
    """Backward-compatible wrapper for callers not yet renamed."""
    return fetch_daily(ticker)
