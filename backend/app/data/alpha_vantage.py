from __future__ import annotations

from typing import Any

import pandas as pd
import requests

from app.config import settings


class AlphaVantageRateLimit(Exception):
    """Raised when Alpha Vantage rate limit is hit."""


class AlphaVantageInvalidTicker(Exception):
    """Raised when Alpha Vantage reports an invalid ticker."""


class AlphaVantageAPIError(Exception):
    """Raised for other Alpha Vantage API errors."""


def _parse_time_series(time_series: dict[str, Any]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    for date_str, values in time_series.items():
        records.append(
            {
                "date": pd.to_datetime(date_str),
                "open": float(values["1. open"]),
                "high": float(values["2. high"]),
                "low": float(values["3. low"]),
                "close": float(values["4. close"]),
                "adj_close": float(values["5. adjusted close"]),
                "volume": int(values["6. volume"]),
            }
        )

    df = pd.DataFrame(records)
    if df.empty:
        raise AlphaVantageAPIError("No data returned from Alpha Vantage")

    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)
    return df


def fetch_daily_adjusted(ticker: str) -> pd.DataFrame:
    if not settings.ALPHA_VANTAGE_API_KEY:
        raise AlphaVantageAPIError("Missing Alpha Vantage API key")

    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": ticker,
        "apikey": settings.ALPHA_VANTAGE_API_KEY,
    }

    try:
        response = requests.get(settings.ALPHA_VANTAGE_BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise AlphaVantageAPIError("Network error while calling Alpha Vantage") from exc
    except ValueError as exc:
        raise AlphaVantageAPIError("Invalid JSON from Alpha Vantage") from exc

    if "Note" in payload or "Information" in payload:
        raise AlphaVantageRateLimit(payload.get("Note") or payload.get("Information"))

    if "Error Message" in payload:
        raise AlphaVantageInvalidTicker(payload["Error Message"])

    time_series = payload.get("Time Series (Daily)")
    if not time_series:
        raise AlphaVantageAPIError("Missing Time Series data in Alpha Vantage response")

    return _parse_time_series(time_series)
