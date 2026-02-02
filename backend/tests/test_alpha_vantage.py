import types

import pandas as pd
import requests

from app.config import settings
from app.data.alpha_vantage import (
    AlphaVantageInvalidTicker,
    AlphaVantageRateLimit,
    fetch_daily_adjusted,
)


def _mock_response(payload, status_code=200):
    response = types.SimpleNamespace()

    def raise_for_status():
        if status_code >= 400:
            raise requests.HTTPError("HTTP error")

    response.raise_for_status = raise_for_status
    response.json = lambda: payload
    return response


def test_fetch_daily_adjusted_success(monkeypatch):
    settings.ALPHA_VANTAGE_API_KEY = "test-key"

    payload = {
        "Time Series (Daily)": {
            "2026-01-31": {
                "1. open": "100.0",
                "2. high": "101.0",
                "3. low": "99.0",
                "4. close": "100.5",
                "5. adjusted close": "100.5",
                "6. volume": "1000000",
            },
            "2026-01-30": {
                "1. open": "98.0",
                "2. high": "100.0",
                "3. low": "97.0",
                "4. close": "99.0",
                "5. adjusted close": "99.0",
                "6. volume": "900000",
            },
        }
    }

    monkeypatch.setattr(
        requests,
        "get",
        lambda *args, **kwargs: _mock_response(payload),
    )

    df = fetch_daily_adjusted("AAPL")
    assert isinstance(df, pd.DataFrame)
    assert "adj_close" in df.columns
    assert len(df) == 2


def test_fetch_daily_adjusted_rate_limit(monkeypatch):
    settings.ALPHA_VANTAGE_API_KEY = "test-key"

    payload = {"Note": "Thank you for using Alpha Vantage"}
    monkeypatch.setattr(
        requests,
        "get",
        lambda *args, **kwargs: _mock_response(payload),
    )

    try:
        fetch_daily_adjusted("AAPL")
        assert False, "Expected AlphaVantageRateLimit"
    except AlphaVantageRateLimit:
        assert True


def test_fetch_daily_adjusted_invalid_ticker(monkeypatch):
    settings.ALPHA_VANTAGE_API_KEY = "test-key"

    payload = {"Error Message": "Invalid API call."}
    monkeypatch.setattr(
        requests,
        "get",
        lambda *args, **kwargs: _mock_response(payload),
    )

    try:
        fetch_daily_adjusted("INVALID")
        assert False, "Expected AlphaVantageInvalidTicker"
    except AlphaVantageInvalidTicker:
        assert True
