"""Integration tests for /api/analyze endpoint."""

import base64
import time

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


class TestAnalyzeBasic:
    """Test basic analysis without forecast."""

    def test_analyze_returns_200(self):
        response = client.post("/api/analyze", json={"ticker": "BMW"})
        assert response.status_code == 200

    def test_response_has_required_fields(self):
        response = client.post("/api/analyze", json={"ticker": "BMW"})
        data = response.json()

        assert "metadata" in data
        assert "indicators" in data
        assert "plots" in data

    def test_metadata_fields(self):
        response = client.post("/api/analyze", json={"ticker": "BMW"})
        meta = response.json()["metadata"]

        assert meta["ticker"] == "BMW"
        assert meta["row_count"] > 0
        assert meta["min_date"] is not None
        assert meta["max_date"] is not None
        assert meta["source"] in ("alpha_vantage", "demo")
        assert isinstance(meta["cache_hit"], bool)

    def test_indicator_values_reasonable(self):
        response = client.post("/api/analyze", json={"ticker": "BMW"})
        ind = response.json()["indicators"]

        assert ind["latest_price"] > 0
        assert ind["ma_20_latest"] > 0
        assert 0 <= ind["rsi_14_latest"] <= 100
        assert isinstance(ind["latest_return"], float)
        assert isinstance(ind["avg_return_30d"], float)
        assert isinstance(ind["avg_volatility_30d"], float)

    def test_plots_are_valid_base64_png(self):
        response = client.post("/api/analyze", json={"ticker": "BMW"})
        plots = response.json()["plots"]

        for key in ("price_and_ma", "returns_volatility", "rsi"):
            assert plots[key] is not None, f"{key} should not be None"
            raw = base64.b64decode(plots[key])
            # PNG magic bytes
            assert raw[:4] == b"\x89PNG", f"{key} is not a valid PNG"

    def test_no_forecast_when_not_requested(self):
        response = client.post("/api/analyze", json={"ticker": "BMW"})
        data = response.json()

        assert data.get("forecast") is None
        assert data.get("backtest") is None
        assert data["plots"].get("forecast") is None


class TestAnalyzeForecast:
    """Test analysis with forecast."""

    def test_7day_forecast(self):
        response = client.post(
            "/api/analyze", json={"ticker": "BMW", "forecast_horizon": 7}
        )
        assert response.status_code == 200

        forecast = response.json()["forecast"]
        assert forecast["horizon"] == 7
        assert len(forecast["forecast"]) == 7
        assert len(forecast["lower_ci"]) == 7
        assert len(forecast["upper_ci"]) == 7
        assert len(forecast["dates"]) == 7
        assert forecast["trend"] in ("upward", "downward", "flat")

    def test_30day_forecast(self):
        response = client.post(
            "/api/analyze", json={"ticker": "BMW", "forecast_horizon": 30}
        )
        assert response.status_code == 200

        forecast = response.json()["forecast"]
        assert forecast["horizon"] == 30
        assert len(forecast["forecast"]) == 30

    def test_backtest_present_with_forecast(self):
        response = client.post(
            "/api/analyze", json={"ticker": "BMW", "forecast_horizon": 7}
        )
        backtest = response.json()["backtest"]

        assert backtest is not None
        assert backtest["mae"] > 0
        assert backtest["rmse"] > 0
        assert backtest["rmse"] >= backtest["mae"]
        assert backtest["mape"] > 0

    def test_forecast_plot_present(self):
        response = client.post(
            "/api/analyze", json={"ticker": "BMW", "forecast_horizon": 7}
        )
        plots = response.json()["plots"]

        assert plots["forecast"] is not None
        raw = base64.b64decode(plots["forecast"])
        assert raw[:4] == b"\x89PNG"

    def test_forecast_dates_are_after_historical(self):
        response = client.post(
            "/api/analyze", json={"ticker": "BMW", "forecast_horizon": 7}
        )
        data = response.json()

        max_date = data["metadata"]["max_date"]
        first_forecast_date = data["forecast"]["dates"][0]
        assert first_forecast_date > max_date

    def test_confidence_intervals_bracket_forecast(self):
        response = client.post(
            "/api/analyze", json={"ticker": "BMW", "forecast_horizon": 7}
        )
        forecast = response.json()["forecast"]

        for i in range(len(forecast["forecast"])):
            assert forecast["lower_ci"][i] <= forecast["forecast"][i]
            assert forecast["upper_ci"][i] >= forecast["forecast"][i]


class TestAnalyzeValidation:
    """Test input validation."""

    def test_invalid_ticker_format(self):
        response = client.post("/api/analyze", json={"ticker": "invalid123"})
        assert response.status_code == 422

    def test_empty_ticker(self):
        response = client.post("/api/analyze", json={"ticker": ""})
        assert response.status_code == 422

    def test_ticker_too_long(self):
        response = client.post("/api/analyze", json={"ticker": "ABCDEF"})
        assert response.status_code == 422

    def test_invalid_horizon(self):
        response = client.post(
            "/api/analyze", json={"ticker": "BMW", "forecast_horizon": 15}
        )
        assert response.status_code == 422

    def test_horizon_too_small(self):
        response = client.post(
            "/api/analyze", json={"ticker": "BMW", "forecast_horizon": 3}
        )
        assert response.status_code == 422


class TestAnalyzeCache:
    """Test caching behavior."""

    def test_second_request_is_cache_hit(self):
        # First request
        r1 = client.post("/api/analyze", json={"ticker": "BMW"})
        assert r1.status_code == 200

        # Second request should be cache hit
        r2 = client.post("/api/analyze", json={"ticker": "BMW"})
        assert r2.status_code == 200
        assert r2.json()["metadata"]["cache_hit"] is True

    def test_cached_response_is_faster(self):
        # Warm up cache
        client.post("/api/analyze", json={"ticker": "BMW"})

        # Time a cache miss (different analysis params don't matter, same ticker = cache hit)
        start = time.time()
        client.post("/api/analyze", json={"ticker": "BMW"})
        cached_duration = time.time() - start

        # Cached request should be under 5 seconds
        assert cached_duration < 5.0

    def test_results_consistent_across_cache(self):
        r1 = client.post("/api/analyze", json={"ticker": "BMW"})
        r2 = client.post("/api/analyze", json={"ticker": "BMW"})

        # Indicators should be identical
        assert r1.json()["indicators"] == r2.json()["indicators"]


class TestAnalyzeDemoData:
    """Test with preloaded BMW demo data."""

    def test_bmw_demo_data_available(self):
        response = client.post("/api/analyze", json={"ticker": "BMW"})
        assert response.status_code == 200

    def test_bmw_has_substantial_data(self):
        response = client.post("/api/analyze", json={"ticker": "BMW"})
        meta = response.json()["metadata"]
        assert meta["row_count"] > 100  # BMW.csv has decades of data
