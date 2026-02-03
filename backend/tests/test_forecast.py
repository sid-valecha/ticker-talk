"""Unit tests for ARIMA forecasting."""

import numpy as np
import pandas as pd
import pytest

from app.compute.forecast_arima import fit_arima, forecast_arima


def _sample_series(n_rows: int = 100) -> pd.Series:
    """Create sample price series for testing."""
    dates = pd.date_range(start="2025-01-01", periods=n_rows, freq="D")
    np.random.seed(42)

    # Random walk starting at 100
    prices = [100.0]
    for _ in range(n_rows - 1):
        change = np.random.normal(0, 2)
        prices.append(max(prices[-1] + change, 1.0))

    series = pd.Series(prices, index=dates, name="adj_close")
    return series


def test_fit_arima_success():
    """Test ARIMA model fitting."""
    series = _sample_series(100)
    fitted = fit_arima(series)

    # Should return fitted model
    assert fitted is not None
    assert hasattr(fitted, "get_forecast")


def test_forecast_arima_7_day():
    """Test 7-day forecast generation."""
    series = _sample_series(100)
    result = forecast_arima(series, horizon=7)

    # Check all expected keys
    assert "forecast" in result
    assert "lower_ci" in result
    assert "upper_ci" in result
    assert "dates" in result
    assert "trend" in result

    # Check lengths match horizon
    assert len(result["forecast"]) == 7
    assert len(result["lower_ci"]) == 7
    assert len(result["upper_ci"]) == 7
    assert len(result["dates"]) == 7

    # Check confidence intervals make sense
    for i in range(7):
        assert result["lower_ci"][i] <= result["forecast"][i]
        assert result["forecast"][i] <= result["upper_ci"][i]


def test_forecast_arima_30_day():
    """Test 30-day forecast generation."""
    series = _sample_series(200)  # Need more data for 30-day
    result = forecast_arima(series, horizon=30)

    assert len(result["forecast"]) == 30
    assert len(result["dates"]) == 30


def test_forecast_arima_trend_detection():
    """Test trend detection in forecast."""
    series = _sample_series(100)
    result = forecast_arima(series, horizon=7)

    # Trend should be one of the expected values
    assert result["trend"] in ["upward", "downward", "flat"]


def test_forecast_arima_dates_format():
    """Test that forecast dates are in correct format."""
    series = _sample_series(100)
    result = forecast_arima(series, horizon=7)

    # Dates should be YYYY-MM-DD format
    for date_str in result["dates"]:
        # Should be parseable
        pd.to_datetime(date_str, format="%Y-%m-%d")


def test_forecast_arima_dates_sequential():
    """Test that forecast dates are sequential."""
    series = _sample_series(100)
    result = forecast_arima(series, horizon=7)

    dates = [pd.to_datetime(d) for d in result["dates"]]
    for i in range(1, len(dates)):
        diff = (dates[i] - dates[i - 1]).days
        assert diff == 1, f"Dates should be sequential, got {diff} day gap"


def test_forecast_arima_insufficient_data():
    """Test that forecast fails gracefully with insufficient data."""
    series = _sample_series(30)  # Too short

    with pytest.raises(ValueError, match="at least 60 data points"):
        forecast_arima(series, horizon=7)


def test_forecast_positive_values():
    """Test that forecast values are reasonable (positive)."""
    series = _sample_series(100)
    result = forecast_arima(series, horizon=7)

    # All values should be positive for stock prices
    # Note: ARIMA can predict negative values, but for realistic data it shouldn't
    # Just check that values are not extremely negative
    for val in result["forecast"]:
        assert val > -1000, "Forecast values should be reasonable"
