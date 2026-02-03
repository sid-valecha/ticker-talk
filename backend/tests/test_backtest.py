"""Unit tests for walk-forward backtesting."""

import numpy as np
import pandas as pd
import pytest

from app.compute.backtest import walk_forward_backtest


def _sample_series(n_rows: int = 200) -> pd.Series:
    """Create sample price series for testing."""
    dates = pd.date_range(start="2024-01-01", periods=n_rows, freq="D")
    np.random.seed(42)

    # Random walk starting at 100
    prices = [100.0]
    for _ in range(n_rows - 1):
        change = np.random.normal(0, 2)
        prices.append(max(prices[-1] + change, 1.0))

    series = pd.Series(prices, index=dates, name="adj_close")
    return series


def test_walk_forward_backtest_7_day():
    """Test 7-day backtest."""
    series = _sample_series(200)
    result = walk_forward_backtest(series, horizon=7)

    # Check all expected keys
    assert "mae" in result
    assert "rmse" in result
    assert "mape" in result
    assert "predictions" in result
    assert "actuals" in result

    # Metrics should be positive
    assert result["mae"] >= 0
    assert result["rmse"] >= 0
    assert result["mape"] >= 0


def test_walk_forward_backtest_30_day():
    """Test 30-day backtest."""
    series = _sample_series(300)  # Need more data for 30-day
    result = walk_forward_backtest(series, horizon=30)

    assert result["mae"] >= 0
    assert result["rmse"] >= 0


def test_backtest_predictions_actuals_match_length():
    """Test that predictions and actuals have matching lengths."""
    series = _sample_series(200)
    result = walk_forward_backtest(series, horizon=7)

    assert len(result["predictions"]) == len(result["actuals"])
    assert len(result["predictions"]) > 0


def test_backtest_rmse_greater_than_mae():
    """Test RMSE >= MAE (mathematical property)."""
    series = _sample_series(200)
    result = walk_forward_backtest(series, horizon=7)

    # RMSE is always >= MAE due to the square root of squared errors
    assert result["rmse"] >= result["mae"] or np.isclose(result["rmse"], result["mae"])


def test_backtest_mape_reasonable():
    """Test MAPE is in reasonable range for stock data."""
    series = _sample_series(200)
    result = walk_forward_backtest(series, horizon=7)

    # MAPE for stock prices should typically be under 50%
    # This is a sanity check, not a hard requirement
    assert result["mape"] < 100, "MAPE should be reasonable for stock data"


def test_backtest_insufficient_data():
    """Test backtest fails gracefully with insufficient data."""
    series = _sample_series(50)  # Too short

    with pytest.raises(ValueError, match="Series too short"):
        walk_forward_backtest(series, horizon=7)


def test_backtest_multiple_splits():
    """Test backtest with different n_splits."""
    series = _sample_series(300)

    result_3 = walk_forward_backtest(series, horizon=7, n_splits=3)
    result_5 = walk_forward_backtest(series, horizon=7, n_splits=5)

    # More splits should generally give more predictions
    assert len(result_5["predictions"]) >= len(result_3["predictions"])


def test_backtest_deterministic():
    """Test backtest gives same results on same data."""
    series = _sample_series(200)

    result1 = walk_forward_backtest(series, horizon=7)
    result2 = walk_forward_backtest(series, horizon=7)

    assert np.isclose(result1["mae"], result2["mae"])
    assert np.isclose(result1["rmse"], result2["rmse"])
