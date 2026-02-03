"""Unit tests for technical indicators."""

import numpy as np
import pandas as pd
import pytest

from app.compute.indicators import (
    compute_all_indicators,
    compute_moving_average,
    compute_returns,
    compute_rsi,
    compute_volatility,
    extract_indicator_summary,
)


def _sample_df(n_rows: int = 100) -> pd.DataFrame:
    """Create sample stock data for testing."""
    dates = pd.date_range(start="2025-01-01", periods=n_rows, freq="D")
    np.random.seed(42)  # Reproducible

    # Start at 100 and random walk
    prices = [100.0]
    for _ in range(n_rows - 1):
        change = np.random.normal(0, 2)
        prices.append(max(prices[-1] + change, 1.0))  # Keep price positive

    df = pd.DataFrame(
        {
            "date": dates,
            "open": prices,
            "high": [p * 1.02 for p in prices],
            "low": [p * 0.98 for p in prices],
            "close": prices,
            "adj_close": prices,
            "volume": [1000000] * n_rows,
        }
    )
    df.set_index("date", inplace=True)
    return df


def test_compute_returns():
    """Test daily returns calculation."""
    data = {"adj_close": [100.0, 102.0, 101.0, 105.0, 103.0]}
    df = pd.DataFrame(data)

    returns = compute_returns(df)

    # First value should be NaN (no previous price)
    assert pd.isna(returns.iloc[0])

    # Second value: (102 - 100) / 100 = 0.02
    assert np.isclose(returns.iloc[1], 0.02, atol=1e-6)

    # Third value: (101 - 102) / 102 = -0.0098...
    assert np.isclose(returns.iloc[2], -0.00980392, atol=1e-6)


def test_compute_moving_average():
    """Test simple moving average."""
    df = _sample_df(30)
    ma = compute_moving_average(df, window=20)

    # First 19 values should be NaN
    assert ma.iloc[:19].isna().all()

    # 20th value should be the average of first 20 adj_close values
    expected = df["adj_close"].iloc[:20].mean()
    assert np.isclose(ma.iloc[19], expected, atol=1e-6)


def test_compute_volatility():
    """Test annualized volatility calculation."""
    df = _sample_df(50)
    vol = compute_volatility(df, window=20)

    # First ~20 values should be NaN (need returns + rolling window)
    assert vol.iloc[:20].isna().all()

    # Volatility should be positive
    vol_clean = vol.dropna()
    assert (vol_clean >= 0).all()


def test_rsi_range():
    """Test RSI is always between 0 and 100."""
    df = _sample_df(100)
    rsi = compute_rsi(df, window=14)

    rsi_clean = rsi.dropna()
    assert len(rsi_clean) > 0
    assert (rsi_clean >= 0).all()
    assert (rsi_clean <= 100).all()


def test_rsi_with_mixed_movement():
    """Test RSI behavior on realistic mixed price movement."""
    # Create prices with mostly upward movement (some down days)
    np.random.seed(123)
    base = 100.0
    prices = [base]
    for _ in range(99):
        # 70% chance of up day, 30% down
        if np.random.random() < 0.7:
            prices.append(prices[-1] + np.random.uniform(0.5, 2.0))
        else:
            prices.append(prices[-1] - np.random.uniform(0.5, 1.0))

    df = pd.DataFrame({"adj_close": prices})
    rsi = compute_rsi(df, window=14).dropna()

    # Should have RSI values
    assert len(rsi) > 0, "RSI series should have values"
    # With mostly up movement, RSI should be elevated (> 50)
    assert rsi.iloc[-1] > 50


def test_compute_all_indicators():
    """Test that compute_all_indicators adds all expected columns."""
    df = _sample_df(50)
    result = compute_all_indicators(df)

    # Original columns should still exist
    assert "adj_close" in result.columns
    assert "volume" in result.columns

    # New indicator columns should be added
    assert "returns" in result.columns
    assert "ma_20" in result.columns
    assert "volatility_20" in result.columns
    assert "rsi_14" in result.columns


def test_extract_indicator_summary():
    """Test extraction of indicator summary dict."""
    df = _sample_df(100)
    df = compute_all_indicators(df)
    summary = extract_indicator_summary(df)

    # Check all expected keys exist
    assert "latest_price" in summary
    assert "latest_return" in summary
    assert "ma_20_latest" in summary
    assert "volatility_20_latest" in summary
    assert "rsi_14_latest" in summary
    assert "avg_return_30d" in summary
    assert "avg_volatility_30d" in summary

    # Check types are correct
    assert isinstance(summary["latest_price"], float)
    assert isinstance(summary["rsi_14_latest"], float)

    # RSI should be in valid range
    assert 0 <= summary["rsi_14_latest"] <= 100


def test_indicators_on_short_series():
    """Test indicators gracefully handle short series."""
    df = _sample_df(5)  # Very short series
    df = compute_all_indicators(df)

    # Should not raise, but most values will be NaN
    assert "returns" in df.columns
    assert "ma_20" in df.columns

    # Summary extraction should still work
    summary = extract_indicator_summary(df)
    assert "latest_price" in summary
