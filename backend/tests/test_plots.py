"""Unit tests for chart generation."""

import base64

import numpy as np
import pandas as pd

from app.compute.indicators import compute_all_indicators
from app.plots.charts import (
    plot_forecast,
    plot_price_and_ma,
    plot_returns_volatility,
    plot_rsi,
)


def _sample_df(n_rows: int = 100) -> pd.DataFrame:
    """Create sample stock data for testing."""
    dates = pd.date_range(start="2025-01-01", periods=n_rows, freq="D")
    np.random.seed(42)

    prices = [100.0]
    for _ in range(n_rows - 1):
        change = np.random.normal(0, 2)
        prices.append(max(prices[-1] + change, 1.0))

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


def _is_valid_base64_png(data: str) -> bool:
    """Check if string is valid base64-encoded PNG."""
    try:
        decoded = base64.b64decode(data)
        # PNG magic bytes
        return decoded[:8] == b"\x89PNG\r\n\x1a\n"
    except Exception:
        return False


def test_plot_price_and_ma_returns_base64():
    """Test price and MA plot returns valid base64 PNG."""
    df = _sample_df(100)
    df = compute_all_indicators(df)

    result = plot_price_and_ma(df, ticker="TEST")

    assert isinstance(result, str)
    assert len(result) > 0
    assert _is_valid_base64_png(result)


def test_plot_price_and_ma_without_ma_column():
    """Test price plot works even without MA column."""
    df = _sample_df(100)
    # Don't compute indicators, so no ma_20 column

    result = plot_price_and_ma(df, ticker="TEST")

    assert _is_valid_base64_png(result)


def test_plot_returns_volatility_returns_base64():
    """Test returns/volatility plot returns valid base64 PNG."""
    df = _sample_df(100)
    df = compute_all_indicators(df)

    result = plot_returns_volatility(df, ticker="TEST")

    assert isinstance(result, str)
    assert _is_valid_base64_png(result)


def test_plot_rsi_returns_base64():
    """Test RSI plot returns valid base64 PNG."""
    df = _sample_df(100)
    df = compute_all_indicators(df)

    result = plot_rsi(df, ticker="TEST")

    assert isinstance(result, str)
    assert _is_valid_base64_png(result)


def test_plot_rsi_without_rsi_column():
    """Test RSI plot handles missing RSI column."""
    df = _sample_df(100)
    # Don't compute indicators

    result = plot_rsi(df, ticker="TEST")

    # Should still return valid base64 (empty plot with message)
    assert _is_valid_base64_png(result)


def test_plot_forecast_returns_base64():
    """Test forecast plot returns valid base64 PNG."""
    df = _sample_df(100)

    forecast_data = {
        "forecast": [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0],
        "lower_ci": [99.0, 99.5, 100.0, 100.5, 101.0, 101.5, 102.0],
        "upper_ci": [103.0, 104.5, 106.0, 107.5, 109.0, 110.5, 112.0],
        "dates": ["2025-04-11", "2025-04-12", "2025-04-13", "2025-04-14", "2025-04-15", "2025-04-16", "2025-04-17"],
    }

    result = plot_forecast(df, forecast_data, ticker="TEST")

    assert isinstance(result, str)
    assert _is_valid_base64_png(result)


def test_plot_forecast_with_short_history():
    """Test forecast plot with limited history days."""
    df = _sample_df(100)

    forecast_data = {
        "forecast": [101.0, 102.0, 103.0],
        "lower_ci": [99.0, 99.5, 100.0],
        "upper_ci": [103.0, 104.5, 106.0],
        "dates": ["2025-04-11", "2025-04-12", "2025-04-13"],
    }

    result = plot_forecast(df, forecast_data, ticker="TEST", history_days=30)

    assert _is_valid_base64_png(result)


def test_plots_with_empty_ticker():
    """Test plots work with empty ticker string."""
    df = _sample_df(50)
    df = compute_all_indicators(df)

    # All should work without ticker
    assert _is_valid_base64_png(plot_price_and_ma(df))
    assert _is_valid_base64_png(plot_returns_volatility(df))
    assert _is_valid_base64_png(plot_rsi(df))


def test_plots_with_minimal_data():
    """Test plots handle minimal data gracefully."""
    df = _sample_df(10)  # Very short series
    df = compute_all_indicators(df)

    # Should not crash, even with NaN values
    assert _is_valid_base64_png(plot_price_and_ma(df, ticker="TEST"))
    assert _is_valid_base64_png(plot_returns_volatility(df, ticker="TEST"))
    assert _is_valid_base64_png(plot_rsi(df, ticker="TEST"))
