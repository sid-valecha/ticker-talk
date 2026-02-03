"""Chart generation for stock analysis.

All charts are returned as base64-encoded PNG strings.
"""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server use

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


def _fig_to_base64(fig: plt.Figure) -> str:
    """Convert matplotlib figure to base64 PNG string."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def plot_price_and_ma(df: pd.DataFrame, ticker: str = "") -> str:
    """Plot price history with 20-day moving average overlay.

    Args:
        df: DataFrame with 'adj_close' and 'ma_20' columns, datetime index
        ticker: Ticker symbol for title

    Returns:
        Base64-encoded PNG string
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot price
    ax.plot(df.index, df["adj_close"], label="Price", linewidth=1.5, color="#2563eb")

    # Plot MA if available
    if "ma_20" in df.columns:
        ax.plot(df.index, df["ma_20"], label="20-day MA", linewidth=1.5, color="#f97316", alpha=0.8)

    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.set_title(f"{ticker} Price History" if ticker else "Price History")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()

    return _fig_to_base64(fig)


def plot_returns_volatility(df: pd.DataFrame, ticker: str = "") -> str:
    """Plot returns and volatility as dual-axis chart.

    Args:
        df: DataFrame with 'returns' and 'volatility_20' columns, datetime index
        ticker: Ticker symbol for title

    Returns:
        Base64-encoded PNG string
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot returns on primary axis
    color1 = "#2563eb"
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Daily Returns", color=color1)
    ax1.plot(df.index, df["returns"], color=color1, alpha=0.6, linewidth=0.8, label="Returns")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

    # Plot volatility on secondary axis
    ax2 = ax1.twinx()
    color2 = "#dc2626"
    ax2.set_ylabel("Volatility (Annualized)", color=color2)
    if "volatility_20" in df.columns:
        ax2.plot(df.index, df["volatility_20"], color=color2, linewidth=1.5, label="20-day Volatility")
    ax2.tick_params(axis="y", labelcolor=color2)

    ax1.set_title(f"{ticker} Returns & Volatility" if ticker else "Returns & Volatility")
    ax1.grid(True, alpha=0.3)

    # Format x-axis dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    return _fig_to_base64(fig)


def plot_forecast(
    df: pd.DataFrame,
    forecast_data: dict,
    ticker: str = "",
    history_days: int = 90,
) -> str:
    """Plot historical prices with forecast and confidence interval.

    Args:
        df: DataFrame with 'adj_close' column, datetime index
        forecast_data: Dict with 'forecast', 'lower_ci', 'upper_ci', 'dates' keys
        ticker: Ticker symbol for title
        history_days: Number of historical days to show (default 90)

    Returns:
        Base64-encoded PNG string
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get recent historical data
    recent = df.tail(history_days)

    # Plot historical prices
    ax.plot(recent.index, recent["adj_close"], label="Historical", linewidth=1.5, color="#2563eb")

    # Parse forecast dates and plot forecast
    forecast_dates = pd.to_datetime(forecast_data["dates"])
    forecast_values = forecast_data["forecast"]

    ax.plot(forecast_dates, forecast_values, label="Forecast", linewidth=2, linestyle="--", color="#f97316")

    # Plot confidence interval
    ax.fill_between(
        forecast_dates,
        forecast_data["lower_ci"],
        forecast_data["upper_ci"],
        alpha=0.2,
        color="#f97316",
        label="95% Confidence",
    )

    # Add vertical line at forecast start
    ax.axvline(x=recent.index[-1], color="gray", linestyle=":", linewidth=1, alpha=0.7)

    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.set_title(f"{ticker} Price Forecast" if ticker else "Price Forecast")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()

    return _fig_to_base64(fig)


def plot_rsi(df: pd.DataFrame, ticker: str = "") -> str:
    """Plot RSI with overbought/oversold zones.

    Args:
        df: DataFrame with 'rsi_14' column, datetime index
        ticker: Ticker symbol for title

    Returns:
        Base64-encoded PNG string
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    if "rsi_14" not in df.columns:
        # Return empty plot if RSI not computed
        ax.text(0.5, 0.5, "RSI not available", ha="center", va="center", transform=ax.transAxes)
        return _fig_to_base64(fig)

    # Plot RSI
    ax.plot(df.index, df["rsi_14"], linewidth=1.5, color="#8b5cf6")

    # Add overbought/oversold zones
    ax.axhline(y=70, color="#dc2626", linestyle="--", linewidth=1, alpha=0.7, label="Overbought (70)")
    ax.axhline(y=30, color="#16a34a", linestyle="--", linewidth=1, alpha=0.7, label="Oversold (30)")
    ax.axhline(y=50, color="gray", linestyle=":", linewidth=0.5, alpha=0.5)

    # Fill zones
    ax.fill_between(df.index, 70, 100, alpha=0.1, color="#dc2626")
    ax.fill_between(df.index, 0, 30, alpha=0.1, color="#16a34a")

    ax.set_xlabel("Date")
    ax.set_ylabel("RSI")
    ax.set_title(f"{ticker} RSI (14-day)" if ticker else "RSI (14-day)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()

    return _fig_to_base64(fig)
