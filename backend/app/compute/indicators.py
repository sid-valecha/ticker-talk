"""Technical indicators for stock analysis.

All computations are deterministic - these functions never call LLMs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_returns(df: pd.DataFrame) -> pd.Series:
    """Daily returns: (price[t] - price[t-1]) / price[t-1]"""
    return df["adj_close"].pct_change()


def compute_moving_average(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Simple moving average (SMA) of adjusted close price."""
    return df["adj_close"].rolling(window=window).mean()


def compute_volatility(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Annualized volatility from rolling std of returns.

    Daily std * sqrt(252 trading days) = annualized volatility.
    """
    returns = compute_returns(df)
    return returns.rolling(window=window).std() * np.sqrt(252)


def compute_rsi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Relative Strength Index (RSI).

    RSI = 100 - (100 / (1 + RS))
    where RS = avg_gain / avg_loss over window.

    RSI ranges from 0-100:
    - RSI > 70: potentially overbought
    - RSI < 30: potentially oversold
    """
    delta = df["adj_close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()

    # Avoid division by zero
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all indicator columns to dataframe.

    Returns a copy with new columns: returns, ma_20, volatility_20, rsi_14.
    """
    df = df.copy()
    df["returns"] = compute_returns(df)
    df["ma_20"] = compute_moving_average(df, window=20)
    df["volatility_20"] = compute_volatility(df, window=20)
    df["rsi_14"] = compute_rsi(df, window=14)
    return df


def extract_indicator_summary(df: pd.DataFrame) -> dict:
    """Extract summary statistics from indicator dataframe.

    Returns dict with latest values and 30-day averages suitable for API response.
    """
    # Ensure indicators are computed
    if "returns" not in df.columns:
        df = compute_all_indicators(df)

    # Get last 30 rows for averaging
    recent = df.tail(30)

    # Latest values (last row, skip NaN)
    latest = df.dropna().iloc[-1] if len(df.dropna()) > 0 else df.iloc[-1]

    return {
        "latest_price": float(df["adj_close"].iloc[-1]),
        "latest_return": float(latest.get("returns", 0)) if pd.notna(latest.get("returns")) else 0.0,
        "ma_20_latest": float(latest.get("ma_20", 0)) if pd.notna(latest.get("ma_20")) else 0.0,
        "volatility_20_latest": float(latest.get("volatility_20", 0)) if pd.notna(latest.get("volatility_20")) else 0.0,
        "rsi_14_latest": float(latest.get("rsi_14", 50)) if pd.notna(latest.get("rsi_14")) else 50.0,
        "avg_return_30d": float(recent["returns"].mean()) if "returns" in recent.columns and not recent["returns"].isna().all() else 0.0,
        "avg_volatility_30d": float(recent["volatility_20"].mean()) if "volatility_20" in recent.columns and not recent["volatility_20"].isna().all() else 0.0,
    }
