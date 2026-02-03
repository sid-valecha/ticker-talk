"""ARIMA forecasting for stock prices.

All computations are deterministic - these functions never call LLMs.
"""

from __future__ import annotations

import warnings
from typing import Any

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def fit_arima(series: pd.Series, order: tuple = (5, 1, 0)) -> Any:
    """Fit ARIMA model to a price series.

    Args:
        series: Time series of adjusted close prices (datetime index, sorted ascending)
        order: ARIMA order (p, d, q) - default (5, 1, 0) works well for daily stock data

    Returns:
        Fitted ARIMA model results
    """
    # Suppress convergence warnings for cleaner output
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        model = ARIMA(series, order=order)
        return model.fit()


def forecast_arima(
    series: pd.Series,
    horizon: int,
    order: tuple = (5, 1, 0),
    confidence: float = 0.95,
) -> dict:
    """Generate price forecast with confidence intervals.

    Args:
        series: Time series of adjusted close prices (datetime index, sorted ascending)
        horizon: Number of days to forecast (7 or 30)
        order: ARIMA order (p, d, q)
        confidence: Confidence level for intervals (default 95%)

    Returns:
        dict with:
            - forecast: list of predicted prices
            - lower_ci: list of lower confidence bounds
            - upper_ci: list of upper confidence bounds
            - dates: list of forecast dates (YYYY-MM-DD strings)
            - trend: "upward", "downward", or "flat"
    """
    if len(series) < 60:
        raise ValueError("Need at least 60 data points for reliable ARIMA forecasting")

    fitted = fit_arima(series, order)
    alpha = 1 - confidence

    forecast_result = fitted.get_forecast(steps=horizon, alpha=alpha)
    predicted = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int(alpha=alpha)

    # Generate forecast dates starting from day after last data point
    last_date = series.index[-1]
    forecast_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=horizon,
        freq="D",
    )

    # Determine trend direction
    first_pred = predicted.iloc[0]
    last_pred = predicted.iloc[-1]
    pct_change = (last_pred - first_pred) / first_pred if first_pred != 0 else 0

    if pct_change > 0.01:  # > 1% increase
        trend = "upward"
    elif pct_change < -0.01:  # > 1% decrease
        trend = "downward"
    else:
        trend = "flat"

    return {
        "forecast": predicted.tolist(),
        "lower_ci": conf_int.iloc[:, 0].tolist(),
        "upper_ci": conf_int.iloc[:, 1].tolist(),
        "dates": forecast_dates.strftime("%Y-%m-%d").tolist(),
        "trend": trend,
    }
