"""Walk-forward backtesting for forecast models.

All computations are deterministic - these functions never call LLMs.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from app.compute.forecast_arima import fit_arima


def walk_forward_backtest(
    series: pd.Series,
    horizon: int,
    n_splits: int = 5,
    order: tuple = (5, 1, 0),
) -> dict:
    """Walk-forward validation for ARIMA model.

    Simulates real-world usage: train on historical data, predict future,
    then move forward and repeat.

    Args:
        series: Time series of adjusted close prices (datetime index, sorted ascending)
        horizon: Forecast horizon in days (7 or 30)
        n_splits: Number of validation windows
        order: ARIMA order (p, d, q)

    Returns:
        dict with:
            - mae: Mean Absolute Error
            - rmse: Root Mean Square Error
            - mape: Mean Absolute Percentage Error
            - predictions: all predictions made
            - actuals: corresponding actual values
    """
    min_train_size = 60  # Minimum data points for ARIMA fitting
    total_length = len(series)

    if total_length < min_train_size + horizon:
        raise ValueError(
            f"Series too short for backtesting. Need at least {min_train_size + horizon} "
            f"data points, got {total_length}"
        )

    predictions = []
    actuals = []

    # Calculate how much data we need for n_splits windows
    # Each window needs: train data + horizon test data
    # We walk forward by horizon steps each time
    test_data_needed = n_splits * horizon
    if total_length < min_train_size + test_data_needed:
        # Reduce n_splits if we don't have enough data
        n_splits = (total_length - min_train_size) // horizon
        if n_splits < 1:
            n_splits = 1

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        for i in range(n_splits):
            # Calculate split points
            # Start from the end and work backwards
            test_end_idx = total_length - (n_splits - i - 1) * horizon
            test_start_idx = test_end_idx - horizon
            train_end_idx = test_start_idx

            if train_end_idx < min_train_size:
                continue  # Not enough training data

            train_data = series.iloc[:train_end_idx]
            test_data = series.iloc[test_start_idx:test_end_idx]

            try:
                fitted = fit_arima(train_data, order)
                forecast = fitted.get_forecast(steps=horizon)
                pred = forecast.predicted_mean.values

                predictions.extend(pred.tolist())
                actuals.extend(test_data.values.tolist())
            except Exception:
                # Skip this split if ARIMA fails to converge
                continue

    if len(predictions) == 0:
        raise ValueError("Backtest failed: no valid predictions were made")

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Calculate metrics
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))

    # MAPE - handle zero values
    nonzero_mask = actuals != 0
    if nonzero_mask.sum() > 0:
        mape = np.mean(np.abs((actuals[nonzero_mask] - predictions[nonzero_mask]) / actuals[nonzero_mask])) * 100
    else:
        mape = 0.0

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "predictions": predictions.tolist(),
        "actuals": actuals.tolist(),
    }
