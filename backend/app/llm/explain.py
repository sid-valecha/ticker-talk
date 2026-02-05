"""LLM explanation generator with guardrails and fallback."""

from __future__ import annotations

import logging
from typing import Any, Dict

from app.llm.client import call_llm_with_fallback

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a financial analysis assistant.\n"
    "\n"
    "STRICT RULES:\n"
    "- Base the explanation ONLY on the provided computed metrics.\n"
    "- Never give trading advice or recommendations (no buy/sell/hold).\n"
    "- Acknowledge uncertainty and model limitations.\n"
    "- Explain what the numbers mean, not what to do.\n"
    "- If forecast exists, describe it as a probabilistic model with uncertainty.\n"
    "- Cite specific metrics from the input (prices, RSI, volatility, errors).\n"
    "- Keep the response to 2-3 short paragraphs.\n"
)


def generate_explanation(analysis_summary: Dict[str, Any]) -> str:
    """Generate an explanation from computed analysis metrics."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_prompt(analysis_summary)},
    ]

    try:
        return call_llm_with_fallback(messages)
    except Exception as exc:  # noqa: BLE001 - graceful degradation
        logger.warning("LLM explanation failed: %s", exc)
        return _fallback_explanation(analysis_summary)


def _build_user_prompt(summary: Dict[str, Any]) -> str:
    """Build the user prompt from analysis summary."""
    lines = [
        "Use only the metrics below to write the explanation.",
        f"Ticker: {summary.get('ticker', 'n/a')}",
        f"Latest price: {_format_price(summary.get('latest_price'))}",
        f"Latest daily return: {_format_pct(summary.get('latest_return'))}",
        f"20-day moving average: {_format_price(summary.get('ma_20_latest'))}",
        f"20-day annualized volatility: {_format_pct(summary.get('volatility_20_latest'))}",
        f"14-day RSI: {_format_rsi(summary.get('rsi_14_latest'))}",
        f"30-day avg daily return: {_format_pct(summary.get('avg_return_30d'))}",
        f"30-day avg annualized volatility: {_format_pct(summary.get('avg_volatility_30d'))}",
    ]

    if "forecast_horizon" in summary:
        lines.extend(
            [
                "Forecast:",
                f"Horizon (days): {summary.get('forecast_horizon')}",
                f"Trend: {summary.get('forecast_trend', 'n/a')}",
                f"Forecast end price: {_format_price(summary.get('forecast_end_price'))}",
                (
                    "Forecast 95% CI: "
                    f"[{_format_price(summary.get('forecast_lower_ci'))}, "
                    f"{_format_price(summary.get('forecast_upper_ci'))}]"
                ),
                f"Backtest MAE: {_format_price(summary.get('backtest_mae'))}",
                f"Backtest RMSE: {_format_price(summary.get('backtest_rmse'))}",
                f"Backtest MAPE: {_format_pct_value(summary.get('backtest_mape'))}",
            ]
        )

    return "\n".join(lines)


def _fallback_explanation(summary: Dict[str, Any]) -> str:
    """Deterministic fallback explanation if LLM fails."""
    ticker = summary.get("ticker", "")
    latest_price = _format_price(summary.get("latest_price"))
    latest_return = _format_pct(summary.get("latest_return"))
    ma_20 = _format_price(summary.get("ma_20_latest"))
    vol_20 = _format_pct(summary.get("volatility_20_latest"))
    rsi = _format_rsi(summary.get("rsi_14_latest"))
    avg_return_30 = _format_pct(summary.get("avg_return_30d"))
    avg_vol_30 = _format_pct(summary.get("avg_volatility_30d"))

    paragraph_1 = (
        f"{ticker} is trading at {latest_price} with a 20-day moving average of {ma_20}. "
        f"The latest daily return is {latest_return}, and the 30-day average daily return is {avg_return_30}. "
        f"Annualized volatility is {vol_20} (30-day average: {avg_vol_30}), and the RSI is {rsi}."
    )

    paragraph_2 = ""
    if "forecast_horizon" in summary:
        horizon = summary.get("forecast_horizon")
        trend = summary.get("forecast_trend", "n/a")
        end_price = _format_price(summary.get("forecast_end_price"))
        lower_ci = _format_price(summary.get("forecast_lower_ci"))
        upper_ci = _format_price(summary.get("forecast_upper_ci"))
        mae = _format_price(summary.get("backtest_mae"))
        rmse = _format_price(summary.get("backtest_rmse"))
        mape = _format_pct_value(summary.get("backtest_mape"))

        paragraph_2 = (
            f"The {horizon}-day ARIMA forecast suggests a {trend} trend toward {end_price}, "
            f"with a 95% confidence interval of {lower_ci} to {upper_ci}. "
            f"Backtesting shows MAE {mae}, RMSE {rmse}, and MAPE {mape}, "
            "which reflects typical model error."
        )

    disclaimer = (
        "This analysis is based on historical data and is for educational purposes only; "
        "it is not financial advice."
    )

    if paragraph_2:
        return f"{paragraph_1}\n\n{paragraph_2}\n\n{disclaimer}"
    return f"{paragraph_1}\n\n{disclaimer}"


def _format_price(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return "n/a"


def _format_pct(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "n/a"


def _format_pct_value(value: Any) -> str:
    """Format value already in percentage points (e.g., MAPE)."""
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.2f}%"
    except (TypeError, ValueError):
        return "n/a"


def _format_rsi(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return "n/a"
