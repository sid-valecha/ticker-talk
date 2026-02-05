"""Intent parsing for natural language stock queries."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional, Set

from app.data.demo_data import DEMO_TICKERS, get_demo_data_dir
from app.llm.client import call_llm_with_fallback

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Extract stock analysis parameters from user's natural language query.

Return ONLY valid JSON with these fields:
- ticker: string (1-5 uppercase letters, e.g., "AAPL", "MSFT")
- forecast_horizon: 7, 30, or null (only if user explicitly requests forecast/prediction)

Common stock name mappings:
- Apple → AAPL
- Advanced Micro Devices → AMD
- Amazon → AMZN
- Boeing → BA
- Bank of America → BAC
- Caterpillar → CAT
- Chevron → CVX
- Disney → DIS
- Google/Alphabet → GOOGL
- Goldman Sachs → GS
- Intel → INTC
- Johnson & Johnson → JNJ
- JPMorgan/JP Morgan → JPM
- Lockheed Martin → LMT
- Meta/Facebook → META
- Microsoft → MSFT
- Netflix → NFLX
- NextEra Energy → NEE
- Northrop Grumman → NOC
- Nvidia → NVDA
- Pfizer → PFE
- Tesla → TSLA
- UnitedHealth → UNH
- Walmart → WMT
- Exxon Mobil → XOM

Rules:
1. If user says "forecast", "predict", "prediction" → include forecast_horizon
2. If no horizon specified, default to 7 days
3. If ticker not recognized, return {"error": "Could not identify ticker symbol"}
4. If query is unclear, return {"error": "Please specify a ticker symbol"}

Examples:
- "forecast AAPL for 30 days" → {"ticker": "AAPL", "forecast_horizon": 30}
- "show me Apple" → {"ticker": "AAPL", "forecast_horizon": null}
- "analyze Microsoft with 7-day prediction" → {"ticker": "MSFT", "forecast_horizon": 7}
- "Tesla forecast" → {"ticker": "TSLA", "forecast_horizon": 7}
- "compare AAPL and MSFT" → {"error": "I can only analyze one ticker at a time"}
"""


def _load_demo_file_tickers() -> Set[str]:
    try:
        demo_dir = get_demo_data_dir()
        return {path.stem.upper() for path in demo_dir.glob("*.csv")}
    except Exception:
        return set()


_DEMO_FILE_TICKERS = _load_demo_file_tickers()
_KNOWN_TICKERS = _DEMO_FILE_TICKERS or set(DEMO_TICKERS)
_TICKER_TOKEN_PATTERN = re.compile(r"\b[a-zA-Z]{1,5}\b")

_COMPANY_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bapple\b", re.IGNORECASE), "AAPL"),
    (
        re.compile(
            r"\badvanced\s+micro\s+devices\b|\badvanced\s+microdevices\b",
            re.IGNORECASE,
        ),
        "AMD",
    ),
    (re.compile(r"\bamazon\b", re.IGNORECASE), "AMZN"),
    (re.compile(r"\bboeing\b|\bba\b", re.IGNORECASE), "BA"),
    (re.compile(r"\bbank\s+of\s+america\b|\bbofa\b", re.IGNORECASE), "BAC"),
    (re.compile(r"\bcaterpillar\b|\bcat\b", re.IGNORECASE), "CAT"),
    (re.compile(r"\bchevron\b", re.IGNORECASE), "CVX"),
    (re.compile(r"\bdisney\b|\bwalt\s+disney\b", re.IGNORECASE), "DIS"),
    (re.compile(r"\bgoogle\b|\balphabet\b", re.IGNORECASE), "GOOGL"),
    (re.compile(r"\bgoldman\s+sachs\b|\bgoldman\b", re.IGNORECASE), "GS"),
    (re.compile(r"\bintel\b", re.IGNORECASE), "INTC"),
    (
        re.compile(
            r"\bjohnson\s*&\s*johnson\b|\bjohnson\s+and\s+johnson\b|\bj&j\b|\bjohnson\b",
            re.IGNORECASE,
        ),
        "JNJ",
    ),
    (
        re.compile(
            r"\bjpmorgan\s+chase\b|\bjp\s*morgan\s+chase\b|\bjp\s*morgan\b|\bj\.?p\.?\s*morgan\b|\bjpmorgan\b",
            re.IGNORECASE,
        ),
        "JPM",
    ),
    (re.compile(r"\blockheed\s+martin\b|\blockheed\b", re.IGNORECASE), "LMT"),
    (re.compile(r"\bmeta\b|\bfacebook\b", re.IGNORECASE), "META"),
    (re.compile(r"\bmicrosoft\b", re.IGNORECASE), "MSFT"),
    (re.compile(r"\bnetflix\b", re.IGNORECASE), "NFLX"),
    (
        re.compile(
            r"\bnextera\s+energy\b|\bnext\s*era\s+energy\b|\bnextera\b",
            re.IGNORECASE,
        ),
        "NEE",
    ),
    (re.compile(r"\bnorthrop\s+grumman\b|\bnorthrop\b", re.IGNORECASE), "NOC"),
    (re.compile(r"\bnvidia\b", re.IGNORECASE), "NVDA"),
    (re.compile(r"\bpfizer\b", re.IGNORECASE), "PFE"),
    (re.compile(r"\btesla\b", re.IGNORECASE), "TSLA"),
    (
        re.compile(
            r"\bunitedhealth\b|\bunited\s+health\b|\bunitedhealth\s+group\b",
            re.IGNORECASE,
        ),
        "UNH",
    ),
    (re.compile(r"\bwalmart\b|\bwal-?mart\b", re.IGNORECASE), "WMT"),
    (re.compile(r"\bexxon\b|\bexxon\s+mobil\b", re.IGNORECASE), "XOM"),
]

_FORECAST_TERMS = re.compile(r"\bforecast\b|\bpredict\b|\bprediction\b", re.IGNORECASE)
_HORIZON_PATTERN = re.compile(r"\b(7|30)\s*-?\s*(day|days|d)?\b", re.IGNORECASE)


def _extract_ticker_candidates(query: str) -> Set[str]:
    candidates: Set[str] = set()

    for token in _TICKER_TOKEN_PATTERN.findall(query):
        token_upper = token.upper()
        if token_upper in _KNOWN_TICKERS:
            candidates.add(token_upper)

    for pattern, ticker in _COMPANY_PATTERNS:
        if pattern.search(query):
            candidates.add(ticker)

    return candidates


def _extract_horizon(query: str) -> Optional[int]:
    match = _HORIZON_PATTERN.search(query)
    if match:
        return int(match.group(1))
    if _FORECAST_TERMS.search(query):
        return 7
    return None


def parse_intent(query: str) -> Dict[str, Any]:
    """Parse natural language query into structured request.

    Args:
        query: User's natural language input

    Returns:
        Dict with:
        - Success: {"ticker": str, "forecast_horizon": int | None}
        - Error: {"error": str}
    """
    if not query or not query.strip():
        return {"error": "Please enter a stock query"}
    if len(query) > 500:
        return {"error": "Query too long (max 500 characters)"}

    candidates = _extract_ticker_candidates(query)
    if len(candidates) > 1:
        return {"error": "I can only analyze one ticker at a time"}
    if len(candidates) == 1:
        ticker = next(iter(candidates))
        return {
            "ticker": ticker,
            "forecast_horizon": _extract_horizon(query),
        }

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    try:
        response = call_llm_with_fallback(messages, max_tokens=100)
        parsed = json.loads(response)

        if "error" in parsed:
            return parsed

        if "ticker" not in parsed:
            return {"error": "Could not identify ticker symbol"}

        ticker = parsed["ticker"].strip().upper()
        if not ticker or len(ticker) > 5 or not ticker.isalpha():
            return {"error": f"Invalid ticker format: {ticker}"}

        horizon = parsed.get("forecast_horizon")
        if horizon is not None and horizon not in [7, 30]:
            return {"error": "Forecast horizon must be 7 or 30 days"}

        return {
            "ticker": ticker,
            "forecast_horizon": horizon,
        }

    except json.JSONDecodeError as e:
        logger.error("Failed to parse LLM response as JSON: %s", e)
        return {"error": "Failed to understand query, please try again"}
    except Exception as e:
        logger.error("Intent parsing failed: %s", e)
        return {"error": "Failed to process query, please try again"}


def get_example_queries() -> list[str]:
    """Return list of example queries for UI hints."""
    return [
        "forecast AAPL for 30 days",
        "show me Tesla",
        "analyze Microsoft with 7-day forecast",
        "GOOGL 30 day prediction",
        "forecast Netflix for 7 days",
    ]
