from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Optional


def latest_expected_market_date(
    now: Optional[datetime] = None,
    lag_days: int = 1,
) -> date:
    """Return the most recent date we expect daily market data to be available."""
    current = (now or datetime.now()).date() - timedelta(days=max(lag_days, 0))
    while current.weekday() >= 5:  # 5=Sat, 6=Sun
        current -= timedelta(days=1)
    return current


def business_days_between(start_date: date, end_date: date) -> int:
    """Count business days strictly after start_date up to end_date inclusive."""
    if start_date >= end_date:
        return 0

    count = 0
    cursor = start_date + timedelta(days=1)
    while cursor <= end_date:
        if cursor.weekday() < 5:
            count += 1
        cursor += timedelta(days=1)
    return count


def stale_business_days(
    max_date_str: str,
    now: Optional[datetime] = None,
    lag_days: int = 1,
) -> Optional[int]:
    """Return stale business-day count for a YYYY-MM-DD max date string."""
    try:
        data_max_date = date.fromisoformat(max_date_str)
    except (TypeError, ValueError):
        return None

    expected = latest_expected_market_date(now=now, lag_days=lag_days)
    return business_days_between(data_max_date, expected)
