from datetime import date, datetime

from app.data.freshness import (
    business_days_between,
    latest_expected_market_date,
    stale_business_days,
)


def test_latest_expected_market_date_uses_lag_and_skips_weekend():
    # Monday with 1-day lag should point to previous Friday.
    now = datetime(2026, 3, 9, 12, 0, 0)  # Monday
    assert latest_expected_market_date(now=now, lag_days=1) == date(2026, 3, 6)


def test_business_days_between_counts_weekdays_only():
    # From Friday to next Tuesday: Monday + Tuesday = 2 business days.
    assert business_days_between(date(2026, 3, 6), date(2026, 3, 10)) == 2


def test_stale_business_days_returns_none_for_bad_date():
    assert stale_business_days("bad-date", now=datetime(2026, 3, 10), lag_days=1) is None


def test_stale_business_days_calculation():
    now = datetime(2026, 3, 10, 9, 0, 0)  # Tuesday
    # With lag=1, expected date is Monday 2026-03-09.
    assert stale_business_days("2026-03-09", now=now, lag_days=1) == 0
    assert stale_business_days("2026-03-06", now=now, lag_days=1) == 1
