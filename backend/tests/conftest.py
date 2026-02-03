"""Shared test fixtures."""

import pytest

from app.main import app
from app.api.analyze import limiter as analyze_limiter


@pytest.fixture(autouse=True)
def _disable_rate_limit():
    """Disable rate limiting for all tests."""
    app.state.limiter.enabled = False
    analyze_limiter.enabled = False
    yield
    app.state.limiter.enabled = True
    analyze_limiter.enabled = True
