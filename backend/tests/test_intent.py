from app.llm.intent import parse_intent


def test_parse_intent_supports_compact_ticker_and_horizon():
    result = parse_intent("AMD30")

    assert result["ticker"] == "AMD"
    assert result["forecast_horizon"] == 30


def test_parse_intent_supports_compact_ticker_and_short_horizon():
    result = parse_intent("AAPL7")

    assert result["ticker"] == "AAPL"
    assert result["forecast_horizon"] == 7


def test_parse_intent_keeps_spaced_horizon_behavior():
    result = parse_intent("AMD 30")

    assert result["ticker"] == "AMD"
    assert result["forecast_horizon"] == 30
