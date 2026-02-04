"""Provider-agnostic LLM client with fallback support."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError

from app.config import settings

logger = logging.getLogger(__name__)


def call_llm_with_fallback(
    messages: List[Dict[str, Any]],
    max_tokens: int = 500,
) -> str:
    """Call primary LLM provider and fall back on rate limits or missing config."""
    primary = settings.LLM_PRIMARY_PROVIDER.lower().strip()
    fallback = settings.LLM_FALLBACK_PROVIDER.lower().strip()

    try:
        return _call_provider(primary, messages, max_tokens)
    except Exception as exc:  # noqa: BLE001 - handled for fallback
        if not _should_fallback(exc) or fallback == primary:
            raise
        logger.warning(
            "Primary LLM provider '%s' failed (%s); trying fallback '%s'.",
            primary,
            exc.__class__.__name__,
            fallback,
        )
        return _call_provider(fallback, messages, max_tokens)


def _call_provider(
    provider: str,
    messages: List[Dict[str, Any]],
    max_tokens: int,
) -> str:
    """Call a specific provider using the OpenAI SDK."""
    provider = provider.lower().strip()

    if provider == "groq":
        api_key = settings.GROQ_API_KEY
        if not api_key:
            raise ValueError("GROQ_API_KEY not set")
        client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        model = settings.GROQ_MODEL
    elif provider == "openai":
        api_key = settings.OPENAI_API_KEY
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        client = OpenAI(api_key=api_key)
        model = settings.OPENAI_MODEL
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=max_tokens,
    )

    content = response.choices[0].message.content if response.choices else None
    if not content:
        raise ValueError("Empty LLM response")

    return content.strip()


def _should_fallback(exc: Exception) -> bool:
    """Determine whether to use fallback provider."""
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, APIStatusError) and exc.status_code == 429:
        return True
    if isinstance(exc, APIConnectionError):
        return True
    if isinstance(exc, ValueError):
        return True
    return False
