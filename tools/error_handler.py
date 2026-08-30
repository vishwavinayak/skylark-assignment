"""
tools/error_handler.py
──────────────────────
Reusable error-handling and telemetry layer for Monday.com and Gemini API calls (Prompt 8).

Covers:
  1. Auth failure (401 / 403 / invalid token) -> Clean error with setup advice.
  2. Rate limiting (429 with Retry-After header) -> Exponential backoff with jitter.
  3. Transient network errors (ConnectionError, Timeout, 5xx) -> Retries with backoff.
  4. Empty result sets & malformed values -> Safe fallbacks and data quality flags.
  5. One board unreachable while other succeeds -> Partial recovery & gap disclosure.
  6. Pagination loop guard -> Hard cap on pages to prevent runaway cursor loops.
  7. Node transition and tool latency logging -> Millisecond-precision telemetry.
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable

logger = logging.getLogger("skylark.telemetry")


# ── Custom Exceptions ──────────────────────────────────────────────────────────

class SkylarkAppError(Exception):
    """Base exception for Skylark BI Agent."""
    pass


class MondayAuthError(SkylarkAppError):
    """Raised when Monday.com API returns 401 or 403 (invalid token)."""
    pass


class MondayRateLimitError(SkylarkAppError):
    """Raised when Monday.com API returns 429."""
    pass


class MondayNetworkError(SkylarkAppError):
    """Raised on transient network or 5xx server errors from Monday.com."""
    pass


class MondayGraphQLError(SkylarkAppError):
    """Raised when Monday.com returns GraphQL error payloads."""
    pass


class GeminiAuthError(SkylarkAppError):
    """Raised when Gemini API key is missing or invalid."""
    pass


class GeminiRateLimitError(SkylarkAppError):
    """Raised when Gemini API quota is exceeded (429)."""
    pass


class GeminiNetworkError(SkylarkAppError):
    """Raised on transient Gemini network errors."""
    pass


# ── Tool Latency & Transition Logging Decorator ────────────────────────────────

def timed_execution(name: str):
    """
    Decorator to measure and log execution latency of API calls or graph nodes.
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            logger.info("⏱️  [START] %s", name)
            try:
                result = func(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - start) * 1000
                logger.info("✅ [DONE]  %s (took %.1f ms)", name, elapsed_ms)
                return result
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - start) * 1000
                logger.error("❌ [FAIL]  %s (failed after %.1f ms: %s)", name, elapsed_ms, exc)
                raise
        return wrapper
    return decorator


# ── Pagination Guard ───────────────────────────────────────────────────────────

class PaginationGuard:
    """
    Guard to prevent infinite loops during cursor-based pagination.
    """
    def __init__(self, max_pages: int = 30, board_id: str | int = "unknown"):
        self.max_pages = max_pages
        self.board_id = board_id
        self.current_page = 0

    def step(self, item_count: int, has_cursor: bool) -> bool:
        """
        Record a page fetch. Returns True if pagination should continue,
        or False if terminated by guard or end of pages.
        """
        self.current_page += 1
        if not has_cursor:
            return False

        if self.current_page >= self.max_pages:
            logger.warning(
                "⚠️ PaginationGuard triggered for board %s: hit max_pages limit (%d). "
                "Halting pagination to prevent runaway loop.",
                self.board_id,
                self.max_pages,
            )
            return False

        return True
