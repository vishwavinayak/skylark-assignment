"""
tools/monday.py
───────────────
Monday.com data-access layer — GraphQL with retry, rate-limiting, and error guards (Prompt 8).

GraphQL API details
───────────────────
• Endpoint:  https://api.monday.com/v2
• API ver:   2024-01  (passed as header)
• Auth:      Authorization: <token>  (no "Bearer" prefix for personal tokens)
• Pagination: items_page / next_items_page cursor pattern, max 500 per page

Column mapping strategy
───────────────────────
Monday.com's v2024+ column_values do NOT expose `title` or `type` on the
ColumnValue interface — only `id`, `text`, and `value`.
We therefore do a one-shot board-schema call first, build an id→title map,
then annotate each column_value with its human-readable title before flattening.

Public API
──────────
  get_board_schema(board_id)    → {name, id, columns: [{id, title, type}]}
  get_deals(board_id)           → list[dict]  — one flat dict per item
  get_work_orders(board_id)     → list[dict]
  MondayClient                  — high-level wrapper with optional MCP probe
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import requests
from dotenv import load_dotenv
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from tools.error_handler import (
    MondayAuthError,
    MondayGraphQLError,
    MondayNetworkError,
    MondayRateLimitError,
    PaginationGuard,
    timed_execution,
)

load_dotenv()

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
MONDAY_API_URL = "https://api.monday.com/v2"
PAGE_SIZE = 500  # Monday.com maximum items per page


def _get_token() -> str:
    token = os.getenv("MONDAY_API_TOKEN", "")
    if not token:
        raise MondayAuthError(
            "MONDAY_API_TOKEN is not set. Please add your Monday.com API token to .env."
        )
    return token


def _headers() -> dict[str, str]:
    return {
        "Authorization": _get_token(),
        "Content-Type": "application/json",
        "API-Version": "2024-01",
    }


# ── Retry wrapper ──────────────────────────────────────────────────────────────

@retry(
    retry=retry_if_exception_type((MondayRateLimitError, MondayNetworkError, requests.exceptions.ConnectionError, requests.exceptions.Timeout)),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _graphql(query: str, variables: dict | None = None) -> dict[str, Any]:
    """Execute a GraphQL query against the Monday.com v2 API with 429 and transient error retries."""
    payload: dict[str, Any] = {"query": query}
    if variables:
        payload["variables"] = variables

    try:
        resp = requests.post(
            MONDAY_API_URL,
            json=payload,
            headers=_headers(),
            timeout=30,
        )
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as net_err:
        logger.warning("Transient network error connecting to Monday.com: %s", net_err)
        raise MondayNetworkError(f"Network error connecting to Monday.com: {net_err}") from net_err

    # Auth failure check
    if resp.status_code in (401, 403):
        raise MondayAuthError(
            f"Monday.com authentication failed (HTTP {resp.status_code}). "
            "Please verify that MONDAY_API_TOKEN in your .env file is valid and has read permissions."
        )

    # Rate limiting check
    if resp.status_code == 429:
        retry_after = int(resp.headers.get("Retry-After", "5"))
        logger.warning("Rate limited by Monday.com (HTTP 429). Waiting %ds before retry.", retry_after)
        time.sleep(retry_after)
        raise MondayRateLimitError(f"Rate limited by Monday.com (HTTP 429). Retry after {retry_after}s.")

    # Server errors (5xx)
    if resp.status_code >= 500:
        logger.warning("Monday.com server error (HTTP %d). Will retry.", resp.status_code)
        raise MondayNetworkError(f"Monday.com 5xx server error: HTTP {resp.status_code}")

    resp.raise_for_status()
    data = resp.json()

    # GraphQL payload error check
    if "errors" in data and data["errors"]:
        err_msgs = [e.get("message", str(e)) for e in data["errors"]]
        err_str = "; ".join(err_msgs)
        if any("not authenticated" in m.lower() or "unauthorized" in m.lower() for m in err_msgs):
            raise MondayAuthError(f"Monday.com authentication error in GraphQL response: {err_str}")
        raise MondayGraphQLError(f"Monday.com GraphQL returned errors: {err_str}")

    return data


# ── Schema Discovery ───────────────────────────────────────────────────────────

@timed_execution("monday.get_board_schema")
def get_board_schema(board_id: str | int) -> dict[str, Any]:
    """
    Return board metadata: name + list of columns with id, title, type.

    Used before item fetching to build the id→title map so we can annotate
    column_values (which only carry `id` in the items API).
    """
    query = """
    query($boardId: [ID!]) {
        boards(ids: $boardId) {
            id
            name
            description
            columns {
                id
                title
                type
                settings_str
            }
        }
    }
    """
    data = _graphql(query, {"boardId": [str(board_id)]})
    boards = data.get("data", {}).get("boards", [])
    if not boards:
        raise ValueError(f"Board {board_id} not found or not accessible.")
    board = boards[0]
    logger.info(
        "Schema: board '%s' (%s) — %d columns.",
        board["name"],
        board["id"],
        len(board["columns"]),
    )
    return board


def _build_col_map(schema: dict) -> dict[str, dict[str, str]]:
    """
    Build {column_id → {title, type}} from a board schema dict.
    Used to annotate column_values during item flattening.
    """
    return {
        col["id"]: {"title": col["title"], "type": col["type"]}
        for col in schema.get("columns", [])
    }


# ── Column value parser ────────────────────────────────────────────────────────

def _parse_column_value(col: dict, col_type: str) -> Any:
    """
    Parse a Monday.com column_value dict into a Python scalar.

    col_type is supplied from the schema map (not from the column_value itself,
    because column_values in the items API don't expose `type`).
    We prefer `text` for display-ready strings and fall back to `value` for
    structured/boolean types.
    """
    text = (col.get("text") or "").strip()
    raw  = col.get("value")     # JSON-encoded string or None

    # Simple text / email / phone / link / long-text
    if col_type in ("text", "email", "phone", "link", "long-text"):
        return text or None

    # Numeric / currency / rating  →  return as-is string; normalizer parses
    if col_type in ("numeric", "rating"):
        return text or None

    # Date  →  raw string; normalizer handles format
    if col_type == "date":
        return text or None

    # Status / dropdown / color  →  label text
    if col_type in ("status", "dropdown", "color"):
        return text or None

    # People (multiple-person)
    if col_type == "multiple-person":
        if raw:
            try:
                parsed = json.loads(raw)
                persons = parsed.get("personsAndTeams", [])
                return ", ".join(str(p.get("id", "")) for p in persons)
            except (json.JSONDecodeError, AttributeError):
                pass
        return text or None

    # Timeline / timerange
    if col_type == "timerange":
        return text or None

    # Formula / mirror / auto-number / board-relation
    if col_type in ("formula", "mirror", "autonumber", "board-relation"):
        return text or None

    # Checkbox / boolean
    if col_type == "boolean":
        if raw:
            try:
                return json.loads(raw).get("checked", False)
            except (json.JSONDecodeError, AttributeError):
                pass
        return text.lower() in ("true", "v", "✓", "yes") if text else False

    # Fallback: return text
    return text or None


# ── Core item fetcher with Pagination Guard ────────────────────────────────────

_COLUMN_VALUES_FRAGMENT = """
    column_values {
        id
        text
        value
    }
"""

_INITIAL_ITEMS_QUERY = """
query($boardId: [ID!], $limit: Int) {
    boards(ids: $boardId) {
        items_page(limit: $limit) {
            cursor
            items {
                id
                name
                state
                created_at
                updated_at
                group { title }
                %s
            }
        }
    }
}
""" % _COLUMN_VALUES_FRAGMENT

_NEXT_ITEMS_QUERY = """
query($cursor: String!, $limit: Int) {
    next_items_page(cursor: $cursor, limit: $limit) {
        cursor
        items {
            id
            name
            state
            created_at
            updated_at
            group { title }
            %s
        }
    }
}
""" % _COLUMN_VALUES_FRAGMENT


def _fetch_all_items(
    board_id: str | int,
    col_map: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    """
    Fetch every item from a board using cursor-based pagination and pagination guard.
    """
    all_items: list[dict[str, Any]] = []
    guard = PaginationGuard(max_pages=30, board_id=board_id)

    # ── Page 1 ─────────────────────────────────────────────────────────────────
    data = _graphql(
        _INITIAL_ITEMS_QUERY,
        {"boardId": [str(board_id)], "limit": PAGE_SIZE},
    )
    board_data = data["data"]["boards"][0]
    page = board_data["items_page"]
    items = page.get("items", [])
    all_items.extend(items)
    cursor = page.get("cursor")

    logger.info(
        "Fetched %d items (page 1) from board %s. has_more=%s",
        len(items),
        board_id,
        bool(cursor),
    )

    # ── Subsequent pages with PaginationGuard ──────────────────────────────────
    while guard.step(item_count=len(items), has_cursor=bool(cursor)):
        data = _graphql(
            _NEXT_ITEMS_QUERY,
            {"cursor": cursor, "limit": PAGE_SIZE},
        )
        page = data["data"].get("next_items_page")
        if not page:
            break
        items = page.get("items", [])
        all_items.extend(items)
        cursor = page.get("cursor")
        logger.info(
            "Fetched %d items (page %d). has_more=%s",
            len(items),
            guard.current_page,
            bool(cursor),
        )

    logger.info("Total: %d items fetched from board %s.", len(all_items), board_id)
    return all_items


def _flatten_item(item: dict[str, Any], col_map: dict[str, dict[str, str]]) -> dict[str, Any]:
    """
    Convert a raw Monday item into a flat dict:
      {
        "monday_id": ...,
        "name": ...,
        "group": ...,
        "created_at": ...,
        "updated_at": ...,
        "<Column Title>": <parsed_value>,
        ...
      }
    """
    flat: dict[str, Any] = {
        "monday_id": item.get("id"),
        "name": item.get("name", ""),
        "state": item.get("state", ""),
        "created_at": item.get("created_at"),
        "updated_at": item.get("updated_at"),
        "group": (
            item.get("group", {}).get("title", "")
            if item.get("group")
            else ""
        ),
    }

    for col in item.get("column_values", []):
        col_id = col.get("id", "")
        meta = col_map.get(col_id, {})
        title = meta.get("title") or col_id
        col_type = meta.get("type", "text")
        flat[title] = _parse_column_value(col, col_type)

    return flat


# ── Public API ─────────────────────────────────────────────────────────────────

@timed_execution("monday.get_deals")
def get_deals(board_id: str | int | None = None) -> list[dict[str, Any]]:
    """
    Fetch all deal items from the Deal Funnel board via live Monday.com GraphQL.
    """
    board_id = board_id or os.getenv("DEALS_BOARD_ID")
    if not board_id:
        raise EnvironmentError(
            "DEALS_BOARD_ID is not set. Pass board_id explicitly or add it to .env."
        )

    logger.info("get_deals: fetching schema for board %s...", board_id)
    schema = get_board_schema(board_id)
    col_map = _build_col_map(schema)

    logger.info("get_deals: fetching items...")
    raw_items = _fetch_all_items(board_id, col_map)
    flattened = [_flatten_item(i, col_map) for i in raw_items]

    logger.info(
        "get_deals: done — %d deals from board '%s'.",
        len(flattened),
        schema.get("name", board_id),
    )
    return flattened


@timed_execution("monday.get_work_orders")
def get_work_orders(board_id: str | int | None = None) -> list[dict[str, Any]]:
    """
    Fetch all work order items from the Work Order Tracker board via live Monday.com GraphQL.
    """
    board_id = board_id or os.getenv("WORK_ORDERS_BOARD_ID")
    if not board_id:
        raise EnvironmentError(
            "WORK_ORDERS_BOARD_ID is not set. Pass board_id explicitly or add it to .env."
        )

    logger.info("get_work_orders: fetching schema for board %s...", board_id)
    schema = get_board_schema(board_id)
    col_map = _build_col_map(schema)

    logger.info("get_work_orders: fetching items...")
    raw_items = _fetch_all_items(board_id, col_map)
    flattened = [_flatten_item(i, col_map) for i in raw_items]

    logger.info(
        "get_work_orders: done — %d work orders from board '%s'.",
        len(flattened),
        schema.get("name", board_id),
    )
    return flattened


# ── High-Level Client ──────────────────────────────────────────────────────────

class MondayClient:
    """High-level client with optional MCP probe and direct GraphQL access."""

    def __init__(self):
        self.use_mcp = os.getenv("USE_MCP", "false").lower() == "true"
        self._mcp_available: bool | None = None

    def _try_mcp_tools(self) -> bool:
        if self._mcp_available is not None:
            return self._mcp_available

        if not self.use_mcp:
            self._mcp_available = False
            return False

        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient  # type: ignore
            import asyncio

            async def _probe():
                client = MultiServerMCPClient(
                    {
                        "monday-platform": {
                            "transport": "http",
                            "url": "https://mcp.monday.com/mcp",
                            "headers": {"Authorization": f"Bearer {_get_token()}"},
                        }
                    }
                )
                tools = await client.get_tools()
                return len(tools) > 0

            available = asyncio.run(_probe())
            self._mcp_available = available
            if available:
                logger.info("Monday.com MCP: available.")
            else:
                logger.warning("MCP probe: 0 tools returned — using GraphQL.")
        except Exception as exc:
            logger.warning("MCP probe failed (%s). Using direct GraphQL.", exc)
            self._mcp_available = False

        return self._mcp_available  # type: ignore[return-value]

    def get_board_schema(self, board_id: str | int) -> dict:
        return get_board_schema(board_id)

    def get_deals(self, board_id: str | int | None = None) -> list[dict]:
        self._try_mcp_tools()
        return get_deals(board_id)

    def get_work_orders(self, board_id: str | int | None = None) -> list[dict]:
        self._try_mcp_tools()
        return get_work_orders(board_id)
