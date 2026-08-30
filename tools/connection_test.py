"""
tools/connection_test.py
─────────────────────────
Standalone connectivity verifier.

Run with:
    python -m tools.connection_test

Checks:
  1. MONDAY_API_TOKEN, DEALS_BOARD_ID, WORK_ORDERS_BOARD_ID are set
  2. Monday.com API is reachable (account query)
  3. Schema can be fetched for both boards
  4. First page of items can be fetched from both boards
  5. Prints a summary table

Exit codes:
  0  — all checks passed
  1  — one or more checks failed
"""

from __future__ import annotations

import os
import sys
import textwrap
import traceback
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()


def _check(label: str, fn) -> tuple[bool, str]:
    """Run fn(), return (success, message)."""
    try:
        result = fn()
        return True, str(result)
    except Exception as exc:
        return False, f"FAILED — {exc}\n{textwrap.indent(traceback.format_exc(), '  ')}"


def run_checks() -> int:
    """Run all connectivity checks. Returns exit code."""
    print("\n" + "=" * 60)
    print("  Skylark BI Agent — Monday.com Connection Test")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60 + "\n")

    # Lazy import so missing dotenv doesn't crash the module-level import
    from tools.monday import get_board_schema, _graphql  # noqa: F401

    checks: list[tuple[str, callable]] = []

    # ── 1. Env vars ────────────────────────────────────────────────────────────
    def check_env():
        missing = [
            v
            for v in ("MONDAY_API_TOKEN", "DEALS_BOARD_ID", "WORK_ORDERS_BOARD_ID")
            if not os.getenv(v)
        ]
        if missing:
            raise EnvironmentError(f"Missing env vars: {', '.join(missing)}")
        return "All required env vars present ✓"

    checks.append(("Environment Variables", check_env))

    # ── 2. API reachability ────────────────────────────────────────────────────
    def check_api():
        data = _graphql("{ me { id name email } }")
        me = data["data"]["me"]
        return f"Authenticated as: {me['name']} <{me['email']}>"

    checks.append(("API Authentication", check_api))

    # ── 3. Deals board schema ──────────────────────────────────────────────────
    def check_deals_schema():
        board_id = os.getenv("DEALS_BOARD_ID")
        schema = get_board_schema(board_id)
        cols = [c["title"] for c in schema["columns"]]
        return (
            f"Board: '{schema['name']}' | "
            f"{len(cols)} columns: {', '.join(cols[:8])}{'...' if len(cols) > 8 else ''}"
        )

    checks.append(("Deals Board Schema", check_deals_schema))

    # ── 4. Work Orders board schema ────────────────────────────────────────────
    def check_wo_schema():
        board_id = os.getenv("WORK_ORDERS_BOARD_ID")
        schema = get_board_schema(board_id)
        cols = [c["title"] for c in schema["columns"]]
        return (
            f"Board: '{schema['name']}' | "
            f"{len(cols)} columns: {', '.join(cols[:8])}{'...' if len(cols) > 8 else ''}"
        )

    checks.append(("Work Orders Board Schema", check_wo_schema))

    # ── 5. Deals first-page fetch ──────────────────────────────────────────────
    def check_deals_items():
        from tools.monday import _fetch_all_items, _flatten_item

        board_id = os.getenv("DEALS_BOARD_ID")
        # Only fetch the first page (don't paginate in a test)
        from tools.monday import _graphql, _COLUMN_VALUES_FRAGMENT, PAGE_SIZE

        q = """
        query($boardId: [ID!], $limit: Int) {
            boards(ids: $boardId) {
                items_page(limit: $limit) {
                    cursor
                    items {
                        id name state created_at updated_at
                        group { title }
                        %s
                    }
                }
            }
        }
        """ % _COLUMN_VALUES_FRAGMENT

        data = _graphql(q, {"boardId": [str(board_id)], "limit": 5})
        items = data["data"]["boards"][0]["items_page"]["items"]
        flat = [_flatten_item(i) for i in items]
        sample_keys = list(flat[0].keys()) if flat else []
        return (
            f"Fetched {len(flat)} sample items. "
            f"Fields: {', '.join(sample_keys[:10])}{'...' if len(sample_keys) > 10 else ''}"
        )

    checks.append(("Deals Board — Sample Items", check_deals_items))

    # ── 6. Work Orders first-page fetch ───────────────────────────────────────
    def check_wo_items():
        from tools.monday import _flatten_item, _graphql, _COLUMN_VALUES_FRAGMENT

        board_id = os.getenv("WORK_ORDERS_BOARD_ID")
        q = """
        query($boardId: [ID!], $limit: Int) {
            boards(ids: $boardId) {
                items_page(limit: $limit) {
                    cursor
                    items {
                        id name state created_at updated_at
                        group { title }
                        %s
                    }
                }
            }
        }
        """ % _COLUMN_VALUES_FRAGMENT

        data = _graphql(q, {"boardId": [str(board_id)], "limit": 5})
        items = data["data"]["boards"][0]["items_page"]["items"]
        flat = [_flatten_item(i) for i in items]
        sample_keys = list(flat[0].keys()) if flat else []
        return (
            f"Fetched {len(flat)} sample items. "
            f"Fields: {', '.join(sample_keys[:10])}{'...' if len(sample_keys) > 10 else ''}"
        )

    checks.append(("Work Orders Board — Sample Items", check_wo_items))

    # ── Run all checks ─────────────────────────────────────────────────────────
    all_passed = True
    for label, fn in checks:
        success, message = _check(label, fn)
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}  {label}")
        if success:
            print(f"         {message}\n")
        else:
            all_passed = False
            print(f"         {message}\n")

    print("=" * 60)
    if all_passed:
        print("  All checks passed. Monday.com connectivity is working! 🚀")
    else:
        print("  One or more checks failed. See details above.")
    print("=" * 60 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(run_checks())
