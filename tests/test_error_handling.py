"""
tests/test_error_handling.py
────────────────────────────
Unit tests for the error handling wrapper and telemetry (Prompt 8).
"""

import pytest
from tools.error_handler import (
    MondayAuthError,
    MondayRateLimitError,
    MondayNetworkError,
    PaginationGuard,
    timed_execution,
)
from graph.agent import data_retrieval


class TestPaginationGuard:
    def test_pagination_guard_stops_at_max_pages(self):
        guard = PaginationGuard(max_pages=5, board_id="123")
        assert guard.step(item_count=100, has_cursor=True) is True
        assert guard.step(item_count=100, has_cursor=True) is True
        assert guard.step(item_count=100, has_cursor=True) is True
        assert guard.step(item_count=100, has_cursor=True) is True
        # 5th page reached max_pages -> should terminate
        assert guard.step(item_count=100, has_cursor=True) is False

    def test_pagination_guard_stops_on_no_cursor(self):
        guard = PaginationGuard(max_pages=10, board_id="123")
        assert guard.step(item_count=50, has_cursor=False) is False


class TestTimedExecution:
    def test_timed_execution_decorator(self):
        @timed_execution("test_function")
        def add(a, b):
            return a + b

        result = add(3, 4)
        assert result == 7

    def test_timed_execution_propagates_exception(self):
        @timed_execution("failing_function")
        def fail():
            raise ValueError("Something went wrong")

        with pytest.raises(ValueError, match="Something went wrong"):
            fail()


class TestPartialBoardFailure:
    def test_partial_board_recovery_in_data_retrieval(self, monkeypatch):
        # Mock monday client where Deals succeeds and Work Orders fails
        class MockMonday:
            def get_deals(self, board_id=None):
                return [{"monday_id": "1", "name": "Deal Alpha", "Masked Deal value": "100000"}]

            def get_work_orders(self, board_id=None):
                raise MondayNetworkError("Connection timed out to Work Orders board")

        monkeypatch.setattr("graph.agent._get_monday", lambda: MockMonday())

        state = {
            "required_boards": ["deals", "workorders"],
            "raw_data": {},
            "assumptions": [],
        }

        result = data_retrieval(state)

        # Deals should be present
        assert "deals" in result["raw_data"]
        assert len(result["raw_data"]["deals"]) == 1

        # Work orders failed but error is captured in assumptions and state.error
        assert "workorders" not in result["raw_data"] or not result["raw_data"]["workorders"]
        assert result["error"] is not None
        assert any("Work Orders board was unreachable" in a for a in result["assumptions"])
