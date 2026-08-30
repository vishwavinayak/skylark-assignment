"""
tests/test_analysis.py
──────────────────────
Unit tests for analysis/bi.py.
Covers:
  - Pipeline health (stages, statuses, closure probability, separate null tracking)
  - Sectoral performance (deals win rates + WO billed/collected values)
  - Ops metrics (completion, billing status, amount receivable, overdue collections)
  - Cross-board health & linkage (1:1 clean rollups vs ambiguous exclusion)
  - Leadership update rollups
"""

import numpy as np
import pandas as pd
import pytest

from analysis.bi import (
    cross_board_health,
    cross_board_linkage,
    leadership_update,
    ops_metrics,
    pipeline_summary,
    sector_breakdown,
    win_loss_rates,
    wo_completion_metrics,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def deals_df():
    """Minimal but realistic deals DataFrame."""
    data = {
        "monday_id": ["1", "2", "3", "4", "5", "6"],
        "name": ["Deal Alpha", "Deal Beta", "Deal Gamma", "Deal Delta", "Deal Epsilon", "Deal Zeta"],
        "deal_stage": ["Proposal", "Closed Won", "Closed Lost", "Negotiation", "Closed Won", "Prospecting"],
        "stage": ["Proposal", "Closed Won", "Closed Lost", "Negotiation", "Closed Won", "Prospecting"],
        "deal_status": ["Open", "Won", "Dead", "Open", "Won", "Open"],
        "sector": ["Energy", "Agriculture", "Energy", "Agriculture", "Energy", "Mining"],
        "deal_value": [500000.0, 1000000.0, 200000.0, 750000.0, 300000.0, np.nan],
        "closure_probability": ["High", "High", None, "Medium", None, "Low"],  # 2 nulls
        "probability": [0.80, 0.80, np.nan, 0.50, np.nan, 0.20],
        "close_date": pd.to_datetime(["2024-09-30", "2024-03-15", "2024-02-28", "2024-10-31", "2024-04-20", None]),
        "created_date": pd.to_datetime(["2024-01-01"] * 6),
        "created_at": pd.to_datetime(["2024-01-01"] * 6),
        "updated_at": pd.to_datetime(["2024-08-01"] * 6),
        "owner_code": ["Alice", "Bob", "Alice", "Bob", "Charlie", None],
        "owner": ["Alice", "Bob", "Alice", "Bob", "Charlie", None],
        "client_code": ["PowerCo", "FarmTech", "SolarInc", "AgriCo", "EnergyLtd", "MinesInc"],
        "currency": ["INR"] * 6,
    }
    return pd.DataFrame(data)


@pytest.fixture
def wo_df():
    """Minimal work orders DataFrame."""
    data = {
        "monday_id": ["101", "102", "103", "104"],
        "name": ["Deal Alpha", "Deal Beta", "WO-Unique", "Deal Delta"],
        "execution_status": ["Completed", "Ongoing", "Completed", "Paused"],
        "status": ["Completed", "Ongoing", "Completed", "Paused"],
        "billing_status": ["Billed", "Partially Billed", "Update Required", "Stuck"],
        "invoice_status": ["Billed", "Not Billed Yet", "Not Billed Yet", "Pending"],
        "sector": ["Energy", "Agriculture", "Energy", "Agriculture"],
        "client": ["PowerCo", "FarmTech", "SolarInc", "MinesInc"],
        "probable_start_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-01-15", "2024-03-01"]),
        "probable_end_date": pd.to_datetime(["2024-03-31", "2024-04-30", "2024-03-15", "2024-04-15"]),
        "data_delivery_date": pd.to_datetime(["2024-03-25", None, "2024-03-20", "2024-05-01"]),
        "revenue": [550000.0, 1000000.0, 200000.0, 700000.0],
        "amount_excl_gst": [550000.0, 1000000.0, 200000.0, 700000.0],
        "billed_excl_gst": [500000.0, 500000.0, 0.0, 0.0],
        "collected_amount": [500000.0, 200000.0, 0.0, 0.0],
        "amount_receivable": [50000.0, 500000.0, 200000.0, 700000.0],
        "is_complete": [True, False, True, False],
        "is_on_time": [True, False, False, False],
        "delay_days": [-6.0, np.nan, 5.0, 16.0],
    }
    return pd.DataFrame(data)


# ── 1. Pipeline Summary (Prompt 6.1) ───────────────────────────────────────────

class TestPipelineSummary:
    def test_returns_dict(self, deals_df):
        result = pipeline_summary(deals_df)
        assert isinstance(result, dict)

    def test_total_deals(self, deals_df):
        result = pipeline_summary(deals_df)
        assert result["total_deals"] == 6

    def test_won_deals(self, deals_df):
        result = pipeline_summary(deals_df)
        assert result["won_deals"] == 2

    def test_pipeline_by_stage_and_status(self, deals_df):
        result = pipeline_summary(deals_df)
        assert "pipeline_by_stage" in result
        assert "pipeline_by_status" in result
        assert len(result["pipeline_by_stage"]) > 0
        assert len(result["pipeline_by_status"]) > 0

    def test_probability_reporting_tracks_nulls_separately(self, deals_df):
        result = pipeline_summary(deals_df)
        assert "probability_reporting" in result
        prob_rep = result["probability_reporting"]
        # In fixture, 2 deals have null closure_probability (Gamma, Epsilon)
        assert prob_rep["deals_with_null_probability_count"] == 2
        assert prob_rep["deals_with_probability_count"] == 4
        # Check that null-probability deals are not dropped from total deals
        assert result["total_deals"] == 6

    def test_empty_df_returns_error(self):
        result = pipeline_summary(pd.DataFrame())
        assert "error" in result


# ── 2. Sector Breakdown (Prompt 6.2) ───────────────────────────────────────────

class TestSectorBreakdown:
    def test_returns_all_sectors(self, deals_df):
        result = sector_breakdown(deals_df)
        sectors = [s["sector"] for s in result["sectors"]]
        assert "Energy" in sectors
        assert "Agriculture" in sectors

    def test_filter_by_sector(self, deals_df):
        result = sector_breakdown(deals_df, filter_sector="Energy")
        assert result["filter_sector"] == "Energy"
        sectors = [s["sector"] for s in result["sectors"]]
        assert sectors == ["Energy"]

    def test_sector_with_work_orders(self, deals_df, wo_df):
        result = sector_breakdown(deals_df, wo_df=wo_df)
        for sec in result["sectors"]:
            assert "wo_count" in sec
            assert "wo_billed_value_fmt" in sec
            assert "wo_collected_value_fmt" in sec
            assert "win_rate_pct" in sec


# ── 3. Win/Loss Rates ──────────────────────────────────────────────────────────

class TestWinLossRates:
    def test_returns_dict(self, deals_df):
        assert isinstance(win_loss_rates(deals_df), dict)

    def test_win_rate_correct(self, deals_df):
        result = win_loss_rates(deals_df)
        # Won: 2, Closed Lost: 1 → win rate = 2/3 ≈ 66.7%
        assert abs(result["overall_win_rate_pct"] - 66.7) < 1.0

    def test_top_closers_list(self, deals_df):
        result = win_loss_rates(deals_df)
        assert isinstance(result["top_closers"], list)


# ── 4. Ops Metrics (Prompt 6.3) ────────────────────────────────────────────────

class TestOpsMetrics:
    def test_returns_dict(self, wo_df):
        result = ops_metrics(wo_df)
        assert isinstance(result, dict)

    def test_completion_rate(self, wo_df):
        result = ops_metrics(wo_df)
        # 2 out of 4 completed → 50%
        assert result["completion_rate_pct"] == 50.0

    def test_billing_status_breakdown(self, wo_df):
        result = ops_metrics(wo_df)
        assert "billing_status_breakdown" in result
        statuses = [b["billing_status"] for b in result["billing_status_breakdown"]]
        assert "Billed" in statuses
        assert "Partially Billed" in statuses

    def test_amount_receivable(self, wo_df):
        result = ops_metrics(wo_df)
        assert "amount_receivable_total" in result
        # 50k + 500k + 200k + 700k = 1,450,000
        assert result["amount_receivable_total"] == 1450000.0
        assert "receivable_by_sector" in result

    def test_overdue_collection_flags(self, wo_df):
        result = ops_metrics(wo_df)
        assert "overdue_collection_flags" in result
        flags = result["overdue_collection_flags"]
        assert "flagged_count" in flags
        assert flags["flagged_count"] >= 1


# ── 5. Cross-Board Linkage (Prompt 4 & 6.4) ────────────────────────────────────

class TestCrossBoardLinkage:
    def test_exact_name_linkage_1_to_1(self, deals_df, wo_df):
        linkage = cross_board_linkage(deals_df, wo_df)
        assert linkage["matched_1_to_1_count"] == 3
        assert len(linkage["clean_linkages"]) == 3
        assert linkage["unmatched_deals_count"] == 3
        assert linkage["unmatched_wo_count"] == 1

        rollup = linkage["numeric_rollup"]
        assert rollup["clean_matched_pairs"] == 3
        assert rollup["clean_total_deal_value"] == 2250000.0

    def test_ambiguous_matches_excluded_from_numeric_rollup(self):
        deals = pd.DataFrame([
            {"monday_id": "1", "name": "Naruto", "deal_value": 500000.0, "stage": "Closed Won", "currency": "INR"},
            {"monday_id": "2", "name": "Naruto", "deal_value": 300000.0, "stage": "Proposal", "currency": "INR"},
            {"monday_id": "3", "name": "Sasuke", "deal_value": 1000000.0, "stage": "Closed Won", "currency": "INR"},
        ])
        wos = pd.DataFrame([
            {"monday_id": "101", "name": "Naruto", "revenue": 450000.0, "billed_excl_gst": 450000.0, "status": "Done"},
            {"monday_id": "102", "name": "Sasuke", "revenue": 1000000.0, "billed_excl_gst": 1000000.0, "status": "Done"},
        ])

        linkage = cross_board_linkage(deals, wos)

        # "Sasuke" is clean 1:1 match
        assert linkage["matched_1_to_1_count"] == 1
        assert len(linkage["clean_linkages"]) == 1

        # "Naruto" is ambiguous (2 deals, 1 WO) -> EXCLUDED from clean rollups
        assert linkage["ambiguous_match_count"] == 1
        assert len(linkage["ambiguous_linkages"]) == 1

        # Rollup includes ONLY Sasuke
        rollup = linkage["numeric_rollup"]
        assert rollup["clean_matched_pairs"] == 1
        assert rollup["clean_total_deal_value"] == 1000000.0
        assert rollup["clean_total_wo_revenue"] == 1000000.0


# ── 6. Cross-Board Health (Prompt 6.4) ─────────────────────────────────────────

class TestCrossBoardHealth:
    def test_returns_dict(self, deals_df, wo_df):
        result = cross_board_health(deals_df, wo_df)
        assert isinstance(result, dict)
        assert "sector_cross_analysis" in result
        assert "name_linkage" in result
        assert "linkage_caveat" in result


# ── 7. Leadership Update ───────────────────────────────────────────────────────

class TestLeadershipUpdate:
    def test_returns_dict(self, deals_df, wo_df):
        result = leadership_update(deals_df, wo_df)
        assert isinstance(result, dict)
        for key in ("pipeline_summary", "win_loss", "operations", "cross_board", "risks_and_flags"):
            assert key in result
        assert isinstance(result["risks_and_flags"], list)
        assert "generated_at" in result
