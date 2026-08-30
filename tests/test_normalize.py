"""
tests/test_normalize.py
────────────────────────
Unit tests for the normalize/clean.py layer.
Covers:
  1. Header-as-value bug (stray re-imported header rows).
  2. "BIlled" typo and Execution Status variant collapsing onto canonical sets.
  3. Blank / missing Masked Deal value (unknown != 0, is_missing flag).
  4. Date parsing defensively to ISO/Timestamp or None.
  5. Scoped data_quality dict with missing % per key downstream field.
"""

import numpy as np
import pandas as pd
import pytest

from normalize.clean import (
    _parse_currency,
    _parse_date,
    _apply_synonym_map,
    drop_header_rows,
    clean_deals,
    clean_work_orders,
    BILLING_STATUS_MAP,
    EXECUTION_STATUS_MAP,
    DEAL_STAGE_MAP,
    SECTOR_MAP,
    STAGE_PROBABILITIES,
)


# ── 1. Currency & Numeric Coercion ─────────────────────────────────────────────

class TestParseCurrency:
    def test_plain_float(self):
        assert _parse_currency("125000.50") == 125000.50

    def test_rupee_symbol(self):
        assert _parse_currency("₹1,50,000") == 150000.0

    def test_dollar_symbol(self):
        assert _parse_currency("$10,000") == 10000.0

    def test_lakh_shorthand(self):
        assert _parse_currency("5L") == 500000.0

    def test_crore_shorthand(self):
        assert _parse_currency("2.5Cr") == 25000000.0

    def test_k_shorthand(self):
        assert _parse_currency("500K") == 500000.0

    def test_range_takes_lower(self):
        assert _parse_currency("10L-15L") == 1000000.0

    def test_none_input_is_none_not_zero(self):
        # Blank / missing must be unknown (None/NaN), never 0
        assert _parse_currency(None) is None

    def test_nan_input_is_none_not_zero(self):
        assert _parse_currency(np.nan) is None

    def test_garbage_input_is_none(self):
        assert _parse_currency("N/A") is None

    def test_empty_string_is_none(self):
        assert _parse_currency("") is None

    def test_commas_and_spaces(self):
        assert _parse_currency("  1,00,000  ") == 100000.0


# ── 2. Date Parsing ────────────────────────────────────────────────────────────

class TestParseDate:
    def test_iso_format(self):
        result = _parse_date("2024-03-15")
        assert result == pd.Timestamp("2024-03-15")

    def test_dmy_slash(self):
        result = _parse_date("15/03/2024")
        assert result == pd.Timestamp("2024-03-15")

    def test_dmy_dash(self):
        result = _parse_date("15-03-2024")
        assert result == pd.Timestamp("2024-03-15")

    def test_iso_datetime(self):
        result = _parse_date("2024-03-15T10:30:00")
        assert result.date() == pd.Timestamp("2024-03-15").date()

    def test_text_format(self):
        result = _parse_date("15 Mar 2024")
        assert result == pd.Timestamp("2024-03-15")

    def test_none_returns_none(self):
        assert _parse_date(None) is None

    def test_na_string_returns_none(self):
        assert _parse_date("N/A") is None

    def test_garbage_returns_none_never_crashes(self):
        assert _parse_date("not-a-date-12345") is None


# ── 3. Synonym Maps & Typo Normalization ────────────────────────────────────────

class TestSynonymMapping:
    def test_billing_status_billed_typo(self):
        # "BIlled" typo from live board must map to "Billed"
        assert _apply_synonym_map("BIlled", BILLING_STATUS_MAP) == "Billed"
        assert _apply_synonym_map("billed", BILLING_STATUS_MAP) == "Billed"

    def test_execution_status_variants_collapse(self):
        # Collapse Execution Status variants onto canonical set
        assert _apply_synonym_map("Executed until current month", EXECUTION_STATUS_MAP) == "Ongoing"
        assert _apply_synonym_map("Pause / struck", EXECUTION_STATUS_MAP) == "Paused"
        assert _apply_synonym_map("Partial Completed", EXECUTION_STATUS_MAP) == "Partial"
        assert _apply_synonym_map("Completed", EXECUTION_STATUS_MAP) == "Completed"
        assert _apply_synonym_map("Not Started", EXECUTION_STATUS_MAP) == "Not Started"

    def test_deal_stage_alphabetic_prefixes(self):
        assert _apply_synonym_map("A. Lead Generated", DEAL_STAGE_MAP) == "Lead"
        assert _apply_synonym_map("B. Sales Qualified Leads", DEAL_STAGE_MAP) == "Qualified"
        assert _apply_synonym_map("E. Proposal/Commercials Sent", DEAL_STAGE_MAP) == "Proposal"
        assert _apply_synonym_map("G. Project Won", DEAL_STAGE_MAP) == "Closed Won"
        assert _apply_synonym_map("L. Project Lost", DEAL_STAGE_MAP) == "Closed Lost"

    def test_sector_synonyms(self):
        assert _apply_synonym_map("energy sector", SECTOR_MAP) == "Energy"
        assert _apply_synonym_map("agri", SECTOR_MAP) == "Agriculture"
        assert _apply_synonym_map("Powerline", SECTOR_MAP) == "Energy"
        assert _apply_synonym_map("Renewables", SECTOR_MAP) == "Renewables"


# ── 4. Header-as-Value Detection ───────────────────────────────────────────────

class TestHeaderAsValueBug:
    def test_drop_header_rows_function(self):
        df = pd.DataFrame([
            {"Sector/service": "Mining", "Deal Stage": "A. Lead Generated", "Masked Deal value": "500000"},
            # Stray header row where cell value == column title
            {"Sector/service": "Sector/service", "Deal Stage": "Deal Stage", "Masked Deal value": "Masked Deal value"},
            {"Sector/service": "Energy", "Deal Stage": "G. Project Won", "Masked Deal value": "1000000"},
        ])
        cleaned_df, n_dropped = drop_header_rows(df)
        assert n_dropped == 1
        assert len(cleaned_df) == 2
        assert "Sector/service" not in cleaned_df["Sector/service"].values
        assert "Deal Stage" not in cleaned_df["Deal Stage"].values

    def test_clean_deals_drops_header_rows(self):
        raw_items = [
            {
                "monday_id": "1",
                "name": "Naruto",
                "Sector/service": "Mining",
                "Deal Stage": "B. Sales Qualified Leads",
                "Masked Deal value": "489360",
                "Deal Status": "Open",
            },
            # Real bug from live Monday.com export: Nezuko & Bugs Bunny header rows
            {
                "monday_id": "999",
                "name": "Nezuko",
                "Sector/service": "Sector/service",
                "Deal Stage": "Deal Stage",
                "Masked Deal value": "Masked Deal value",
                "Deal Status": "Deal Status",
            },
            {
                "monday_id": "2",
                "name": "Sasuke",
                "Sector/service": "Powerline",
                "Deal Stage": "G. Project Won",
                "Masked Deal value": "17616960",
                "Deal Status": "Won",
            },
        ]
        df, report = clean_deals(raw_items)
        assert len(df) == 2
        assert report["header_rows_dropped"] == 1
        assert "Nezuko" not in df["name"].values
        assert "Naruto" in df["name"].values
        assert "Sasuke" in df["name"].values


# ── 5. Blank / Missing Masked Deal Value ───────────────────────────────────────

class TestBlankMaskedDealValue:
    def test_blank_masked_deal_value_is_nan_and_flagged(self):
        raw_items = [
            {
                "monday_id": "1",
                "name": "Deal with Value",
                "Sector/service": "Mining",
                "Deal Stage": "G. Project Won",
                "Masked Deal value": "500000",
            },
            {
                "monday_id": "2",
                "name": "Deal with Blank Value",
                "Sector/service": "Mining",
                "Deal Stage": "A. Lead Generated",
                "Masked Deal value": "",  # blank string
            },
            {
                "monday_id": "3",
                "name": "Deal with None Value",
                "Sector/service": "Mining",
                "Deal Stage": "A. Lead Generated",
                "Masked Deal value": None,  # None
            },
        ]
        df, report = clean_deals(raw_items)
        assert len(df) == 3

        # Row 1 has valid float
        assert df.loc[df["name"] == "Deal with Value", "deal_value"].iloc[0] == 500000.0
        assert df.loc[df["name"] == "Deal with Value", "is_missing_deal_value"].iloc[0] == False

        # Rows 2 & 3 must be NaN (unknown, NOT coerced to 0)
        assert pd.isna(df.loc[df["name"] == "Deal with Blank Value", "deal_value"].iloc[0])
        assert df.loc[df["name"] == "Deal with Blank Value", "is_missing_deal_value"].iloc[0] == True

        assert pd.isna(df.loc[df["name"] == "Deal with None Value", "deal_value"].iloc[0])
        assert df.loc[df["name"] == "Deal with None Value", "is_missing_deal_value"].iloc[0] == True

        # Quality report reflects missingness
        assert "deal_value" in report["fields"]
        assert report["fields"]["deal_value"]["missing_count"] == 2
        assert report["fields"]["deal_value"]["missing_pct"] == 66.7


# ── 6. Work Orders: "BIlled" Typo & Status Collapsing ──────────────────────────

class TestWorkOrdersNormalization:
    def test_billed_typo_and_execution_variants(self):
        raw_items = [
            {
                "monday_id": "101",
                "name": "WO Alpha",
                "Execution Status": "Completed",
                "Billing Status": "BIlled",  # live board typo!
                "Sector": "Mining",
                "Amount in Rupees (Excl of GST) (Masked)": "264398.08",
                "Probable Start Date": "2024-01-10",
                "Probable End Date": "2024-03-31",
                "Data Delivery Date": "2024-04-05",
            },
            {
                "monday_id": "102",
                "name": "WO Beta - Recurring",
                "Execution Status": "Executed until current month",  # variant
                "Billing Status": "Partially Billed",
                "Sector": "Powerline",
                "Amount in Rupees (Excl of GST) (Masked)": "154150",
            },
            {
                "monday_id": "103",
                "name": "WO Gamma - Paused",
                "Execution Status": "Pause / struck",  # variant
                "Billing Status": "Update Required",
                "Sector": "Railways",
                "Amount in Rupees (Excl of GST) (Masked)": None,
            },
        ]
        df, report = clean_work_orders(raw_items)
        assert len(df) == 3

        # Typo fixed
        assert df.loc[df["name"] == "WO Alpha", "billing_status"].iloc[0] == "Billed"

        # Execution status collapsed
        assert df.loc[df["name"] == "WO Alpha", "execution_status"].iloc[0] == "Completed"
        assert df.loc[df["name"] == "WO Beta - Recurring", "execution_status"].iloc[0] == "Ongoing"
        assert df.loc[df["name"] == "WO Gamma - Paused", "execution_status"].iloc[0] == "Paused"

        # Derived fields
        assert df.loc[df["name"] == "WO Alpha", "is_complete"].iloc[0] == True
        assert df.loc[df["name"] == "WO Beta - Recurring", "is_complete"].iloc[0] == False
        assert df.loc[df["name"] == "WO Alpha", "delay_days"].iloc[0] == 5.0

        # Quality report
        assert "board" in report
        assert report["board"] == "work_orders"
        assert "fields" in report
        assert "amount_excl_gst" in report["fields"]
        assert report["fields"]["amount_excl_gst"]["missing_count"] == 1
