"""
analysis/bi.py
──────────────
Deterministic Business Intelligence calculations (pandas only, no LLM math).

Core BI functions:
  1. pipeline_summary(deals_df)
     - Pipeline health: deal count/value by Deal Stage and Deal Status.
     - Weighted by Closure Probability where present.
     - Reports null-probability deals separately rather than dropping them.
  2. sector_breakdown(deals_df, wo_df=None, filter_sector=None)
     - Sectoral performance: deal value/win rate by Sector/service.
     - Work order completion and billed/collected value by Sector.
  3. ops_metrics(wo_df) / wo_completion_metrics(wo_df)
     - Ops metrics: completion rate, Billing Status breakdown.
     - Total Amount Receivable (with sector breakdown).
     - Overdue-collection flags (Data Delivery Date passed, Invoice Status not final).
  4. cross_board_linkage(deals_df, wo_df) & cross_board_health(deals_df, wo_df)
     - Cross-board sector view: open-deal pipeline vs active work-order load per sector.
     - Uses only the non-ambiguous 1:1 links from Prompt 4.
     - Excludes ambiguous codename matches (>1 deal or >1 WO) from numeric rollups by default with explicit caveat.
  5. leadership_update(deals_df, wo_df)
     - Board-level executive KPI rollups.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from normalize.clean import CLOSURE_PROBABILITY_MAP, STAGE_PROBABILITIES

logger = logging.getLogger(__name__)

# Numeric weight mapping for live board closure probability strings
_PROBABILITY_WEIGHTS: dict[str, float] = {
    "High": 0.80,
    "Medium": 0.50,
    "Low": 0.20,
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division that returns `default` instead of ZeroDivisionError / NaN."""
    if denominator == 0 or pd.isna(denominator):
        return default
    result = numerator / denominator
    return float(result) if not pd.isna(result) else default


def _fmt_currency(value: float | None, currency: str = "INR") -> str:
    """Format a float as a human-readable currency string."""
    if value is None or pd.isna(value):
        return "N/A"
    if currency == "INR":
        if abs(value) >= 1_00_00_000:
            return f"₹{value / 1_00_00_000:.2f} Cr"
        if abs(value) >= 1_00_000:
            return f"₹{value / 1_00_00_000 * 100:.2f} L" if abs(value) >= 1_00_00_000 else f"₹{value / 1_00_000:.2f} L"
        return f"₹{value:,.0f}"
    return f"${value:,.0f}"


def _current_quarter() -> tuple[int, int]:
    """Return (year, quarter) for today's date."""
    now = datetime.now(timezone.utc)
    return now.year, (now.month - 1) // 3 + 1


def _in_current_quarter(date_series: pd.Series) -> pd.Series:
    """Boolean mask: True if date falls in the current calendar quarter."""
    year, q = _current_quarter()
    q_start = pd.Timestamp(year=year, month=(q - 1) * 3 + 1, day=1)
    q_end = q_start + pd.offsets.QuarterEnd(0)
    return (date_series >= q_start) & (date_series <= q_end)


# ══════════════════════════════════════════════════════════════════════════════
# 1. PIPELINE HEALTH (Prompt 6.1)
# ══════════════════════════════════════════════════════════════════════════════

def pipeline_summary(deals_df: pd.DataFrame) -> dict[str, Any]:
    """
    Pipeline health metrics:
    - Deal count and value by Deal Stage and Deal Status.
    - Weighted by Closure Probability where present (or stage probability).
    - Reports null-probability deals separately rather than dropping them.
    """
    if deals_df.empty:
        return {"error": "No deal data available."}

    df = deals_df.copy()
    currency = df["currency"].iloc[0] if "currency" in df.columns else "INR"

    stage_col = "deal_stage" if "deal_stage" in df.columns else "stage"
    status_col = "deal_status" if "deal_status" in df.columns else "status"

    # ── Closure Probability Weighting ──────────────────────────────────────────
    # If Closure Probability string column exists (High/Medium/Low), map it.
    # If null, check stage probability or leave as null probability.
    has_closure_prob = "closure_probability" in df.columns
    if has_closure_prob:
        df["closure_prob_weight"] = df["closure_probability"].map(_PROBABILITY_WEIGHTS)
        df["has_explicit_prob"] = df["closure_prob_weight"].notna()
    else:
        df["closure_prob_weight"] = np.nan
        df["has_explicit_prob"] = False

    # Effective probability: explicit closure prob if present, else stage probability fallback
    df["effective_prob"] = np.where(
        df["has_explicit_prob"],
        df["closure_prob_weight"],
        df[stage_col].map(STAGE_PROBABILITIES),
    )
    df["calculated_weighted_value"] = df["deal_value"] * df["effective_prob"]

    active = df[df[stage_col] != "Closed Lost"]
    won = df[df[stage_col] == "Closed Won"]

    # Separate tracking for null probability
    null_prob_mask = df["closure_probability"].isna() if has_closure_prob else df["effective_prob"].isna()
    null_prob_deals = df[null_prob_mask]
    with_prob_deals = df[~null_prob_mask]

    null_prob_count = int(len(null_prob_deals))
    null_prob_val = float(null_prob_deals["deal_value"].sum() or 0)
    with_prob_count = int(len(with_prob_deals))
    with_prob_val = float(with_prob_deals["deal_value"].sum() or 0)

    # ── Breakdown by Deal Stage ────────────────────────────────────────────────
    stage_breakdown: list[dict] = []
    for stage_val, grp in df.groupby(stage_col, dropna=False):
        stage_name = str(stage_val) if not pd.isna(stage_val) else "Unknown"
        total_val = float(grp["deal_value"].sum() or 0)
        weighted_val = float(grp["calculated_weighted_value"].sum() or 0)
        null_grp = grp[grp["closure_probability"].isna()] if has_closure_prob else grp[grp["effective_prob"].isna()]

        stage_breakdown.append({
            "stage": stage_name,
            "count": int(len(grp)),
            "total_value": round(total_val, 2),
            "total_value_fmt": _fmt_currency(total_val, currency),
            "weighted_value": round(weighted_val, 2),
            "weighted_value_fmt": _fmt_currency(weighted_val, currency),
            "null_probability_count": int(len(null_grp)),
            "null_probability_value_fmt": _fmt_currency(float(null_grp["deal_value"].sum() or 0), currency),
            "avg_value_fmt": _fmt_currency(_safe_div(total_val, len(grp)), currency),
        })

    # ── Breakdown by Deal Status ───────────────────────────────────────────────
    status_breakdown: list[dict] = []
    if status_col in df.columns:
        for stat_val, grp in df.groupby(status_col, dropna=False):
            stat_name = str(stat_val) if not pd.isna(stat_val) else "Unknown"
            s_val = float(grp["deal_value"].sum() or 0)
            s_weighted = float(grp["calculated_weighted_value"].sum() or 0)
            status_breakdown.append({
                "status": stat_name,
                "count": int(len(grp)),
                "total_value": round(s_val, 2),
                "total_value_fmt": _fmt_currency(s_val, currency),
                "weighted_value": round(s_weighted, 2),
                "weighted_value_fmt": _fmt_currency(s_weighted, currency),
            })

    # Current quarter
    date_col = "close_date" if "close_date" in df.columns else "tentative_close_date"
    cq_mask = _in_current_quarter(df[date_col]) if date_col in df.columns else pd.Series([False] * len(df))
    cq_df = df[cq_mask]
    year, q = _current_quarter()

    total_pipeline = float(active["deal_value"].sum() or 0)
    weighted_pipeline = float(active["calculated_weighted_value"].sum() or 0)

    return {
        "total_deals": int(len(df)),
        "active_deals": int(len(active)),
        "won_deals": int(len(won)),
        "currency": currency,
        "total_pipeline_value": round(total_pipeline, 2),
        "total_pipeline_value_fmt": _fmt_currency(total_pipeline, currency),
        "weighted_pipeline_value": round(weighted_pipeline, 2),
        "weighted_pipeline_value_fmt": _fmt_currency(weighted_pipeline, currency),
        "avg_deal_size": round(_safe_div(float(df["deal_value"].sum() or 0), len(df[df["deal_value"].notna()])), 2),
        "avg_deal_size_fmt": _fmt_currency(_safe_div(float(df["deal_value"].sum() or 0), len(df[df["deal_value"].notna()])), currency),
        "probability_reporting": {
            "deals_with_probability_count": with_prob_count,
            "deals_with_probability_value_fmt": _fmt_currency(with_prob_val, currency),
            "deals_with_null_probability_count": null_prob_count,
            "deals_with_null_probability_value_fmt": _fmt_currency(null_prob_val, currency),
            "note": "Null-probability deals are tracked and reported separately without being dropped from pipeline counts or totals.",
        },
        "pipeline_by_stage": stage_breakdown,
        "pipeline_by_status": status_breakdown,
        "current_quarter": f"Q{q} {year}",
        "current_quarter_deal_count": int(len(cq_df)),
        "current_quarter_pipeline_value_fmt": _fmt_currency(float(cq_df["deal_value"].sum() or 0), currency),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. SECTORAL PERFORMANCE (Prompt 6.2)
# ══════════════════════════════════════════════════════════════════════════════

def sector_breakdown(
    deals_df: pd.DataFrame,
    wo_df: pd.DataFrame | None = None,
    filter_sector: str | None = None,
) -> dict[str, Any]:
    """
    Sectoral performance analysis:
    - Deal value and win rate by Sector/service.
    - Work order completion and billed/collected value by Sector (when wo_df is provided).
    """
    if deals_df.empty:
        return {"error": "No deal data available."}

    df = deals_df.copy()
    currency = df["currency"].iloc[0] if "currency" in df.columns else "INR"
    stage_col = "deal_stage" if "deal_stage" in df.columns else "stage"

    if filter_sector:
        df = df[df["sector"].str.lower() == filter_sector.lower()]
        if df.empty:
            return {
                "error": f"No deals found for sector '{filter_sector}'.",
                "available_sectors": sorted(deals_df["sector"].dropna().unique().tolist()),
            }

    # Pre-aggregate work order sector data if available
    wo_by_sector: dict[str, dict] = {}
    if wo_df is not None and not wo_df.empty and "sector" in wo_df.columns:
        w_df = wo_df.copy()
        if filter_sector:
            w_df = w_df[w_df["sector"].str.lower() == filter_sector.lower()]

        for sec, grp in w_df.groupby("sector", dropna=False):
            sec_name = str(sec) if not pd.isna(sec) else "Unknown"
            comp_cnt = int(grp["is_complete"].sum() if "is_complete" in grp.columns else 0)
            tot_cnt = int(len(grp))
            billed_val = float(grp["billed_excl_gst"].sum() if "billed_excl_gst" in grp.columns else 0)
            collected_val = float(grp["collected_amount"].sum() if "collected_amount" in grp.columns else 0)
            rev_val = float(grp["revenue"].sum() if "revenue" in grp.columns else grp.get("amount_excl_gst", pd.Series([0])).sum() or 0)

            wo_by_sector[sec_name] = {
                "wo_count": tot_cnt,
                "wo_completed": comp_cnt,
                "wo_completion_rate_pct": round(_safe_div(comp_cnt, tot_cnt) * 100, 1),
                "wo_billed_value_fmt": _fmt_currency(billed_val, currency),
                "wo_collected_value_fmt": _fmt_currency(collected_val, currency),
                "wo_revenue_fmt": _fmt_currency(rev_val, currency),
            }

    sectors: list[dict] = []
    for sector, grp in df.groupby("sector", dropna=False):
        sector_name = str(sector) if not pd.isna(sector) else "Unknown"
        won = grp[grp[stage_col] == "Closed Won"]
        active = grp[grp[stage_col] != "Closed Lost"]
        win_rate = _safe_div(len(won), len(grp)) * 100
        tot_val = float(grp["deal_value"].sum() or 0)

        item: dict[str, Any] = {
            "sector": sector_name,
            "deal_count": int(len(grp)),
            "total_value": round(tot_val, 2),
            "total_value_fmt": _fmt_currency(tot_val, currency),
            "won_value": round(float(won["deal_value"].sum() or 0), 2),
            "won_value_fmt": _fmt_currency(float(won["deal_value"].sum() or 0), currency),
            "active_pipeline_fmt": _fmt_currency(float(active["deal_value"].sum() or 0), currency),
            "win_rate_pct": round(win_rate, 1),
            "avg_deal_size_fmt": _fmt_currency(_safe_div(tot_val, grp["deal_value"].notna().sum()), currency),
        }

        # Attach work order metrics if available
        if sector_name in wo_by_sector:
            item.update(wo_by_sector[sector_name])
        elif wo_df is not None:
            item.update({
                "wo_count": 0,
                "wo_completed": 0,
                "wo_completion_rate_pct": 0.0,
                "wo_billed_value_fmt": "₹0",
                "wo_collected_value_fmt": "₹0",
                "wo_revenue_fmt": "₹0",
            })

        sectors.append(item)

    sectors.sort(key=lambda x: x["total_value"], reverse=True)

    return {
        "filter_sector": filter_sector,
        "currency": currency,
        "sectors": sectors,
        "top_sector": sectors[0]["sector"] if sectors else None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. WIN / LOSS RATES
# ══════════════════════════════════════════════════════════════════════════════

def win_loss_rates(deals_df: pd.DataFrame) -> dict[str, Any]:
    """
    Win rate analysis, average deal cycle time, and top closers.
    """
    if deals_df.empty:
        return {"error": "No deal data available."}

    df = deals_df.copy()
    currency = df["currency"].iloc[0] if "currency" in df.columns else "INR"
    stage_col = "deal_stage" if "deal_stage" in df.columns else "stage"

    closed = df[df[stage_col].isin(["Closed Won", "Closed Lost"])].copy()
    won = closed[closed[stage_col] == "Closed Won"]

    overall_win_rate = _safe_div(len(won), len(closed)) * 100

    # Average deal cycle: close_date - created_date / created_at
    date_close = "close_date" if "close_date" in df.columns else "tentative_close_date"
    date_create = "created_date" if "created_date" in df.columns else "created_at"
    if date_close in df.columns and date_create in df.columns:
        won_with_dates = won.dropna(subset=[date_close, date_create]).copy()
        if len(won_with_dates):
            won_with_dates["cycle_days"] = (
                won_with_dates[date_close] - won_with_dates[date_create]
            ).dt.days
            avg_cycle = float(won_with_dates["cycle_days"].mean())
        else:
            avg_cycle = None
    else:
        avg_cycle = None

    # Top closers (by owner / owner_code)
    owner_col = "owner_code" if "owner_code" in df.columns else "owner"
    top_closers: list[dict] = []
    if owner_col in df.columns:
        for owner, grp in df.groupby(owner_col, dropna=False):
            if pd.isna(owner):
                continue
            owner_won = grp[grp[stage_col] == "Closed Won"]
            owner_closed = grp[grp[stage_col].isin(["Closed Won", "Closed Lost"])]
            top_closers.append({
                "owner": str(owner),
                "deals_won": int(len(owner_won)),
                "deals_closed": int(len(owner_closed)),
                "win_rate_pct": round(_safe_div(len(owner_won), len(owner_closed)) * 100, 1),
                "revenue_won_fmt": _fmt_currency(float(owner_won["deal_value"].sum() or 0), currency),
            })
        top_closers.sort(key=lambda x: x["deals_won"], reverse=True)

    return {
        "total_deals": int(len(df)),
        "total_closed": int(len(closed)),
        "total_won": int(len(won)),
        "total_lost": int(len(closed) - len(won)),
        "overall_win_rate_pct": round(overall_win_rate, 1),
        "avg_deal_cycle_days": round(avg_cycle, 1) if avg_cycle is not None else None,
        "revenue_from_won_fmt": _fmt_currency(float(won["deal_value"].sum() or 0), currency),
        "top_closers": top_closers[:10],
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. OPS METRICS & OVERDUE COLLECTIONS (Prompt 6.3)
# ══════════════════════════════════════════════════════════════════════════════

def ops_metrics(wo_df: pd.DataFrame) -> dict[str, Any]:
    """
    Operational metrics (Prompt 6.3):
    - Completion rate.
    - Billing Status breakdown.
    - Amount Receivable (total and by sector / customer).
    - Overdue-collection flags: Data Delivery Date passed AND Invoice Status not final.
    """
    if wo_df.empty:
        return {"error": "No work order data available."}

    df = wo_df.copy()
    currency = "INR"
    total = len(df)

    # Completion
    is_comp_col = "is_complete" if "is_complete" in df.columns else "status"
    if is_comp_col == "is_complete":
        completed = df[df["is_complete"] == True]
    else:
        completed = df[df["status"].isin(["Completed", "Done"])]
    completion_rate = _safe_div(len(completed), total) * 100

    # On-time & delays
    on_time = df[df["is_on_time"] == True] if "is_on_time" in df.columns else pd.DataFrame()
    on_time_rate = _safe_div(len(on_time), len(completed)) * 100 if len(completed) else 0.0
    delayed = df[(df["delay_days"] > 0)] if "delay_days" in df.columns else pd.DataFrame()
    avg_delay = float(delayed["delay_days"].mean()) if len(delayed) else 0.0

    # ── Billing Status Breakdown ───────────────────────────────────────────────
    bill_col = "billing_status" if "billing_status" in df.columns else "Billing Status"
    billing_counts: list[dict] = []
    if bill_col in df.columns:
        for b_status, grp in df.groupby(bill_col, dropna=False):
            b_name = str(b_status) if not pd.isna(b_status) else "Unknown"
            b_val = float(grp["revenue"].sum() if "revenue" in grp.columns else grp.get("amount_excl_gst", pd.Series([0])).sum() or 0)
            billing_counts.append({
                "billing_status": b_name,
                "count": int(len(grp)),
                "pct": round(_safe_div(len(grp), total) * 100, 1),
                "amount_fmt": _fmt_currency(b_val, currency),
            })
        billing_counts.sort(key=lambda x: x["count"], reverse=True)

    # ── Amount Receivable ──────────────────────────────────────────────────────
    rec_col = "amount_receivable" if "amount_receivable" in df.columns else "Amount Receivable (Masked)"
    total_receivable = float(df[rec_col].sum() or 0) if rec_col in df.columns else 0.0

    # Receivable by sector
    receivable_by_sector: list[dict] = []
    if rec_col in df.columns and "sector" in df.columns:
        for sec, grp in df.groupby("sector", dropna=False):
            sec_rec = float(grp[rec_col].sum() or 0)
            if sec_rec > 0:
                receivable_by_sector.append({
                    "sector": str(sec) if not pd.isna(sec) else "Unknown",
                    "amount_receivable": round(sec_rec, 2),
                    "amount_receivable_fmt": _fmt_currency(sec_rec, currency),
                    "work_order_count": int(len(grp)),
                })
        receivable_by_sector.sort(key=lambda x: x["amount_receivable"], reverse=True)

    # ── Overdue Collection Flags ───────────────────────────────────────────────
    # Condition: Data Delivery Date has passed (< today) AND Invoice Status not final
    # (i.e. invoice_status is not 'Billed' / 'Closed', e.g. 'Not billed yet' or missing)
    now_date = pd.Timestamp.now().normalize()
    deliv_col = "data_delivery_date" if "data_delivery_date" in df.columns else "probable_end_date"
    inv_col = "invoice_status" if "invoice_status" in df.columns else "billing_status"

    overdue_flags: list[dict] = []
    if deliv_col in df.columns:
        # Delivery date passed
        delivered_mask = df[deliv_col].notna() & (df[deliv_col] < now_date)
        # Invoice status not finalized
        if inv_col in df.columns:
            not_invoiced_mask = ~df[inv_col].fillna("").astype(str).str.lower().isin(["billed", "closed", "not billable"])
        else:
            not_invoiced_mask = pd.Series(True, index=df.index)

        overdue_df = df[delivered_mask & not_invoiced_mask].copy()

        for _, row in overdue_df.iterrows():
            amt = float(row.get("amount_excl_gst") or row.get("revenue") or 0)
            overdue_flags.append({
                "name": str(row.get("name", "Unknown")),
                "sector": str(row.get("sector", "Unknown")),
                "data_delivery_date": str(row[deliv_col].date()) if pd.notna(row[deliv_col]) else "N/A",
                "days_since_delivery": int((now_date - row[deliv_col]).days) if pd.notna(row[deliv_col]) else 0,
                "invoice_status": str(row.get(inv_col, "Unknown")),
                "billing_status": str(row.get("billing_status", "Unknown")),
                "amount_at_risk_fmt": _fmt_currency(amt, currency),
            })
        overdue_flags.sort(key=lambda x: x["days_since_delivery"], reverse=True)

    # Execution status counts
    status_counts: list[dict] = []
    exec_col = "execution_status" if "execution_status" in df.columns else "status"
    if exec_col in df.columns:
        for status, grp in df.groupby(exec_col, dropna=False):
            status_counts.append({
                "status": str(status) if not pd.isna(status) else "Unknown",
                "count": int(len(grp)),
                "pct": round(_safe_div(len(grp), total) * 100, 1),
            })
        status_counts.sort(key=lambda x: x["count"], reverse=True)

    return {
        "total_work_orders": total,
        "completed": int(len(completed)),
        "completion_rate_pct": round(completion_rate, 1),
        "on_time_rate_pct": round(on_time_rate, 1),
        "delayed_count": int(len(delayed)),
        "avg_delay_days": round(avg_delay, 1),
        "execution_status_breakdown": status_counts,
        "billing_status_breakdown": billing_counts,
        "amount_receivable_total": round(total_receivable, 2),
        "amount_receivable_total_fmt": _fmt_currency(total_receivable, currency),
        "receivable_by_sector": receivable_by_sector,
        "overdue_collection_flags": {
            "flagged_count": len(overdue_flags),
            "flagged_orders": overdue_flags[:15],
            "note": "Flagged where Data Delivery Date has passed but Invoice Status is not final (unbilled delivered work).",
        },
    }


# Alias for backward compatibility
wo_completion_metrics = ops_metrics


# ══════════════════════════════════════════════════════════════════════════════
# 5. CROSS-BOARD SECTOR VIEW (Prompt 6.4)
# ══════════════════════════════════════════════════════════════════════════════

def cross_board_linkage(deals_df: pd.DataFrame, wo_df: pd.DataFrame) -> dict[str, Any]:
    """
    Exact-match cross-board linkage between Deals and Work Orders on normalized name (Prompt 4).

    Rules:
    1. Join ONLY on normalized (trim + case-fold) Deal name masked == Deal Name.
    2. Exact normalized-string matches only (no fuzzy matching).
    3. If a name maps to >1 Deal or >1 Work Order, flag it as AMBIGUOUS.
    4. Ambiguous matches are EXCLUDED from cross-board numeric rollups by default.
    """
    if deals_df.empty or wo_df.empty:
        return {
            "error": "Both boards must have data for cross-board linkage.",
            "total_deals": len(deals_df),
            "total_work_orders": len(wo_df),
            "matched_1_to_1_count": 0,
            "ambiguous_match_count": 0,
            "unmatched_deals_count": len(deals_df),
            "unmatched_wo_count": len(wo_df),
            "linkage_coverage_pct": 0.0,
            "clean_linkages": [],
            "ambiguous_linkages": [],
            "numeric_rollup": {},
            "linkage_caveat": "Cross-board linkage could not be performed because one or both boards are empty.",
        }

    currency = "INR"
    if "currency" in deals_df.columns and not deals_df.empty:
        currency = deals_df["currency"].iloc[0]

    d = deals_df.copy()
    w = wo_df.copy()

    # Normalise name: trim and lowercase
    d["norm_name"] = d["name"].astype(str).str.strip().str.lower()
    w["norm_name"] = w["name"].astype(str).str.strip().str.lower()

    # Filter out empty/null names
    d = d[d["norm_name"].str.len() > 0]
    w = w[w["norm_name"].str.len() > 0]

    total_deals = len(d)
    total_wos = len(w)

    deal_name_counts = d["norm_name"].value_counts()
    wo_name_counts = w["norm_name"].value_counts()

    common_names = set(deal_name_counts.index) & set(wo_name_counts.index)

    clean_1_to_1_names = {
        name for name in common_names
        if deal_name_counts[name] == 1 and wo_name_counts[name] == 1
    }
    ambiguous_names = {
        name for name in common_names
        if deal_name_counts[name] > 1 or wo_name_counts[name] > 1
    }
    unmatched_deal_names = set(deal_name_counts.index) - common_names
    unmatched_wo_names = set(wo_name_counts.index) - common_names

    clean_deal_rows = d[d["norm_name"].isin(clean_1_to_1_names)]
    clean_wo_rows = w[w["norm_name"].isin(clean_1_to_1_names)]

    ambiguous_deal_rows = d[d["norm_name"].isin(ambiguous_names)]
    ambiguous_wo_rows = w[w["norm_name"].isin(ambiguous_names)]

    unmatched_deal_rows = d[d["norm_name"].isin(unmatched_deal_names)]
    unmatched_wo_rows = w[w["norm_name"].isin(unmatched_wo_names)]

    matched_1_to_1_count = len(clean_1_to_1_names)
    ambiguous_match_count = len(ambiguous_names)
    ambiguous_deals_count = len(ambiguous_deal_rows)
    ambiguous_wo_count = len(ambiguous_wo_rows)
    unmatched_deals_count = len(unmatched_deal_rows)
    unmatched_wo_count = len(unmatched_wo_rows)

    deal_coverage_pct = round(_safe_div(len(clean_deal_rows), total_deals) * 100, 1)
    wo_coverage_pct = round(_safe_div(len(clean_wo_rows), total_wos) * 100, 1)

    # ── Clean 1:1 Linkages ─────────────────────────────────────────────────────
    clean_linkages: list[dict] = []
    clean_deal_val_sum = 0.0
    clean_wo_val_sum = 0.0
    clean_wo_billed_sum = 0.0

    if not clean_deal_rows.empty and not clean_wo_rows.empty:
        merged_clean = pd.merge(
            clean_deal_rows,
            clean_wo_rows,
            on="norm_name",
            suffixes=("_deal", "_wo"),
        )

        for _, row in merged_clean.iterrows():
            d_val_raw = row.get("deal_value_deal", row.get("deal_value"))
            w_rev_raw = row.get("revenue_wo", row.get("amount_excl_gst", row.get("revenue")))
            w_billed_raw = row.get("billed_excl_gst_wo", row.get("billed_excl_gst"))

            d_val = float(d_val_raw) if pd.notna(d_val_raw) else 0.0
            w_rev = float(w_rev_raw) if pd.notna(w_rev_raw) else 0.0
            w_billed = float(w_billed_raw) if pd.notna(w_billed_raw) else 0.0

            clean_deal_val_sum += d_val
            clean_wo_val_sum += w_rev
            clean_wo_billed_sum += w_billed

            clean_linkages.append({
                "name": str(row.get("name_deal", row.get("name", row["norm_name"]))),
                "sector": str(row.get("sector_deal") or row.get("sector_wo") or row.get("sector") or "Unknown"),
                "deal_stage": str(row.get("deal_stage_deal") or row.get("deal_stage") or row.get("stage_deal") or row.get("stage") or "Unknown"),
                "deal_value": d_val,
                "deal_value_fmt": _fmt_currency(d_val, currency),
                "wo_execution_status": str(row.get("execution_status_wo") or row.get("execution_status") or row.get("status_wo") or row.get("status") or "Unknown"),
                "wo_billing_status": str(row.get("billing_status_wo") or row.get("billing_status") or "Unknown"),
                "wo_revenue": w_rev,
                "wo_revenue_fmt": _fmt_currency(w_rev, currency),
                "wo_billed": w_billed,
                "wo_billed_fmt": _fmt_currency(w_billed, currency),
                "variance": round(w_rev - d_val, 2),
                "variance_fmt": _fmt_currency(w_rev - d_val, currency),
            })

    # ── Ambiguous Linkages (excluded from rollup) ──────────────────────────────
    ambiguous_linkages: list[dict] = []
    for name in sorted(ambiguous_names):
        d_subset = d[d["norm_name"] == name]
        w_subset = w[w["norm_name"] == name]

        d_val = float(d_subset["deal_value"].sum() or 0)
        w_rev = float(w_subset["revenue"].sum() if "revenue" in w_subset.columns else w_subset.get("amount_excl_gst", pd.Series([0])).sum() or 0)

        ambiguous_linkages.append({
            "name": str(d_subset["name"].iloc[0] if len(d_subset) else name),
            "deal_count": int(len(d_subset)),
            "work_order_count": int(len(w_subset)),
            "deal_stages": d_subset.get("deal_stage", d_subset.get("stage", pd.Series())).dropna().unique().tolist(),
            "wo_statuses": w_subset.get("execution_status", w_subset.get("status", pd.Series())).dropna().unique().tolist(),
            "total_deal_value_fmt": _fmt_currency(d_val, currency),
            "total_wo_revenue_fmt": _fmt_currency(w_rev, currency),
            "exclusion_reason": "Ambiguous 1:N / N:M codename match — excluded from numeric rollups to avoid double-counting.",
        })

    # ── Numeric Rollup for Clean 1:1 Matches ───────────────────────────────────
    numeric_rollup = {
        "clean_matched_pairs": matched_1_to_1_count,
        "clean_total_deal_value": round(clean_deal_val_sum, 2),
        "clean_total_deal_value_fmt": _fmt_currency(clean_deal_val_sum, currency),
        "clean_total_wo_revenue": round(clean_wo_val_sum, 2),
        "clean_total_wo_revenue_fmt": _fmt_currency(clean_wo_val_sum, currency),
        "clean_total_wo_billed": round(clean_wo_billed_sum, 2),
        "clean_total_wo_billed_fmt": _fmt_currency(clean_wo_billed_sum, currency),
        "clean_variance": round(clean_wo_val_sum - clean_deal_val_sum, 2),
        "clean_variance_fmt": _fmt_currency(clean_wo_val_sum - clean_deal_val_sum, currency),
    }

    caveat = (
        f"Cross-board exact name linkage identified {matched_1_to_1_count} unambiguous 1:1 matches "
        f"({deal_coverage_pct}% deal coverage, {wo_coverage_pct}% WO coverage). "
        f"{ambiguous_match_count} codenames matched ambiguously across multiple deals/orders "
        f"({ambiguous_deals_count} deal rows, {ambiguous_wo_count} WO rows) and are excluded from numeric rollups. "
        f"{unmatched_deals_count} deals and {unmatched_wo_count} work orders remain unmatched."
    )

    return {
        "total_deals": total_deals,
        "total_work_orders": total_wos,
        "matched_1_to_1_count": matched_1_to_1_count,
        "ambiguous_match_count": ambiguous_match_count,
        "ambiguous_deals_count": ambiguous_deals_count,
        "ambiguous_wo_count": ambiguous_wo_count,
        "unmatched_deals_count": unmatched_deals_count,
        "unmatched_wo_count": unmatched_wo_count,
        "deal_coverage_pct": deal_coverage_pct,
        "wo_coverage_pct": wo_coverage_pct,
        "numeric_rollup": numeric_rollup,
        "clean_linkages": clean_linkages,
        "ambiguous_linkages": ambiguous_linkages,
        "linkage_caveat": caveat,
    }


def cross_board_health(deals_df: pd.DataFrame, wo_df: pd.DataFrame) -> dict[str, Any]:
    """
    Cross-board sector view (Prompt 6.4):
    - Open-deal pipeline vs active work-order load per sector.
    - Uses only the non-ambiguous 1:1 links from Prompt 4 for project rollups.
    """
    if deals_df.empty and wo_df.empty:
        return {"error": "Both boards returned no data."}

    currency = "INR"
    if not deals_df.empty and "currency" in deals_df.columns:
        currency = deals_df["currency"].iloc[0]

    stage_col = "deal_stage" if "deal_stage" in deals_df.columns else "stage"
    exec_col = "execution_status" if "execution_status" in wo_df.columns else "status"

    # Open deals
    open_deals = deals_df[~deals_df[stage_col].isin(["Closed Won", "Closed Lost"])] if not deals_df.empty else pd.DataFrame()
    won_deals = deals_df[deals_df[stage_col] == "Closed Won"] if not deals_df.empty else pd.DataFrame()

    # Active work orders (not completed / cancelled)
    active_wos = wo_df[~wo_df[exec_col].isin(["Completed", "Cancelled"])] if not wo_df.empty else pd.DataFrame()

    # Exact name linkage
    linkage = cross_board_linkage(deals_df, wo_df)
    clean_linked_df = pd.DataFrame(linkage.get("clean_linkages", []))

    # All unique sectors
    all_sectors = set()
    if not deals_df.empty and "sector" in deals_df.columns:
        all_sectors.update(deals_df["sector"].dropna().unique().tolist())
    if not wo_df.empty and "sector" in wo_df.columns:
        all_sectors.update(wo_df["sector"].dropna().unique().tolist())

    sector_view: list[dict] = []
    for sector in sorted(all_sectors):
        sec_open_deals = open_deals[open_deals["sector"] == sector] if not open_deals.empty and "sector" in open_deals.columns else pd.DataFrame()
        sec_won_deals = won_deals[won_deals["sector"] == sector] if not won_deals.empty and "sector" in won_deals.columns else pd.DataFrame()
        sec_active_wos = active_wos[active_wos["sector"] == sector] if not active_wos.empty and "sector" in active_wos.columns else pd.DataFrame()
        sec_total_wos = wo_df[wo_df["sector"] == sector] if not wo_df.empty and "sector" in wo_df.columns else pd.DataFrame()

        sec_open_val = float(sec_open_deals["deal_value"].sum() or 0)
        sec_won_val = float(sec_won_deals["deal_value"].sum() or 0)
        sec_active_wo_val = float(sec_active_wos["revenue"].sum() if "revenue" in sec_active_wos.columns else sec_active_wos.get("amount_excl_gst", pd.Series([0])).sum() or 0)

        # 1:1 clean linked metrics for this sector
        if not clean_linked_df.empty and "sector" in clean_linked_df.columns:
            sec_clean = clean_linked_df[clean_linked_df["sector"] == sector]
            clean_pairs = int(len(sec_clean))
            clean_deal_v = float(sec_clean["deal_value"].sum() or 0)
            clean_wo_v = float(sec_clean["wo_revenue"].sum() or 0)
        else:
            clean_pairs = 0
            clean_deal_v = 0.0
            clean_wo_v = 0.0

        sector_view.append({
            "sector": str(sector),
            "open_deal_count": int(len(sec_open_deals)),
            "open_pipeline_value_fmt": _fmt_currency(sec_open_val, currency),
            "won_deal_count": int(len(sec_won_deals)),
            "won_revenue_fmt": _fmt_currency(sec_won_val, currency),
            "active_work_order_count": int(len(sec_active_wos)),
            "active_wo_value_fmt": _fmt_currency(sec_active_wo_val, currency),
            "total_work_orders": int(len(sec_total_wos)),
            "clean_linked_pairs": clean_pairs,
            "clean_linked_deal_val_fmt": _fmt_currency(clean_deal_v, currency),
            "clean_linked_wo_rev_fmt": _fmt_currency(clean_wo_v, currency),
        })

    sector_view.sort(key=lambda x: int(x["open_deal_count"]) + int(x["active_work_order_count"]), reverse=True)

    return {
        "currency": currency,
        "total_open_pipeline_fmt": _fmt_currency(float(open_deals["deal_value"].sum() or 0) if not open_deals.empty else 0, currency),
        "total_won_revenue_fmt": _fmt_currency(float(won_deals["deal_value"].sum() or 0) if not won_deals.empty else 0, currency),
        "total_active_work_orders": int(len(active_wos)),
        "sector_cross_analysis": sector_view,
        "name_linkage": linkage,
        "linkage_caveat": linkage.get("linkage_caveat", ""),
        "data_note": "Cross-board view compares open pipeline against active work-order load per sector, using only non-ambiguous 1:1 codename linkages for deal-to-order rollups.",
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6. LEADERSHIP UPDATE (Board-Level Executive Report)
# ══════════════════════════════════════════════════════════════════════════════

def leadership_update(deals_df: pd.DataFrame, wo_df: pd.DataFrame) -> dict[str, Any]:
    """
    Compile a comprehensive KPI dict for the leadership update.
    All numbers are deterministic pandas calculations.
    """
    pipeline = pipeline_summary(deals_df) if not deals_df.empty else {"error": "No deals"}
    win_loss = win_loss_rates(deals_df) if not deals_df.empty else {"error": "No deals"}
    sectors = sector_breakdown(deals_df, wo_df=wo_df) if not deals_df.empty else {"error": "No deals"}
    ops = ops_metrics(wo_df) if not wo_df.empty else {"error": "No WOs"}
    cross = cross_board_health(deals_df, wo_df)

    stage_col = "deal_stage" if "deal_stage" in deals_df.columns else "stage"

    # Identify risks
    risks: list[str] = []

    if not deals_df.empty:
        late_stage = deals_df[deals_df[stage_col].isin(["Negotiation", "Proposal"])]
        if len(late_stage):
            currency = deals_df.get("currency", pd.Series(["INR"])).iloc[0]
            risks.append(
                f"{len(late_stage)} deals in Proposal/Negotiation stage worth "
                f"{_fmt_currency(float(late_stage['deal_value'].sum() or 0), currency)} — "
                "needs follow-up to convert."
            )
        date_upd = "updated_at" if "updated_at" in deals_df.columns else "Created Date"
        if date_upd in deals_df.columns:
            stale = deals_df[
                (deals_df[date_upd] < (pd.Timestamp.now() - pd.Timedelta(days=30)))
                & (~deals_df[stage_col].isin(["Closed Won", "Closed Lost"]))
            ]
            if len(stale):
                risks.append(f"{len(stale)} active deals haven't been updated in 30+ days.")

    if not wo_df.empty and "delay_days" in wo_df.columns:
        overdue = wo_df[(wo_df["delay_days"] > 14) & (wo_df["is_complete"] == False)]
        if len(overdue):
            risks.append(f"{len(overdue)} work orders are overdue by 2+ weeks.")

    # Overdue collections
    if isinstance(ops, dict) and "overdue_collection_flags" in ops:
        flagged_cnt = ops["overdue_collection_flags"].get("flagged_count", 0)
        if flagged_cnt > 0:
            risks.append(f"{flagged_cnt} delivered work orders have unfinalized billing/invoice status (collection risk).")

    now = datetime.now(timezone.utc)
    return {
        "generated_at": now.isoformat(),
        "period": f"As of {now.strftime('%d %b %Y')}",
        "pipeline_summary": pipeline,
        "win_loss": win_loss,
        "sector_performance": sectors,
        "operations": ops,
        "cross_board": cross,
        "risks_and_flags": risks,
    }
