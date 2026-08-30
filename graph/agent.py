"""
graph/agent.py
──────────────
LangGraph orchestrator for the Skylark BI Agent (Prompt 7 & Prompt 8).

Graph architecture
──────────────────
  ┌─────────────────────────────────────────────────────────────────┐
  │                           [START]                               │
  │                              │                                  │
  │                     _route_entry                                │
  │                   ├── leadership ──► leadership_entry           │
  │                   └── query ───────► query_understanding        │
  │                                            │                    │
  │                                 _route_clarification            │
  │                   ├── ambiguous ──► ask_user ──► END            │
  │                   └── clear ──────► data_retrieval ◄── START   │
  │                                          │         (leadership) │
  │                                    normalization                │
  │                                          │                      │
  │                             [Data Quality Check]               │
  │                                   ├── insufficient ──┐          │
  │                                   └── sufficient  ───┤          │
  │                                                       ▼         │
  │                                                   analysis      │
  │                                                       │         │
  │                                             insight_generation  │
  │                                                       │         │
  │                                                      END        │
  └─────────────────────────────────────────────────────────────────┘

Node responsibilities
─────────────────────
  query_understanding  Gemini: extract intent, filters, required_boards, ambiguity
  ask_user             Gemini: craft clarifying question → final_answer → END
  leadership_entry     Preset intent="leadership", required_boards=["deals", "workorders"]
  data_retrieval       Monday.com GraphQL: fetch raw_data based on required_boards (with partial board recovery)
  normalization        pandas: clean → normalized_data + data_quality + has_caveat
  analysis             pandas: compute metrics (deterministic BI calculations, no LLM math)
  insight_generation   Gemini: narrative from metrics + quality + assumptions → final_answer
"""

from __future__ import annotations

import logging
import os
from typing import Any

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

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
from graph.state import AgentState
from llm.gemini import GeminiClient
from normalize.clean import clean_deals, clean_work_orders
from tools.error_handler import timed_execution
from tools.monday import MondayClient

logger = logging.getLogger(__name__)

# ── Singletons ──────────────────────────────────────────────────────────────────
_monday: MondayClient | None = None
_gemini: GeminiClient | None = None


def _get_monday() -> MondayClient:
    global _monday
    if _monday is None:
        _monday = MondayClient()
    return _monday


def _get_gemini() -> GeminiClient:
    global _gemini
    if _gemini is None:
        _gemini = GeminiClient()
    return _gemini


# ══════════════════════════════════════════════════════════════════════════════
# NODE 1 — Query Understanding
# ══════════════════════════════════════════════════════════════════════════════

@timed_execution("graph.query_understanding")
def query_understanding(state: AgentState) -> dict[str, Any]:
    """
    Parse the user's question and populate:
      intent, filters, required_boards, is_ambiguous, ambiguity_reason, assumptions
    """
    query = state.get("query", "")
    if not query:
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, HumanMessage):
                query = msg.content
                break

    try:
        result = _get_gemini().understand_query(query)
        return {
            "query": query,
            "intent": result.get("intent", "general"),
            "filters": result.get("filters", {}),
            "required_boards": result.get("required_boards", []),
            "is_ambiguous": result.get("is_ambiguous", False),
            "ambiguity_reason": result.get("ambiguity_reason", ""),
            "assumptions": result.get("assumptions", []),
            "error": None,
        }
    except Exception as exc:
        logger.exception("query_understanding failed: %s", exc)
        return {
            "query": query,
            "intent": "general",
            "filters": {},
            "required_boards": [],
            "is_ambiguous": False,
            "ambiguity_reason": "",
            "assumptions": [f"Query understanding failed ({exc}); proceeding with general analysis."],
            "error": str(exc),
        }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 2 — Ask User (Clarification)
# ══════════════════════════════════════════════════════════════════════════════

@timed_execution("graph.ask_user")
def ask_user(state: AgentState) -> dict[str, Any]:
    """
    Generate a single focused clarifying question and return it as final_answer.
    The graph routes to END after this node.
    """
    try:
        question = _get_gemini().ask_clarification(
            state.get("query", ""),
            state.get("ambiguity_reason", "The question is ambiguous."),
        )
    except Exception as exc:
        question = (
            "Could you clarify what you're looking for? "
            "For example: which sector, time period, or metric are you interested in?"
        )
        logger.warning("ask_user fallback triggered: %s", exc)

    return {
        "final_answer": question,
        "messages": [AIMessage(content=question)],
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 3 — Data Retrieval (with Partial Board Recovery)
# ══════════════════════════════════════════════════════════════════════════════

@timed_execution("graph.data_retrieval")
def data_retrieval(state: AgentState) -> dict[str, Any]:
    """
    Fetch raw items from Monday.com based on required_boards.
    Handles partial board failures: if one board is unreachable while the other
    succeeds, proceeds with available data and documents the gap in assumptions.
    """
    required = state.get("required_boards", [])
    existing_raw = state.get("raw_data") or {}
    raw: dict[str, list] = dict(existing_raw)
    new_assumptions: list[str] = list(state.get("assumptions", []))
    fetch_errors: list[str] = []

    deals_id = os.getenv("DEALS_BOARD_ID")
    wo_id = os.getenv("WORK_ORDERS_BOARD_ID")

    # ── Fetch Deals ────────────────────────────────────────────────────────────
    if "deals" in required and not raw.get("deals"):
        try:
            logger.info("Fetching deals (board %s)...", deals_id)
            raw["deals"] = _get_monday().get_deals(deals_id)
            logger.info("Fetched %d deal items.", len(raw["deals"]))
        except Exception as exc:
            logger.exception("Deals board fetch failed: %s", exc)
            fetch_errors.append(f"Deals board ({deals_id}) fetch failed: {exc}")
            new_assumptions.append(f"Deals board was unreachable ({exc}). Sales pipeline data is unavailable.")

    # ── Fetch Work Orders ──────────────────────────────────────────────────────
    if "workorders" in required and not raw.get("workorders"):
        try:
            logger.info("Fetching work orders (board %s)...", wo_id)
            raw["workorders"] = _get_monday().get_work_orders(wo_id)
            logger.info("Fetched %d WO items.", len(raw["workorders"]))
        except Exception as exc:
            logger.exception("Work orders board fetch failed: %s", exc)
            fetch_errors.append(f"Work Orders board ({wo_id}) fetch failed: {exc}")
            new_assumptions.append(f"Work Orders board was unreachable ({exc}). Project execution data is unavailable.")

    # Check if all required boards failed
    if required and all(b not in raw or not raw[b] for b in required):
        err_msg = "; ".join(fetch_errors) or "Failed to retrieve required board data from Monday.com."
        return {
            "raw_data": raw,
            "error": err_msg,
            "assumptions": new_assumptions,
        }

    err_state = "; ".join(fetch_errors) if fetch_errors else None
    return {"raw_data": raw, "assumptions": new_assumptions, "error": err_state}


# ══════════════════════════════════════════════════════════════════════════════
# NODE 4 — Normalization / Data Quality
# ══════════════════════════════════════════════════════════════════════════════

_QUALITY_THRESHOLD = 65.0   # % overall completeness
_WARNING_THRESHOLD = 4      # number of warnings that triggers caveat flag


@timed_execution("graph.normalization")
def normalization(state: AgentState) -> dict[str, Any]:
    """
    Clean raw_data with pandas → normalized_data.
    Compute data_quality report.
    Set has_caveat = True if quality is below threshold.
    """
    raw = state.get("raw_data") or {}
    normalized: dict[str, list] = {}
    per_board: dict[str, dict] = {}
    all_warnings: list[str] = []

    try:
        if "deals" in raw and raw["deals"]:
            df, q = clean_deals(raw["deals"])
            normalized["deals"] = df.to_dict(orient="records") if not df.empty else []
            per_board["deals"] = q
            all_warnings.extend(q.get("warnings", []))

        if "workorders" in raw and raw["workorders"]:
            df, q = clean_work_orders(raw["workorders"])
            normalized["workorders"] = df.to_dict(orient="records") if not df.empty else []
            per_board["workorders"] = q
            all_warnings.extend(q.get("warnings", []))

        completeness_values = [
            v.get("overall_completeness", 100.0)
            for v in per_board.values()
        ]
        overall = (
            round(sum(completeness_values) / len(completeness_values), 1)
            if completeness_values else 100.0
        )

        has_caveat = (
            overall < _QUALITY_THRESHOLD or len(all_warnings) >= _WARNING_THRESHOLD
        )

        quality: dict[str, Any] = {
            "overall_completeness": overall,
            "has_caveat": has_caveat,
            "warnings": all_warnings,
            "per_board": per_board,
        }

        new_assumptions: list[str] = list(state.get("assumptions", []))
        if has_caveat:
            new_assumptions.append(
                f"Data quality is below threshold ({overall:.0f}% complete). "
                "Results should be treated as directional estimates."
            )

        logger.info(
            "Normalization complete: %d boards, %.1f%% complete, caveat=%s.",
            len(per_board), overall, has_caveat,
        )

        return {
            "normalized_data": normalized,
            "data_quality": quality,
            "assumptions": new_assumptions,
            "error": state.get("error"),
        }

    except Exception as exc:
        logger.exception("normalization failed: %s", exc)
        quality = {
            "overall_completeness": 0.0,
            "has_caveat": True,
            "warnings": [f"Normalization error: {exc}"],
            "per_board": {},
        }
        return {
            "normalized_data": {},
            "data_quality": quality,
            "error": str(exc),
        }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 5 — Analysis (pandas, deterministic BI)
# ══════════════════════════════════════════════════════════════════════════════

@timed_execution("graph.analysis")
def analysis(state: AgentState) -> dict[str, Any]:
    """
    Run deterministic pandas BI calculations based on intent and filters.
    """
    intent = state.get("intent", "general")
    filters = state.get("filters") or {}
    normalized = state.get("normalized_data") or {}
    new_assumptions: list[str] = list(state.get("assumptions", []))

    deals_df = _records_to_df(normalized.get("deals", []))
    orders_df = _records_to_df(normalized.get("workorders", []))

    try:
        metrics: dict[str, Any] = {}

        # Sector filter
        sector = filters.get("sector")
        if sector and not deals_df.empty:
            deals_df = _apply_sector_filter(deals_df, sector)
            new_assumptions.append(f"Filtered deal data to sector: {sector}.")
        if sector and not orders_df.empty:
            orders_df = _apply_sector_filter(orders_df, sector)

        # Stage / status filter
        if filters.get("stage") and not deals_df.empty:
            stg_col = "deal_stage" if "deal_stage" in deals_df.columns else "stage"
            deals_df = deals_df[
                deals_df.get(stg_col, pd.Series(dtype=str))
                .str.lower()
                .eq(filters["stage"].lower())
            ]
            new_assumptions.append(f"Filtered to deal stage: {filters['stage']}.")

        if filters.get("status") and not orders_df.empty:
            stat_col = "execution_status" if "execution_status" in orders_df.columns else "status"
            orders_df = orders_df[
                orders_df.get(stat_col, pd.Series(dtype=str))
                .str.lower()
                .eq(filters["status"].lower())
            ]
            new_assumptions.append(f"Filtered to WO status: {filters['status']}.")

        # Owner filter
        owner_val = filters.get("owner")
        if owner_val and not deals_df.empty:
            own_col = "owner_code" if "owner_code" in deals_df.columns else "owner"
            if own_col in deals_df.columns:
                deals_df = deals_df[
                    deals_df[own_col].astype(str).str.lower().str.contains(
                        owner_val.lower(), na=False
                    )
                ]
                new_assumptions.append(f"Filtered to owner: {owner_val}.")

        # Compute metrics
        if intent in ("deals", "cross", "leadership", "general"):
            if not deals_df.empty:
                metrics["pipeline"] = pipeline_summary(deals_df)
                metrics["win_loss"] = win_loss_rates(deals_df)
                metrics["sector_analysis"] = sector_breakdown(
                    deals_df,
                    wo_df=orders_df if not orders_df.empty else None,
                    filter_sector=sector,
                )

        if intent in ("workorders", "cross", "leadership", "general"):
            if not orders_df.empty:
                metrics["operations"] = ops_metrics(orders_df)

        if intent in ("cross", "leadership"):
            metrics["cross_board"] = cross_board_health(deals_df, orders_df)

        if intent == "leadership":
            metrics["leadership_kpis"] = leadership_update(deals_df, orders_df)

        if not metrics:
            metrics["note"] = "No specific BI metrics could be computed for this query."
            new_assumptions.append("Insufficient data available to compute specific metrics.")

        return {"metrics": metrics, "assumptions": new_assumptions, "error": state.get("error")}

    except Exception as exc:
        logger.exception("analysis node failed: %s", exc)
        return {
            "metrics": {"error": str(exc)},
            "assumptions": new_assumptions + [f"Analysis encountered an error: {exc}"],
            "error": str(exc),
        }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 6 — Insight Generation (Gemini)
# ══════════════════════════════════════════════════════════════════════════════

@timed_execution("graph.insight_generation")
def insight_generation(state: AgentState) -> dict[str, Any]:
    """
    Translate pre-computed metrics into a human narrative via Gemini.
    If intent == "leadership", call generate_leadership_update instead.
    Never invents or modifies figures.
    """
    intent = state.get("intent", "general")
    query = state.get("query", "")
    metrics = state.get("metrics", {})
    quality = state.get("data_quality") or {}
    assumptions = state.get("assumptions", [])

    try:
        if intent == "leadership":
            kpis = metrics.get("leadership_kpis", metrics)
            answer = _get_gemini().generate_leadership_update(
                kpis,
                quality=quality,
                assumptions=assumptions,
            )
        else:
            answer = _get_gemini().explain_results(
                question=query,
                metrics=metrics,
                quality=quality,
                assumptions=assumptions,
            )

        if state.get("error"):
            answer = (
                f"⚠️ *Note: I encountered an issue fetching or processing some data "
                f"(`{state['error']}`). The following analysis may be incomplete.*\n\n"
            ) + answer

        return {
            "final_answer": answer,
            "messages": [AIMessage(content=answer)],
        }

    except Exception as exc:
        logger.exception("insight_generation failed: %s", exc)
        fallback = (
            "⚠️ I computed the metrics but couldn't generate a narrative response.\n\n"
            f"Error: `{exc}`\n\n"
            "Please try again or rephrase your question."
        )
        return {
            "final_answer": fallback,
            "messages": [AIMessage(content=fallback)],
        }


# ══════════════════════════════════════════════════════════════════════════════
# LEADERSHIP ENTRY — Parallel Entry Point (Prompt 7)
# ══════════════════════════════════════════════════════════════════════════════

@timed_execution("graph.leadership_entry")
def leadership_entry(state: AgentState) -> dict[str, Any]:
    """
    Parallel entry node for leadership updates.
    Pre-sets intent="leadership", required_boards=["deals", "workorders"],
    bypassing query_understanding and clarification_check to route straight
    to data_retrieval.
    """
    query = state.get("query") or "Generate a leadership update with current KPIs and risks."
    return {
        "intent": "leadership",
        "required_boards": ["deals", "workorders"],
        "filters": state.get("filters") or {},
        "is_ambiguous": False,
        "ambiguity_reason": "",
        "assumptions": ["Leadership update path active — fetching both Deals and Work Orders boards."],
        "query": query,
        "error": None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# CONDITIONAL EDGE ROUTERS
# ══════════════════════════════════════════════════════════════════════════════

def _route_entry(state: AgentState) -> str:
    """START conditional edge: leadership vs query path."""
    trigger = state.get("trigger", "query")
    if trigger == "leadership":
        return "leadership"
    return "query"


def _route_clarification(state: AgentState) -> str:
    """query_understanding → Clarification Check conditional edge."""
    if state.get("is_ambiguous"):
        return "ambiguous"
    return "clear"


def _route_data_quality(state: AgentState) -> str:
    """normalization → Data Quality Check conditional edge."""
    quality = state.get("data_quality") or {}
    if quality.get("has_caveat", False):
        return "insufficient"
    return "sufficient"


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _records_to_df(records: list[dict]) -> pd.DataFrame:
    """Convert list-of-records back to a pandas DataFrame with datetime columns."""
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    for col in df.columns:
        if "date" in col.lower() or col in ("created_at", "updated_at"):
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _apply_sector_filter(df: pd.DataFrame, sector: str) -> pd.DataFrame:
    """Case-insensitive sector filter that falls back to full df on no match."""
    if "sector" not in df.columns:
        return df
    mask = df["sector"].str.lower() == sector.lower()
    filtered = df[mask]
    if filtered.empty:
        logger.warning("Sector filter '%s' matched 0 rows — returning unfiltered.", sector)
        return df
    return filtered


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_graph():
    """
    Construct and compile the LangGraph StateGraph.
    """
    builder = StateGraph(AgentState)

    # Register Nodes
    builder.add_node("query_understanding", query_understanding)
    builder.add_node("ask_user",            ask_user)
    builder.add_node("leadership_entry",    leadership_entry)
    builder.add_node("data_retrieval",      data_retrieval)
    builder.add_node("normalization",       normalization)
    builder.add_node("analysis",            analysis)
    builder.add_node("insight_generation",  insight_generation)

    # Entry Router (Parallel Entry Point)
    builder.add_conditional_edges(
        START,
        _route_entry,
        {
            "query":      "query_understanding",
            "leadership": "leadership_entry",
        },
    )

    # Leadership Entry -> Data Retrieval
    builder.add_edge("leadership_entry", "data_retrieval")

    # Clarification Check
    builder.add_conditional_edges(
        "query_understanding",
        _route_clarification,
        {
            "ambiguous": "ask_user",
            "clear":     "data_retrieval",
        },
    )

    # Ask User -> END
    builder.add_edge("ask_user", END)

    # Data Retrieval -> Normalization
    builder.add_edge("data_retrieval", "normalization")

    # Data Quality Check
    builder.add_conditional_edges(
        "normalization",
        _route_data_quality,
        {
            "insufficient": "analysis",
            "sufficient":   "analysis",
        },
    )

    # Analysis -> Insight Generation -> END
    builder.add_edge("analysis", "insight_generation")
    builder.add_edge("insight_generation", END)

    return builder.compile()
