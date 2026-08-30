"""
graph/state.py
──────────────
LangGraph AgentState — canonical schema for the Skylark BI Agent.

Field contract
──────────────
  query           Raw user question (str)
  intent          Classified intent: "deals" | "workorders" | "cross" | "leadership" | "general"
  filters         Extracted query filters (sector, date_range, stage, status, owner)
  required_boards Which Monday.com boards to fetch: ["deals"] | ["workorders"] | ["deals","workorders"]
  raw_data        Raw items from Monday.com {"deals": [...], "workorders": [...]}
  normalized_data Cleaned pandas records {"deals": [...], "workorders": [...]}
  metrics         Computed BI metrics dict (all pandas, no LLM math)
  data_quality    Quality report {overall_completeness, warnings, has_caveat, per_board}
  assumptions     List of assumptions the agent made while answering
  final_answer    The finished answer string shown to the user

Control fields (internal)
  messages        LangGraph message list (conversation history)
  trigger         Entry-point signal: "query" | "leadership"
  is_ambiguous    True if the question needs clarification before proceeding
  ambiguity_reason Why the question is ambiguous (used in ask_user node)
  error           Last error message, if any
"""

from __future__ import annotations

from typing import Annotated, Any

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class Filters(TypedDict, total=False):
    """Structured filters extracted from the user's question."""
    sector: str | None          # e.g. "Energy", "Agriculture"
    date_range: str | None      # e.g. "Q3 2024", "this quarter", "last 30 days"
    stage: str | None           # e.g. "Proposal", "Closed Won"
    status: str | None          # e.g. "In Progress", "Done"
    owner: str | None           # e.g. "Alice"


class DataQuality(TypedDict, total=False):
    """Data quality summary returned by the normalization node."""
    overall_completeness: float     # 0–100 %
    has_caveat: bool                # True if quality is too low for reliable analysis
    warnings: list[str]             # Human-readable warning messages
    per_board: dict[str, dict]      # Per-board quality details


class AgentState(TypedDict):
    # ── Conversation history ───────────────────────────────────────────────────
    messages: Annotated[list, add_messages]

    # ── Input / routing ────────────────────────────────────────────────────────
    query: str                      # the user's raw question
    trigger: str                    # "query" | "leadership" (controls entry point)

    # ── Query understanding ────────────────────────────────────────────────────
    intent: str                     # "deals" | "workorders" | "cross" | "leadership" | "general"
    filters: Filters                # extracted query filters
    required_boards: list[str]      # ["deals"] | ["workorders"] | ["deals", "workorders"]
    assumptions: list[str]          # agent assumptions (populated during understanding + analysis)

    # ── Clarification ──────────────────────────────────────────────────────────
    is_ambiguous: bool              # True → route to ask_user, False → route to data_retrieval
    ambiguity_reason: str           # reason surfaced to the user

    # ── Data layer ─────────────────────────────────────────────────────────────
    raw_data: dict[str, list]       # {"deals": [...], "workorders": [...]}
    normalized_data: dict[str, list]# {"deals": [...records...], "workorders": [...records...]}

    # ── Analysis layer ─────────────────────────────────────────────────────────
    metrics: dict[str, Any]         # pandas-computed BI metrics

    # ── Quality layer ──────────────────────────────────────────────────────────
    data_quality: DataQuality       # quality summary from normalization node

    # ── Output ────────────────────────────────────────────────────────────────
    final_answer: str | None        # finished response shown to user

    # ── Error handling ─────────────────────────────────────────────────────────
    error: str | None
