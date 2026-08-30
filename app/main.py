"""
app/main.py
───────────
Streamlit conversational UI for the Skylark BI Agent (Prompt 9).

Features:
  - Full chat message history with expandable data quality badges & assumptions
  - Sidebar with Live Connected Boards status & Last Sync Time
  - Manual "🔄 Sync / Refresh Data" button to clear session cache & re-sync from Monday.com
  - Scoped Leadership Update generator (Sector / Owner / Time range)
  - Native Streamlit Cloud secrets support (st.secrets) — runs with no local .env
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── Ensure project root is in sys.path so modules (graph, tools, etc.) resolve ──
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# ── Load local .env if present (ignored on Streamlit Cloud where st.secrets is used) ──
load_dotenv(ROOT_DIR / ".env")

# ── Streamlit Cloud Secrets bridge: safely copy st.secrets → os.environ ──────────────
try:
    if hasattr(st, "secrets"):
        for _k in (
            "MONDAY_API_TOKEN", "DEALS_BOARD_ID", "WORK_ORDERS_BOARD_ID",
            "GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_MODEL", "USE_MCP",
        ):
            if _k in st.secrets and not os.getenv(_k):
                os.environ[_k] = str(st.secrets[_k])

        # Inter-operability between GOOGLE_API_KEY and GEMINI_API_KEY
        if "GOOGLE_API_KEY" in st.secrets and not os.getenv("GEMINI_API_KEY"):
            os.environ["GEMINI_API_KEY"] = str(st.secrets["GOOGLE_API_KEY"])
        elif "GEMINI_API_KEY" in st.secrets and not os.getenv("GOOGLE_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = str(st.secrets["GEMINI_API_KEY"])
except Exception:
    # On local environments without secrets.toml, st.secrets raises StreamlitSecretNotFoundError.
    # We gracefully fall back to variables already loaded by load_dotenv().
    pass

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Page config & CSS ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Skylark BI Agent",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stChatMessage"] { border-radius: 12px; margin-bottom: 8px; }
[data-testid="stSidebar"] { background: #0f1117; }
.quality-badge { display:inline-block; padding:2px 8px; border-radius:9999px;
                 font-size:0.75rem; font-weight:600; }
.q-good { background:#22c55e20; color:#22c55e; }
.q-warn { background:#f59e0b20; color:#f59e0b; }
.q-bad  { background:#ef444420; color:#ef4444; }
.board-card { background:#1e2230; border:1px solid #2d3348; border-radius:8px; padding:10px 12px; margin-bottom:8px; }
.board-title { font-size:0.85rem; font-weight:600; color:#f1f5f9; }
.board-meta { font-size:0.75rem; color:#94a3b8; margin-top:2px; }
.sync-badge { font-size:0.75rem; color:#38bdf8; font-weight:500; }
</style>
""", unsafe_allow_html=True)


# ── Initial state ──────────────────────────────────────────────────────────────
_EMPTY_AGENT_STATE: dict = {
    "messages": [],
    "query": "",
    "trigger": "query",
    "intent": "",
    "filters": {},
    "required_boards": [],
    "assumptions": [],
    "is_ambiguous": False,
    "ambiguity_reason": "",
    "raw_data": {},
    "normalized_data": {},
    "metrics": {},
    "data_quality": {},
    "final_answer": None,
    "error": None,
}


def _init_session():
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "graph" not in st.session_state:
        st.session_state.graph = None
    if "agent_state" not in st.session_state:
        st.session_state.agent_state = dict(_EMPTY_AGENT_STATE)
    if "connection_ok" not in st.session_state:
        st.session_state.connection_ok = None
    if "last_sync_time" not in st.session_state:
        st.session_state.last_sync_time = None
    if "boards_info" not in st.session_state:
        st.session_state.boards_info = {}


def _load_graph():
    if st.session_state.graph is None:
        with st.spinner("Initialising agent graph…"):
            from graph.agent import build_graph
            st.session_state.graph = build_graph()
    return st.session_state.graph


def _check_env() -> list[str]:
    required = ["MONDAY_API_TOKEN", "DEALS_BOARD_ID", "WORK_ORDERS_BOARD_ID"]
    gemini = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    missing = [k for k in required if not os.getenv(k)]
    if not gemini:
        missing.append("GEMINI_API_KEY (or GOOGLE_API_KEY)")
    return missing


def _quality_html(completeness: float | None, has_caveat: bool, warnings: list[str]) -> str:
    c = completeness or 100.0
    if has_caveat or c < 70:
        cls, label = "q-bad",  f"✗ {c:.0f}% — insufficient quality"
    elif c < 90:
        cls, label = "q-warn", f"⚠ {c:.0f}% complete"
    else:
        cls, label = "q-good", f"✓ {c:.0f}% complete"
    badge = f'<span class="quality-badge {cls}">{label}</span>'
    if warnings:
        items = "".join(f"<li>{w}</li>" for w in warnings)
        badge += f"<ul style='margin-top:6px;font-size:0.8rem;opacity:0.8'>{items}</ul>"
    return badge


# ── Sidebar ────────────────────────────────────────────────────────────────────

def _render_sidebar():
    with st.sidebar:
        st.markdown("# 🚁 Skylark BI")
        st.caption("Monday.com · LangGraph · Gemini · pandas")
        st.divider()

        # ── Connected Boards Section ───────────────────────────────────────────
        st.markdown("### 🔌 Connected Boards")

        deals_id = os.getenv("DEALS_BOARD_ID", "Not Set")
        wo_id = os.getenv("WORK_ORDERS_BOARD_ID", "Not Set")

        deals_info = st.session_state.boards_info.get("deals", {})
        wo_info = st.session_state.boards_info.get("workorders", {})

        deals_name = deals_info.get("name", "Deals / Deal Funnel")
        wo_name = wo_info.get("name", "Work Orders / Tracker")

        deals_cnt = len(st.session_state.agent_state.get("raw_data", {}).get("deals", []))
        wo_cnt = len(st.session_state.agent_state.get("raw_data", {}).get("workorders", []))

        st.markdown(f"""
        <div class="board-card">
            <div class="board-title">📊 {deals_name}</div>
            <div class="board-meta">ID: <code>{deals_id}</code> | Cached Items: <b>{deals_cnt}</b></div>
        </div>
        <div class="board-card">
            <div class="board-title">🚚 {wo_name}</div>
            <div class="board-meta">ID: <code>{wo_id}</code> | Cached Items: <b>{wo_cnt}</b></div>
        </div>
        """, unsafe_allow_html=True)

        # Last sync status
        sync_t = st.session_state.last_sync_time
        if sync_t:
            st.markdown(f"<div class='sync-badge'>🕒 Last Sync: {sync_t}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='sync-badge'>🕒 Last Sync: <i>Not synced yet</i></div>", unsafe_allow_html=True)

        # Connection & Sync Buttons
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔌 Test API", use_container_width=True):
                _test_connection()
        with c2:
            if st.button("🔄 Sync Data", use_container_width=True):
                _sync_data()

        st.divider()

        # ── Leadership Update Section ──────────────────────────────────────────
        st.markdown("### 📋 Leadership Update")
        with st.expander("⚙️ Scope Update (Optional)", expanded=False):
            scope_sector = st.selectbox(
                "Sector filter",
                options=["All Sectors", "Energy", "Mining", "Renewables", "Railways", "Powerline", "Construction", "Defence", "Tender", "DSP", "Aviation", "Manufacturing"],
                index=0,
                key="scope_sector",
            )
            scope_owner = st.text_input("Owner / Rep code", placeholder="e.g. USER_001", key="scope_owner")
            scope_quarter = st.selectbox(
                "Time window",
                options=["All Time (Current Active)", "Current Quarter (Q3 2026)", "Last 30 Days"],
                index=0,
                key="scope_quarter",
            )

        if st.button(
            "📊 Generate Leadership Update",
            use_container_width=True,
            type="primary",
            key="btn_leadership",
        ):
            query_parts = ["Generate a leadership update with current KPIs, operations, and risks."]
            sec_val = st.session_state.get("scope_sector")
            if sec_val and sec_val != "All Sectors":
                query_parts.append(f"Scope specifically to sector: {sec_val}.")
            own_val = st.session_state.get("scope_owner")
            if own_val and own_val.strip():
                query_parts.append(f"Scope to owner: {own_val.strip()}.")
            q_val = st.session_state.get("scope_quarter")
            if q_val and "Current Quarter" in q_val:
                query_parts.append("Filter to current quarter.")
            elif q_val and "Last 30 Days" in q_val:
                query_parts.append("Filter to last 30 days.")

            full_query = " ".join(query_parts)
            _run_query(
                question=full_query,
                trigger="leadership",
            )
            st.rerun()

        st.divider()

        # ── Sample Questions ───────────────────────────────────────────────────
        st.markdown("### 💡 Sample Questions")
        _samples = [
            "How's our pipeline looking this quarter?",
            "What's the energy sector pipeline value?",
            "Show me win rates by deal stage",
            "Which work orders are delayed?",
            "Compare deals won vs WOs completed by sector",
            "What's our overall WO completion rate and amount receivable?",
        ]
        for q in _samples:
            if st.button(q, use_container_width=True, key=f"sample_{hash(q)}"):
                _run_query(q, trigger="query")
                st.rerun()

        st.divider()

        if st.button("🗑 Clear Conversation", use_container_width=True):
            st.session_state.chat_messages = []
            for k in ("intent", "filters", "metrics", "data_quality",
                      "final_answer", "assumptions", "error"):
                st.session_state.agent_state[k] = _EMPTY_AGENT_STATE[k]
            st.rerun()

        st.divider()
        st.caption(
            "Monday.com is the sole runtime source of truth.\n"
            "All calculations are deterministic pandas.\n"
            "LLM performs reasoning & narrative only."
        )


def _test_connection():
    with st.spinner("Testing Monday.com connection…"):
        try:
            from tools.monday import get_board_schema
            deals_id = os.getenv("DEALS_BOARD_ID")
            wo_id = os.getenv("WORK_ORDERS_BOARD_ID")
            d = get_board_schema(deals_id)
            w = get_board_schema(wo_id)
            st.session_state.connection_ok = True
            st.session_state.boards_info["deals"] = d
            st.session_state.boards_info["workorders"] = w
            st.sidebar.success(
                f"✓ Deals: **{d['name']}** ({len(d['columns'])} cols)\n"
                f"✓ WOs:   **{w['name']}** ({len(w['columns'])} cols)"
            )
        except Exception as exc:
            st.session_state.connection_ok = False
            st.sidebar.error(f"Connection failed: {exc}")


def _sync_data():
    with st.spinner("Syncing latest live data from Monday.com…"):
        try:
            from tools.monday import get_deals, get_work_orders, get_board_schema
            deals_id = os.getenv("DEALS_BOARD_ID")
            wo_id = os.getenv("WORK_ORDERS_BOARD_ID")

            d_schema = get_board_schema(deals_id)
            w_schema = get_board_schema(wo_id)
            st.session_state.boards_info["deals"] = d_schema
            st.session_state.boards_info["workorders"] = w_schema

            deals_data = get_deals(deals_id)
            wo_data = get_work_orders(wo_id)

            # Store in agent state
            st.session_state.agent_state["raw_data"] = {
                "deals": deals_data,
                "workorders": wo_data,
            }
            # Clear normalized data to force re-normalization
            st.session_state.agent_state["normalized_data"] = {}

            now = datetime.now()
            st.session_state.last_sync_time = now.strftime("%H:%M:%S, %d %b %Y")
            st.session_state.connection_ok = True
            st.sidebar.success(f"✓ Synced {len(deals_data)} deals and {len(wo_data)} work orders.")
            st.rerun()
        except Exception as exc:
            st.sidebar.error(f"Sync failed: {exc}")


# ── Chat area ──────────────────────────────────────────────────────────────────

def _render_chat():
    st.markdown("## 🚁 Skylark BI Agent")
    st.caption(
        "Ask founder-level business questions. "
        "Data is pulled live from Monday.com and calculated deterministically with pandas."
    )

    missing = _check_env()
    if missing:
        st.error(
            f"⚠️ Missing configuration: **{', '.join(missing)}**\n\n"
            "Set these in `.env` (local) or Streamlit Cloud **App Settings → Secrets**.",
            icon="🔑",
        )
        with st.expander("Setup instructions"):
            st.code(
                "# Streamlit Cloud Secrets format:\n"
                'MONDAY_API_TOKEN = "your_token_here"\n'
                'DEALS_BOARD_ID = "5030966166"\n'
                'WORK_ORDERS_BOARD_ID = "5030966117"\n'
                'GEMINI_API_KEY = "your_gemini_api_key"\n',
                language="toml",
            )
        return

    # Render history
    for msg in st.session_state.chat_messages:
        avatar = "👤" if msg["role"] == "user" else "🚁"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg.get("quality"):
                q = msg["quality"]
                with st.expander("📊 Data Quality", expanded=False):
                    st.markdown(
                        _quality_html(
                            q.get("overall_completeness"),
                            q.get("has_caveat", False),
                            q.get("warnings", []),
                        ),
                        unsafe_allow_html=True,
                    )
            if msg.get("assumptions"):
                with st.expander("📝 Assumptions", expanded=False):
                    for a in msg["assumptions"]:
                        st.markdown(f"- {a}")

    # Chat input (handles /leadership-update slash command or standard questions)
    if prompt := st.chat_input("Ask a business question or type /leadership-update…"):
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        st.session_state.chat_messages.append({"role": "user", "content": prompt})

        p_lower = prompt.strip().lower()
        if p_lower.startswith("/leadership") or "prepare this week's update" in p_lower or "weekly update for the founders" in p_lower:
            _run_query(prompt, trigger="leadership")
        else:
            _run_query(prompt, trigger="query")


# ── Agent runner ───────────────────────────────────────────────────────────────

_STEP_LABELS = {
    "query_understanding":  "🧠 Understanding your question…",
    "ask_user":             "💬 Preparing clarification…",
    "leadership_entry":     "📋 Preparing leadership update…",
    "data_retrieval":       "📡 Fetching from Monday.com…",
    "normalization":        "🧹 Cleaning and validating data…",
    "analysis":             "🔢 Running pandas calculations…",
    "insight_generation":   "✍️  Generating insights with Gemini…",
}


def _run_query(question: str, trigger: str = "query"):
    """Build agent input state and stream the graph, then append the response."""
    graph = _load_graph()
    agent_state = st.session_state.agent_state

    # Build input — carry cached raw/normalised data forward
    input_state = {
        **_EMPTY_AGENT_STATE,
        "messages":        [HumanMessage(content=question)],
        "query":           question,
        "trigger":         trigger,
        "raw_data":        agent_state.get("raw_data") or {},
        "normalized_data": agent_state.get("normalized_data") or {},
    }

    with st.chat_message("assistant", avatar="🚁"):
        step_ph  = st.empty()
        answer_ph = st.empty()

        try:
            final_state: dict = {}

            for event in graph.stream(input_state, stream_mode="updates"):
                for node_name, node_output in event.items():
                    step_ph.markdown(
                        f'<div style="opacity:0.55;font-size:0.82rem">'
                        f'{_STEP_LABELS.get(node_name, f"⚙️ {node_name}…")}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    if isinstance(node_output, dict):
                        for k in ("raw_data", "normalized_data", "intent", "filters",
                                  "required_boards", "metrics", "data_quality",
                                  "assumptions", "error"):
                            if node_output.get(k) is not None:
                                agent_state[k] = node_output[k]
                        final_state.update(node_output)

            step_ph.empty()

            # Record sync time if data was retrieved in this turn
            if "data_retrieval" in str(final_state) or agent_state.get("raw_data"):
                if not st.session_state.last_sync_time:
                    st.session_state.last_sync_time = datetime.now().strftime("%H:%M:%S, %d %b %Y")

            answer = (
                final_state.get("final_answer")
                or agent_state.get("final_answer")
                or "I couldn't generate a response. Please try again."
            )
            answer_ph.markdown(answer)

            quality = agent_state.get("data_quality") or {}
            assumptions = agent_state.get("assumptions") or []

            if quality.get("warnings") or quality.get("has_caveat"):
                with st.expander("📊 Data Quality", expanded=False):
                    st.markdown(
                        _quality_html(
                            quality.get("overall_completeness"),
                            quality.get("has_caveat", False),
                            quality.get("warnings", []),
                        ),
                        unsafe_allow_html=True,
                    )

            if assumptions:
                with st.expander("📝 Assumptions", expanded=False):
                    for a in assumptions:
                        st.markdown(f"- {a}")

            st.session_state.chat_messages.append({
                "role":        "assistant",
                "content":     answer,
                "_for":        question,
                "quality":     quality or None,
                "assumptions": assumptions or None,
            })

        except Exception as exc:
            step_ph.empty()
            logger.exception("Agent stream failed: %s", exc)
            err = (
                f"⚠️ Something went wrong:\n\n```\n{exc}\n```\n\n"
                "Check your credentials and try again."
            )
            answer_ph.markdown(err)
            st.session_state.chat_messages.append({
                "role": "assistant", "content": err, "_for": question,
            })


# ── Entry ──────────────────────────────────────────────────────────────────────

def main():
    _init_session()
    _render_sidebar()
    _render_chat()


if __name__ == "__main__":
    main()
