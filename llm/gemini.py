"""
llm/gemini.py
─────────────
Gemini client wrapper for the Skylark BI Agent (Prompt 7 & Prompt 8).

Public API
──────────
  understand_query(question)          → {intent, filters, required_boards,
                                          is_ambiguous, ambiguity_reason, assumptions}
  ask_clarification(question, reason) → clarifying question string
  explain_results(question, metrics,  → narrative answer string
                  quality, assumptions)
  generate_leadership_update(kpis,    → structured markdown report string
                             quality, assumptions)
  answer_general(question)            → general answer string

The LLM never does math. It receives pre-calculated dicts and writes prose.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from tools.error_handler import (
    GeminiAuthError,
    GeminiNetworkError,
    GeminiRateLimitError,
    timed_execution,
)

load_dotenv()

logger = logging.getLogger(__name__)

# ── System prompt ──────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are Skylark BI, an expert Business Intelligence analyst for Skylark Drones \
— a drone-services company operating across Energy, Agriculture, Industrial, \
Mining, Defence, Government, Construction, Telecom, Logistics, and Forestry sectors.

You help founders and executives get fast, accurate answers about the business. \
You have access to two live data sources pulled from Monday.com:
  1. Deal Funnel   — sales pipeline: stages, values, sectors, close dates, owners, closure probability.
  2. Work Orders   — project execution: status, timelines, revenue, delays, billing status, amount receivable.

Non-negotiable rules:
- Every number you reference was computed by deterministic pandas code. \
  DO NOT recalculate, estimate, or modify any number — only narrate them.
- Clearly surface data-quality warnings, null-probability disclosures, and assumptions in your response.
- Be concise and executive-friendly. Use bullet points and clear KPIs.
- If a question is genuinely ambiguous, ask ONE focused clarifying question.
- Never fabricate data. If a metric is unavailable, say so explicitly.
- Cross-board linkages: only reference 1:1 clean project rollups and always disclose ambiguous matches as caveats.
- Leadership updates: headline → pipeline snapshot → ops/delivery snapshot → sector highlights → risks & recommendations → data caveats.\
"""

# ── Query Understanding prompt ─────────────────────────────────────────────────

_UNDERSTAND_PROMPT = """\
Analyze this business question for Skylark Drones and extract structured information.

System Context:
- Current Date: {current_date} (Current Quarter: {current_quarter})
- Deal Funnel board fields:
  * Name (codename)
  * Masked Deal value (numeric pipeline value)
  * Deal Stage (A. Lead Generated -> Lead, B. Sales Qualified Leads -> Qualified, C. Demo Done -> Demo, E. Proposal/Commercials Sent -> Proposal, F. Negotiations -> Negotiation, G. Project Won / H. Work Order Received -> Closed Won, L. Project Lost -> Closed Lost, M. Projects On Hold -> On Hold)
  * Deal Status (Open, Won, Dead, On Hold)
  * Sector/service (Renewables, Mining, Powerline/Energy, Railways, Construction, DSP, Tender, Manufacturing, Defence, Aviation, Others)
  * Tentative Close Date, Close Date (A), Created Date
  * Owner code, Client Code
- Work Order Tracker board fields:
  * Name (Deal name masked)
  * Execution Status (Completed, Ongoing [includes 'Executed until current month'], Paused [includes 'Pause / struck'], Partial [includes 'Partial Completed'], Not Started, Pending)
  * Billing Status (Billed [handles 'BIlled' typo], Partially Billed, Not Billable, Update Required, Stuck)
  * WO Status (billed) (Open, Closed)
  * Sector (Renewables, Mining, Powerline/Energy, Railways, Construction, etc.)
  * Amount in Rupees (Excl of GST) (Masked) [Contract / order amount]
  * Billed Value in Rupees (Excl of GST.) (Masked) [Invoiced amount]
  * Collected Amount in Rupees (Incl of GST.) (Masked) [Realized cash collection]
  * Probable Start Date, Probable End Date, Data Delivery Date

Question: {question}

Return ONLY a valid JSON object — no markdown fences, no commentary:
{{
  "intent": "<deals | workorders | cross | leadership | general>",
  "filters": {{
    "sector":     "<canonical sector name (e.g. 'Energy', 'Mining', 'Renewables', 'Railways') or null>",
    "date_range": "<resolved time window, e.g. 'Q3 2026', 'last 30 days', '2026-07-01 to 2026-09-30', or null>",
    "stage":      "<canonical deal stage or null>",
    "status":     "<canonical WO execution status or null>",
    "owner":      "<person/owner code or null>",
    "target_metric": "<e.g. 'deal_value', 'billed_value', 'collected_amount', 'order_amount', 'all_revenue', or null>"
  }},
  "required_boards": ["deals" | "workorders" | "deals","workorders"],
  "is_ambiguous":    false,
  "ambiguity_reason": "",
  "assumptions":     ["<explicit assumption 1>", "<explicit assumption 2>", ...]
}}

Rules:
1. Intent & Boards:
   - Pipeline, sales, deal stage, win rate, sales rep queries -> intent "deals", required_boards = ["deals"]
   - Work order, execution status, operational delays, billing status, project delivery -> intent "workorders", required_boards = ["workorders"]
   - Comparing deals vs work orders, contract realization, deal-to-order linkage, cross-board revenue -> intent "cross", required_boards = ["deals","workorders"]
   - Executive report, founder update, high-level business health -> intent "leadership", required_boards = ["deals","workorders"]
   - General company knowledge / chit-chat -> intent "general", required_boards = []

2. Loose Founder Language & Synonym Mapping:
   - "energy" -> map to Sector "Energy" (which encompasses Powerline and Energy).
   - "solar" / "wind" / "green" -> map to Sector "Renewables".
   - "trains" / "rail" -> map to Sector "Railways".
   - "defense" / "military" / "surveillance" -> map to Sector "Defence".
   - "this quarter" -> resolve against {current_quarter} ({current_date}).
   - Ambiguous metric words: "revenue" can mean 'Masked Deal value' (pipeline), 'Amount (Excl GST)' (order value), 'Billed Value' (invoiced), or 'Collected Amount' (cash). If the question is about pipeline, assume Masked Deal value. If about operations, assume Amount (Excl GST). Always document this in 'assumptions'.

3. Clarification vs Assumption:
   - Set is_ambiguous = true ONLY if the question is genuinely underspecified and cannot be answered with reasonable business assumptions (e.g. "How is everything going?", "What about that client?").
   - If proceeding with an assumption (e.g. assuming 'revenue' means deal value or 'this quarter' means {current_quarter}), keep is_ambiguous = false and add the exact assumption to the 'assumptions' list so it is surfaced to the user.
"""

# ── Intent-only prompt (fast path, not used for full understanding) ────────────

_INTENT_ONLY_PROMPT = """\
Classify the intent of this business question (one word only, no explanation):
deals | workorders | cross | leadership | general

Question: {question}
Intent:\
"""


# ── Client ─────────────────────────────────────────────────────────────────────

class GeminiClient:
    """Thin LangChain wrapper around Gemini 1.5 Pro / Flash with error handling and telemetry."""

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise GeminiAuthError(
                "GEMINI_API_KEY (or GOOGLE_API_KEY) is not set. "
                "Please add it to your .env file."
            )
        model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.2,
            max_output_tokens=4096,
        )
        logger.info("GeminiClient initialised with model: %s", model_name)

    def _invoke(self, system: str, user: str) -> str:
        """Send a prompt pair and return stripped text response with exception mapping."""
        try:
            response = self.llm.invoke([
                SystemMessage(content=system),
                HumanMessage(content=user),
            ])
            content = response.content
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict) and "text" in item:
                        parts.append(item["text"])
                    elif hasattr(item, "text"):
                        parts.append(item.text)
                    else:
                        parts.append(str(item))
                return "".join(parts).strip()
            return str(content).strip()

        except Exception as exc:
            err_msg = str(exc).lower()
            if "invalid api key" in err_msg or "unauthenticated" in err_msg or "401" in err_msg or "403" in err_msg:
                raise GeminiAuthError(f"Gemini API authentication failed: {exc}") from exc
            if "quota" in err_msg or "resourceexhausted" in err_msg or "429" in err_msg or "rate limit" in err_msg:
                raise GeminiRateLimitError(f"Gemini API rate limit or quota exceeded: {exc}") from exc
            if "connection" in err_msg or "timeout" in err_msg or "503" in err_msg or "unavailable" in err_msg:
                raise GeminiNetworkError(f"Transient network error connecting to Gemini: {exc}") from exc
            raise

    def _parse_json(self, raw: str) -> dict:
        """Strip markdown fences and parse JSON, returning {} on failure."""
        text = raw.strip()
        if "```" in text:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
            if match:
                text = match.group(1).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("LLM returned non-JSON: %.200r", raw)
            return {}

    # ── Primary: full query understanding ─────────────────────────────────────

    @timed_execution("gemini.understand_query")
    def understand_query(self, question: str) -> dict[str, Any]:
        """
        Deep query understanding in a single LLM call.
        """
        now = datetime.now(timezone.utc)
        current_date = now.strftime("%Y-%m-%d")
        current_quarter = f"Q{(now.month - 1) // 3 + 1} {now.year}"
        prompt = _UNDERSTAND_PROMPT.format(
            question=question,
            current_date=current_date,
            current_quarter=current_quarter,
        )
        raw = self._invoke(_SYSTEM_PROMPT, prompt)
        result = self._parse_json(raw)

        # Validate / normalise
        intent = result.get("intent", "general")
        if intent not in ("deals", "workorders", "cross", "leadership", "general"):
            intent = "general"

        # Ensure required_boards is consistent with intent
        required: list[str] = result.get("required_boards") or []
        if intent == "deals" and "deals" not in required:
            required = ["deals"]
        elif intent == "workorders" and "workorders" not in required:
            required = ["workorders"]
        elif intent in ("cross", "leadership") and len(required) < 2:
            required = ["deals", "workorders"]
        elif intent == "general":
            required = []

        return {
            "intent":           intent,
            "filters":          result.get("filters") or {},
            "required_boards":  required,
            "is_ambiguous":     bool(result.get("is_ambiguous", False)),
            "ambiguity_reason": result.get("ambiguity_reason") or "",
            "assumptions":      result.get("assumptions") or [],
        }

    # ── Clarification ──────────────────────────────────────────────────────────

    @timed_execution("gemini.ask_clarification")
    def ask_clarification(self, question: str, ambiguity: str) -> str:
        """Generate a single focused clarifying question."""
        prompt = (
            f'The user asked: "{question}"\n\n'
            f"The question is ambiguous because: {ambiguity}\n\n"
            "Write ONE concise, friendly clarifying question to resolve the ambiguity. "
            "Do not answer the original question yet."
        )
        return self._invoke(_SYSTEM_PROMPT, prompt)

    # ── Result narration (Prompt 7) ───────────────────────────────────────────

    @timed_execution("gemini.explain_results")
    def explain_results(
        self,
        question: str,
        metrics: dict[str, Any],
        quality: dict[str, Any] | None = None,
        assumptions: list[str] | None = None,
    ) -> str:
        """
        Narrate pre-computed metrics as an executive-friendly answer.
        Never invents or modifies figures.
        """
        # Build quality + assumptions section
        extra = ""
        if quality:
            completeness = quality.get("overall_completeness", 100)
            has_caveat = quality.get("has_caveat", False)
            warnings = quality.get("warnings", [])

            if has_caveat or warnings:
                parts = [
                    f"\n\n⚠️ **Data Quality & Caveats** (overall completeness: {completeness:.0f}%):",
                ]
                parts += [f"- {w}" for w in warnings]
                extra += "\n".join(parts)

        if assumptions:
            extra += "\n\n📝 **Assumptions Made:**\n"
            extra += "\n".join(f"- {a}" for a in assumptions)

        prompt = (
            f'The user asked: "{question}"\n\n'
            "Pre-computed analytics from live Monday.com boards (DO NOT recalculate or modify numbers):\n"
            f"```json\n{json.dumps(metrics, indent=2, default=str)}\n```"
            f"{extra}\n\n"
            "Write a concise, founder-ready answer adhering strictly to these requirements:\n"
            "• 1-2 sentence executive headline summarizing key takeaways\n"
            "• Bullet points for key metrics (formatted currency, counts, win rates, delivery rates)\n"
            "• Plain business language, no technical jargon\n"
            "• If cross-board project linkage or probability reporting is present, mention coverage and null-probability counts explicitly\n"
            "• Include data quality caveats and assumptions at the end\n"
            "• DO NOT recalculate or modify any numbers from the JSON."
        )
        return self._invoke(_SYSTEM_PROMPT, prompt)

    # ── Leadership update (Prompt 7) ───────────────────────────────────────────

    @timed_execution("gemini.generate_leadership_update")
    def generate_leadership_update(
        self,
        kpi_dict: dict[str, Any],
        quality: dict[str, Any] | None = None,
        assumptions: list[str] | None = None,
    ) -> str:
        """
        Generate a structured executive-summary leadership report (Prompt 7):
        - Pipeline snapshot
        - Delivery/ops snapshot
        - Sector highlights
        - Risks & Recommendations
        - Data caveats section
        """
        quality_str = ""
        if quality and quality.get("warnings"):
            quality_str = "\nData quality warnings:\n" + "\n".join(f"- {w}" for w in quality["warnings"])
        if assumptions:
            quality_str += "\nAssumptions:\n" + "\n".join(f"- {a}" for a in assumptions)

        prompt = (
            "Generate a professional, board-ready weekly leadership update for Skylark Drones.\n\n"
            "Use the following pre-computed KPIs (DO NOT recalculate or modify any figures):\n"
            f"```json\n{json.dumps(kpi_dict, indent=2, default=str)}\n```"
            f"{quality_str}\n\n"
            "Structure the report in clean Markdown exactly as follows:\n\n"
            "## 📊 Skylark Drones — Leadership Business Update\n"
            "**Period:** [from data period]\n\n"
            "### 🎯 Pipeline Snapshot\n"
            "- Total active pipeline value, weighted pipeline value, and total deals count\n"
            "- Deal stage distribution and overall win rate\n"
            "- Closure probability breakdown and separate disclosure of null-probability deals\n\n"
            "### 🚚 Delivery & Operations Snapshot\n"
            "- Work order volume and completion rate\n"
            "- On-time delivery rate and average delay days\n"
            "- Billing status breakdown and Amount Receivable summary\n"
            "- Overdue collections / unbilled delivery flags\n\n"
            "### 🌐 Sector Highlights & Cross-Board Alignment\n"
            "- Top performing sectors by pipeline and won revenue\n"
            "- Sector-level open pipeline vs active work order load\n"
            "- Exact-match project linkage summary (1:1 clean linked rollups vs excluded ambiguous matches)\n\n"
            "### ⚠️ Risks & Flags\n"
            "- Concrete bullet points for late-stage stall, overdue orders, collection risks, or pipeline concentration\n\n"
            "### 💡 Strategic Recommendations\n"
            "- 2-3 specific, actionable recommendations for the founders\n\n"
            "### 🔍 Data Quality & Caveats\n"
            "- Data completeness percentage, key missing fields, header-as-value stripping note, and linkage coverage disclosure\n\n"
            "---\n"
            "*Data sourced live from Monday.com. All calculations are deterministic pandas rollups.*\n\n"
            "Keep the report executive-ready, polished, and under 600 words."
        )
        return self._invoke(_SYSTEM_PROMPT, prompt)

    # ── General / fallback ─────────────────────────────────────────────────────

    @timed_execution("gemini.answer_general")
    def answer_general(self, question: str) -> str:
        """Answer a general question that doesn't need live board data."""
        prompt = (
            f'The user asked: "{question}"\n\n'
            "Answer politely as Skylark BI. If the question was asking about specific "
            "metrics (deals, work orders, revenue, pipeline), remind the user that they "
            "can ask specific questions about the sales pipeline, work orders, or request "
            "a leadership update."
        )
        return self._invoke(_SYSTEM_PROMPT, prompt)
