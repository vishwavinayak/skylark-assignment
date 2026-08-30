# 🚁 Skylark Drones — Founder Business Intelligence Agent

> An enterprise-grade, LangGraph-orchestrated Business Intelligence agent that answers founder- and executive-level questions about Skylark Drones' commercial pipeline and operational delivery.
> 
> Built on **live Monday.com data**, calculated deterministically with **pandas**, reasoned with **Google Gemini**, and delivered via a **Streamlit** conversational interface.

---

## 📑 Table of Contents
1. [Executive Overview & Approach](#-executive-overview--approach)
2. [System Architecture](#-system-architecture)
3. [Core Assumptions & Disambiguations](#-core-assumptions--disambiguations)
4. [Key Engineering Trade-offs](#-key-engineering-trade-offs)
5. [AI Tools & Frameworks Used](#-ai-tools--frameworks-used)
6. [Challenges Faced & Solutions](#-challenges-faced--solutions)
7. [Cross-Board Linkage Strategy](#-cross-board-linkage-strategy)
8. [Leadership Update Feature](#-leadership-update-feature)
9. [Potential Future Improvements](#-potential-future-improvements)
10. [Local Setup & Streamlit Cloud Deployment](#-local-setup--streamlit-cloud-deployment)

---

## 🎯 Executive Overview & Approach

Founders need reliable, rapid visibility into company performance across two distinct datasets:
1. **Deal Funnel (CRM)**: Sales pipeline, stages, deal values, win/loss rates, closure probabilities, and sales reps.
2. **Work Order Tracker (Operations)**: Project delivery status, completion rates, operational delays, billing milestones, and cash collections.

### Our Approach
* **Zero Math Hallucinations**: LLMs are notoriously unreliable at arithmetic, floating-point aggregations, and multi-table joins. We strictly enforce that **100% of numeric calculations are executed deterministically in Python/pandas**. The LLM (Gemini) is restricted solely to query understanding, filter extraction, and executive narration.
* **Live Monday.com Source of Truth**: The original Excel sheets are never loaded at runtime. Live data is fetched dynamically via cursor-paginated Monday.com GraphQL API calls with automatic schema discovery.
* **Quality-Gated Transparency**: Incomplete fields and data anomalies are surfaced in an expandable **Data Quality** badge and **Assumptions** drawer with every response, guaranteeing complete auditability.

---

## 🏗 System Architecture

The application is powered by a **LangGraph `StateGraph`** with typed state threading, conditional routing, and error resilience:

```
                            ┌──────────────── [START] ────────────────┐
                            │                                         │
                   _route_entry (trigger)                             │
                 ├── leadership ──► leadership_entry                  │
                 └── query ───────► query_understanding               │
                                           │                          │
                                _route_clarification                  │
                          ├── ambiguous ──► ask_user ──► [END]        │
                          └── clear ──────► data_retrieval ◄──────────┘
                                                  │
                                            normalization
                                                  │
                                       [Data Quality Check]
                                                  │
                                              analysis (pandas)
                                                  │
                                          insight_generation (Gemini)
                                                  │
                                                [END]
```

### State Schema (`AgentState`)
Every node reads and updates a shared typed state containing:
* `query`: Natural language question from the founder.
* `intent`: Classified domain (`deals`, `workorders`, `cross`, `leadership`, `general`).
* `filters`: Structured parameters (`sector`, `date_range`, `stage`, `status`, `owner`, `target_metric`).
* `required_boards`: Dynamically identified boards (`["deals"]`, `["workorders"]`, or both).
* `raw_data`: Live items retrieved from Monday.com (cached at session level).
* `normalized_data`: Cleaned DataFrames with canonical types and stripped header rows.
* `metrics`: Deterministic pandas outputs (pipeline sums, win rates, delay distributions, linked rollups).
* `data_quality`: Scoped completeness metrics, warning lists, and `has_caveat` boolean flag.
* `assumptions`: Explicit inferences logged during understanding or calculation.
* `final_answer`: Polished markdown output presented to the founder.

---

## 📝 Core Assumptions & Disambiguations

### 1. The Meaning of "Revenue"
Monday.com contains four separate financial columns across different stages of the commercial lifecycle:
* **`Masked Deal value` (Deals)**: *Pipeline Opportunity Value* (Pre-sale potential).
* **`Amount in Rupees (Excl of GST) (Masked)` (Work Orders)**: *Contract / Order Book Value* (Committed delivery amount).
* **`Billed Value in Rupees (Excl of GST.) (Masked)` (Work Orders)**: *Invoiced Revenue* (Recognized delivery work).
* **`Collected Amount in Rupees (Incl of GST.) (Masked)` (Work Orders)**: *Realized Cash* (Cash in bank).

> **Heuristic**: When a founder asks for "revenue", the agent contextually maps to Deal Value (for sales pipeline questions), Order Amount (for delivery volume questions), or Billed/Collected Amount (for financial performance), and **always surfaces the exact interpretation in the `Assumptions` drawer**.

### 2. Missing Value Policy (Never Coerce to Zero)
* Blank cells, `"-"`, `"N/A"`, or unparseable values are preserved as `NaN`/`None`.
* **We never coerce missing numbers to `0`**. Coercing missing deal values to zero would severely corrupt average deal sizes, pipeline totals, and win rate analytics.
* Deals with null `Closure Probability` are tracked and reported separately in probability summaries rather than discarded.

### 3. Date Parsing & Current Context
* Loose relative dates (e.g. *"this quarter"*, *"last month"*) are resolved against current UTC system time (`Q3 2026`).
* 10+ date formats (ISO, `DD/MM/YYYY`, `DD-MM-YYYY`, text dates like `"15 Mar 2024"`) are parsed into ISO strings defensively without ever crashing.

---

## ⚖️ Key Engineering Trade-offs

| Decision | Chosen Path | Alternative Considered | Rationale |
|---|---|---|---|
| **Orchestration** | LangGraph StateGraph | Unbounded ReAct loop / Simple chain | Guarantees deterministic execution flow; enables explicit clarification routing and quality gates. |
| **Numeric Computations** | Pure pandas Engine | LLM Code Interpreter / LLM Arithmetic | Completely eliminates hallucinated numbers, math drift, and incorrect floating-point rollups. |
| **Data Access** | Direct GraphQL API (v2024-01) with MCP Probe | Pure MCP-only dependency | MCP requires enterprise admin connector setup. GraphQL works universally on all Monday accounts with zero blocker. |
| **Cross-Board Join** | Exact-Match Normalized String Join | Levenshtein / Fuzzy Matching | Fuzzy matching risks false-positive joins between unrelated deals with generic codenames (e.g. *Batman-1* vs *Batman-2*). In corporate BI, false joins are far worse than unlinked rows. |
| **Ambiguous Codenames** | Exclude from numeric financial rollups; disclose as caveat | Force 1:many join or arbitrary pick | Eliminates distorted double-counting when codenames repeat across unrelated deals/orders. |

---

## 🤖 AI Tools & Frameworks Used

* **Google Gemini (Gemini 1.5 Pro / Flash via `ChatGoogleGenerativeAI`)**:
  * Used for **Query Understanding** (extracting structured intent, canonical filters, and detecting ambiguity).
  * Used for **Clarification Generation** (crafting a single targeted question when user input is underspecified).
  * Used for **Insight Generation** (synthesizing pandas metrics into executive-ready markdown prose without changing numbers).
* **LangGraph (`langgraph`)**: Stateful graph topology, conditional branching (`_route_clarification`, `_route_data_quality`, `_route_entry`), and state persistence.
* **LangChain Core (`langchain-core`)**: Structured message primitives (`HumanMessage`, `AIMessage`, `SystemMessage`).
* **Tenacity (`tenacity`)**: Exponential backoff and retry decorators for API rate limiting (`429`) and transient network resilience.
* **Streamlit (`streamlit`)**: Conversational UI, session cache persistence, and secrets bridging.

---

## 🛠 Challenges Faced & Solutions

### 1. Re-imported "Header-as-Value" Row Artifacts
* **Challenge**: Excel exports re-imported into Monday.com contained duplicate header rows inside data rows (e.g. `Sector/service == "Sector/service"`).
* **Solution**: Implemented `drop_header_rows(df)` which dynamically compares cell content against column titles. In live data, this cleanly purged 2 invalid rows (*Nezuko* and *Bugs Bunny*).

### 2. Typo Normalization & Status Collapse
* **Challenge**: Typo in production Monday board (`"BIlled"`) and fragmented execution statuses (`"Executed until current month"`, `"Pause / struck"`, `"Partial Completed"`).
* **Solution**: Created declarative synonym maps in [`normalize/clean.py`](file:///Users/vishwavinayak/Documents/Development/skylark-assignment/normalize/clean.py) that map variants onto a canonical status set (`Billed`, `Ongoing`, `Paused`, `Partial`).

### 3. Non-Overlapping Code Namespaces
* **Challenge**: Work Orders use `Customer Name Code` (`WOCOMPANY_xxx`) and Deals use `Client Code` (`COMPANYxxx`). The ID numbering schemes do not correspond.
* **Solution**: Joined on normalized exact project codenames (`Deal Name == Deal name masked`) with case-folding and whitespace trimming.

### 4. Monday.com GraphQL v2024 Column Schema Quirks
* **Challenge**: API version `2024-01` does not expose `title` on `column_values` in the `items_page` query.
* **Solution**: Implemented two-phase schema discovery: fetched board columns metadata first to build an `id -> title` dictionary, then annotated item values dynamically.

---

## 🔗 Cross-Board Linkage Strategy

Cross-board queries evaluate commercial pipeline commitments against actual operational execution:

```
Deals Board (344 items)               Work Orders Board (176 items)
  ├── Deal Name: "Sasuke"  ─────────────► Deal name masked: "Sasuke"   (1:1 Clean Match ✅)
  ├── Deal Name: "Naruto"  ─────────────► Deal name masked: "Naruto"   (1:1 Clean Match ✅)
  └── Deal Name: "scooby-doo" (12 deals) ─► Deal name masked: "scooby-doo" (8 WOs)  (Ambiguous ⚠️)
```

### Linkage Coverage Summary (Live Data)
* **16 Clean 1:1 Linked Projects**: Rolled up into exact deal-to-work-order realization and variance metrics.
* **37 Ambiguous Codenames**: Codenames mapped to $>1$ Deal or $>1$ Order (e.g. *scooby-doo*). **Excluded from numeric rollups by default** to protect financial accuracy; reported in an explicit caveat section.
* **Unmatched Rows**: 291 early-stage pipeline deals (no work order issued yet) and 123 direct/legacy work orders.

---

## 📋 Leadership Update Feature

Triggered via **`"📊 Generate Leadership Update"`** in the sidebar, typing **`/leadership-update`**, or natural requests (*"prepare this week's update for the founders"*).

### Report Structure
1. **🎯 Pipeline Snapshot**: Active pipeline value (₹110.69 Cr), weighted value (₹43.89 Cr), stage distribution, win rate (55.9%), and null-probability disclosure.
2. **🚚 Delivery & Operations Snapshot**: 176 work orders, 66.5% completion rate, 80.3% on-time execution, delay analysis, and ₹3.63 Cr Amount Receivable breakdown.
3. **🌐 Sector Highlights & Alignment**: Cross-board comparison of open pipeline demand vs. active delivery capacity per sector.
4. **⚠️ Risks & Flags**: Immediate alerts on overdue delivered work without final invoices, unbilled milestones, and stale late-stage proposals.
5. **💡 Strategic Recommendations**: Concrete, high-impact founder actions.
6. **🔍 Data Quality & Caveats**: Full disclosure of data completeness %, dropped header rows, and linkage coverage.

---

## 🚀 Potential Future Improvements

With additional development time, the following high-value enhancements would be implemented:
1. **Upstream Foreign Key in Monday.com**: Configure a native Monday.com `board-relation` / `mirror` column linking each Work Order directly to its originating Deal Item ID, replacing string codename heuristics with relational integrity.
2. **Multi-Factor Confidence-Scored Entity Matching**: Train or configure a probabilistic entity-resolution pipeline weighting multiple signals (codename, client domain, contract value $\pm 5\%$, assigned owner, date proximity) to provide a linkage confidence score (0.00 – 1.00) with human review for borderline matches.
3. **Time-Series Velocity & Cohort Tracking**: Ingest Monday.com activity audit logs to track stage velocity (average days spent in *Proposal $\rightarrow$ Negotiation $\rightarrow$ Won*) and quarterly revenue cohort realization.
4. **Webhook-Driven Real-Time Sync**: Replace poll/sync buttons with Monday.com Webhooks (`item_created`, `column_value_changed`) delivering change events to a streaming state store.

---

## 💻 Local Setup & Streamlit Cloud Deployment

### 1. Prerequisites
* Python 3.10+
* [uv](https://docs.astral.sh/uv/) package manager
* Monday.com API Token & Board IDs
* Google Gemini API Key

### 2. Local Installation
```bash
# Clone the repository
git clone https://github.com/vishwavinayak/skylark-assignment.git
cd skylark-assignment

# Install dependencies via uv
uv sync

# Configure your environment
cp .env.example .env
# Fill in: MONDAY_API_TOKEN, DEALS_BOARD_ID, WORK_ORDERS_BOARD_ID, GEMINI_API_KEY
```

### 3. Run Automated Tests
```bash
uv run pytest -v
# 54 passed in ~1s
```

### 4. Run the Streamlit App Locally
```bash
uv run streamlit run app/main.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### 5. Deploy to Streamlit Community Cloud
1. Push your repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and create a **New App**.
3. Point to `streamlit_app.py` (or `app/main.py`).
4. In **App Settings $\rightarrow$ Secrets**, paste:
```toml
MONDAY_API_TOKEN = "your_monday_api_token"
DEALS_BOARD_ID = "5030966166"
WORK_ORDERS_BOARD_ID = "5030966117"
GEMINI_API_KEY = "your_gemini_api_key"
GEMINI_MODEL = "gemini-1.5-pro"
USE_MCP = "false"
```
5. Click **Deploy!**
