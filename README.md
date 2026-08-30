# 🚁 Skylark BI Agent

> A LangGraph-powered Business Intelligence agent that answers founder-level questions about your Skylark Drones operations — pulling live data from Monday.com, cleaning it deterministically with pandas, and synthesising plain-English insights via Gemini.

---

## Live Demo

> Hosted on Streamlit Community Cloud — link provided separately.

---

## Architecture Overview

```
User (Streamlit Chat)
        │
        ▼
┌────────────────────────────────────────────────────────┐
│               LangGraph StateGraph                     │
│                                                        │
│  START → [router] → intent classification              │
│             │                                          │
│    ┌────────┼────────────┐                            │
│    ▼        ▼            ▼                            │
│ [clarify] [general]   [fetch] ← Monday.com API        │
│    │         │           │                            │
│    END      END    [normalize] ← pandas cleaning      │
│                        │                              │
│                   [analyze] ← deterministic math      │
│                        │                              │
│              ┌─────────┴──────────┐                   │
│              ▼                    ▼                   │
│          [report]           [synthesize]               │
│         (leadership)             │                    │
│              └──────────────────►│                    │
│                              [END] ← Gemini narrative  │
└────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Rationale |
|---|---|
| **pandas for all math** | Eliminates LLM hallucination on numeric results. LLM only writes prose. |
| **MCP probe + GraphQL fallback** | MCP requires admin-enabled AI Connectors. GraphQL works universally. |
| **Board IDs via env vars** | Avoids brittle name-based discovery; IDs are stable. |
| **`(df, quality_report)` tuple** | Forces data-quality surfacing at every query. |
| **LangGraph StateGraph** | Explicit routing makes conditional logic auditable and testable. |
| **Session-level data cache** | Fetched data persists within a Streamlit session — no redundant API calls. |

---

## Project Structure

```
skylark-assignment/
├── app/
│   └── main.py                  # Streamlit chat UI
├── graph/
│   ├── state.py                 # AgentState TypedDict
│   └── agent.py                 # LangGraph StateGraph + all nodes
├── tools/
│   ├── monday.py                # Monday.com data access (GraphQL + MCP)
│   └── connection_test.py       # Standalone connectivity verifier
├── normalize/
│   └── clean.py                 # pandas cleaning + quality report
├── analysis/
│   └── bi.py                    # Deterministic BI calculations
├── llm/
│   └── gemini.py                # Gemini client wrapper
├── tests/
│   ├── test_normalize.py        # Unit tests (no Monday.com required)
│   └── test_analysis.py
├── .streamlit/
│   └── config.toml              # Dark theme + server config
├── .env.example                 # Template for local development
├── requirements.txt
└── README.md
```

---

## Monday.com Board Setup

### Import the provided Excel files

1. In Monday.com, click **+ Add** → **Import data** → **Excel**
2. Import **Deal Funnel Data.xlsx** → name the board **Deal Funnel** (or any name)
3. Import **Work Order Tracker Data.xlsx** → name the board **Work Order Tracker** (or any name)

### Find your Board IDs

The board ID is in the URL when viewing a board:
```
https://your-company.monday.com/boards/1234567890
                                        ^^^^^^^^^^
                                        This is your board ID
```

### Recommended column types for Deal Funnel

| Column | Monday Type |
|---|---|
| Deal Value / Amount | Numbers |
| Stage / Status | Status |
| Sector / Industry | Dropdown |
| Close Date | Date |
| Owner / Sales Rep | Person |
| Company / Client | Text |

### Recommended column types for Work Order Tracker

| Column | Monday Type |
|---|---|
| Status | Status |
| Start Date / End Date | Date |
| Revenue / Contract Value | Numbers |
| Sector | Dropdown |
| Client / Customer | Text |
| Owner / Assigned To | Person |

> **Note:** The normalizer uses fuzzy column-name matching, so exact column names don't matter as long as they're reasonably descriptive. See `normalize/clean.py` for the full synonym list.

---

## Local Development Setup

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (fast Python package manager)
- A Monday.com account with API access
- A Google AI Studio / Gemini API key

### Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Installation

```bash
# Clone / extract the repo
cd skylark-assignment

# Install all dependencies (creates .venv automatically)
uv sync --dev

# Configure environment
cp .env.example .env
# Edit .env with your actual values
```

> **Streamlit Cloud note:** `requirements.txt` is auto-generated from `pyproject.toml` via
> `uv export --no-hashes --no-dev -o requirements.txt`. Commit it alongside `pyproject.toml`.
> Streamlit Cloud uses it to install dependencies with pip.

### Configure `.env`

```env
MONDAY_API_TOKEN=your_monday_api_token
DEALS_BOARD_ID=1234567890
WORK_ORDERS_BOARD_ID=0987654321
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-1.5-pro
USE_MCP=false
```

**Getting your Monday.com API token:**
1. Click your avatar → **Administration** → **Connections** → **API**
2. Copy your **Personal API Token** (v2)

### Test connectivity

```bash
uv run python -m tools.connection_test
```

### Run locally

```bash
uv run streamlit run app/main.py
```

Open [http://localhost:8501](http://localhost:8501)

### Run tests

```bash
uv run pytest
```

### Update dependencies

```bash
# Add a package
uv add some-package

# Regenerate requirements.txt for Streamlit Cloud
uv export --no-hashes --no-dev -o requirements.txt
```

---

## Deployment — Streamlit Community Cloud

1. **Push to GitHub** (public or private repo)

2. **Go to** [share.streamlit.io](https://share.streamlit.io) → **New app**

3. **Settings:**
   - Repository: `your-github-username/skylark-assignment`
   - Branch: `main`
   - Main file path: `app/main.py`

4. **Add Secrets** (App Settings → Secrets):
   ```toml
   MONDAY_API_TOKEN = "your_monday_api_token"
   DEALS_BOARD_ID = "1234567890"
   WORK_ORDERS_BOARD_ID = "0987654321"
   GEMINI_API_KEY = "your_gemini_api_key"
   GEMINI_MODEL = "gemini-1.5-pro"
   USE_MCP = "false"
   ```

5. **Deploy** — Streamlit Cloud installs `requirements.txt` automatically.

---

## Sample Questions

| Category | Example Question |
|---|---|
| Pipeline | "How's our pipeline looking this quarter?" |
| Sector | "What's the energy sector pipeline value?" |
| Win rates | "Show me win rates by deal stage" |
| Operations | "Which work orders are delayed by more than 2 weeks?" |
| Cross-board | "Compare deals won vs work orders completed by sector" |
| Leadership | "Generate a leadership update" |
| Clarification | "How are we doing?" ← agent asks a clarifying question |

---

## Data Quality Handling

Every query surfaces a **Data Quality** expander showing:
- Overall completeness % across the board
- Per-field warnings (e.g. "42% of deals are missing a close date")

The agent communicates these limitations explicitly rather than silently computing on incomplete data.

### Messy data handled

- Currency: `₹`, `$`, `,`, `1L`, `2.5Cr`, `500K`, ranges like `10L-15L`
- Dates: 10+ formats (ISO, DD/MM/YYYY, "15 Mar 2024", etc.)
- Sectors: synonyms (`agri` → Agriculture, `oil & gas` → Energy, etc.)
- Stages: synonyms (`won` → Closed Won, `lost` → Closed Lost, etc.)
- Status: (`wip` → In Progress, `done` → Done, etc.)
- Nulls: `""`, `"-"`, `"N/A"`, `"na"`, `"TBD"` → `NaN`
- Duplicates: exact-row deduplication

---

## Leadership Update Feature

Click **"📋 Generate Leadership Update"** in the sidebar (or ask naturally) to get a structured executive report covering:

- Pipeline & Sales KPIs
- Operations summary (completion rates, delays)
- Financial highlights by sector
- Risks & flags (stale deals, overdue WOs)
- Actionable recommendations

> **Interpretation:** The leadership update is a synthesis of live Monday.com data into a board-ready summary. All numbers are pandas-computed; Gemini writes the prose narrative. The agent flags data quality limitations inline.

---

## Tech Stack Justification

| Component | Choice | Reason |
|---|---|---|
| Agent framework | LangGraph | Explicit graph = auditable routing, easy to add nodes |
| LLM | Gemini 1.5 Pro | Strong reasoning + 1M token context for large datasets |
| Data access | Monday.com GraphQL API | Reliable, no admin setup; MCP available as opt-in upgrade |
| Data cleaning | pandas | Deterministic, testable, handles real-world mess |
| BI calculations | pandas | No LLM math = no hallucinations on numbers |
| UI | Streamlit | Fastest path to a shareable conversational interface |
| Deployment | Streamlit Community Cloud | Zero infra, GitHub-native, free tier |

---

## Environment Variables Reference

| Variable | Required | Description |
|---|---|---|
| `MONDAY_API_TOKEN` | ✅ | Monday.com Personal API Token v2 |
| `DEALS_BOARD_ID` | ✅ | Numeric ID of your Deal Funnel board |
| `WORK_ORDERS_BOARD_ID` | ✅ | Numeric ID of your Work Order Tracker board |
| `GEMINI_API_KEY` | ✅ | Google AI Studio API key (also accepted as `GOOGLE_API_KEY`) |
| `GEMINI_MODEL` | ⬜ | Model name (default: `gemini-1.5-pro`) |
| `USE_MCP` | ⬜ | `true` to attempt Monday.com MCP before GraphQL (default: `false`) |
