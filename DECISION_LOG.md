# 📑 Architectural Decision Log: Skylark Business Intelligence Agent

**Project:** Skylark Drones Technical Assignment — Founder BI Agent  
**Author:** AI Engineer / Candidate  
**Date:** August 30, 2026  
**Stack:** Python 3.13, LangGraph, Gemini (Google GenAI), Monday.com GraphQL API, pandas, Streamlit  

---

## Executive Summary

This document details the core architectural choices, data engineering trade-offs, defensive design patterns, and operational assumptions made while designing and building the **Skylark Business Intelligence Agent**.

The agent is designed to answer founder- and executive-level business questions across sales pipeline health and project execution operations. The implementation adheres strictly to a foundational principle: **Monday.com is the sole runtime source of truth, pandas performs 100% of deterministic business calculations, and the LLM handles semantic understanding and executive narration without hallucinating numbers.**

---

## 1. Orchestration: LangGraph vs. Simple Wrapper / ReAct Loop

### Context & Problem
Founder queries range from simple status checks (*"What is our win rate in Mining?"*) to ambiguous exploratory questions (*"How are things going?"*), complex cross-board calculations (*"Compare deals won vs active work order load by sector"*), and scheduled executive summaries (*"/leadership-update"*).

### Choice
We selected **LangGraph** with a strongly typed `AgentState` schema over a naive linear chain or an unconstrained ReAct agent loop.

```
                     ┌─────────────── [START] ───────────────┐
                     │                                       │
            _route_entry (trigger)                           │
          ├── leadership ──► leadership_entry                │
          └── query ───────► query_understanding             │
                                    │                        │
                         _route_clarification                │
                   ├── ambiguous ──► ask_user ──► [END]      │
                   └── clear ──────► data_retrieval ◄────────┘
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

### Rationale
1. **Auditable, Deterministic Routing**: Unlike unbounded ReAct loops that can drift into repetitive tool-calling cycles, LangGraph's explicit graph topology guarantees that execution progresses deterministically through defined lifecycle stages: `Understanding` $\rightarrow$ `Retrieval` $\rightarrow$ `Normalization` $\rightarrow$ `Analysis` $\rightarrow$ `Insight Generation`.
2. **First-Class Clarification Path (`ask_user`)**: If a founder question is genuinely ambiguous (e.g. *"Tell me about that project"*), the graph conditionally routes to `ask_user` and terminates immediately with a targeted question rather than hallucinating an answer on guessed parameters.
3. **Quality-Gated Downstream Flow**: If the normalization layer discovers that data completeness falls below threshold ($\le 65\%$), the `Data Quality Check` router sets a `has_caveat` signal that forces the insight generator to disclose limitations prominently.
4. **Parallel Leadership Fast-Path**: The `/leadership-update` command enters via `leadership_entry`, pre-populates required boards (`["deals", "workorders"]`), and bypasses redundant ambiguity checks to run straight through retrieval and executive analysis.

---

## 2. Division of Labor: pandas for Deterministic Math vs. Gemini for Reasoning

### Context & Problem
Large Language Models excel at natural language synthesis, fuzzy synonym resolution, and semantic framing. However, LLMs are fundamentally non-deterministic when performing arithmetic, floating-point aggregations, multi-column joins, or weighted percentages over hundreds of rows.

### Choice
A strict separation of concerns was enforced:
* **pandas Engine (Deterministic)**: 100% of row filtering, duplicate dropping, currency parsing, weighted pipeline calculations, win/loss rates, delay computations, billing breakdowns, and exact joins.
* **Gemini LLM (Reasoning & Narration)**: 
  1. *Query Understanding*: Maps loose founder terms (*"solar"*, *"last month"*, *"revenue"*) into structured filters.
  2. *Insight Generation*: Receives structured JSON output from pandas and translates it into an executive-ready narrative, strictly forbidden from modifying or computing figures.

```
  Founder Query ──► Gemini (Intent & Synonyms) ──► Structured Filters
                                                          │
  Live Monday.com Data ──► pandas (Clean & Math) ◄────────┘
                                 │
                         Computed Metrics JSON
                                 │
                    Gemini (Executive Narrative) ──► Founder Answer
```

### Result
Eliminated numeric hallucination. Every currency figure, percentage, and deal count presented to the founder is mathematically verifiable against the underlying pandas DataFrames.

---

## 3. Cross-Board Linkage: Exact Normalized String Match vs. Fuzzy Matching

### Context & Problem
The two Monday.com boards originate from separate business domains:
* **Deal Funnel**: Uses `Client Code` (`COMPANYxxx`) and `Deal Name` (e.g. *Sasuke, Naruto, scooby-doo*).
* **Work Order Tracker**: Uses `Customer Name Code` (`WOCOMPANY_xxx`) and `Deal name masked` (e.g. *Sasuke, Naruto, scooby-doo*).

Crucially, **the client code namespaces do not correspond** (`COMPANY001` does not map to `WOCOMPANY_001`). Therefore, cross-board linkage had to rely on project codenames.

### Choice: Exact-Match Only (`trim` + `case-fold`)
We deliberately rejected fuzzy matching (Levenshtein distance, Jaro-Winkler, token-set ratio) in favor of **exact normalized string matching** on `Deal Name == Deal name masked`, combined with **strict cardinality-based ambiguity exclusion**.

### Linkage Mechanics & Coverage Reality
1. **Ambiguity Exclusion Rule**: Because fictional codenames repeat across unrelated deals and orders, any codename that maps to $>1$ Deal or $>1$ Work Order is flagged as `ambiguous`.
2. **Numeric Rollup Guard**: Ambiguous matches are strictly excluded from numeric financial totals (e.g. Deal Value vs. Invoiced Revenue) and reported only in a dedicated caveat section.
3. **Live Data Findings**:
   * **Total Matching Codenames**: 53
   * **Clean 1:1 Linked Projects**: 16 projects (e.g. *Sasuke, Naruto, Goku, Vegeta*).
   * **Ambiguous / Multi-Match Codenames**: 37 codenames (e.g. *scooby-doo* which appeared across 12 distinct deals and 8 work orders).
   * **Unmatched Deals**: 291 deals (early-stage pipeline that never reached operational execution).
   * **Unmatched Work Orders**: 123 work orders (legacy or direct work orders without active CRM deal entries).

### Why Fuzzy Matching Was Rejected
In corporate BI, a false-positive join is significantly more damaging than an unlinked row. Fuzzy matching would have incorrectly joined distinct project codenames (e.g. *Batman-Phase1* to *Batman-Phase2* or *Ironman* to *Antman*), corrupting financial realization metrics without founder visibility. Exact-match plus explicit coverage disclosure ensures 100% data integrity.

---

## 4. Normalization & Data Quality Strategies

Monday.com exports often contain dirty inputs, inconsistent casings, human typos, and export artifacts. The normalizer handles these deterministically on every fetch:

### A. Header-as-Value Row Stripping
* **Issue**: Re-imported Excel sheets often contain stray header rows embedded inside the dataset (e.g. a row where `Sector/service == "Sector/service"` and `Deal Stage == "Deal Stage"`).
* **Solution**: Implemented `drop_header_rows(df)` which tests each row against column headers. In live data, this cleanly removed **2 invalid rows** (*Nezuko* and *Bugs Bunny*).

### B. Editable Synonym Maps & Typo Normalization
* **Billing Status**: Normalized the notorious `"BIlled"` typo $\rightarrow$ `"Billed"`.
* **Execution Status**: Collapsed fragmented operational variants into a canonical set:
  * `"Executed until current month"` $\rightarrow$ `"Ongoing"`
  * `"Pause / struck"` $\rightarrow$ `"Paused"`
  * `"Partial Completed"` $\rightarrow$ `"Partial"`
* **Deal Stages**: Stripped alphabetic sort prefixes (`"A. Lead Generated"` $\rightarrow$ `"Lead"`, `"G. Project Won"` $\rightarrow$ `"Closed Won"`, `"L. Project Lost"` $\rightarrow$ `"Closed Lost"`).
* **Sectors**: Unified sector synonyms (`"Powerline"` $\rightarrow$ `"Energy"`, `"Solar"/"Wind"` $\rightarrow$ `"Renewables"`, `"Train"` $\rightarrow$ `"Railways"`).

### C. Defensive Currency & Date Parsing
* **Currency**: Handled `₹`, `$`, commas, shorthand notation (`1.5L` $\rightarrow 150,000$, `2.5Cr` $\rightarrow 25,000,000$, `50K` $\rightarrow 50,000$), and ranges (`10L-15L` $\rightarrow 1,000,000$).
* **Null Rule**: **Blanks, sentinel strings (`"N/A"`, `"-"`), and unparseable values are preserved as `NaN`/`None` and NEVER coerced to `0`**. Coercing missing values to zero would severely corrupt average deal sizes and financial reporting.
* **Missingness Tracking**: Created per-field boolean masks (`is_missing_<field>`) and scoped `data_quality` dictionaries attached to every normalized DataFrame.

---

## 5. Metric Disambiguation: What "Revenue" Means

### Context & Problem
Founders frequently ask: *"What is our revenue this quarter?"* However, the Monday.com boards maintain four distinct monetary fields across two lifecycles:

| Field Name | Board | Business Meaning |
|---|---|---|
| `Masked Deal value` | Deals | **Pipeline Opportunity Value** (Pre-sale potential) |
| `Amount in Rupees (Excl of GST) (Masked)` | Work Orders | **Contract / Order Book Value** (Committed commercial value) |
| `Billed Value in Rupees (Excl of GST.) (Masked)` | Work Orders | **Invoiced Revenue** (Recognized work performed) |
| `Collected Amount in Rupees (Incl of GST.) (Masked)` | Work Orders | **Cash Collection** (Realized cash in bank) |

### Resolution Heuristic
1. **Context-Driven Mapping**:
   * If the question discusses *pipeline, sales targets, or won deals* $\rightarrow$ mapped to `Masked Deal value`.
   * If the question discusses *operations, projects, or contract size* $\rightarrow$ mapped to `Amount (Excl GST)`.
   * If the question discusses *billing, delivery, or realized financial performance* $\rightarrow$ mapped to `Billed Value` and `Collected Amount`.
2. **Mandatory Assumption Disclosure**: Whenever "revenue" is used loosely, the agent writes an explicit assumption into `state.assumptions` (e.g. *"Interpreted 'revenue' as Invoiced Billed Value (₹X) and Collected Cash (₹Y)"*) which is surfaced in the UI drawer and final answer.

---

## 6. Interpretation of "Leadership Updates"

The `/leadership-update` feature was designed not as generic prose, but as a **structured, board-ready executive briefing**:

1. **Parallel Entry Path**: Bypasses conversational back-and-forth and executes full cross-board ingestion (`deals` + `workorders`).
2. **Standard Executive Taxonomy**:
   * **🎯 Pipeline Snapshot**: Active pipeline value (₹110.69 Cr), probability-weighted value (₹43.89 Cr), stage distribution, win rate (55.9%), and explicit disclosure of null-probability deal volume.
   * **🚚 Delivery & Operations Snapshot**: 176 total orders, 66.5% completion rate, 80.3% on-time execution, average delay of 24.8 days on stalled jobs, and ₹3.63 Cr total Amount Receivable.
   * **🌐 Sector Highlights & Alignment**: Cross-board comparison of open pipeline demand vs. active delivery capacity per sector.
   * **⚠️ Risks & Actionable Recommendations**: Immediate alerts on overdue delivered work without final invoices, unbilled milestones, and stale late-stage proposals.
   * **🔍 Data Quality & Caveats**: Complete transparency on data completeness percentage, dropped header rows, and linkage coverage.
3. **Pre-Generation Scoping**: Founders can optionally scope the update by Sector (e.g. *Mining only*), Sales Owner, or Time Horizon via the UI expander.

---

## 7. Production Readiness, Error Handling & Telemetry

Implemented in `tools/error_handler.py` and `tools/monday.py`:
* **Auth Failure Handling**: Catches 401/403 errors and presents actionable setup instructions for `MONDAY_API_TOKEN` / `GEMINI_API_KEY`.
* **Rate Limiting (429)**: Respects `Retry-After` headers and performs exponential backoff retries (up to 5 attempts).
* **Transient Network Errors**: Automatic retry on connection drops, timeouts, and 5xx server issues.
* **Pagination Loop Guard**: Hard-capped pagination (`max_pages=30`) preventing runaway infinite cursor loops.
* **Partial Board Failure Recovery**: If Work Orders is temporarily down while Deals succeeds, the agent **proceeds with available data** and explicitly documents the operational data gap in `state.assumptions`.
* **Telemetry**: `@timed_execution` decorator logs node entry, exit, and millisecond-level tool latency for every GraphQL and LLM interaction.

---

## 8. What I Would Improve With More Time

1. **Upstream Schema Alignment in Monday.com**:
   * Implement a native Monday.com `board-relation` / `mirror` column linking each Work Order directly to its originating Deal Item ID. This replaces string codename heuristics with foreign-key integrity.
2. **Multi-Factor Confidence-Scored Entity Matching**:
   * Build a probabilistic entity-resolution model that weights multiple signals (normalized codename, company domain, contract value matching $\pm 5\%$, assigned owner, and tentative start date proximity) to provide a linkage confidence score (0.00 – 1.00) rather than binary match/reject.
3. **Time-Series Velocity & Cohort Tracking**:
   * Ingest Monday.com activity logs / audit trails to calculate deal velocity (average days spent in Proposal $\rightarrow$ Negotiation $\rightarrow$ Won) and cohort retention across quarters.
4. **Webhook-Driven Real-Time Sync**:
   * Replace manual sync / poll patterns with Monday.com Webhooks (`item_created`, `column_value_changed`) delivering change events to a streaming state store.
