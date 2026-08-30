"""
normalize/clean.py
──────────────────
Pandas-based normalisation for live Monday.com board items.

Both `clean_deals` and `clean_work_orders` return:
    (df: pd.DataFrame, quality_report: dict)

Design principles
-----------------
* Defensive:  never raise on bad data — bad values → NaN with a log warning.
* Idempotent: safe to call multiple times on the same input.
* No LLM:     all cleaning is rule-based / regex / synonym maps.
* is_missing: tracked per numeric/currency field so blank != 0.
* data_quality dict attached to every normalised frame, scoped to fields
  that are actually used downstream (not every column).

Header-as-value detection
--------------------------
When Monday.com data is re-imported from Excel, stray header rows appear
where a cell's value equals the column's own title string.  We detect and
drop these rows before any other processing.  Detection is exact-string
match between cell value and its column header — no fuzzy matching.

Synonym maps (editable)
------------------------
All synonym maps live at module level and are plain dicts so they can be
patched in tests or extended at runtime without touching core logic.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# NULL SENTINELS
# ══════════════════════════════════════════════════════════════════════════════

_NULL_STRINGS: frozenset[str] = frozenset({
    "", "-", "--", "---", "n/a", "na", "none", "null",
    "undefined", "nan", "#n/a", "tbd", "unknown", "?",
})


# ══════════════════════════════════════════════════════════════════════════════
# SYNONYM MAPS  (editable — patch or extend at runtime)
# ══════════════════════════════════════════════════════════════════════════════

# ── Sector / service ───────────────────────────────────────────────────────────
# Keys: lowercase stripped variants.  Values: canonical display name.
SECTOR_MAP: dict[str, str] = {
    # Energy / power
    "energy": "Energy",
    "energy sector": "Energy",
    "oil & gas": "Energy",
    "oil and gas": "Energy",
    "power": "Energy",
    "powerline": "Energy",
    "power line": "Energy",
    "renewables": "Renewables",
    "renewable energy": "Renewables",
    "renewable": "Renewables",
    "solar": "Energy",
    # Agriculture
    "agri": "Agriculture",
    "agriculture": "Agriculture",
    "agricultural": "Agriculture",
    "farming": "Agriculture",
    "agro": "Agriculture",
    # Industrial / manufacturing
    "industrial": "Industrial",
    "industry": "Industrial",
    "manufacturing": "Manufacturing",
    # Infrastructure
    "infra": "Infrastructure",
    "infrastructure": "Infrastructure",
    "railways": "Railways",
    "railway": "Railways",
    "rail": "Railways",
    # Mining
    "mining": "Mining",
    "mines": "Mining",
    # Government / Defence
    "government": "Government",
    "govt": "Government",
    "defense": "Defence",
    "defence": "Defence",
    "military": "Defence",
    "security and surveillance": "Defence",
    "security & surveillance": "Defence",
    # Construction
    "construction": "Construction",
    # Telecom
    "telecom": "Telecom",
    "telecommunications": "Telecom",
    # Logistics
    "logistics": "Logistics",
    "supply chain": "Logistics",
    # Forestry
    "forestry": "Forestry",
    "forest": "Forestry",
    # Aviation
    "aviation": "Aviation",
    # DSP / Spectra
    "dsp": "DSP",
    # Tender
    "tender": "Tender",
    # Catch-all
    "others": "Others",
    "other": "Others",
    "miscellaneous": "Others",
    "misc": "Others",
}

# ── Deal stage — based on actual board values ──────────────────────────────────
# Monday board uses alphabetic prefixes: "A. Lead Generated", "G. Project Won"
# We normalise to clean canonical names for BI calculations.
DEAL_STAGE_MAP: dict[str, str] = {
    # Direct board values (lowercase)
    "a. lead generated": "Lead",
    "b. sales qualified leads": "Qualified",
    "c. demo done": "Demo",
    "d. feasibility": "Demo",
    "e. proposal/commercials sent": "Proposal",
    "f. negotiations": "Negotiation",
    "g. project won": "Closed Won",
    "h. work order received": "Closed Won",
    "i. poc": "Demo",
    "j. invoice sent": "Closed Won",
    "k. amount accrued": "Closed Won",
    "l. project lost": "Closed Lost",
    "m. projects on hold": "On Hold",
    "n. not relevant at the moment": "Closed Lost",
    "o. not relevant at all": "Closed Lost",
    "project completed": "Closed Won",
    # Generic aliases
    "prospecting": "Prospecting",
    "prospect": "Prospecting",
    "lead": "Lead",
    "lead generated": "Lead",
    "qualified": "Qualified",
    "qualification": "Qualified",
    "proposal": "Proposal",
    "demo": "Demo",
    "demo / proposal": "Demo",
    "negotiation": "Negotiation",
    "negotiating": "Negotiation",
    "closed won": "Closed Won",
    "won": "Closed Won",
    "closed": "Closed Won",
    "closed lost": "Closed Lost",
    "lost": "Closed Lost",
    "on hold": "On Hold",
}

# ── Deal status (overall) ──────────────────────────────────────────────────────
DEAL_STATUS_MAP: dict[str, str] = {
    "open": "Open",
    "active": "Open",
    "won": "Won",
    "closed won": "Won",
    "dead": "Dead",
    "lost": "Dead",
    "closed lost": "Dead",
    "on hold": "On Hold",
    "hold": "On Hold",
    "not relevant at the moment": "Dead",
    "not relevant at all": "Dead",
}

# ── Closure probability ────────────────────────────────────────────────────────
CLOSURE_PROBABILITY_MAP: dict[str, str] = {
    "high": "High",
    "medium": "Medium",
    "med": "Medium",
    "low": "Low",
    "very high": "High",
    "very low": "Low",
}

# ── Execution status — collapsed to a small canonical set ─────────────────────
# "Executed until current month" → Ongoing (recurring project running)
# "Pause / struck"               → Paused
# "Partial Completed"            → Partial
# "Details pending from Client"  → Pending
EXECUTION_STATUS_MAP: dict[str, str] = {
    "completed": "Completed",
    "complete": "Completed",
    "done": "Completed",
    "ongoing": "Ongoing",
    "in progress": "Ongoing",
    "in-progress": "Ongoing",
    "executed until current month": "Ongoing",   # recurring project running this month
    "partial completed": "Partial",
    "partially completed": "Partial",
    "partial": "Partial",
    "pause / struck": "Paused",
    "paused": "Paused",
    "on hold": "Paused",
    "hold": "Paused",
    "not started": "Not Started",
    "pending": "Not Started",
    "details pending from client": "Pending",
    "details pending": "Pending",
    "cancelled": "Cancelled",
    "canceled": "Cancelled",
}

# ── Billing status ─────────────────────────────────────────────────────────────
# "BIlled" is a real typo in the live board — must be in the map.
BILLING_STATUS_MAP: dict[str, str] = {
    "billed": "Billed",
    "billed ": "Billed",     # trailing-space variant
    "billed.": "Billed",
    "fully billed": "Billed",
    "partially billed": "Partially Billed",
    "partial billed": "Partially Billed",
    "not billable": "Not Billable",
    "not billable ": "Not Billable",
    "update required": "Update Required",
    "stuck": "Stuck",
    "pending": "Pending",
}

# ── WO billed status ───────────────────────────────────────────────────────────
WO_BILLED_STATUS_MAP: dict[str, str] = {
    "open": "Open",
    "closed": "Closed",
    "close": "Closed",
}

# ── Invoice status ─────────────────────────────────────────────────────────────
INVOICE_STATUS_MAP: dict[str, str] = {
    "not billed yet": "Not Billed Yet",
    "billed": "Billed",
    "partially billed": "Partially Billed",
    "pending": "Pending",
    "cancelled": "Cancelled",
}

# ── Win-probability by canonical stage ────────────────────────────────────────
STAGE_PROBABILITIES: dict[str, float] = {
    "Prospecting": 0.10,
    "Lead":        0.15,
    "Qualified":   0.30,
    "Demo":        0.45,
    "Proposal":    0.60,
    "Negotiation": 0.80,
    "Closed Won":  1.00,
    "Closed Lost": 0.00,
    "On Hold":     0.20,
}


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _nullify(series: pd.Series) -> pd.Series:
    """Replace known null-sentinel strings with np.nan."""
    return series.map(
        lambda x: np.nan
        if (isinstance(x, str) and x.strip().lower() in _NULL_STRINGS)
        or x is None
        else x
    )


def _parse_currency(value: Any) -> float | None:
    """
    Parse a currency/numeric string into a Python float.

    Handles:
    - Indian currency symbols: ₹
    - Commas in Indian number formatting: 1,00,000
    - Shorthand: 1L = 100 000, 1Cr = 10 000 000, 1K = 1 000
    - Ranges (take lower bound): "10L-15L" → 1 000 000
    - Blank / unknown → None  (never zero)
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return None if pd.isna(value) else float(value)
    s = str(value).strip()
    if not s or s.lower() in _NULL_STRINGS:
        return None
    # Strip currency symbols and spaces
    s = re.sub(r"[₹$€£,\s]", "", s)
    # Ranges — take lower bound
    s = s.split("-")[0].strip()
    if not s:
        return None
    multiplier = 1.0
    upper = s.upper()
    if upper.endswith("CR"):
        multiplier = 10_000_000
        s = s[:-2]
    elif upper.endswith("L"):
        multiplier = 100_000
        s = s[:-1]
    elif upper.endswith("K"):
        multiplier = 1_000
        s = s[:-1]
    try:
        return float(s) * multiplier
    except (ValueError, AttributeError):
        return None


def _parse_date(value: Any) -> pd.Timestamp | None:
    """
    Try multiple date formats defensively; return pd.Timestamp or None.
    Never raises — bad formats are logged at DEBUG level and return None.
    """
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value if not pd.isna(value) else None
    s = str(value).strip()
    if not s or s.lower() in _NULL_STRINGS:
        return None
    _DATE_FORMATS = [
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%m/%d/%Y",
        "%d %b %Y",
        "%d %B %Y",
        "%b %d, %Y",
        "%B %d, %Y",
        "%d-%b-%Y",
        "%d-%B-%Y",
    ]
    for fmt in _DATE_FORMATS:
        try:
            return pd.Timestamp(datetime.strptime(s, fmt))
        except ValueError:
            continue
    # Last resort: let pandas infer
    try:
        return pd.Timestamp(s)
    except Exception:
        logger.debug("Could not parse date: %r", s)
        return None


def _apply_synonym_map(value: Any, synonym_map: dict[str, str]) -> str | None:
    """
    Apply a synonym map (case-insensitive, stripped).
    If no match: title-case the original value.
    If null/empty: return None.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip()
    if not s or s.lower() in _NULL_STRINGS:
        return None
    return synonym_map.get(s.lower(), s.strip())


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """
    Return the first column in `df` matching any candidate
    (case-insensitive, ignoring spaces/underscores/hyphens).
    """
    normalised = {re.sub(r"[\s_\-/]", "", c).lower(): c for c in df.columns}
    for cand in candidates:
        key = re.sub(r"[\s_\-/]", "", cand).lower()
        if key in normalised:
            return normalised[key]
    return None


# ══════════════════════════════════════════════════════════════════════════════
# HEADER-AS-VALUE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def drop_header_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Drop stray re-imported header rows.

    A row is considered a header row if ANY of its cell values exactly equals
    the column's own title string (e.g. Sector/service == "Sector/service",
    Deal Stage == "Deal Stage").

    This is an exact string comparison — not fuzzy — so legitimate values
    that happen to be short words are never mistakenly dropped.

    Returns:
        (cleaned_df, n_dropped)
    """
    cols = df.columns.tolist()
    # Build a boolean mask: True where ANY cell value == its own column header
    is_header_row = pd.Series(False, index=df.index)
    for col in cols:
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            is_header_row |= df[col].astype(str).str.strip() == str(col).strip()

    n_dropped = int(is_header_row.sum())
    if n_dropped:
        logger.warning(
            "Dropped %d header-as-value row(s). "
            "These are stray re-imported header rows from Excel export.",
            n_dropped,
        )
    return df[~is_header_row].copy(), n_dropped


# ══════════════════════════════════════════════════════════════════════════════
# DATA QUALITY REPORT
# ══════════════════════════════════════════════════════════════════════════════

def _quality_report(
    df: pd.DataFrame,
    key_fields: list[str],
    board: str,
    n_header_rows_dropped: int = 0,
    extra_warnings: list[str] | None = None,
) -> dict[str, Any]:
    """
    Compute a data_quality dict scoped to `key_fields` — the fields
    actually used downstream by the analysis layer.

    Structure:
        {
          "board": str,
          "total_rows": int,
          "header_rows_dropped": int,
          "overall_completeness": float,   # 0–100 %, key fields only
          "has_caveat": bool,
          "fields": {
            "<field>": {
              "missing_count": int,
              "missing_pct": float,
              "is_missing": bool,          # True if > threshold (30%)
            }
          },
          "warnings": [str, ...]
        }
    """
    total = len(df)
    warnings: list[str] = list(extra_warnings or [])
    field_stats: dict[str, dict] = {}

    if total == 0:
        return {
            "board": board,
            "total_rows": 0,
            "header_rows_dropped": n_header_rows_dropped,
            "overall_completeness": 0.0,
            "has_caveat": True,
            "fields": {},
            "warnings": warnings + ["No rows after cleaning."],
        }

    completeness_values: list[float] = []

    for field in key_fields:
        if field not in df.columns:
            field_stats[field] = {
                "missing_count": total,
                "missing_pct": 100.0,
                "is_missing": True,
            }
            warnings.append(f"Field '{field}' not found in board data.")
            completeness_values.append(0.0)
            continue

        n_missing = int(df[field].isna().sum())
        pct = round(100.0 * n_missing / total, 1)
        is_missing_flag = pct > 30.0
        field_stats[field] = {
            "missing_count": n_missing,
            "missing_pct": pct,
            "is_missing": is_missing_flag,
        }
        completeness_values.append(100.0 - pct)
        if is_missing_flag:
            warnings.append(
                f"{pct:.0f}% of rows are missing '{field}' — "
                "downstream calculations for this field may be incomplete."
            )

    if n_header_rows_dropped:
        warnings.append(
            f"{n_header_rows_dropped} stray header row(s) were removed "
            "(cells whose value equalled the column title)."
        )

    overall = round(sum(completeness_values) / len(completeness_values), 1) \
        if completeness_values else 0.0

    has_caveat = overall < 65.0 or len(warnings) >= 4

    return {
        "board": board,
        "total_rows": total,
        "header_rows_dropped": n_header_rows_dropped,
        "overall_completeness": overall,
        "has_caveat": has_caveat,
        "fields": field_stats,
        "warnings": warnings,
    }


# ══════════════════════════════════════════════════════════════════════════════
# DEALS NORMALIZATION
# ══════════════════════════════════════════════════════════════════════════════

# Fields that the analysis layer actually reads from the deals DataFrame.
# Quality report is scoped to these — not every column.
_DEALS_KEY_FIELDS = [
    "deal_value",
    "sector",
    "deal_stage",
    "deal_status",
    "closure_probability",
    "close_date",
    "owner_code",
    "client_code",
]


def clean_deals(raw_items: list[dict[str, Any]]) -> tuple[pd.DataFrame, dict]:
    """
    Normalise raw deal items fetched from Monday.com into a tidy DataFrame.

    Steps:
      1. Drop pure duplicate rows
      2. Detect & drop header-as-value rows (re-import artefact)
      3. Nullify sentinel strings ("", "-", "n/a", etc.)
      4. Map column names to canonical names (fuzzy + known board titles)
      5. Apply synonym maps: sector, deal_stage, deal_status, closure_probability
      6. Parse dates defensively
      7. Coerce deal_value to float; track is_missing_deal_value flag
      8. Compute probability + weighted_value from deal_stage
      9. Attach data_quality report scoped to key fields

    Returns:
        (df, quality_report)
    """
    if not raw_items:
        return pd.DataFrame(), {
            "board": "deals",
            "total_rows": 0,
            "header_rows_dropped": 0,
            "overall_completeness": 0.0,
            "has_caveat": True,
            "fields": {},
            "warnings": ["No deal items returned from Monday.com."],
        }

    df = pd.DataFrame(raw_items)

    # ── 1. Drop pure-duplicate rows ────────────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates()
    if len(df) < before:
        logger.info("Deals: dropped %d duplicate rows.", before - len(df))

    # ── 2. Drop header-as-value rows ───────────────────────────────────────────
    df, n_header_dropped = drop_header_rows(df)

    # ── 3. Nullify sentinel strings ────────────────────────────────────────────
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = _nullify(df[col])

    # ── 4. Column name → canonical mapping ────────────────────────────────────
    # Real Monday.com board column names (from schema probe):
    #   "Masked Deal value", "Deal Stage", "Sector/service",
    #   "Deal Status", "Closure Probability", "Close Date (A)",
    #   "Tentative Close Date", "Owner code", "Client Code"

    # Deal value
    val_col = _find_col(df, [
        "Masked Deal value", "Deal Value", "Value", "Amount",
        "Revenue", "deal_value", "DealValue", "Contract Value",
    ])
    if val_col:
        df["deal_value"] = df[val_col].apply(_parse_currency)
        df["is_missing_deal_value"] = df["deal_value"].isna()
        if val_col != "deal_value":
            df = df.drop(columns=[val_col], errors="ignore")
    else:
        df["deal_value"] = np.nan
        df["is_missing_deal_value"] = True
        logger.warning("Deals: no deal-value column found.")

    # Sector
    sector_col = _find_col(df, [
        "Sector/service", "Sector", "Industry", "Vertical",
        "Category", "sector",
    ])
    if sector_col:
        df["sector"] = df[sector_col].apply(
            lambda x: _apply_synonym_map(x, SECTOR_MAP)
        )
        if sector_col != "sector":
            df = df.drop(columns=[sector_col], errors="ignore")
    else:
        df["sector"] = None

    # Deal stage
    stage_col = _find_col(df, [
        "Deal Stage", "Stage", "Pipeline Stage", "deal_stage",
    ])
    if stage_col:
        df["deal_stage"] = df[stage_col].apply(
            lambda x: _apply_synonym_map(x, DEAL_STAGE_MAP)
        )
        if stage_col != "deal_stage":
            df = df.drop(columns=[stage_col], errors="ignore")
    else:
        df["deal_stage"] = None

    # Deal status (Won / Dead / Open / On Hold)
    status_col = _find_col(df, ["Deal Status", "deal_status", "Status"])
    if status_col:
        df["deal_status"] = df[status_col].apply(
            lambda x: _apply_synonym_map(x, DEAL_STATUS_MAP)
        )
        if status_col != "deal_status":
            df = df.drop(columns=[status_col], errors="ignore")
    else:
        df["deal_status"] = None

    # Closure probability
    prob_col = _find_col(df, [
        "Closure Probability", "closure_probability", "Win Probability", "Probability",
    ])
    if prob_col:
        df["closure_probability"] = df[prob_col].apply(
            lambda x: _apply_synonym_map(x, CLOSURE_PROBABILITY_MAP)
        )
        if prob_col != "closure_probability":
            df = df.drop(columns=[prob_col], errors="ignore")
    else:
        df["closure_probability"] = None

    # ── 5. Dates ───────────────────────────────────────────────────────────────
    close_col = _find_col(df, [
        "Close Date (A)", "Close Date", "Expected Close",
        "Closing Date", "CloseDate", "close_date",
    ])
    if close_col:
        df["close_date"] = df[close_col].apply(_parse_date)
        if close_col != "close_date":
            df = df.drop(columns=[close_col], errors="ignore")
    else:
        df["close_date"] = pd.NaT

    tent_col = _find_col(df, [
        "Tentative Close Date", "Tentative Close", "tentative_close_date",
    ])
    if tent_col:
        df["tentative_close_date"] = df[tent_col].apply(_parse_date)
        if tent_col != "tentative_close_date":
            df = df.drop(columns=[tent_col], errors="ignore")
    else:
        df["tentative_close_date"] = pd.NaT

    # System dates
    for sys_col in ("created_at", "updated_at"):
        if sys_col in df.columns:
            df[sys_col] = df[sys_col].apply(_parse_date)

    created_col = _find_col(df, ["Created Date", "created_date"])
    if created_col and created_col not in ("created_at",):
        df["created_date"] = df[created_col].apply(_parse_date)
        if created_col != "created_date":
            df = df.drop(columns=[created_col], errors="ignore")

    # ── 6. Owner / Client codes ────────────────────────────────────────────────
    owner_col = _find_col(df, [
        "Owner code", "Owner Code", "Owner", "Account Owner",
        "Sales Rep", "Assigned To", "Person", "owner",
    ])
    if owner_col and owner_col != "owner_code":
        df["owner_code"] = df[owner_col]
        df = df.drop(columns=[owner_col], errors="ignore")
    elif owner_col == "owner_code":
        pass
    else:
        df["owner_code"] = None

    client_col = _find_col(df, [
        "Client Code", "Client code", "Company", "Client",
        "Account", "Organization", "Customer", "client_code",
    ])
    if client_col and client_col != "client_code":
        df["client_code"] = df[client_col]
        df = df.drop(columns=[client_col], errors="ignore")
    elif client_col == "client_code":
        pass
    else:
        df["client_code"] = None

    # ── 7. Win-probability + weighted value & aliases ──────────────────────────
    df["stage"] = df["deal_stage"]
    df["owner"] = df["owner_code"]
    df["company"] = df["client_code"]
    df["probability"] = df["deal_stage"].map(STAGE_PROBABILITIES)
    df["weighted_value"] = df["deal_value"] * df["probability"]

    # ── 8. Data quality report ─────────────────────────────────────────────────
    report = _quality_report(
        df,
        key_fields=_DEALS_KEY_FIELDS,
        board="deals",
        n_header_rows_dropped=n_header_dropped,
    )

    logger.info(
        "Deals cleaned: %d rows (-%d headers), %.1f%% complete on key fields.",
        len(df),
        n_header_dropped,
        report["overall_completeness"],
    )
    return df, report


# ══════════════════════════════════════════════════════════════════════════════
# WORK ORDERS NORMALIZATION
# ══════════════════════════════════════════════════════════════════════════════

# Fields used downstream by the analysis layer.
_WO_KEY_FIELDS = [
    "execution_status",
    "sector",
    "amount_excl_gst",
    "billed_excl_gst",
    "billing_status",
    "wo_billed_status",
    "probable_start_date",
    "probable_end_date",
    "data_delivery_date",
]


def clean_work_orders(raw_items: list[dict[str, Any]]) -> tuple[pd.DataFrame, dict]:
    """
    Normalise raw work-order items fetched from Monday.com into a tidy DataFrame.

    Steps:
      1. Drop pure duplicate rows
      2. Detect & drop header-as-value rows (re-import artefact)
      3. Nullify sentinel strings
      4. Map column names + apply synonym maps:
           Execution Status  → execution_status  (canonical set)
           Billing Status    → billing_status     ("BIlled" → "Billed", etc.)
           WO Status (billed)→ wo_billed_status
           Invoice Status    → invoice_status
           Sector            → sector
      5. Parse all date columns defensively
      6. Coerce numeric/currency columns; blank = NaN (not 0); is_missing flags
      7. Compute derived: is_complete, delay_days (if dates available)
      8. Attach data_quality report scoped to key fields

    Returns:
        (df, quality_report)
    """
    if not raw_items:
        return pd.DataFrame(), {
            "board": "work_orders",
            "total_rows": 0,
            "header_rows_dropped": 0,
            "overall_completeness": 0.0,
            "has_caveat": True,
            "fields": {},
            "warnings": ["No work order items returned from Monday.com."],
        }

    df = pd.DataFrame(raw_items)

    # ── 1. Drop pure-duplicate rows ────────────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates()
    if len(df) < before:
        logger.info("WOs: dropped %d duplicate rows.", before - len(df))

    # ── 2. Drop header-as-value rows ───────────────────────────────────────────
    df, n_header_dropped = drop_header_rows(df)

    # ── 3. Nullify sentinel strings ────────────────────────────────────────────
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = _nullify(df[col])

    # ── 4. Categorical columns ─────────────────────────────────────────────────

    # Execution status
    exec_col = _find_col(df, [
        "Execution Status", "execution_status", "Work Status", "Status",
    ])
    if exec_col:
        df["execution_status"] = df[exec_col].apply(
            lambda x: _apply_synonym_map(x, EXECUTION_STATUS_MAP)
        )
        if exec_col != "execution_status":
            df = df.drop(columns=[exec_col], errors="ignore")
    else:
        df["execution_status"] = None

    # Billing status — "BIlled" typo handled in BILLING_STATUS_MAP
    bill_col = _find_col(df, ["Billing Status", "billing_status"])
    if bill_col:
        df["billing_status"] = df[bill_col].apply(
            lambda x: _apply_synonym_map(x, BILLING_STATUS_MAP)
        )
        if bill_col != "billing_status":
            df = df.drop(columns=[bill_col], errors="ignore")
    else:
        df["billing_status"] = None

    # WO billed status
    wo_stat_col = _find_col(df, [
        "WO Status (billed)", "WO Status", "wo_billed_status",
    ])
    if wo_stat_col:
        df["wo_billed_status"] = df[wo_stat_col].apply(
            lambda x: _apply_synonym_map(x, WO_BILLED_STATUS_MAP)
        )
        if wo_stat_col != "wo_billed_status":
            df = df.drop(columns=[wo_stat_col], errors="ignore")
    else:
        df["wo_billed_status"] = None

    # Invoice status
    inv_col = _find_col(df, ["Invoice Status", "invoice_status"])
    if inv_col:
        df["invoice_status"] = df[inv_col].apply(
            lambda x: _apply_synonym_map(x, INVOICE_STATUS_MAP)
        )
        if inv_col != "invoice_status":
            df = df.drop(columns=[inv_col], errors="ignore")
    else:
        df["invoice_status"] = None

    # Sector
    sector_col = _find_col(df, ["Sector", "Industry", "Vertical", "sector"])
    if sector_col:
        df["sector"] = df[sector_col].apply(
            lambda x: _apply_synonym_map(x, SECTOR_MAP)
        )
        if sector_col != "sector":
            df = df.drop(columns=[sector_col], errors="ignore")
    else:
        df["sector"] = None

    # Nature of work / type
    nature_col = _find_col(df, ["Nature of Work", "Type of Work", "nature_of_work"])
    if nature_col:
        df["nature_of_work"] = df[nature_col].apply(
            lambda x: str(x).strip() if x and not pd.isna(x) else None
        )
        if nature_col != "nature_of_work":
            df = df.drop(columns=[nature_col], errors="ignore")

    # ── 5. Dates ───────────────────────────────────────────────────────────────
    _date_cols = [
        (["Probable Start Date", "Start Date", "start_date", "probable_start_date"], "probable_start_date"),
        (["Probable End Date", "End Date", "end_date", "probable_end_date"], "probable_end_date"),
        (["Data Delivery Date", "Delivery Date", "data_delivery_date"], "data_delivery_date"),
        (["Date of PO/LOI", "PO Date", "po_date"], "po_date"),
        (["Last invoice date", "Last Invoice Date", "last_invoice_date"], "last_invoice_date"),
    ]
    for candidates, target in _date_cols:
        src = _find_col(df, candidates)
        if src:
            df[target] = df[src].apply(_parse_date)
            if src != target:
                df = df.drop(columns=[src], errors="ignore")
        else:
            df[target] = pd.NaT

    # System dates
    for sys_col in ("created_at", "updated_at"):
        if sys_col in df.columns:
            df[sys_col] = df[sys_col].apply(_parse_date)

    # ── 6. Numeric / currency columns ─────────────────────────────────────────
    # All are "Masked" in the live board to protect client identity
    _num_cols = [
        (
            ["Amount in Rupees (Excl of GST) (Masked)", "Amount in Rupees (Excl of GST)", "amount_excl_gst"],
            "amount_excl_gst",
            "is_missing_amount",
        ),
        (
            ["Amount in Rupees (Incl of GST) (Masked)", "Amount in Rupees (Incl of GST)", "amount_incl_gst"],
            "amount_incl_gst",
            "is_missing_amount_incl",
        ),
        (
            ["Billed Value in Rupees (Excl of GST.) (Masked)", "Billed Value (Excl GST)", "billed_excl_gst"],
            "billed_excl_gst",
            "is_missing_billed",
        ),
        (
            ["Billed Value in Rupees (Incl of GST.) (Masked)", "Billed Value (Incl GST)", "billed_incl_gst"],
            "billed_incl_gst",
            "is_missing_billed_incl",
        ),
        (
            ["Collected Amount in Rupees (Incl of GST.) (Masked)", "Collected Amount", "collected_amount"],
            "collected_amount",
            "is_missing_collected",
        ),
        (
            ["Amount Receivable (Masked)", "Amount Receivable", "amount_receivable"],
            "amount_receivable",
            "is_missing_receivable",
        ),
        (
            ["Amount to be billed in Rs. (Exl. of GST) (Masked)", "Amount to be billed", "amount_to_bill"],
            "amount_to_bill",
            "is_missing_to_bill",
        ),
        (
            ["Quantity by Ops", "quantity_by_ops"],
            "quantity_by_ops",
            "is_missing_qty",
        ),
    ]
    for candidates, target, missing_flag in _num_cols:
        src = _find_col(df, candidates)
        if src:
            df[target] = df[src].apply(_parse_currency)
            df[missing_flag] = df[target].isna()
            if src != target:
                df = df.drop(columns=[src], errors="ignore")
        else:
            df[target] = np.nan
            df[missing_flag] = True

    # ── 7. Derived fields & aliases ───────────────────────────────────────────
    df["status"] = df["execution_status"]
    df["revenue"] = df["amount_excl_gst"]

    # is_complete
    df["is_complete"] = df["execution_status"].isin(["Completed"])

    # delay_days: positive = delivered late
    if "probable_end_date" in df.columns and "data_delivery_date" in df.columns:
        both_present = df["probable_end_date"].notna() & df["data_delivery_date"].notna()
        df["delay_days"] = np.where(
            both_present,
            (df["data_delivery_date"] - df["probable_end_date"]).dt.days,
            np.nan,
        )
    else:
        df["delay_days"] = np.nan

    df["is_on_time"] = df["is_complete"] & (df["delay_days"].fillna(0) <= 0)

    # ── 8. Data quality report ─────────────────────────────────────────────────
    report = _quality_report(
        df,
        key_fields=_WO_KEY_FIELDS,
        board="work_orders",
        n_header_rows_dropped=n_header_dropped,
    )

    logger.info(
        "WOs cleaned: %d rows (-%d headers), %.1f%% complete on key fields.",
        len(df),
        n_header_dropped,
        report["overall_completeness"],
    )
    return df, report
