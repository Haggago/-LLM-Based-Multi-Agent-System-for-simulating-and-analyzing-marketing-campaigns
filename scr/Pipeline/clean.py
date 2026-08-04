"""Deterministic, vectorized cleaning -- runs on all 2.68M rows, no LLM involved.

Everything here is rule-based on purpose: it's free, fast, and deterministic.
Only the genuinely ambiguous residue (messy category strings that don't match
any rule) gets handed to the LLM cleaning agent later, on unique values only.
"""

from __future__ import annotations

import logging
import re

import pandas as pd

try:
    import phonenumbers
except ImportError:  # pragma: no cover
    phonenumbers = None

logger = logging.getLogger(__name__)

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def normalize_phone(raw: str | None, default_region: str = "DE") -> str | None:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)) or not phonenumbers:
        return None
    try:
        parsed = phonenumbers.parse(raw, default_region)
        if phonenumbers.is_valid_number(parsed):
            return phonenumbers.format_number(
                parsed, phonenumbers.PhoneNumberFormat.E164
            )
    except Exception:
        pass
    return None


def validate_email(raw: str | None) -> tuple[str | None, bool]:
    """Returns (normalized_email_or_None, is_valid_syntax)."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)) or not str(raw).strip():
        return None, False
    raw = raw.strip().lower()
    is_valid = bool(EMAIL_RE.match(raw))
    return (raw if is_valid else None), is_valid


def normalize_category_rule_based(raw: str | None) -> str | None:
    """Cheap rule-based category normalization. Anything this can't confidently
    map is left as-is and flows to the LLM cleaning agent on unique values.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)) or not str(raw).strip():
        return None
    return str(raw).strip().title()


class CleaningPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.default_region = config.get("cleaning", {}).get("default_country_code", "DE")

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        n_start = len(df)

        # --- required-field sanity ---
        min_len = self.config.get("cleaning", {}).get("min_name_length", 2)
        if "name" in df.columns:
            df = df[df["name"].notna() & (df["name"].str.len() >= min_len)]

        # --- phone ---
        if "phone" in df.columns:
            df["phone_normalized"] = df["phone"].map(
                lambda v: normalize_phone(v, self.default_region)
            )

        # --- email ---
        if "email" in df.columns:
            normalized_valid = df["email"].map(validate_email)
            df["email_normalized"] = normalized_valid.map(lambda t: t[0])
            df["email_syntax_valid"] = normalized_valid.map(lambda t: t[1])

        # --- category (rule-based pass; LLM agent refines later on uniques) ---
        if "category" in df.columns:
            df["category_normalized"] = df["category"].map(normalize_category_rule_based)

        # --- closed-business handling ---
        if "is_permanently_closed" in df.columns:
            if self.config.get("cleaning", {}).get("drop_permanently_closed", False):
                before = len(df)
                df = df[~df["is_permanently_closed"]]
                logger.info("Dropped %d permanently-closed businesses", before - len(df))

        permanently_closed = df["is_permanently_closed"] if "is_permanently_closed" in df.columns \
            else pd.Series(False, index=df.index)
        temporarily_closed = df["is_temporarily_closed"] if "is_temporarily_closed" in df.columns \
            else pd.Series(False, index=df.index)
        df["is_active"] = ~permanently_closed.astype(bool) & ~temporarily_closed.astype(bool)

        # --- numeric sanity clamps ---
        if "score" in df.columns:
            df.loc[(df["score"] < 0) | (df["score"] > 5), "score"] = pd.NA
        if "lat" in df.columns:
            df.loc[(df["lat"] < -90) | (df["lat"] > 90), "lat"] = pd.NA
        if "lng" in df.columns:
            df.loc[(df["lng"] < -180) | (df["lng"] > 180), "lng"] = pd.NA

        logger.info("Cleaning: %d -> %d rows", n_start, len(df))
        return df.reset_index(drop=True)
