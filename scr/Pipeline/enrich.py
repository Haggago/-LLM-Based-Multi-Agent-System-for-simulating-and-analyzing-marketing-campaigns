"""Feature engineering for segmentation and marketing scoring.

All vectorized, all cheap -- runs on every row without touching an LLM.
"""

from __future__ import annotations

import json
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def _safe_has(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[col].notna() & (df[col].astype(str).str.strip().ne("")) & (
        df[col].astype(str).str.lower().ne("nan")
    )


def _parse_opening_hours_days(raw) -> int:
    """Count distinct days with hours listed. Handles the common
    JSON-ish or delimited formats scrapers export; degrades gracefully."""
    if not raw or (isinstance(raw, float) and pd.isna(raw)):
        return 0
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return sum(1 for v in parsed.values() if v)
            if isinstance(parsed, list):
                return len(parsed)
        except (json.JSONDecodeError, TypeError):
            # fallback: count weekday-name occurrences
            days = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
            return sum(1 for d in days if d in raw.lower())
    return 0


def enrich(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    df = df.copy()

    # --- digital presence signals ---
    df["has_website"] = _safe_has(df, "website")
    df["has_email"] = _safe_has(df, "email_normalized") | _safe_has(df, "email")
    df["has_phone"] = _safe_has(df, "phone_normalized") | _safe_has(df, "phone")
    df["has_facebook"] = _safe_has(df, "facebook")
    df["has_instagram"] = _safe_has(df, "instagram")
    df["has_booking_link"] = _safe_has(df, "booking_link")
    df["has_description"] = _safe_has(df, "description")
    df["has_main_image"] = _safe_has(df, "main_image_url")

    social_cols = ["has_facebook", "has_instagram"]
    df["social_media_count"] = df[social_cols].sum(axis=1)

    presence_cols = [
        "has_website", "has_email", "has_phone", "has_facebook",
        "has_instagram", "has_booking_link",
    ]
    df["digital_presence_score"] = df[presence_cols].sum(axis=1) / len(presence_cols)

    # --- data completeness (useful for prioritizing enrichment/outreach) ---
    completeness_cols = [
        "name", "category", "street_address", "city", "zip_code", "phone",
        "email", "website", "description", "main_image_url",
    ]
    present = [c for c in completeness_cols if c in df.columns]
    df["completeness_score"] = sum(_safe_has(df, c).astype(int) for c in present) / max(len(present), 1)

    # --- opening hours ---
    if "opening_hours" in df.columns:
        df["days_open_per_week"] = df["opening_hours"].map(_parse_opening_hours_days)

    # --- description length (proxy for marketing content investment) ---
    if "description" in df.columns:
        df["description_length"] = df["description"].fillna("").astype(str).str.len()

    # --- price bucket ---
    if "price" in df.columns:
        buckets = config.get("enrichment", {}).get("price_buckets", ["unknown", "$", "$$", "$$$", "$$$$"])
        df["price_bucket"] = df["price"].where(df["price"].isin(buckets), "unknown")

    # --- review volume/quality tiering (helps lead scoring later) ---
    if "ratings" in df.columns:
        df["review_volume_tier"] = pd.cut(
            df["ratings"].fillna(0),
            bins=[-1, 0, 5, 20, 100, float("inf")],
            labels=["none", "very_low", "low", "moderate", "high"],
        )

    logger.info("Enrichment complete: %d features added", len(df.columns))
    return df
