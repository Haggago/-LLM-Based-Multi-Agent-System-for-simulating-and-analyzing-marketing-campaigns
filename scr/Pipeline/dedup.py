"""Entity resolution.

Google Maps scrapes across overlapping search grids (INPUT CITY / INPUT
REGION / INPUT DISTRICT) routinely re-collect the same physical business.
Two-stage approach:

  1. Exact dedup on PLACE ID / CID -- these are Google's own stable unique
     identifiers, so this stage is cheap, safe, and catches the vast majority
     of duplicates in a scraped dataset.
  2. Fuzzy fallback, ONLY for rows missing both IDs -- blocked by zip_code so
     we never do an O(n^2) comparison across the full 2.68M rows.
"""

from __future__ import annotations

import logging

import pandas as pd
from rapidfuzz import fuzz

from src.utils.schema import ID_COLUMNS_PRIORITY

logger = logging.getLogger(__name__)


def dedup_exact(df: pd.DataFrame) -> pd.DataFrame:
    """Dedup on place_id, then cid, keeping the most-recently-collected row."""
    sort_col = "collected_at" if "collected_at" in df.columns else None
    if sort_col:
        df = df.sort_values(sort_col, ascending=False)

    for id_col in ID_COLUMNS_PRIORITY:
        if id_col not in df.columns:
            continue
        has_id = df[id_col].notna() & (df[id_col].astype(str).str.strip() != "")
        with_id = df[has_id].drop_duplicates(subset=[id_col], keep="first")
        without_id = df[~has_id]
        n_before = len(df)
        df = pd.concat([with_id, without_id], ignore_index=True)
        logger.info(
            "Exact dedup on %s: removed %d duplicate rows", id_col, n_before - len(df)
        )
    return df


def _normalize_key(row, fields) -> str:
    parts = []
    for f in fields:
        v = row.get(f)
        parts.append(str(v).strip().lower() if pd.notna(v) else "")
    return "|".join(parts)


def dedup_fuzzy(
    df: pd.DataFrame,
    block_by: str = "zip_code",
    threshold: int = 92,
    id_columns=ID_COLUMNS_PRIORITY,
) -> pd.DataFrame:
    """Fuzzy dedup restricted to rows missing every ID column, blocked by
    `block_by` to keep comparisons tractable (e.g. only compare rows sharing
    a zip code, rather than the whole dataset).
    """
    has_any_id = pd.Series(False, index=df.index)
    for id_col in id_columns:
        if id_col in df.columns:
            has_any_id |= df[id_col].notna() & (df[id_col].astype(str).str.strip() != "")

    no_id_df = df[~has_any_id]
    rest_df = df[has_any_id]

    if no_id_df.empty or block_by not in df.columns:
        return df

    to_drop = set()
    for _, group in no_id_df.groupby(block_by, dropna=True):
        if len(group) < 2:
            continue
        rows = group.to_dict("records")
        idxs = list(group.index)
        keys = [f"{r.get('name','')} {r.get('street_address', r.get('address',''))}" for r in rows]
        for i in range(len(rows)):
            if idxs[i] in to_drop:
                continue
            for j in range(i + 1, len(rows)):
                if idxs[j] in to_drop:
                    continue
                score = fuzz.token_sort_ratio(keys[i], keys[j])
                if score >= threshold:
                    to_drop.add(idxs[j])

    logger.info("Fuzzy dedup: removed %d likely-duplicate rows (no ID present)", len(to_drop))
    no_id_df = no_id_df.drop(index=list(to_drop))
    return pd.concat([rest_df, no_id_df], ignore_index=True)


def resolve_entities(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    df = dedup_exact(df)
    df = dedup_fuzzy(
        df,
        block_by=config.get("dedup", {}).get("block_by", "zip_code"),
        threshold=config.get("dedup", {}).get("fuzzy_match_threshold", 92),
    )
    return df.reset_index(drop=True)
