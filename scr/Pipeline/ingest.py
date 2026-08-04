"""Ingestion: load the raw export and normalize column names/dtypes.

Designed for a 2.68M-row CSV. Uses chunked reads + pyarrow where available so
it doesn't require an enormous amount of RAM on a laptop.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.utils.schema import normalize_columns, BOOLEAN_COLUMNS, NUMERIC_COLUMNS

logger = logging.getLogger(__name__)

TRUE_STRINGS = {"true", "1", "yes", "y"}
FALSE_STRINGS = {"false", "0", "no", "n", ""}


def _coerce_boolean(series: pd.Series) -> pd.Series:
    def _to_bool(v):
        if pd.isna(v):
            return False
        s = str(v).strip().lower()
        if s in TRUE_STRINGS:
            return True
        if s in FALSE_STRINGS:
            return False
        return bool(v)

    return series.map(_to_bool)


def load_data(path: str | Path, nrows: int | None = None, chunksize: int | None = None):
    """Load and normalize the raw dataset.

    Args:
        path: path to CSV (or parquet -- inferred from extension).
        nrows: optional cap, useful for local testing.
        chunksize: if set, returns a generator of normalized chunks instead of
            a single DataFrame. Recommended for the full 2.68M-row file when
            RAM is limited; downstream pipeline steps accept either.

    Returns:
        pd.DataFrame, or an iterator of pd.DataFrame chunks if chunksize is set.
    """
    path = Path(path)
    logger.info("Loading data from %s", path)

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
        if nrows:
            df = df.head(nrows)
        return _normalize(df)

    read_kwargs = dict(dtype=str, keep_default_na=False, na_values=[""])
    if chunksize:
        def _gen():
            for chunk in pd.read_csv(path, chunksize=chunksize, nrows=nrows, **read_kwargs):
                yield _normalize(chunk)
        return _gen()

    df = pd.read_csv(path, nrows=nrows, **read_kwargs)
    return _normalize(df)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = normalize_columns(df.columns)
    df = df.rename(columns=rename_map)

    for col in BOOLEAN_COLUMNS:
        if col in df.columns:
            df[col] = _coerce_boolean(df[col])

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "collected_at" in df.columns:
        df["collected_at"] = pd.to_datetime(df["collected_at"], errors="coerce")

    # strip whitespace on the main text fields
    for col in ["name", "category", "street_address", "address", "city",
                "region", "description", "email", "website"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().replace({"nan": None, "None": None})

    return df
