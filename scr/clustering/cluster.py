"""Clustering: combines text embeddings with numeric business features and
fits MiniBatchKMeans, which scales comfortably to millions of rows (unlike
plain KMeans or HDBSCAN on the full dataset).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

from src.clustering.embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)


def build_feature_matrix(df: pd.DataFrame, config: dict) -> np.ndarray:
    cfg = config.get("clustering", {})
    text_fields = [f for f in cfg.get("text_fields", []) if f in df.columns]
    numeric_fields = [f for f in cfg.get("numeric_features", []) if f in df.columns]

    embedder = EmbeddingGenerator(cfg.get("embedding_model", "all-MiniLM-L6-v2"))
    text_matrix = embedder.embed_dataframe(df, text_fields) if text_fields else None

    numeric_matrix = None
    if numeric_fields:
        numeric_df = df[numeric_fields].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        numeric_matrix = StandardScaler().fit_transform(numeric_df.values)

    if text_matrix is not None and numeric_matrix is not None:
        # downweight the (high-dim) text embedding relative to numeric features
        text_matrix = text_matrix * 0.5
        return np.hstack([text_matrix, numeric_matrix])
    return text_matrix if text_matrix is not None else numeric_matrix


def fit_clusters(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    cfg = config.get("clustering", {})
    X = build_feature_matrix(df, config)
    if X is None:
        raise ValueError("No usable text or numeric fields found for clustering.")

    n_clusters = cfg.get("n_clusters", 12)
    logger.info("Fitting MiniBatchKMeans with k=%d on %d rows", n_clusters, len(df))

    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=cfg.get("random_state", 42),
        batch_size=min(10000, max(len(df), 256)),
        n_init="auto",
    )
    labels = model.fit_predict(X)

    df = df.copy()
    df["cluster_id"] = labels

    cluster_meta = {"n_clusters": n_clusters, "inertia": float(model.inertia_)}
    return df, cluster_meta


def cluster_summary(df: pd.DataFrame, cluster_id: int, numeric_fields: list[str]) -> dict:
    sub = df[df["cluster_id"] == cluster_id]
    stats = {"size": len(sub), "pct_of_total": round(100 * len(sub) / len(df), 2)}
    for f in numeric_fields:
        if f in sub.columns:
            stats[f"avg_{f}"] = round(float(pd.to_numeric(sub[f], errors="coerce").mean()), 3)
    if "category_llm" in sub.columns:
        stats["top_category"] = sub["category_llm"].mode().iloc[0] if not sub["category_llm"].mode().empty else None
    return stats
