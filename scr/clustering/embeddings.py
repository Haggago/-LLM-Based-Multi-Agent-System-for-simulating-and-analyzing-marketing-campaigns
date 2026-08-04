"""Text embeddings for clustering, generated locally (no API cost, no rate
limits) via sentence-transformers. Again -- we embed UNIQUE text combos, not
every row, since category/description text repeats heavily in a scraped
directory (chains, franchises, copy-pasted listings).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_dataframe(self, df: pd.DataFrame, text_fields: list[str]) -> np.ndarray:
        """Builds a combined text field, embeds UNIQUE values once, joins back.
        Returns an (n_rows, dim) array aligned with df's row order.
        """
        combined = df[text_fields].fillna("").agg(" | ".join, axis=1)
        unique_texts = combined.drop_duplicates()

        logger.info(
            "Embedding %d unique text combinations out of %d total rows (%.1f%% reduction)",
            len(unique_texts), len(df), 100 * (1 - len(unique_texts) / max(len(df), 1)),
        )

        vectors = self.model.encode(
            unique_texts.tolist(), batch_size=256, show_progress_bar=True, convert_to_numpy=True,
        )
        lookup = dict(zip(unique_texts, vectors))

        dim = vectors.shape[1]
        out = np.zeros((len(df), dim), dtype=np.float32)
        for i, text in enumerate(combined.values):
            out[i] = lookup[text]
        return out
