"""Cleaning Agent (LLM-assisted).

Key cost-control idea: with 2.68M rows there might be only a few thousand
DISTINCT raw category strings (chains, franchises, and repeated scraper
noise collapse heavily). We canonicalize the UNIQUE values through the LLM
once, cache every response, and join the result back onto all rows. This
turns a 2.68M-call problem into a low-thousands-call problem.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.agents.base_agent import BaseAgent
from src.llm.client import LLMClient
from src.llm.prompts import CATEGORY_TAXONOMY_SYSTEM, category_taxonomy_prompt

logger = logging.getLogger(__name__)


class CleaningAgent(BaseAgent):
    name = "cleaning_agent"

    def __init__(self, config: dict, llm_client: LLMClient):
        super().__init__(config)
        self.llm = llm_client
        self.taxonomy = config.get("taxonomy", {}).get("categories", [])
        self.max_calls = config.get("llm", {}).get("max_bulk_llm_calls", 20000)
        self.use_batch = config.get("llm", {}).get("use_batch_api", True)
        self.bulk_model = config.get("llm", {}).get("bulk_model", "claude-haiku-4-5-20251001")

    def run(self, df: pd.DataFrame, context: dict) -> tuple[pd.DataFrame, dict]:
        self._log_start(df)

        if "category_normalized" not in df.columns or not self.taxonomy:
            logger.warning("[%s] skipped: missing category_normalized column or empty taxonomy", self.name)
            return df, context

        # unique (category, description-snippet) pairs -- description helps
        # disambiguate generic labels like "Health" or "Clinic"
        df["_desc_snippet"] = df.get("description", pd.Series(dtype=str)).fillna("").astype(str).str.slice(0, 150)
        unique_pairs = (
            df[["category_normalized", "_desc_snippet"]]
            .drop_duplicates(subset=["category_normalized"])  # description snippet varies; key off category
            .dropna(subset=["category_normalized"])
        )

        if len(unique_pairs) > self.max_calls:
            logger.warning(
                "[%s] %d unique categories exceeds max_bulk_llm_calls=%d -- "
                "truncating to the %d most frequent (rest keep rule-based normalization)",
                self.name, len(unique_pairs), self.max_calls, self.max_calls,
            )
            freq = df["category_normalized"].value_counts()
            keep = set(freq.head(self.max_calls).index)
            unique_pairs = unique_pairs[unique_pairs["category_normalized"].isin(keep)]

        logger.info("[%s] canonicalizing %d unique category strings via %s",
                    self.name, len(unique_pairs), self.bulk_model)

        prompts = [
            category_taxonomy_prompt(row.category_normalized, row._desc_snippet, self.taxonomy)
            for row in unique_pairs.itertuples()
        ]
        custom_ids = [f"cat_{i}" for i in range(len(prompts))]

        if self.use_batch:
            responses = self.llm.batch_complete(
                prompts, system=CATEGORY_TAXONOMY_SYSTEM, model=self.bulk_model, custom_ids=custom_ids,
            )
            texts = [responses.get(cid, "").strip() for cid in custom_ids]
        else:
            texts = [
                self.llm.complete(p, system=CATEGORY_TAXONOMY_SYSTEM, model=self.bulk_model).strip()
                for p in prompts
            ]

        # guard against hallucinated labels outside the taxonomy
        valid_set = set(self.taxonomy)
        texts = [t if t in valid_set else "Other / Unclear" for t in texts]

        mapping = dict(zip(unique_pairs["category_normalized"], texts))
        df["category_llm"] = df["category_normalized"].map(mapping)
        df["category_llm"] = df["category_llm"].fillna("Other / Unclear")
        df = df.drop(columns=["_desc_snippet"])

        context["category_mapping_size"] = len(mapping)
        self._log_end(df)
        return df, context
