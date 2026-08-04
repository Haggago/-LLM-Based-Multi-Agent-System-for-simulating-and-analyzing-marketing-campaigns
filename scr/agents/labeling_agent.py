"""Labeling Agent (LLM-assisted).

Runs at most n_clusters times (dozens, not millions) -- uses the reasoning
model since quality matters more than volume here.
"""

from __future__ import annotations

import json
import logging

import pandas as pd

from src.agents.base_agent import BaseAgent
from src.clustering.cluster import cluster_summary
from src.llm.client import LLMClient
from src.llm.prompts import CLUSTER_LABELING_SYSTEM, cluster_labeling_prompt

logger = logging.getLogger(__name__)


class LabelingAgent(BaseAgent):
    name = "labeling_agent"

    def __init__(self, config: dict, llm_client: LLMClient):
        super().__init__(config)
        self.llm = llm_client
        self.reasoning_model = config.get("llm", {}).get("reasoning_model", "claude-sonnet-5")

    def run(self, df: pd.DataFrame, context: dict) -> tuple[pd.DataFrame, dict]:
        self._log_start(df)
        if "cluster_id" not in df.columns:
            raise ValueError("labeling_agent requires cluster_id -- run clustering first")

        numeric_fields = self.config.get("clustering", {}).get("numeric_features", [])
        labels: dict[int, dict] = {}

        for cluster_id in sorted(df["cluster_id"].unique()):
            stats = cluster_summary(df, cluster_id, numeric_fields)
            sample_rows = (
                df[df["cluster_id"] == cluster_id]
                .sample(min(5, stats["size"]), random_state=42)
                .to_dict("records")
            )
            prompt = cluster_labeling_prompt(cluster_id, stats, sample_rows)
            raw = self.llm.complete(prompt, system=CLUSTER_LABELING_SYSTEM, model=self.reasoning_model)
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Cluster %d: failed to parse LLM JSON, using fallback label", cluster_id)
                parsed = {"segment_name": f"Segment {cluster_id}", "segment_description": raw[:300]}
            labels[cluster_id] = {**parsed, "stats": stats}

        df = df.copy()
        df["segment_name"] = df["cluster_id"].map(lambda c: labels[c]["segment_name"])
        df["segment_description"] = df["cluster_id"].map(lambda c: labels[c]["segment_description"])

        context["cluster_labels"] = labels
        self._log_end(df)
        return df, context
