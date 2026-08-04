"""Marketing Intelligence Agent (LLM-assisted).

Same low-volume pattern as the labeling agent: one call per segment.
Produces the final actionable output the rest of the system exists to serve.
"""

from __future__ import annotations

import json
import logging

import pandas as pd

from src.agents.base_agent import BaseAgent
from src.llm.client import LLMClient
from src.llm.prompts import MARKETING_INTEL_SYSTEM, marketing_intel_prompt

logger = logging.getLogger(__name__)


class MarketingAgent(BaseAgent):
    name = "marketing_agent"

    def __init__(self, config: dict, llm_client: LLMClient):
        super().__init__(config)
        self.llm = llm_client
        self.reasoning_model = config.get("llm", {}).get("reasoning_model", "claude-sonnet-5")

    def run(self, df: pd.DataFrame, context: dict) -> tuple[pd.DataFrame, dict]:
        self._log_start(df)
        cluster_labels = context.get("cluster_labels")
        if not cluster_labels:
            raise ValueError("marketing_agent requires context['cluster_labels'] -- run labeling_agent first")

        briefs = {}
        for cluster_id, info in cluster_labels.items():
            prompt = marketing_intel_prompt(info["segment_name"], info["segment_description"], info["stats"])
            raw = self.llm.complete(prompt, system=MARKETING_INTEL_SYSTEM, model=self.reasoning_model)
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Cluster %d: failed to parse marketing brief JSON", cluster_id)
                parsed = {
                    "priority_tier": "Unknown", "key_gaps": [], "recommended_approach": raw[:300],
                    "estimated_deal_rationale": "",
                }
            briefs[cluster_id] = parsed

        df = df.copy()
        df["marketing_priority_tier"] = df["cluster_id"].map(lambda c: briefs[c].get("priority_tier"))
        df["marketing_recommended_approach"] = df["cluster_id"].map(
            lambda c: briefs[c].get("recommended_approach")
        )

        context["marketing_briefs"] = briefs
        self._log_end(df)
        return df, context
