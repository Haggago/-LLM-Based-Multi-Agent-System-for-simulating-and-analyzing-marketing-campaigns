"""Orchestrator: runs the deterministic pipeline stages and the LLM agents in
sequence, and writes out the final enriched dataset + a markdown report.

This is intentionally a plain sequential pipeline rather than a heavyweight
graph framework -- the stage dependencies here are linear (clean -> dedup ->
enrich -> categorize -> cluster -> label -> report), so a DAG framework would
add complexity without adding capability. If you later want conditional
branching (e.g. re-routing low-confidence rows through a different agent),
LangGraph is a natural upgrade path -- each agent here already exposes the
same run(df, context) -> (df, context) interface a graph node would use.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from src.agents.cleaning_agent import CleaningAgent
from src.agents.labeling_agent import LabelingAgent
from src.agents.marketing_agent import MarketingAgent
from src.clustering.cluster import fit_clusters
from src.llm.client import LLMClient
from src.pipeline.clean import CleaningPipeline
from src.pipeline.dedup import resolve_entities
from src.pipeline.enrich import enrich

logger = logging.getLogger(__name__)


class MultiAgentOrchestrator:
    def __init__(self, config: dict, skip_llm: bool = False):
        self.config = config
        self.skip_llm = skip_llm
        self.llm_client = None if skip_llm else LLMClient(
            config, cache_dir=config.get("paths", {}).get("cache_dir", "data/processed/.cache")
        )

    def run(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        context: dict = {}

        logger.info("=== Stage 1/6: Deterministic cleaning ===")
        df = CleaningPipeline(self.config).run(df)

        logger.info("=== Stage 2/6: Entity resolution (dedup) ===")
        df = resolve_entities(df, self.config)

        logger.info("=== Stage 3/6: Feature enrichment ===")
        df = enrich(df, self.config)

        if not self.skip_llm:
            logger.info("=== Stage 4/6: LLM cleaning agent (category canonicalization) ===")
            df, context = CleaningAgent(self.config, self.llm_client).run(df, context)
        else:
            logger.info("=== Stage 4/6: SKIPPED (skip_llm=True) ===")
            df["category_llm"] = df.get("category_normalized")

        logger.info("=== Stage 5/6: Segmentation (clustering) ===")
        df, cluster_meta = fit_clusters(df, self.config)
        context["cluster_meta"] = cluster_meta

        if not self.skip_llm:
            logger.info("=== Stage 6/6: LLM labeling + marketing intelligence agents ===")
            df, context = LabelingAgent(self.config, self.llm_client).run(df, context)
            df, context = MarketingAgent(self.config, self.llm_client).run(df, context)
        else:
            logger.info("=== Stage 6/6: SKIPPED (skip_llm=True) ===")

        return df, context

    def export(self, df: pd.DataFrame, context: dict) -> None:
        processed_dir = Path(self.config.get("paths", {}).get("processed_dir", "data/processed"))
        reports_dir = Path(self.config.get("paths", {}).get("reports_dir", "reports"))
        processed_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)

        out_path = processed_dir / "businesses_enriched.parquet"
        df.to_parquet(out_path, index=False)
        logger.info("Wrote enriched dataset to %s (%d rows, %d cols)", out_path, *df.shape)

        context_path = reports_dir / "run_context.json"
        context_path.write_text(json.dumps(_json_safe(context), indent=2))

        if "cluster_labels" in context:
            self._write_markdown_report(df, context, reports_dir / "marketing_intelligence_report.md")

    def _write_markdown_report(self, df: pd.DataFrame, context: dict, path: Path) -> None:
        lines = ["# Marketing Intelligence Report", ""]
        lines.append(f"Total businesses analyzed: **{len(df):,}**")
        lines.append(f"Segments identified: **{len(context.get('cluster_labels', {}))}**\n")

        for cluster_id, info in sorted(context.get("cluster_labels", {}).items()):
            brief = context.get("marketing_briefs", {}).get(cluster_id, {})
            lines.append(f"## Segment {cluster_id}: {info.get('segment_name')}")
            lines.append(f"{info.get('segment_description')}\n")
            lines.append(f"- **Size:** {info['stats'].get('size'):,} ({info['stats'].get('pct_of_total')}%)")
            lines.append(f"- **Priority tier:** {brief.get('priority_tier', 'N/A')}")
            gaps = brief.get("key_gaps", [])
            if gaps:
                lines.append(f"- **Key gaps:** {', '.join(gaps)}")
            lines.append(f"- **Recommended approach:** {brief.get('recommended_approach', 'N/A')}")
            lines.append(f"- **Deal rationale:** {brief.get('estimated_deal_rationale', 'N/A')}\n")

        path.write_text("\n".join(lines))
        logger.info("Wrote marketing intelligence report to %s", path)


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)
