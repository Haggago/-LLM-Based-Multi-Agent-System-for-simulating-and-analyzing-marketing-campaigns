"""Prompt templates. Kept as plain functions so they're easy to test and tune
independently of the agents that call them.
"""

CATEGORY_TAXONOMY_SYSTEM = """You are a data-cleaning specialist for a healthcare business directory. \
You canonicalize messy, free-text Google Maps category strings into a fixed taxonomy. \
Always respond with ONLY the single best-matching category label from the provided list, \
verbatim, with no extra words, quotes, or punctuation. If nothing fits well, respond with \
"Other / Unclear"."""


def category_taxonomy_prompt(raw_category: str, description: str | None, taxonomy: list[str]) -> str:
    taxonomy_block = "\n".join(f"- {t}" for t in taxonomy)
    desc_block = f'\nBusiness description: "{description}"' if description else ""
    return (
        f"Raw category label: \"{raw_category}\"{desc_block}\n\n"
        f"Allowed categories:\n{taxonomy_block}\n\n"
        f"Which single category best matches?"
    )


CLUSTER_LABELING_SYSTEM = """You are a marketing data analyst. You are given summary statistics and \
representative examples for one cluster out of many produced by unsupervised clustering of businesses. \
Produce a concise, human-readable segment name (3-6 words) and a 2-3 sentence description of what \
characterizes this segment. Respond as JSON with keys "segment_name" and "segment_description". \
No markdown, no extra commentary."""


def cluster_labeling_prompt(cluster_id: int, stats: dict, sample_rows: list[dict]) -> str:
    stats_block = "\n".join(f"- {k}: {v}" for k, v in stats.items())
    samples_block = "\n".join(
        f"  {i+1}. {r.get('name','?')} | category: {r.get('category_llm') or r.get('category_normalized')} "
        f"| description: {(r.get('description') or '')[:150]}"
        for i, r in enumerate(sample_rows)
    )
    return (
        f"Cluster {cluster_id} summary statistics:\n{stats_block}\n\n"
        f"Representative businesses in this cluster:\n{samples_block}\n\n"
        f"Generate the segment name and description."
    )


MARKETING_INTEL_SYSTEM = """You are a senior marketing strategist advising an outreach/sales team that \
sells digital marketing and web services to healthcare businesses (addiction treatment, therapy, and \
related providers). For each business segment you are given, produce a short, actionable marketing \
intelligence brief. Be concrete and specific to the data provided -- avoid generic filler. \
Respond as JSON with keys: "priority_tier" (High/Medium/Low), "key_gaps" (list of 2-4 short strings), \
"recommended_approach" (2-3 sentences), and "estimated_deal_rationale" (1 sentence)."""


def marketing_intel_prompt(segment_name: str, segment_description: str, stats: dict) -> str:
    stats_block = "\n".join(f"- {k}: {v}" for k, v in stats.items())
    return (
        f"Segment: {segment_name}\n"
        f"Description: {segment_description}\n\n"
        f"Segment statistics:\n{stats_block}\n\n"
        f"Generate the marketing intelligence brief for a sales/outreach team targeting this segment."
    )
