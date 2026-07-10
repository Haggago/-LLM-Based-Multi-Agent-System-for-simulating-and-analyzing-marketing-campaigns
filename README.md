# LLM-Based Multi-Agent System for Business Segmentation & Marketing Intelligence

Built for a 2.68M-row Google Maps export of drug/addiction-treatment businesses,
but the taxonomy in `config/config.yaml` is the only vertical-specific piece —
swap it and the pipeline works for any business directory.

## Design principle

At 2.68M rows, the biggest engineering risk isn't "which agents to build" —
it's calling an LLM per row. That's 2.68M API calls: expensive, slow, and the
wrong tool for bulk data cleaning anyway. This system splits work by what
actually needs an LLM:

| Stage | Runs on | Method | Cost |
|---|---|---|---|
| Cleaning (phone/email/address normalization, closed-business flags) | All 2.68M rows | Deterministic rules (pandas/regex/`phonenumbers`) | Free |
| Entity resolution (dedup) | All 2.68M rows | Exact match on Google's `PLACE ID`/`CID`, fuzzy fallback blocked by zip | Free |
| Feature engineering | All 2.68M rows | Vectorized pandas | Free |
| Category canonicalization | **Unique** category strings only (typically low thousands, not millions) | LLM (Claude Haiku 4.5, Batch API) | Low |
| Clustering / segmentation | All 2.68M rows | Local embeddings (`sentence-transformers`) + `MiniBatchKMeans` | Free |
| Segment naming & description | Once per cluster (dozens) | LLM (Claude Sonnet 5) | Negligible |
| Marketing intelligence briefs | Once per cluster (dozens) | LLM (Claude Sonnet 5) | Negligible |

Every LLM call is cached to disk (`data/processed/.cache/`) keyed by
`(model, prompt)` hash, so re-running the pipeline never re-bills for a
prompt it's already answered — important when iterating on later stages.

## Architecture

```
Raw CSV (2.68M rows, 54 cols)
        │
        ▼
┌─────────────────────┐
│  Ingestion           │  src/pipeline/ingest.py — schema normalization, dtypes
└─────────┬────────────┘
          ▼
┌─────────────────────┐
│  Cleaning            │  src/pipeline/clean.py — phone/email/address rules
└─────────┬────────────┘
          ▼
┌─────────────────────┐
│  Entity Resolution   │  src/pipeline/dedup.py — PLACE ID/CID + fuzzy fallback
└─────────┬────────────┘
          ▼
┌─────────────────────┐
│  Enrichment          │  src/pipeline/enrich.py — digital presence, completeness
└─────────┬────────────┘
          ▼
┌─────────────────────┐
│  🤖 Cleaning Agent    │  src/agents/cleaning_agent.py — category canonicalization
│     (Claude Haiku)   │  via Batch API, on UNIQUE values only
└─────────┬────────────┘
          ▼
┌─────────────────────┐
│  Clustering           │  src/clustering/ — local embeddings + MiniBatchKMeans
└─────────┬────────────┘
          ▼
┌─────────────────────┐
│  🤖 Labeling Agent    │  src/agents/labeling_agent.py — names/describes each
│     (Claude Sonnet)  │  cluster from summary stats + samples
└─────────┬────────────┘
          ▼
┌─────────────────────┐
│  🤖 Marketing Agent   │  src/agents/marketing_agent.py — priority tier, gaps,
│     (Claude Sonnet)  │  outreach strategy per segment
└─────────┬────────────┘
          ▼
   businesses_enriched.parquet
   marketing_intelligence_report.md
```

All agents share the same interface (`run(df, context) -> (df, context)`),
orchestrated sequentially by `src/agents/orchestrator.py`. It's a plain
pipeline, not a graph framework — the dependency chain here is linear.
If you need conditional routing later (e.g. re-processing low-confidence
rows through a different agent), each agent already exposes the interface
a LangGraph node would use, so that's a drop-in upgrade, not a rewrite.

## Setup

```bash
git clone <your-repo-url>
cd llm-marketing-intel
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your ANTHROPIC_API_KEY
```

## Quick test (no API key needed, no cost)

```bash
python data/sample/generate_sample.py --rows 2000 --out data/raw/businesses.csv
python -m src.main --skip-llm
```

This runs cleaning, dedup, and enrichment only — verify those stages against
your real data's quirks before spending anything on LLM calls.

## Full run

```bash
# Put your real export at data/raw/businesses.csv, or:
python -m src.main --input /path/to/your/export.csv

# Tune cluster count if the default 12 segments doesn't fit your data:
python -m src.main --input /path/to/your/export.csv --n-clusters 15
```

For the full 2.68M-row file, consider first running with `--sample 100000`
to sanity-check the LLM agent outputs (category taxonomy fit, segment
labels) before committing to the full run.

Outputs land in `data/processed/businesses_enriched.parquet` (all 2.68M
rows with all engineered features + cluster assignment) and
`reports/marketing_intelligence_report.md` (the human-readable brief).

## Cost control

- `llm.max_bulk_llm_calls` in `config/config.yaml` is a hard cap on distinct
  LLM calls in the cleaning agent — protects against a config or dedup bug
  silently exploding the unique-value count.
- `llm.use_batch_api: true` routes the cleaning agent through Anthropic's
  Message Batches API (~50% cheaper, async).
- Disk cache means the segmentation/labeling/marketing stages can be
  re-run freely while you tune `n_clusters` without re-billing the
  cleaning agent.

## Adapting to a different vertical

Everything specific to "drug therapy centers" lives in one place:
`taxonomy.categories` in `config/config.yaml`. Change that list and the
prompts in `src/llm/prompts.py` will canonicalize against your new taxonomy
automatically.

## Repository structure

```
config/config.yaml          all tunable parameters (paths, thresholds, models)
data/sample/                synthetic data generator for testing
src/utils/schema.py         raw column → normalized column mapping
src/pipeline/               deterministic stages (no LLM, run on all rows)
src/clustering/             local embeddings + MiniBatchKMeans
src/llm/                    Anthropic client (sync + batch), prompt templates
src/agents/                 the three LLM agents + orchestrator
src/main.py                 CLI entrypoint
tests/                      pytest unit tests for the deterministic stages
```

## Tests

```bash
pytest tests/ -v
```
