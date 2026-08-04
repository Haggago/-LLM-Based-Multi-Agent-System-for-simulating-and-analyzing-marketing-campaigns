"""CLI entrypoint.

Examples:
    # Quick local test on a 500-row sample, no LLM calls, no API key needed:
    python -m src.main --sample 500 --skip-llm

    # Full run on the real 2.68M-row export:
    python -m src.main --input data/raw/businesses.csv

    # Full run with a custom cluster count:
    python -m src.main --input data/raw/businesses.csv --n-clusters 15
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.agents.orchestrator import MultiAgentOrchestrator
from src.pipeline.ingest import load_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config/config.yaml", help="Path to YAML config")
    p.add_argument("--input", default=None, help="Override paths.raw_data from config")
    p.add_argument("--sample", type=int, default=None, help="Only load the first N rows (testing)")
    p.add_argument("--n-clusters", type=int, default=None, help="Override clustering.n_clusters")
    p.add_argument("--skip-llm", action="store_true", help="Run only the deterministic stages (no API key needed, no cost)")
    return p.parse_args()


def main():
    load_dotenv()
    args = parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())

    if args.input:
        config["paths"]["raw_data"] = args.input
    if args.n_clusters:
        config["clustering"]["n_clusters"] = args.n_clusters

    input_path = config["paths"]["raw_data"]
    if not Path(input_path).exists():
        logger.error(
            "Input file not found: %s\n"
            "Either place your export there, pass --input <path>, or generate a "
            "test sample with: python data/sample/generate_sample.py",
            input_path,
        )
        sys.exit(1)

    df = load_data(input_path, nrows=args.sample)
    logger.info("Loaded %d rows, %d columns", *df.shape)

    orchestrator = MultiAgentOrchestrator(config, skip_llm=args.skip_llm)
    df, context = orchestrator.run(df)
    orchestrator.export(df, context)

    logger.info("Done. See %s/ for outputs.", config["paths"]["processed_dir"])


if __name__ == "__main__":
    main()
