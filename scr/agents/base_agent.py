"""Minimal agent interface. Deliberately lightweight -- no heavy framework
dependency. Each agent has a single job, takes a DataFrame (+ context) in,
and returns a DataFrame (+ context) out, so they compose into a simple
pipeline the Orchestrator runs in sequence.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import pandas as pd

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    name: str = "base_agent"

    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def run(self, df: pd.DataFrame, context: dict) -> tuple[pd.DataFrame, dict]:
        """Execute the agent. Returns (possibly-modified df, updated context)."""
        raise NotImplementedError

    def _log_start(self, df: pd.DataFrame):
        logger.info("[%s] starting on %d rows", self.name, len(df))

    def _log_end(self, df: pd.DataFrame):
        logger.info("[%s] done -- %d rows", self.name, len(df))
