"""Thin wrapper around the Anthropic API with:
  - on-disk response caching, keyed by (model, prompt) hash -- so re-running
    the pipeline never re-bills for a prompt it's already answered.
  - simple retry/backoff.
  - both synchronous (`complete`) and Message Batches API (`batch_complete`)
    paths, since the cleaning agent's volume (thousands of unique category
    strings) benefits from the ~50% batch discount and async processing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None


class LLMClient:
    def __init__(self, config: dict, cache_dir: str | Path):
        self.config = config.get("llm", {})
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY not set. Copy .env.example to .env and fill it in."
            )
        if anthropic is None:
            raise ImportError("Run `pip install anthropic` first.")

        self.client = anthropic.Anthropic(api_key=api_key)
        self.max_retries = self.config.get("max_retries", 5)
        self.max_tokens = self.config.get("max_output_tokens", 1024)
        self.temperature = self.config.get("temperature", 0.2)

    # ------------------------------------------------------------------ #
    # caching
    # ------------------------------------------------------------------ #
    def _cache_key(self, model: str, system: str, prompt: str) -> str:
        raw = f"{model}::{system}::{prompt}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _read_cache(self, key: str) -> str | None:
        path = self._cache_path(key)
        if path.exists():
            return json.loads(path.read_text())["response"]
        return None

    def _write_cache(self, key: str, response: str) -> None:
        self._cache_path(key).write_text(json.dumps({"response": response}))

    # ------------------------------------------------------------------ #
    # synchronous single call (used for low-volume agents: cluster naming,
    # marketing report generation)
    # ------------------------------------------------------------------ #
    def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.config.get("reasoning_model", "claude-sonnet-5")
        key = self._cache_key(model, system, prompt)
        cached = self._read_cache(key)
        if cached is not None:
            return cached

        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.messages.create(
                    model=model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system=system,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = "".join(
                    block.text for block in resp.content if block.type == "text"
                )
                self._write_cache(key, text)
                return text
            except Exception as e:  # noqa: BLE001
                last_err = e
                wait = 2 ** attempt
                logger.warning("LLM call failed (attempt %d): %s -- retrying in %ds",
                                attempt + 1, e, wait)
                time.sleep(wait)
        raise RuntimeError(f"LLM call failed after {self.max_retries} attempts") from last_err

    # ------------------------------------------------------------------ #
    # batch API (used for the high-volume bulk cleaning agent)
    # ------------------------------------------------------------------ #
    def batch_complete(
        self, prompts: list[str], system: str = "", model: str | None = None,
        custom_ids: list[str] | None = None, poll_interval: int = 10,
    ) -> dict[str, str]:
        """Submits all prompts as one Message Batch, polls until done, returns
        {custom_id: response_text}. Uncached prompts only -- cached ones are
        filled in without hitting the API at all.
        """
        model = model or self.config.get("bulk_model", "claude-haiku-4-5-20251001")
        custom_ids = custom_ids or [str(i) for i in range(len(prompts))]

        results: dict[str, str] = {}
        requests = []
        for cid, prompt in zip(custom_ids, prompts):
            key = self._cache_key(model, system, prompt)
            cached = self._read_cache(key)
            if cached is not None:
                results[cid] = cached
                continue
            requests.append(
                {
                    "custom_id": cid,
                    "params": {
                        "model": model,
                        "max_tokens": self.max_tokens,
                        "temperature": self.temperature,
                        "system": system,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                }
            )

        if not requests:
            logger.info("batch_complete: all %d prompts served from cache", len(prompts))
            return results

        logger.info("Submitting Message Batch with %d requests (%d served from cache)",
                    len(requests), len(prompts) - len(requests))
        batch = self.client.messages.batches.create(requests=requests)

        while True:
            batch = self.client.messages.batches.retrieve(batch.id)
            if batch.processing_status == "ended":
                break
            time.sleep(poll_interval)

        id_to_prompt = {r["custom_id"]: r["params"]["messages"][0]["content"] for r in requests}
        for entry in self.client.messages.batches.results(batch.id):
            cid = entry.custom_id
            if entry.result.type == "succeeded":
                text = "".join(
                    b.text for b in entry.result.message.content if b.type == "text"
                )
                results[cid] = text
                key = self._cache_key(model, system, id_to_prompt[cid])
                self._write_cache(key, text)
            else:
                logger.warning("Batch request %s failed: %s", cid, entry.result.type)
                results[cid] = ""

        return results
