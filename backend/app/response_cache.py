"""Exact-match response cache (M11-T4, ADR-046): in-process, LRU + TTL, no new infrastructure.

A hit needs the SAME question text (Unicode-normalized, outer whitespace trimmed), the same mode,
and the same system identity: models, efforts, prompt hashes, corpus hash and gate threshold. A
one-character-different question is a miss, and there is no similar-question matching, ever
(medical correctness). Only complete, validated, answered results are stored: never errors,
refusals or `rerank_skipped` results. The personal-advice screen and the rate limits run before
any lookup. Evaluation runners call the pipelines directly and never import this module.
"""

import hashlib
import json
import threading
import time
import unicodedata
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

from backend.app.agents import critic, planner, reviser
from backend.app.baseline import synthesizer
from backend.app.config import (
    CRITIC_MODEL,
    EVIDENCE_GATE_THRESHOLD,
    LLM_MODEL,
    PLANNER_MODEL,
    REPO_ROOT,
)

DEFAULT_MAX_ENTRIES = 256
DEFAULT_TTL_SECONDS = 3600.0
CORPUS_MANIFEST = REPO_ROOT / "data" / "manifests" / "corpus.json"


def normalize_question(question: str) -> str:
    return unicodedata.normalize("NFC", question).strip()


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def system_identity() -> dict[str, Any]:
    """Everything that, if changed, could change an answer (so it must change the key)."""
    try:
        corpus = hashlib.sha256(CORPUS_MANIFEST.read_bytes()).hexdigest()
    except OSError:
        corpus = "unknown"
    return {
        "models": {"synthesizer": LLM_MODEL, "planner": PLANNER_MODEL, "critic": CRITIC_MODEL},
        "efforts": {
            "baseline_synthesizer": synthesizer.SYNTHESIZER_EFFORT,
            "agent_synthesizer": reviser.AGENT_SYNTHESIZER_EFFORT,
            "planner": planner.PLANNER_EFFORT,
            "critic": critic.CRITIC_EFFORT,
        },
        "prompts": {
            "synthesizer": _sha(synthesizer.SYSTEM_PROMPT + synthesizer.USER_TEMPLATE),
            "planner": _sha(planner.SYSTEM_PROMPT + planner.USER_TEMPLATE),
            "critic": _sha(critic.SYSTEM_PROMPT + critic.USER_TEMPLATE),
            "retry": _sha(reviser.RETRY_TEMPLATE),
        },
        "corpus_sha256": corpus,
        "gate_threshold": EVIDENCE_GATE_THRESHOLD,
    }


def make_key(question: str, mode: str, identity: dict[str, Any] | None = None) -> str:
    payload = {
        "question": normalize_question(question),
        "mode": mode,
        "system": identity if identity is not None else system_identity(),
    }
    return _sha(json.dumps(payload, sort_keys=True, ensure_ascii=True))


def storable(*, abstained: bool, error: str | None, rerank_skipped: bool) -> bool:
    """Only a complete, validated, answered result may be stored."""
    return not abstained and error is None and not rerank_skipped


class ResponseCache:
    def __init__(
        self,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.max_entries, self.ttl, self.clock = max_entries, ttl_seconds, clock
        self._items: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._items.get(key)
            if entry is None:
                return None
            stored_at, value = entry
            if self.clock() - stored_at > self.ttl:
                del self._items[key]
                return None
            self._items.move_to_end(key)
            return value

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._items[key] = (self.clock(), value)
            self._items.move_to_end(key)
            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        return len(self._items)
