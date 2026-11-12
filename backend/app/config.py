"""Settings, model IDs and prices (ARCHITECTURE §0, §10, §12).

Secrets come from environment variables; a local `.env` (never committed) is loaded if present.
"""

import os
import re
from pathlib import Path
from urllib.parse import unquote

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

# Models.
EMBEDDING_MODEL = "voyage-4"
EMBEDDING_DIMS = 1024
RERANK_MODEL = "rerank-3"
LLM_MODEL = "claude-sonnet-5-5"  # every LLM call in the measured pipelines
# The API's personal-advice screen only, outside the benchmark.
ADVICE_MODEL = "claude-haiku-5-5"

# USD per million tokens, checked 2026-10-21. Free allocations are not subtracted here.
PRICE_PER_M_TOKENS = {
    EMBEDDING_MODEL: 0.06,
    RERANK_MODEL: 0.05,
}
# Anthropic list prices, USD per million tokens. Thinking tokens bill as output.
# Haiku 5.5: prompts up to 100,000 tokens (the screen's are ~600), checked 2026-11-02.
# `cache_read` is 0.1x and `cache_write` (5-minute entries) 1.25x the input price
# (M11, checked 2026-11-11).
LLM_PRICE_PER_M_TOKENS = {
    LLM_MODEL: {"input": 2.00, "output": 10.00, "cache_read": 0.20, "cache_write": 2.50},
    ADVICE_MODEL: {"input": 0.10, "output": 0.50, "cache_read": 0.01, "cache_write": 0.125},
}

# Per-stage model routing and prompt caching (M11, ADR-046). Defaults reproduce v1.1.0 exactly:
# every stage on LLM_MODEL, no caching. Env overrides exist so an evaluation candidate is one
# process setting, not a code edit; the evaluation runner records them in run_config.json.
_STAGES = ("planner", "synthesizer", "critic")


def _stage_model(var: str) -> str:
    value = os.environ.get(var, "").strip() or LLM_MODEL
    if value not in LLM_PRICE_PER_M_TOKENS:
        raise RuntimeError(f"{var} must be one of {sorted(LLM_PRICE_PER_M_TOKENS)}")
    return value


def _cache_stages() -> frozenset[str]:
    names = frozenset(
        s.strip() for s in os.environ.get("PROMPT_CACHE_STAGES", "").split(",") if s.strip()
    )
    if not names <= set(_STAGES):
        raise RuntimeError(f"PROMPT_CACHE_STAGES must be a comma list from {_STAGES}")
    return names


PLANNER_MODEL = _stage_model("PLANNER_MODEL")
CRITIC_MODEL = _stage_model("CRITIC_MODEL")
PROMPT_CACHE_STAGES = _cache_stages()


SECRET_ENV_VARS = (
    "DATABASE_URL",
    "DATABASE_ADMIN_URL",
    "VOYAGE_API_KEY",
    "ANTHROPIC_API_KEY",
    "EVALUATOR_API_KEY",
)
_URL_PASSWORD = re.compile(r"(://[^:/@\s]*:)[^@\s]+@")
_KEYWORD_PASSWORD = re.compile(r"(password\s*=\s*)('[^']*'|\S+)", re.IGNORECASE)


def redact(text: str) -> str:
    """Remove secrets from text before it is printed or logged.

    Driver errors can echo a whole connection string, so this masks every secret env value, every
    '://user:password@' and 'password=...' password, and (for URLs) the password on its own, both
    as written and percent-decoded.
    """
    for name in SECRET_ENV_VARS:
        value = os.environ.get(name, "").strip()
        if not value:
            continue
        text = text.replace(value, f"<{name}>")
        match = _URL_PASSWORD.search(value)
        if match:
            password = value[match.end(1) : match.end() - 1]
            for form in {password, unquote(password)}:
                text = text.replace(form, "***")
    text = _KEYWORD_PASSWORD.sub(r"\1***", text)
    return _URL_PASSWORD.sub(r"\1***@", text)


def env(name: str) -> str:
    """Return a required environment variable, failing with its name only (never its value)."""
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is not set; add it to the local .env file (see .env.example)")
    return value


# Evidence gate (ARCHITECTURE §7): abstain when the top rerank score is below this.
# Chosen on dev data only by eval/runners/gate_calibration.py; frozen before any test run.
EVIDENCE_GATE_THRESHOLD = 0.72  # top rerank-3 score
EVIDENCE_GATE_DENSE_THRESHOLD = 0.62  # top cosine score, used only when reranking failed
EVIDENCE_GATE_DEV_RUN = "dev-gate-20261023-030256-4455e98"

# API spend protection. Values come from the environment on the backend host.
PER_IP_QUERIES_PER_HOUR = 10
DEFAULT_DAILY_QUERY_CAP = 100
LOCAL_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]
