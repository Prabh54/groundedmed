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

# Models (ADR-019).
EMBEDDING_MODEL = "voyage-4"
EMBEDDING_DIMS = 1024
RERANK_MODEL = "rerank-3"

# USD per million tokens, checked 2026-10-21 (ADR-019). Free allocations are not subtracted here.
PRICE_PER_M_TOKENS = {
    EMBEDDING_MODEL: 0.06,
    RERANK_MODEL: 0.05,
}


SECRET_ENV_VARS = (
    "DATABASE_URL",
    "DATABASE_ADMIN_URL",
    "VOYAGE_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
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
