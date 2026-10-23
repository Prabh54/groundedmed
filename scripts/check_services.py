"""Connectivity check (M1-T0, M2-T0): Voyage key, Supabase admin connection, Anthropic key.

Run: uv run python -m scripts.check_services
Prints OK/FAIL per service and never prints keys, URLs or passwords. Costs a few Voyage tokens
and one tiny Claude call (well under $0.01).
"""

import sys

import anthropic
import voyageai

from backend.app.config import (
    EMBEDDING_DIMS,
    EMBEDDING_MODEL,
    LLM_MODEL,
    RERANK_MODEL,
    env,
    redact,
)
from backend.app.retrieval.vector_search import open_connection


def check_voyage() -> str:
    client = voyageai.Client(api_key=env("VOYAGE_API_KEY"), max_retries=1)
    emb = client.embed(["connectivity check"], model=EMBEDDING_MODEL, input_type="query")
    dims = len(emb.embeddings[0])
    if dims != EMBEDDING_DIMS:
        raise RuntimeError(f"{EMBEDDING_MODEL} returned {dims} dims, expected {EMBEDDING_DIMS}")
    rr = client.rerank("connectivity check", ["a short document"], model=RERANK_MODEL, top_k=1)
    tokens = emb.total_tokens + rr.total_tokens
    return f"{EMBEDDING_MODEL} {dims} dims, {RERANK_MODEL} ok ({tokens} tokens)"


def check_database() -> str:
    with open_connection(env("DATABASE_ADMIN_URL")) as conn:
        major = conn.execute("show server_version").fetchone()[0].split(".")[0]
        row = conn.execute(
            "select default_version from pg_available_extensions where name = 'vector'"
        ).fetchone()
    if row is None:
        raise RuntimeError("pgvector extension is not available on this database")
    return f"Postgres {major}, pgvector {row[0]} available"


def check_anthropic() -> str:
    client = anthropic.Anthropic(api_key=env("ANTHROPIC_API_KEY"), max_retries=1, timeout=60)
    msg = client.messages.create(
        model=LLM_MODEL,
        max_tokens=64,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": "Reply with the single word: ok"}],
    )
    if msg.stop_reason == "refusal":
        raise RuntimeError("test call was refused")
    return f"{msg.model} answered ({msg.usage.input_tokens} in / {msg.usage.output_tokens} out)"


def main() -> int:
    failed = False
    checks = [
        ("Voyage", check_voyage),
        ("Supabase", check_database),
        ("Anthropic", check_anthropic),
    ]
    for name, check in checks:
        try:
            print(f"{name}: OK - {check()}")
        except Exception as exc:  # driver errors can echo connection strings, so always redact
            failed = True
            msg = redact(str(exc)).splitlines()[0] if str(exc) else ""
            print(f"{name}: FAIL - {type(exc).__name__}: {msg[:200]}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
