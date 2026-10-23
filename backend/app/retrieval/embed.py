"""Voyage embeddings (ARCHITECTURE §3–4, ADR-019)."""

from typing import Literal

import voyageai

from backend.app.config import EMBEDDING_DIMS, EMBEDDING_MODEL, env


def voyage_client(*, max_retries: int = 1, timeout: float = 15) -> voyageai.Client:
    """Query-time defaults are tight so a rerank outage falls back to dense order within ~30 s
    instead of minutes (M1 review); bulk ingestion passes longer limits."""
    return voyageai.Client(api_key=env("VOYAGE_API_KEY"), max_retries=max_retries, timeout=timeout)


def embed_texts(
    client: voyageai.Client, texts: list[str], input_type: Literal["document", "query"]
) -> tuple[list[list[float]], int]:
    """Embed texts with voyage-4; returns (vectors, tokens billed).

    `truncation=False` makes an over-long input fail loudly instead of being silently cut.
    """
    result = client.embed(texts, model=EMBEDDING_MODEL, input_type=input_type, truncation=False)
    vectors = result.embeddings
    if len(vectors) != len(texts) or any(len(v) != EMBEDDING_DIMS for v in vectors):
        raise RuntimeError(f"{EMBEDDING_MODEL} returned an unexpected number or size of vectors")
    return vectors, result.total_tokens
