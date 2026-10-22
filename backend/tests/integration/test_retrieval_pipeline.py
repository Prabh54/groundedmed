"""Live embed -> store -> retrieve -> rerank (PLAN.md M1-T4). Needs VOYAGE_API_KEY and DATABASE_URL.

Queries are built from distractor (unlabeled) papers, so no benchmark question is used.
"""

import json
import os

import pytest

from backend.app.retrieval.retrieve import K_CANDIDATES, K_FINAL, retrieve
from scripts.ingest import CORPUS_MANIFEST, load_corpus

pytestmark = pytest.mark.skipif(
    not (os.environ.get("VOYAGE_API_KEY") and os.environ.get("DATABASE_URL")),
    reason="Voyage key or database URL not set",
)


def distractor_queries(n: int) -> list[tuple[str, str]]:
    """(pmid, first sentence of its passage) for the first n distractors in the manifest."""
    manifest = json.loads(CORPUS_MANIFEST.read_text(encoding="utf-8"))
    wanted = [d["pmid"] for d in manifest["documents"] if d["source"] == "pqa_u"][:n]
    passages = {r["pmid"]: r["passage"] for r in load_corpus()}
    return [(pmid, passages[pmid].split(". ")[0]) for pmid in wanted]


@pytest.mark.parametrize("pmid,query", distractor_queries(3))
def test_embed_store_retrieve_rerank(pmid, query):
    result = retrieve(query)
    assert not result.rerank_skipped, result.rerank_error
    assert len(result.candidates) == K_CANDIDATES
    assert len(result.evidence) == K_FINAL
    assert pmid in [c.pmid for c in result.evidence]  # its own opening sentence finds it
    dense = [c.dense_score for c in result.candidates]
    assert dense == sorted(dense, reverse=True)
    rerank = [c.rerank_score for c in result.evidence]
    assert rerank == sorted(rerank, reverse=True)
    assert {c.pmid for c in result.evidence} <= {c.pmid for c in result.candidates}
    assert all(u.tokens > 0 for u in result.usage)
