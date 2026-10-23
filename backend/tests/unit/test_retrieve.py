"""retrieve() ordering and the rerank failure path, with fakes for Voyage and the DB (M1-T4)."""

from types import SimpleNamespace

import pytest

from backend.app.models.retrieval import Candidate, RetrievalError
from backend.app.retrieval import retrieve as retrieve_module
from backend.app.retrieval.rerank import rerank_candidates
from backend.app.retrieval.retrieve import retrieve


def candidates(n: int) -> list[Candidate]:
    return [
        Candidate(pmid=str(100 + i), passage=f"passage {i}", sections=[], dense_score=1 - i / 100,
                  dense_rank=i)
        for i in range(1, n + 1)
    ]  # fmt: skip


class FakeVoyage:
    """embed() returns a 1024-dim vector; rerank() scores documents by `scores` or raises."""

    def __init__(self, scores=None, rerank_error=None):
        self.scores, self.rerank_error, self.rerank_calls = scores, rerank_error, 0

    def embed(self, texts, **kwargs):
        return SimpleNamespace(embeddings=[[0.0] * 1024 for _ in texts], total_tokens=7)

    def rerank(self, query, documents, **kwargs):
        self.rerank_calls += 1
        if self.rerank_error:
            raise self.rerank_error
        results = [SimpleNamespace(index=i, relevance_score=s) for i, s in enumerate(self.scores)]
        return SimpleNamespace(results=results[::-1], total_tokens=50)


@pytest.fixture
def fake_search(monkeypatch):
    monkeypatch.setattr(retrieve_module, "dense_search", lambda conn, vec, k: candidates(k))


def test_rerank_reorders_and_keeps_dense_fields():
    scores = [0.1, 0.9, 0.5, 0.9]  # tie at 0.9: dense order (index 1 before 3) wins
    out, tokens = rerank_candidates(FakeVoyage(scores), "q", candidates(4))
    assert tokens == 50
    assert [c.pmid for c in out] == ["101", "102", "103", "104"]  # still dense order
    assert [c.rerank_rank for c in out] == [4, 1, 3, 2]
    assert [c.dense_rank for c in out] == [1, 2, 3, 4]


def test_rerank_rejects_incomplete_results():
    client = FakeVoyage([0.5])
    with pytest.raises(RuntimeError):
        rerank_candidates(client, "q", candidates(2))


def test_retrieve_returns_reranked_top_k(fake_search):
    scores = [i / 20 for i in range(20)]  # last dense candidate scores highest
    result = retrieve("q", conn=object(), client=FakeVoyage(scores))
    assert not result.rerank_skipped and result.rerank_error is None
    assert [c.pmid for c in result.evidence] == ["120", "119", "118", "117", "116"]
    assert [c.rerank_rank for c in result.evidence] == [1, 2, 3, 4, 5]
    assert len(result.candidates) == 20
    assert [u.model for u in result.usage] == ["voyage-4", "rerank-3"]
    assert set(result.timings_ms) == {"embed", "search", "rerank", "total"}


def test_rerank_failure_falls_back_to_dense_order_and_is_flagged(fake_search, monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "pa-secret-test-key")
    error = RuntimeError("upstream 503 for key pa-secret-test-key")
    result = retrieve("q", conn=object(), client=FakeVoyage(rerank_error=error))
    assert result.rerank_skipped
    assert result.rerank_error.startswith("RuntimeError: upstream 503")
    assert "pa-secret-test-key" not in result.rerank_error  # errors are redacted
    assert [c.pmid for c in result.evidence] == ["101", "102", "103", "104", "105"]
    assert all(c.rerank_score is None for c in result.candidates)
    assert [u.model for u in result.usage] == ["voyage-4"]  # no rerank tokens billed


def test_embedding_failure_raises(fake_search):
    class Broken(FakeVoyage):
        def embed(self, texts, **kwargs):
            raise ConnectionError("voyage down")

    with pytest.raises(RetrievalError, match="ConnectionError: voyage down"):
        retrieve("q", conn=object(), client=Broken())
