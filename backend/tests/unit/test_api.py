"""API schemas and error paths with fakes for the pipeline and DB (PLAN.md M2-T4)."""

import pytest
from fastapi.testclient import TestClient

from backend.app import main
from backend.app.guardrails.rate_limit import RateLimiter
from backend.app.models.answer import Claim, FinalAnswer, ValidationReport
from backend.app.models.retrieval import Candidate, RetrievalError, Section


def evidence():
    return [
        Candidate(pmid=str(100 + i), passage=f"passage {i}",
                  sections=[Section(label="RESULTS", text=f"passage {i}")],
                  dense_score=0.8, dense_rank=i, rerank_score=0.9 - i / 10, rerank_rank=i)
        for i in range(1, 6)
    ]  # fmt: skip


def final_answer(**overrides) -> FinalAnswer:
    fields = dict(
        question="Does X help?", verdict="yes", answer="X helped.",
        claims=[Claim(id="c1", text="X helped.", citation_ids=["PMID:101"])],
        uncertainty="Small study.", abstained=False, abstain_reason=None, evidence=evidence(),
        rerank_skipped=False, gate_passed=True, top_rerank_score=0.8,
        validation=ValidationReport(), timing_ms={"retrieval": 400, "synthesis": 7000,
                                                  "total": 7400},
        usage=[], estimated_cost_usd=0.015, prompt_versions={"synthesizer": "v1"},
    )  # fmt: skip
    return FinalAnswer(**{**fields, **overrides})


class FakeConn:
    def __init__(self, fail=False):
        self.fail = fail

    def __enter__(self):
        if self.fail:
            raise RetrievalError("database connection failed: postgresql://u:secret@host/db")
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, *args):
        return self


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "connect", lambda: FakeConn())
    monkeypatch.setattr(main, "clients", lambda: (object(), object()))
    monkeypatch.setattr(main, "answer_baseline", lambda q, **kw: final_answer(question=q))
    monkeypatch.setattr(main, "limiter", RateLimiter(per_ip_per_hour=100, daily_cap=100))
    return TestClient(main.app)


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok" and r.json()["db"] == "ok" and r.json()["version"]


def test_health_503_when_db_unreachable(client, monkeypatch):
    monkeypatch.setattr(main, "connect", lambda: FakeConn(fail=True))
    r = client.get("/health")
    assert r.status_code == 503 and r.json()["db"] == "unreachable"


def test_query_returns_cited_answer_schema(client):
    r = client.post("/query", json={"question": "Does X help?", "mode": "baseline"})
    assert r.status_code == 200
    body = r.json()
    assert body["verdict"] == "yes" and body["mode"] == "baseline"
    assert body["claims"][0]["citation_ids"] == ["PMID:101"]
    assert [c["id"] for c in body["citations"]] == [f"PMID:{100 + i}" for i in range(1, 6)]
    assert body["citations"][0]["pubmed_url"] == "https://pubmed.ncbi.nlm.nih.gov/101/"
    assert {c for cl in body["claims"] for c in cl["citation_ids"]} <= {
        c["id"] for c in body["citations"]
    }
    assert [s["name"] for s in body["stages"]] == ["retriever", "synthesizer"]
    assert set(body["timing_ms"]) == {"planner", "retrieval", "synthesis", "critic", "total"}
    assert body["prompt_versions"] == {"synthesizer": "v1"}


def test_weak_evidence_abstention_is_200_not_error(client, monkeypatch):
    weak = final_answer(verdict="insufficient_evidence", claims=[], abstained=True,
                        abstain_reason="weak_evidence", gate_passed=False)  # fmt: skip
    monkeypatch.setattr(main, "answer_baseline", lambda q, **kw: weak)
    r = client.post("/query", json={"question": "Sourdough recipe?"})
    assert r.status_code == 200
    assert r.json()["abstained"] and r.json()["abstain_reason"] == "weak_evidence"
    assert r.json()["stages"][1]["status"] == "skipped"


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "abc"},  # too short
        {"question": "x" * 501},  # too long
        {"question": "Does X help?", "mode": "agent"},  # not available until M3
        {"question": "Does X help?", "mode": "other"},
        {},
    ],
)
def test_invalid_input_is_422(client, payload):
    assert client.post("/query", json=payload).status_code == 422


def test_per_ip_limit_returns_429(client, monkeypatch):
    monkeypatch.setattr(main, "limiter", RateLimiter(per_ip_per_hour=2, daily_cap=100))
    codes = [client.post("/query", json={"question": "Does X help?"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_daily_cap_returns_429_with_friendly_message(client, monkeypatch):
    monkeypatch.setattr(main, "limiter", RateLimiter(per_ip_per_hour=100, daily_cap=1))
    client.post("/query", json={"question": "Does X help?"})
    r = client.post("/query", json={"question": "Does X help?"})
    assert r.status_code == 429 and r.json()["detail"] == main.LIMIT_MESSAGE


def test_retrieval_failure_is_503_without_leaking_secrets(client, monkeypatch, capsys):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:secret@host/db")
    monkeypatch.setattr(main, "connect", lambda: FakeConn(fail=True))
    r = client.post("/query", json={"question": "Does X help?"})
    assert r.status_code == 503 and "secret" not in r.text
    assert "secret" not in capsys.readouterr().out


def test_llm_failure_is_503_never_partial_answer(client, monkeypatch):
    failed = final_answer(verdict="insufficient_evidence", claims=[], abstained=True,
                          abstain_reason="error", error="APIConnectionError")  # fmt: skip
    monkeypatch.setattr(main, "answer_baseline", lambda q, **kw: failed)
    r = client.post("/query", json={"question": "Does X help?"})
    assert r.status_code == 503 and "claims" not in r.json()


def test_source_found(client, monkeypatch):
    doc = {"pmid": "101", "passage": "p", "sections": [{"label": "RESULTS", "text": "p"}],
           "meshes": ["Humans"]}  # fmt: skip
    monkeypatch.setattr(main, "get_document", lambda conn, pmid: doc if pmid == "101" else None)
    r = client.get("/sources/101")
    assert r.status_code == 200
    assert r.json()["id"] == "PMID:101" and r.json()["meshes"] == ["Humans"]
    assert client.get("/sources/102").status_code == 404


@pytest.mark.parametrize("pmid", ["abc", "1; drop table documents", "1234567890", "1e5"])
def test_source_rejects_non_numeric_pmid(client, monkeypatch, pmid):
    monkeypatch.setattr(main, "get_document", lambda conn, p: pytest.fail("reached the database"))
    assert client.get(f"/sources/{pmid}").status_code == 422


def test_cors_allows_localhost_and_refuses_other_origins(client):
    ok = client.options("/query", headers={"Origin": "http://localhost:3000",
                                           "Access-Control-Request-Method": "POST"})  # fmt: skip
    bad = client.options("/query", headers={"Origin": "https://evil.example",
                                            "Access-Control-Request-Method": "POST"})  # fmt: skip
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:3000"
    assert "access-control-allow-origin" not in bad.headers


def test_rate_limiter_window_and_daily_reset():
    now = [1_700_000_000.0]
    rl = RateLimiter(per_ip_per_hour=1, daily_cap=3, clock=lambda: now[0])
    assert rl.check("a") is None and rl.check("a") == "per_ip"
    now[0] += 3601
    assert rl.check("a") is None  # hour window passed
    assert rl.check("b") is None and rl.check("c") == "daily_cap"
    now[0] += 86400
    assert rl.check("c") is None  # new UTC day


def test_whitespace_only_question_is_422(client):
    assert client.post("/query", json={"question": "      \n\t   "}).status_code == 422


def test_unexpected_bug_is_500_not_disguised_as_503(monkeypatch):
    def broken(q, **kw):
        raise KeyError("bug in pipeline")

    monkeypatch.setattr(main, "connect", lambda: FakeConn())
    monkeypatch.setattr(main, "clients", lambda: (object(), object()))
    monkeypatch.setattr(main, "answer_baseline", broken)
    monkeypatch.setattr(main, "limiter", RateLimiter(per_ip_per_hour=100, daily_cap=100))
    r = TestClient(main.app, raise_server_exceptions=False).post(
        "/query", json={"question": "Does X help?"}
    )
    assert r.status_code == 500


def test_allowed_origins_parsing():
    assert main.allowed_origins("") == main.LOCAL_ORIGINS
    assert main.allowed_origins(" https://a.vercel.app/ , ,https://b.app") == [
        "https://a.vercel.app",
        "https://b.app",
        *main.LOCAL_ORIGINS,
    ]


@pytest.mark.parametrize("value,expected", [(None, 100), ("", 100), ("abc", 100), ("0", 100),
                                            ("-5", 100), ("25", 25)])  # fmt: skip
def test_daily_cap_parsing(value, expected):
    assert main.daily_cap(value) == expected


def test_rate_limiter_forgets_idle_ips_at_day_change():
    now = [1_700_000_000.0]
    rl = RateLimiter(per_ip_per_hour=5, daily_cap=100, clock=lambda: now[0])
    rl.check("a")
    now[0] += 86400
    rl.check("b")
    assert set(rl._hits) == {"b"}
