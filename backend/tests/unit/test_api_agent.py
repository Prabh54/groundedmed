"""Agent mode in /query and /query/stream, run through the real graph with a fake model,
fake Voyage and a fake database (no network). Baseline mode is covered unchanged in test_api.py."""

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend.app import main
from backend.app.agents import critic as critic_module
from backend.app.agents import planner as planner_module
from backend.app.agents import retriever as retriever_module
from backend.app.baseline import synthesizer as synth_module
from backend.app.guardrails.rate_limit import RateLimiter
from backend.app.models.retrieval import Candidate, RetrievalError, Section
from backend.app.retrieval import retrieve as baseline_retrieve_module

QUESTION = "Does early mobilization reduce delirium in ICU patients?"

OUTPUTS = {
    "planner": {"subquestions": [{"id": "q1", "query": "mobilization delirium trial"}]},
    "synthesizer": {
        "verdict": "yes",
        "claims": [
            {"id": "c1", "text": "Delirium was shorter.", "citation_ids": ["PMID:101"]},
            {"id": "c2", "text": "Mortality fell by half.", "citation_ids": ["PMID:102"]},
        ],
        "limitations": [{"id": "l1", "text": "Single centre.", "citation_ids": ["PMID:101"]}],
    },
    "critic": {
        "results": [
            {"claim_id": "c1", "supported": True, "citation_valid": True, "action": "keep",
             "reason": "SECRET-REASONING matches the passage"},
            {"claim_id": "c2", "supported": False, "citation_valid": True, "action": "remove",
             "reason": "SECRET-REASONING mortality not reported"},
            {"claim_id": "l1", "supported": True, "citation_valid": True, "action": "keep",
             "reason": "SECRET-REASONING ok"},
        ],
        "answer_supported": True,
    },
}  # fmt: skip


class FakeLLM:
    def __init__(self, fail: str | None = None):
        self.fail, self.calls = fail, []
        self.stage_of = {
            planner_module.SYSTEM_PROMPT: "planner",
            synth_module.SYSTEM_PROMPT: "synthesizer",
            critic_module.SYSTEM_PROMPT: "critic",
        }
        self.messages = SimpleNamespace(create=self.create)

    def create(self, **kwargs):
        stage = self.stage_of[kwargs["system"]]
        self.calls.append(stage)
        refused = stage == self.fail
        text = "no" if refused else json.dumps(OUTPUTS[stage])
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)],
            stop_reason="refusal" if refused else "end_turn", stop_details=None,
            usage=SimpleNamespace(input_tokens=1000, output_tokens=100),
        )  # fmt: skip


class FakeVoyage:
    def __init__(self, score: float = 0.9):
        self.score = score

    def embed(self, texts, **kwargs):
        # vector 0 = the question, 1 = a sub-query (dense_search routes on it)
        return SimpleNamespace(embeddings=[[0.0 if t == QUESTION else 1.0] * 1024 for t in texts],
                               total_tokens=10)  # fmt: skip

    def rerank(self, query, documents, **kwargs):
        results = [SimpleNamespace(index=i, relevance_score=self.score - i / 100)
                   for i in range(len(documents))]  # fmt: skip
        return SimpleNamespace(results=results, total_tokens=100)


class FakeConn:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def dense_search(conn, vector, k):
    pmids = ["101", "102", "103", "104", "105"] if vector[0] == 0 else ["106", "101"]
    return [Candidate(pmid=p, passage=f"passage {p}",
                      sections=[Section(label="RESULTS", text=f"Result of paper {p}.")],
                      dense_score=0.8, dense_rank=i)
            for i, p in enumerate(pmids, start=1)]  # fmt: skip


@pytest.fixture
def setup(monkeypatch):
    fakes = SimpleNamespace(llm=FakeLLM(), voyage=FakeVoyage())
    monkeypatch.setattr(main, "connect", lambda: FakeConn())
    monkeypatch.setattr(main, "clients", lambda: (fakes.voyage, fakes.llm))
    monkeypatch.setattr(retriever_module, "dense_search", dense_search)
    monkeypatch.setattr(baseline_retrieve_module, "dense_search", dense_search)
    monkeypatch.setattr(main, "limiter", RateLimiter(per_ip_per_hour=100, daily_cap=100))
    fakes.client = TestClient(main.app)
    return fakes


def post(fakes, path="/query", mode="agent"):
    return fakes.client.post(path, json={"question": QUESTION, "mode": mode})


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


# --- /query ----------------------------------------------------------------------------------


def test_agent_query_returns_the_checked_answer(setup, capsys):
    r = post(setup)
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "agent" and body["verdict"] == "yes" and not body["abstained"]
    assert [c["id"] for c in body["claims"]] == ["c1"]  # c2 removed by the critic
    assert body["answer"] == "Delirium was shorter."
    assert {c for cl in body["claims"] for c in cl["citation_ids"]} <= {
        c["id"] for c in body["citations"]
    }
    assert [s["name"] for s in body["stages"]] == [
        "planner", "retriever", "gate", "synthesizer", "critic", "validator"
    ]  # fmt: skip
    assert all(s["status"] == "done" for s in body["stages"])
    messages = [s["message"] for s in body["stages"]]
    assert messages[1] == "Retrieved 6 candidate passages; selected 5 evidence passages"
    assert messages[5] == "Kept 1 cited claims; removed 1 unsupported items"
    assert body["prompt_versions"] == {"planner": "v1", "synthesizer": "v3", "critic": "v1",
                                       "advice_check": "v1"}  # fmt: skip
    assert set(body["timing_ms"]) >= {"planner", "retrieval", "synthesis", "critic", "total"}
    assert body["timing_ms"]["total"] >= body["timing_ms"]["planner"]
    assert "SECRET-REASONING" not in r.text  # critic reasons never reach the client

    (line,) = [json.loads(x) for x in capsys.readouterr().out.splitlines() if '"mode"' in x]
    assert line["mode"] == "agent" and line["candidates"] == 6
    assert line["subqueries"] == ["mobilization delirium trial"]
    assert line["critic_removed"] == ["c2"] and line["planner_added_evidence"] >= 0
    assert line["cost_usd"] > 0


def test_agent_low_evidence_refusal_is_200_with_skipped_stages(setup):
    setup.voyage.score = 0.3
    body = post(setup).json()
    assert body["abstained"] and body["abstain_reason"] == "weak_evidence"
    status = {s["name"]: s["status"] for s in body["stages"]}
    assert status["synthesizer"] == status["critic"] == "skipped"
    validator = next(s for s in body["stages"] if s["name"] == "validator")
    assert validator["message"] == "No answer: the evidence scored below the relevance threshold"
    assert setup.llm.calls == ["planner"]


@pytest.mark.parametrize("stage", ["planner", "synthesizer", "critic"])
def test_agent_provider_failure_is_a_controlled_503(setup, stage):
    setup.llm.fail = stage
    r = post(setup)
    assert r.status_code == 503 and r.json()["detail"] == main.UNAVAILABLE_MESSAGE
    assert "claims" not in r.json()  # never a partial answer


def test_agent_retrieval_failure_is_503_and_logs_the_planner_cost(setup, monkeypatch, capsys):
    def broken(conn, vector, k):
        raise RetrievalError("db down")

    monkeypatch.setattr(retriever_module, "dense_search", broken)
    assert post(setup).status_code == 503
    (line,) = [json.loads(x) for x in capsys.readouterr().out.splitlines() if '"error"' in x]
    assert line["stage"] == "retrieval" and line["cost_usd"] > 0  # the planner call was paid
    # the advice screen (free in unit tests) ran first, then the planner
    assert [u["model"] for u in line["usage"]] == ["claude-haiku-5-5", "claude-sonnet-5-5"]


def test_baseline_mode_is_still_the_default_and_unchanged(setup):
    r = setup.client.post("/query", json={"question": QUESTION})
    body = r.json()
    assert r.status_code == 200 and body["mode"] == "baseline"
    assert setup.llm.calls == ["synthesizer"]  # no planner, no critic
    assert [c["id"] for c in body["claims"]] == ["c1", "c2"]  # no critic removal in baseline
    assert [s["name"] for s in body["stages"]] == ["retriever", "synthesizer"]
    assert body["prompt_versions"] == {"synthesizer": "v3", "advice_check": "v1"}


# --- /query/stream ---------------------------------------------------------------------------


def test_stream_sends_stage_events_then_the_result(setup):
    r = post(setup, "/query/stream")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(r.text)
    kinds = [k for k, _ in events]
    assert kinds[-1] == "result" and set(kinds[:-1]) == {"stage"}
    stages = [(d["name"], d["status"]) for k, d in events if k == "stage"]
    expected = [(n, s) for n in ("planner", "retriever", "gate", "synthesizer", "critic",
                                 "validator") for s in ("started", "done")]  # fmt: skip
    assert stages == expected
    messages = [d["message"] for k, d in events if k == "stage"]
    assert "Planning evidence needs" in messages
    assert "Checking 3 claims and limitations" in messages

    def stable(body):  # drop what differs between two runs: request ID and timings
        drop = {"request_id", "timing_ms"}
        stages = [{k: v for k, v in st.items() if k != "ms"} for st in body["stages"]]
        return {k: v for k, v in body.items() if k not in drop} | {"stages": stages}

    main.response_cache.clear()  # the stream stored its answer; /query must run for real here
    assert stable(events[-1][1]) == stable(post(setup).json())  # same body as /query
    assert "SECRET-REASONING" not in r.text


def test_stream_provider_failure_ends_with_an_error_event(setup):
    setup.llm.fail = "critic"
    events = parse_sse(post(setup, "/query/stream").text)
    assert events[-1] == ("error", {"message": main.UNAVAILABLE_MESSAGE})
    assert ("critic", "failed") in [(d["name"], d["status"]) for k, d in events if k == "stage"]
    stages = [d for k, d in events if k == "stage"]
    validator_done = [
        d["message"] for d in stages if (d["name"], d["status"]) == ("validator", "done")
    ]
    assert validator_done == [
        "No answer: a processing step failed"
    ]  # not "evidence did not support"
    assert all(k != "result" for k, _ in events)


def test_stream_bug_ends_with_a_plain_error_event(setup, monkeypatch, capsys):
    def bug(*args, **kwargs):
        raise KeyError("internal detail")

    monkeypatch.setattr(main, "answer_agent", bug)
    events = parse_sse(post(setup, "/query/stream").text)
    assert events == [("error", {"message": main.INTERNAL_MESSAGE})]
    assert "internal detail" in capsys.readouterr().out  # logged server-side only


def test_stream_baseline_mode_reports_its_stages(setup):
    events = parse_sse(post(setup, "/query/stream", mode="baseline").text)
    assert [(k, d.get("name")) for k, d in events] == [
        ("stage", "retriever"), ("stage", "synthesizer"), ("result", None)
    ]  # fmt: skip
    assert events[-1][1]["mode"] == "baseline" and setup.llm.calls == ["synthesizer"]


def test_stream_rate_limit_is_http_429_before_streaming(setup, monkeypatch):
    monkeypatch.setattr(main, "limiter", RateLimiter(per_ip_per_hour=1, daily_cap=100))
    post(setup, "/query/stream")
    assert post(setup, "/query/stream").status_code == 429


# --- exact-match response cache (M11-T4, ADR-046) --------------------------------------------


def test_identical_question_is_answered_from_the_cache_without_any_model_call(setup):
    first = post(setup).json()
    calls_after_first = list(setup.llm.calls)
    second = post(setup).json()
    assert setup.llm.calls == calls_after_first  # no planner, synthesizer or critic call
    assert second["request_id"] != first["request_id"]
    for same in ("verdict", "answer", "claims", "citations", "limitations", "prompt_versions"):
        assert second[same] == first[same]
    assert second["estimated_cost_usd"] == 0.0 < first["estimated_cost_usd"]
    assert set(second["timing_ms"]) == {"cache_lookup", "total"}
    assert all(s["status"] == "done" and s["ms"] == 0.0 for s in second["stages"])


def test_one_character_different_question_misses_the_cache(setup):
    post(setup)
    calls = len(setup.llm.calls)
    r = setup.client.post("/query", json={"question": QUESTION + " ", "mode": "agent"})
    assert len(setup.llm.calls) == calls  # outer whitespace is trimmed: still a hit
    r = setup.client.post("/query", json={"question": QUESTION[:-1], "mode": "agent"})
    assert r.status_code in (200, 503) and len(setup.llm.calls) > calls  # a miss: models ran


def test_mode_is_part_of_the_cache_key(setup):
    post(setup, mode="agent")
    agent_calls = list(setup.llm.calls)
    post(setup, mode="baseline")
    assert len(setup.llm.calls) > len(agent_calls)  # the baseline did not reuse the agent's answer


def test_refusals_are_never_stored(setup):
    setup.voyage.score = 0.3  # weak evidence: abstains
    assert post(setup).json()["abstain_reason"] == "weak_evidence"
    post(setup)
    assert setup.llm.calls == ["planner", "planner"]  # the second one searched again
    assert len(main.response_cache) == 0


def test_errors_are_never_stored(setup):
    setup.llm.fail = "critic"
    assert post(setup).status_code == 503
    assert len(main.response_cache) == 0
    setup.llm.fail = None
    assert post(setup).status_code == 200  # a real answer after the error, not a stored failure


def test_cached_answer_also_works_on_the_stream(setup):
    post(setup)
    calls = list(setup.llm.calls)
    r = post(setup, "/query/stream")
    events = parse_sse(r.text)
    assert setup.llm.calls == calls
    assert events[-1][0] == "result" and events[-1][1]["verdict"] == "yes"


def test_evaluation_and_pipelines_never_touch_the_answer_cache():
    import pathlib

    root = pathlib.Path(main.__file__).resolve().parents[2]
    for rel in ("backend/app/agents", "backend/app/baseline", "eval/runners", "eval"):
        for path in (root / rel).glob("*.py"):
            assert "response_cache" not in path.read_text(), path
