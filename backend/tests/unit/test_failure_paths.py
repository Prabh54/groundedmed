"""Failure paths end to end, with fakes (final M2 review, should-fix 2): an LLM failure inside the
pipeline becomes a controlled abstention with its cost counted, and an outage on one question does
not stop an eval run."""

import json
from types import SimpleNamespace

from backend.app.baseline import pipeline
from backend.app.models.answer import FinalAnswer, ValidationReport
from backend.app.models.retrieval import Candidate, RetrievalError, RetrievalResult, Section
from eval.runners import run_system


def evidence():
    return [
        Candidate(pmid=str(100 + i), passage="p", sections=[Section(label="RESULTS", text="p")],
                  dense_score=0.8, dense_rank=i, rerank_score=0.95, rerank_rank=i)
        for i in range(1, 6)
    ]  # fmt: skip


def fake_retrieval(question, conn=None, client=None):
    ev = evidence()
    return RetrievalResult(query=question, candidates=ev, evidence=ev, rerank_skipped=False,
                           timings_ms={"total": 1.0}, usage=[])  # fmt: skip


class RefusingLLM:
    """A model that declines: HTTP 200, stop_reason "refusal", tokens billed."""

    def __init__(self):
        self.messages = SimpleNamespace(create=self.create)

    def create(self, **kwargs):
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="I can't help with that.")],
            stop_reason="refusal", stop_details=SimpleNamespace(category="bio"),
            usage=SimpleNamespace(input_tokens=3000, output_tokens=100),
        )  # fmt: skip


def test_llm_failure_in_pipeline_abstains_with_cost_counted(monkeypatch):
    monkeypatch.setattr(pipeline, "retrieve", fake_retrieval)
    answer = pipeline.answer_baseline("Does X help?", llm=RefusingLLM())
    assert answer.abstained and answer.abstain_reason == "error"
    assert answer.verdict == "insufficient_evidence"
    assert answer.claims == [] and answer.limitations == []
    assert answer.answer == pipeline.ERROR_SUMMARY and "declined" in answer.error
    assert answer.estimated_cost_usd == (3000 * 2 + 100 * 10) / 1e6  # the refused call is billed


class FakeConn:
    closed = 0

    def close(self):
        FakeConn.closed += 1


def ok_answer(question: str) -> FinalAnswer:
    return FinalAnswer(
        question=question, verdict="yes", answer="A.", claims=[], limitations=[],
        abstained=False, abstain_reason=None, evidence=evidence(), rerank_skipped=False,
        gate_passed=True, top_rerank_score=0.95, validation=ValidationReport(),
        timing_ms={"retrieval": 1.0, "synthesis": 1.0, "total": 2.0}, usage=[],
        estimated_cost_usd=0.01, prompt_versions={"synthesizer": "v3"},
    )  # fmt: skip


def test_eval_run_survives_an_outage_on_one_question(tmp_path, monkeypatch):
    calls, connects = [], []

    def answer(question, **kwargs):
        calls.append(question)
        if len(calls) == 2:
            raise RetrievalError("ConnectionError: database connection failed")
        return ok_answer(question)

    def connect():
        connects.append(1)
        return FakeConn()

    monkeypatch.setattr(run_system, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(run_system, "REPO_ROOT", tmp_path)  # for the final "Saved to" line
    clean = {"git_commit": "f" * 40, "git_dirty": False, "git_diff_sha256": None}
    monkeypatch.setattr(run_system, "git_state", lambda: clean)
    monkeypatch.setattr(run_system, "voyage_client", lambda **kw: object())
    monkeypatch.setattr(run_system, "anthropic_client", lambda: object())
    monkeypatch.setattr(run_system, "connect", connect)
    monkeypatch.setattr(run_system, "answer_baseline", answer)

    run_system.main(["--system", "baseline", "--split", "dev", "--limit", "3"])

    (run_dir,) = tmp_path.iterdir()
    rows = [json.loads(line) for line in (run_dir / "raw.jsonl").open(encoding="utf-8")]
    assert len(rows) == 3 and len(calls) == 3  # the run continued after the failure
    assert rows[1]["abstain_reason"] == "error" and not rows[1]["correct"]
    assert "database connection failed" in rows[1]["error"]
    assert rows[0]["error"] is None and rows[2]["error"] is None
    assert len(connects) == 2  # reconnected after the failure
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["errors"] == 1 and summary["n_questions"] == 3
