"""Closed-book control (ADR-010, M4-T3b): question only, verdict only, no retrieval. Fake model."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from eval import closed_book
from eval.runners import run_system


class FakeLLM:
    def __init__(self, text='{"verdict": "no"}', stop_reason="end_turn"):
        self.text, self.stop_reason, self.requests = text, stop_reason, []
        self.messages = SimpleNamespace(create=self.create)

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self.text)],
            stop_reason=self.stop_reason, stop_details=None,
            usage=SimpleNamespace(input_tokens=300, output_tokens=400),
        )  # fmt: skip


@pytest.fixture
def runner(tmp_path, monkeypatch):
    monkeypatch.setattr(run_system, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(run_system, "REPO_ROOT", tmp_path)
    clean = {"git_commit": "d" * 40, "git_dirty": False, "git_diff_sha256": None}
    monkeypatch.setattr(run_system, "git_state", lambda: clean)

    def no_retrieval(*a, **k):
        raise AssertionError("closed-book must not touch retrieval or the database")

    monkeypatch.setattr(run_system, "voyage_client", no_retrieval)
    monkeypatch.setattr(run_system, "connect", no_retrieval)
    llm = FakeLLM()
    monkeypatch.setattr(run_system, "anthropic_client", lambda: llm)

    def run(*extra):
        run_system.main(["--system", "closed_book", "--split", "dev", *extra])
        (d,) = tmp_path.iterdir()
        rows = [json.loads(x) for x in (d / "raw.jsonl").read_text().splitlines()]
        load = lambda n: json.loads((d / n).read_text())  # noqa: E731
        return rows, load("summary.json"), load("run_config.json")

    return run, llm


def test_closed_book_sends_only_the_question_and_records_a_verdict(runner):
    run, llm = runner
    rows, summary, config = run("--limit", "3")
    assert len(llm.requests) == 3
    sent = llm.requests[0]
    assert sent["system"] == closed_book.SYSTEM_PROMPT and "EVIDENCE" not in sent["messages"][0][
        "content"]  # fmt: skip
    assert sent["output_config"]["effort"] == "medium"
    assert sent["output_config"]["format"]["schema"]["properties"]["verdict"]["enum"] == [
        "yes", "no", "maybe"]  # fmt: skip
    assert [r["verdict"] for r in rows] == ["no"] * 3
    assert all(r["correct"] == (r["gold_label"] == "no") for r in rows)
    assert all(r["claims"] == [] and r["evidence_pmids"] == [] and not r["abstained"] for r in rows)
    assert rows[0]["cost_usd"] == pytest.approx((300 * 2 + 400 * 10) / 1e6)
    assert summary["hit_at_1"] == summary["claim_support"] == "not applicable"
    assert summary["coverage"] == 1.0 and summary["complete"] is True
    assert config["closed_book_prompt"] == "v1"
    assert config["closed_book_prompt_sha256"] == hashlib.sha256(
        closed_book.SYSTEM_PROMPT.encode()).hexdigest()  # fmt: skip
    assert "closed_book_prompt_sha256" in run_system.IDENTITY_KEYS


def test_a_refused_call_is_an_error_row_with_its_cost_counted(runner):
    run, llm = runner
    llm.stop_reason = "refusal"
    rows, summary, _ = run("--limit", "2")
    assert [r["abstain_reason"] for r in rows] == ["error", "error"]
    assert not any(r["correct"] for r in rows) and summary["errors"] == 2
    assert rows[0]["cost_usd"] == pytest.approx((300 * 2 + 400 * 10) / 1e6)


def test_closed_book_estimate_is_its_own():
    assert run_system.estimated_cost("closed_book", 200) == pytest.approx(1.0)
