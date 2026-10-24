"""M3-T8: the eval runner's agent system and its hard cost cap, over the real graph with a fake
model, fake Voyage and a fake database (no network)."""

import json
from types import SimpleNamespace

import pytest

from backend.app.agents import critic as critic_module
from backend.app.agents import planner as planner_module
from backend.app.agents import retriever as retriever_module
from backend.app.baseline import synthesizer as synth_module
from backend.app.models.retrieval import Candidate, Section
from eval.runners import run_system

OUTPUTS = {
    "planner": {"subquestions": [{"id": "q1", "query": "SUB query"}]},
    "synthesizer": {
        "verdict": "yes",
        "claims": [
            {"id": "c1", "text": "Finding one.", "citation_ids": ["PMID:101"]},
            {"id": "c2", "text": "Finding two.", "citation_ids": ["PMID:102"]},
        ],
        "limitations": [],
    },
    "critic": {
        "results": [
            {"claim_id": "c1", "supported": True, "citation_valid": True, "action": "keep",
             "reason": "ok"},
            {"claim_id": "c2", "supported": False, "citation_valid": True, "action": "remove",
             "reason": "not in passage"},
        ],
        "answer_supported": True,
    },
}  # fmt: skip


class FakeLLM:
    def __init__(self):
        self.stage_of = {
            planner_module.SYSTEM_PROMPT: "planner",
            synth_module.SYSTEM_PROMPT: "synthesizer",
            critic_module.SYSTEM_PROMPT: "critic",
        }
        self.messages = SimpleNamespace(create=self.create)

    def create(self, **kwargs):
        return SimpleNamespace(
            content=[SimpleNamespace(type="text",
                                     text=json.dumps(OUTPUTS[self.stage_of[kwargs["system"]]]))],
            stop_reason="end_turn", stop_details=None,
            usage=SimpleNamespace(input_tokens=1000, output_tokens=100),
        )  # fmt: skip


class FakeVoyage:
    def embed(self, texts, **kwargs):
        return SimpleNamespace(embeddings=[[1.0 if t.startswith("SUB") else 0.0] * 1024
                                           for t in texts], total_tokens=10)  # fmt: skip

    def rerank(self, query, documents, **kwargs):
        # the sub-query-only paper 200 scores highest -> planner_added_evidence 1
        results = [SimpleNamespace(index=i, relevance_score=0.99 if "200" in d else 0.9 - i / 100)
                   for i, d in enumerate(documents)]  # fmt: skip
        return SimpleNamespace(results=results, total_tokens=100)


class FakeConn:
    def close(self):
        pass


def dense_search(conn, vector, k):
    pmids = [str(p) for p in range(101, 121)] if vector[0] == 0 else ["200", "999"]
    return [Candidate(pmid=p, passage=f"passage {p}",
                      sections=[Section(label="RESULTS", text=f"Result of {p}.")],
                      dense_score=0.8, dense_rank=i)
            for i, p in enumerate(pmids, start=1)]  # fmt: skip


@pytest.fixture
def runner(tmp_path, monkeypatch):
    monkeypatch.setattr(run_system, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(run_system, "REPO_ROOT", tmp_path)
    clean = {"git_commit": "f" * 40, "git_dirty": False, "git_diff_sha256": None}
    monkeypatch.setattr(run_system, "git_state", lambda: clean)
    monkeypatch.setattr(run_system, "voyage_client", lambda **kw: FakeVoyage())
    monkeypatch.setattr(run_system, "anthropic_client", lambda: FakeLLM())
    monkeypatch.setattr(run_system, "connect", lambda: FakeConn())
    monkeypatch.setattr(retriever_module, "dense_search", dense_search)

    def run(*extra):
        run_system.main(["--system", "agent", "--split", "dev", *extra])
        (run_dir,) = tmp_path.iterdir()
        rows = [json.loads(line) for line in (run_dir / "raw.jsonl").open(encoding="utf-8")]
        load = lambda name: json.loads((run_dir / name).read_text(encoding="utf-8"))  # noqa: E731
        return rows, load("summary.json"), load("run_config.json")

    return run


def test_agent_run_stores_agent_diagnostics(runner):
    rows, summary, config = runner("--limit", "3")
    assert len(rows) == 3 and summary["system"] == "agent" and summary["errors"] == 0
    row = rows[0]
    assert row["subqueries"] == ["SUB query"] and row["n_candidates"] == 22
    assert row["planner_added_evidence"] == 1 and row["evidence_pmids"][0] == "200"
    assert row["critic_removed"] == ["c2"] and [c["id"] for c in row["claims"]] == ["c1"]
    assert row["prompt_versions"] == {"planner": "v1", "synthesizer": "v3", "critic": "v1"}
    assert row["gold_in_question_top20"] is False and row["gold_in_pool"] is False
    assert row["correct"] == (row["verdict"] == row["gold_label"])

    agent = summary["agent"]
    assert agent["planner_added_evidence"] == {"mean": 1.0, "share_questions_gt0": 1.0,
                                               "questions_gt0": 3,
                                               "gold_found_only_via_subqueries": 0}  # fmt: skip
    assert agent["critic"]["items_checked"] == 6 and agent["critic"]["items_removed"] == 3
    assert agent["critic"]["answers_abstained_by_critic"] == 0
    assert agent["mean_candidates"] == 22
    assert config["planner_prompt"] == "v1" and config["critic_prompt"] == "v1"
    assert config["planner_effort"] == config["critic_effort"] == "low"
    assert len(config["critic_prompt_sha256"]) == 64


def test_cost_cap_stops_before_exceeding(runner):
    # each fake question costs 3 x $0.003 + Voyage list price; allowance $0.05 per question
    rows, summary, config = runner("--limit", "10", "--max-cost", "0.07")
    assert len(rows) == 3 and summary["n_questions"] == 3
    assert summary["stopped_at_cost_cap"] is True and config["max_cost_usd"] == 0.07
    assert sum(r["cost_usd"] for r in rows) <= 0.07


def test_no_cap_hit_is_reported_as_such(runner):
    _, summary, _ = runner("--limit", "2", "--max-cost", "5")
    assert summary["stopped_at_cost_cap"] is False and summary["n_questions"] == 2


def test_non_positive_cap_is_refused(runner):
    with pytest.raises(SystemExit):
        runner("--limit", "1", "--max-cost", "0")


def test_failed_question_still_counts_the_planner_cost(runner, monkeypatch):
    """M3 review S3: retrieval fails after the paid planner call -> the error row keeps its cost."""
    calls = []

    def flaky(conn, vector, k):
        calls.append(1)
        if len(calls) <= 2:  # question 1: both of its searches fail
            raise OSError("connection reset")
        return dense_search(conn, vector, k)

    monkeypatch.setattr(retriever_module, "dense_search", flaky)
    rows, summary, _ = runner("--limit", "2")
    assert rows[0]["abstain_reason"] == "error" and rows[0]["error"]
    assert [u["model"] for u in rows[0]["usage"]] == ["claude-sonnet-5-5"]  # the planner call
    assert rows[0]["cost_usd"] == pytest.approx((1000 * 2 + 100 * 10) / 1e6)
    assert summary["cost_usd"]["total"] == pytest.approx(rows[0]["cost_usd"] + rows[1]["cost_usd"],
                                                         abs=1e-4)  # fmt: skip


def test_cost_stop_margin_grows_with_the_most_expensive_question(runner, monkeypatch):
    """Margin = max($0.05, 2 x the most expensive question so far)."""
    monkeypatch.setattr(run_system, "PER_QUESTION_ALLOWANCE", 0.001)
    # each fake question costs ~$0.009, so the margin is ~$0.018: q3 would need 0.018 + 0.018
    rows, summary, _ = runner("--limit", "10", "--max-cost", "0.03")
    assert len(rows) == 2 and summary["stopped_at_cost_cap"] is True
    assert sum(r["cost_usd"] for r in rows) <= 0.03


# --- approval guard (owner instruction 2026-10-24) ---------------------------------------------

APPROVAL = "Yes, run the agent on the 100 dev questions with a hard cap of $5."


@pytest.mark.parametrize(
    "extra",
    [[], ["--max-cost", "5"], ["--approval", APPROVAL], ["--max-cost", "5", "--approval", "ok"]],
)
def test_full_run_without_cap_and_approval_text_is_refused(runner, tmp_path, extra):
    with pytest.raises(SystemExit, match="max-cost|approval"):
        runner(*extra)
    assert list(tmp_path.iterdir()) == []  # refused before any folder or paid call


def test_full_run_saves_the_owner_approval_word_for_word(runner):
    rows, summary, config = runner("--max-cost", "5", "--approval", f"  {APPROVAL}  ")
    assert len(rows) == 100 and config["limit"] is None
    assert config["owner_approval"] == summary["owner_approval"] == APPROVAL
    assert config["max_cost_usd"] == 5.0


def test_smoke_run_needs_no_approval_and_records_none(runner):
    rows, summary, config = runner("--limit", "1")
    assert len(rows) == 1 and config["owner_approval"] is None and summary["owner_approval"] is None


# --- cost-based trigger (owner instruction 2026-10-24: --limit 100 must not skip the guard) -----


@pytest.mark.parametrize("limit", ["100", "20", "15"])  # agent est. $3.50, $0.70, $0.525
def test_runs_estimated_above_the_threshold_need_approval_whatever_the_limit(
    runner, tmp_path, limit
):
    with pytest.raises(SystemExit, match=r"estimated at \$"):
        runner("--limit", limit)
    assert list(tmp_path.iterdir()) == []  # refused before any folder or paid call


def test_run_at_or_below_the_threshold_needs_no_approval(runner):
    rows, summary, config = runner("--limit", "14")  # 14 x $0.035 = $0.49
    assert len(rows) == 14 and config["owner_approval"] is None
    assert config["estimated_cost_usd"] == summary["estimated_cost_usd"] == pytest.approx(0.49)


def test_limit_100_with_cap_and_approval_runs_and_saves_both(runner):
    rows, summary, config = runner("--limit", "100", "--max-cost", "5", "--approval", APPROVAL)
    assert len(rows) == 100 and config["owner_approval"] == APPROVAL
    assert config["estimated_cost_usd"] == pytest.approx(3.5)


@pytest.mark.parametrize(
    "system,n,needs",
    [("baseline", 33, False), ("baseline", 34, True), ("agent", 14, False), ("agent", 15, True)],
)
def test_threshold_is_per_system_estimate(system, n, needs):
    estimate = run_system.estimated_cost(system, n)
    if needs:
        with pytest.raises(SystemExit, match="estimated"):
            run_system.check_approval(estimate, None, None)
    else:
        run_system.check_approval(estimate, None, None)  # no error
