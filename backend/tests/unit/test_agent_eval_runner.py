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
