"""scripts/check_examples.py: single-question mode (--only/--cap), refusals and the merge into the
record file. The agent, model and database are stubs; the record file is a temporary copy.
No network, $0."""

import json
import sys
from types import SimpleNamespace

import pytest

from scripts import check_examples as ce

REAL_RECORD = ce.OUT  # docs/design/example-questions-check.json, never written by these tests


@pytest.fixture
def run(monkeypatch, tmp_path):
    """Returns run(*args) -> (record after the run, questions the stub agent was asked)."""
    out = tmp_path / "check.json"
    out.write_text(REAL_RECORD.read_text(encoding="utf-8"), encoding="utf-8")
    asked: list[str] = []
    answers: dict[str, dict] = {}  # pmid -> overrides of the stub's good answer (see agent())

    class Conn:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def agent(question, **kwargs):
        asked.append(question)
        pmid = next(p for p, q in ce.CANDIDATES if q == question)
        # by default a good answer: gold verdict, not abstained, 2 claims citing the source paper
        a = {"verdict": GOLD[pmid], "abstained": False, "n_claims": 2, "cites": pmid}
        a |= answers.get(pmid, {})
        claim = SimpleNamespace(text="claim", citation_ids=[f"PMID:{a['cites']}"])
        return SimpleNamespace(verdict=a["verdict"], abstained=a["abstained"], abstain_reason=None,
                               claims=[claim] * a["n_claims"], estimated_cost_usd=0.03,
                               timing_ms={"total": 5.0}), None  # fmt: skip

    monkeypatch.setattr(ce, "OUT", out)
    monkeypatch.setattr(ce, "answer_agent", agent)
    monkeypatch.setattr(ce, "anthropic_client", lambda: None)
    monkeypatch.setattr(ce, "voyage_client", lambda: None)
    monkeypatch.setattr(ce, "connect", lambda: Conn())
    monkeypatch.setattr(ce.Path, "relative_to", lambda self, other: self)  # tmp file: outside repo

    def go(*args, overrides=None):
        answers.clear()
        answers.update(overrides or {})
        monkeypatch.setattr(sys, "argv", ["check_examples", *args])
        ce.main()
        return json.loads(out.read_text(encoding="utf-8")), asked

    go.before = json.loads(out.read_text(encoding="utf-8"))
    go.out = out
    return go


DEV_RUN = ce.REPO_ROOT / "eval/results/dev-agent-20261023-222333-871b22d/raw.jsonl"
GOLD = {
    row["pmid"]: row["gold_label"]
    for row in map(json.loads, DEV_RUN.read_text(encoding="utf-8").splitlines())
}
BY_PMID = lambda record: {r["pmid"]: r for r in record["results"]}  # noqa: E731


def test_only_checks_one_question_and_keeps_every_other_entry(run):
    after, asked = run("--only", "17598882", "--cap", "0.05")
    assert asked == ["Is breast cancer prognosis inherited?"]
    old, new = BY_PMID(run.before), BY_PMID(after)
    assert set(new) == set(old) | {"17598882"}
    assert all(new[p] == old[p] for p in old if p != "17598882")  # untouched, incl. rt-PA's record
    entry = new["17598882"]
    assert entry["keep"] is True and entry["cost_usd"] == 0.03 and entry["cap_usd"] == 0.05
    assert {"checked_at", "model", "commit"} <= set(entry)
    assert after["spent_usd"] == pytest.approx(run.before["spent_usd"] + 0.03)
    assert after["cap_usd"] == run.before["cap_usd"]


def test_rechecking_a_pmid_replaces_only_its_entry(run):
    after, _ = run(
        "--only", "26370095", "--cap", "0.05", overrides={"26370095": {"verdict": "maybe"}}
    )
    old, new = BY_PMID(run.before), BY_PMID(after)
    assert len(after["results"]) == len(run.before["results"])
    assert new["26370095"]["verdict"] == "maybe" and new["26370095"]["keep"] is False
    assert all(new[p] == old[p] for p in old if p != "26370095")


@pytest.mark.parametrize("pmid", ["99999999", "8245806"])  # unknown; a frozen TEST paper
def test_non_candidates_are_refused_before_any_call(run, pmid):
    with pytest.raises(SystemExit, match="not a candidate"):
        run("--only", pmid, "--cap", "0.05")
    _, asked = run("--only", "17598882", "--cap", "0.05")  # nothing was asked before this valid run
    assert asked == ["Is breast cancer prognosis inherited?"]


@pytest.mark.parametrize("cap", ["0", "-1"])
def test_a_cap_that_is_not_positive_is_refused(run, cap):
    with pytest.raises(SystemExit, match="positive"):
        run("--only", "17598882", "--cap", cap)


def test_candidates_are_dev_only():
    manifests = ce.REPO_ROOT / "data/manifests"
    dev = {i["pmid"] for i in json.loads((manifests / "dev.json").read_text())["items"]}
    test = {i["pmid"] for i in json.loads((manifests / "test.json").read_text())["items"]}
    assert all(p in dev and p not in test for p, _ in ce.CANDIDATES)


def test_full_run_merges_and_its_cap_stops_before_overspending(run):
    # $1 cap, $0.08 reserve, $0.03 per question: the pre-check allows every candidate
    after, asked = run()
    assert len(asked) == len(ce.CANDIDATES)
    assert "24669960" in BY_PMID(after)  # rt-PA is no longer a candidate but its record survives
    # a cap of $0.10 admits one question ($0 + 0.08 <= 0.10), then stops ($0.03 + 0.08 > 0.10)
    asked.clear()
    after2, asked = run("--cap", "0.10")
    assert len(asked) == 1
    assert len(after2["results"]) == len(after["results"])


def test_a_run_that_checks_nothing_leaves_the_file_unchanged(run, monkeypatch):
    monkeypatch.setattr(ce, "PER_QUESTION_RESERVE_USD", 1.0)  # reserve > cap: nothing starts
    after, asked = run("--cap", "0.50")
    assert asked == [] and after == run.before


@pytest.mark.parametrize(
    "override",
    [
        {"verdict": "no"},  # a clear yes/no, but not the dev label (17598882 is "yes")
        {"abstained": True},
        {"n_claims": 1},
        {"cites": "26370095"},  # cites another paper, not its own source
    ],
)
def test_keep_needs_every_rule(run, override):
    after, _ = run("--only", "17598882", "--cap", "0.05", overrides={"17598882": override})
    assert BY_PMID(after)["17598882"]["keep"] is False


def test_a_test_paper_among_the_candidates_stops_the_run_before_any_call(run, monkeypatch):
    monkeypatch.setattr(ce, "CANDIDATES", [*ce.CANDIDATES, ("8245806", "A frozen TEST question?")])
    with pytest.raises(SystemExit, match="not a dev-only question"):
        run("--only", "8245806", "--cap", "0.05")
    with pytest.raises(SystemExit, match="not a dev-only question"):
        run()  # full mode checks every candidate first
    assert run.out.read_text(encoding="utf-8") == REAL_RECORD.read_text(encoding="utf-8")


def test_cap_given_on_the_command_line_is_recorded(run):
    after, _ = run("--cap", "0.10")
    checked = [r for r in after["results"] if r.get("cap_usd") == 0.10]
    assert len(checked) == 1


def test_nothing_checked_and_no_record_yet_writes_no_file(run, monkeypatch):
    run.out.unlink()
    monkeypatch.setattr(ce, "PER_QUESTION_RESERVE_USD", 1.0)
    monkeypatch.setattr(sys, "argv", ["check_examples", "--cap", "0.50"])
    ce.main()
    assert not run.out.exists()
