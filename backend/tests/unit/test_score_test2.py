"""The test2 scorer (M15-T9), on small made-up result files in tmp_path only.

The real test2 manifest and results are never read: MANIFESTS, RESULTS_DIR and git state are
replaced by fakes. Expected numbers below are worked out by hand."""

import hashlib
import json
from pathlib import Path

import pytest

from eval.runners import run_system, score_test2

ROOT = Path(__file__).resolve().parents[3]
CLEAN = {"git_commit": "b" * 40, "git_dirty": False, "git_diff_sha256": None}
SECRET = "SECRET-QUESTION-TEXT"
IE = "insufficient_evidence"

# 10 made-up questions: gold yes for 1-6, no for 7-10.
GOLD = {str(i): ("yes" if i <= 6 else "no") for i in range(1, 11)}
# baseline right on 1,2,3,4,5,7,10 (7/10); wrong on 6 (maybe), 8 (yes), 9 (refused)
BASELINE = {"1": "yes", "2": "yes", "3": "yes", "4": "yes", "5": "yes", "6": "maybe",
            "7": "no", "8": "yes", "9": IE, "10": "no"}  # fmt: skip
# agent right on 1,2,3,4,6,8,9,10 (8/10); wrong on 5 (refused), 7 (yes)
AGENT = {"1": "yes", "2": "yes", "3": "yes", "4": "yes", "5": IE, "6": "yes",
         "7": "yes", "8": "no", "9": "no", "10": "no"}  # fmt: skip
# retried: 5 (retry failed, first pass stood), 8 (maybe -> no), 9 (yes -> no)
RETRIED = {"5": ("yes", "timeout"), "8": ("maybe", None), "9": ("yes", None)}


def draft(verdict: str, pmid: str) -> dict:
    claim = {"id": "c1", "text": f"SECRET-CLAIM {pmid}", "citation_ids": [f"PMID:{pmid}"]}
    return {"verdict": verdict, "claims": [claim], "limitations": []}


def critic(action: str) -> list[dict]:
    supported = action == "keep"
    return [{"claim_id": "c1", "supported": supported, "citation_valid": True,
             "action": action, "reason": "made up"}]  # fmt: skip


def row(pmid: str, verdict: str, system: str, ms: float, gold: dict, retried: dict) -> dict:
    r = {
        "pmid": pmid, "question": f"{SECRET} {pmid}", "gold_label": gold[pmid],
        "verdict": verdict, "correct": verdict == gold[pmid], "abstained": verdict == IE,
        "abstain_reason": "no_supported_claims" if verdict == IE else None,
        "error": None, "cost_usd": 0.013 if system == "baseline" else 0.03,
        "timings_ms": {"total": ms}, "evidence_pmids": [pmid],
    }  # fmt: skip
    if system == "agent":
        r |= {"retry_used": False, "retry_error": None, "first_draft_answer": None,
              "first_critic_results": [], "first_answer_supported": None}  # fmt: skip
        if pmid in retried:
            first_verdict, error = retried[pmid]
            r |= {"retry_used": True, "retry_error": error, "cost_usd": 0.06,
                  "first_draft_answer": draft(first_verdict, pmid),
                  "first_critic_results": critic("remove"),  # every claim removed -> refusal
                  "first_answer_supported": False}  # fmt: skip
    return r


def make_rows(system: str, verdicts=None, gold=GOLD, retried=RETRIED) -> list[dict]:
    verdicts = verdicts or (BASELINE if system == "baseline" else AGENT)
    ms = {"baseline": lambda i: 1000.0, "agent": lambda i: 100.0 * i}[system]
    return [row(p, verdicts[p], system, ms(i), gold, retried) for i, p in enumerate(gold, 1)]


def stored_summary(rows: list[dict]) -> dict:
    verdicts = [r["verdict"] for r in rows]
    return {"n_questions": len(rows), "accuracy": sum(r["correct"] for r in rows) / len(rows),
            "verdict_counts": {v: verdicts.count(v) for v in set(verdicts)},
            "errors": sum(r["error"] is not None for r in rows),
            "cost_usd": {"total": round(sum(r["cost_usd"] for r in rows), 4)}}  # fmt: skip


def write_run(results, run_id, system, rows, manifest_sha, split="test2", summary=None):
    d = results / run_id
    d.mkdir()
    config = {"run_id": run_id, "system": system, "split": split, "manifest_sha256": manifest_sha,
              "question_ids": [r["pmid"] for r in rows]}  # fmt: skip
    if system == "agent":
        config["agent_rules_sha256"] = score_test2.rules_sha256()
    (d / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
    (d / "raw.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    summary = stored_summary(rows) if summary is None else summary
    (d / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return d


@pytest.fixture
def env(tmp_path, monkeypatch):
    manifests, results = tmp_path / "manifests", tmp_path / "results"
    manifests.mkdir()
    results.mkdir()
    items = [{"pmid": p, "question": f"{SECRET} {p}", "final_decision": g} for p, g in GOLD.items()]
    body = json.dumps({"split": "test2", "items": items}).encode()
    sha = hashlib.sha256(body).hexdigest()
    (manifests / "test2.json").write_bytes(body)
    (manifests / "test2.sha256").write_text(f"{sha}  test2.json\n", encoding="utf-8")
    for module in (run_system, score_test2):
        monkeypatch.setattr(module, "MANIFESTS", manifests)
    for module in (run_system, score_test2):
        monkeypatch.setattr(module, "RESULTS_DIR", results)
    monkeypatch.setattr(score_test2, "git_state", lambda: CLEAN)
    return {"results": results, "sha": sha}


def score(env, baseline_rows=None, agent_rows=None) -> dict:
    b = baseline_rows or make_rows("baseline")
    a = agent_rows or make_rows("agent")
    write_run(env["results"], "test2-baseline-x", "baseline", b, env["sha"])
    write_run(env["results"], "test2-agent-x", "agent", a, env["sha"])
    score_test2.main(["--baseline-run", "test2-baseline-x", "--agent-run", "test2-agent-x"])
    (out,) = [d for d in env["results"].iterdir() if d.name.startswith("m15-test2-score-")]
    return json.loads((out / "summary.json").read_text(encoding="utf-8"))


def test_hand_checked_numbers(env, capsys):
    s = score(env)
    b, a = s["systems"]["baseline"], s["systems"]["agent"]
    assert (b["correct"], b["refused"], b["accuracy_pct"]) == (7, 1, 70.0)
    assert (a["correct"], a["refused"], a["accuracy_pct"]) == (8, 1, 80.0)
    assert b["cost_usd_per_question"] == 0.013 and a["cost_usd_per_question"] == 0.039
    assert b["median_time_per_answer_ms"] == 1000.0 and a["median_time_per_answer_ms"] == 550.0
    assert b["accuracy_wilson95_pct"][0] < 70.0 < b["accuracy_wilson95_pct"][1]
    p = s["agent_vs_baseline"]
    assert (p["agent_only_right"], p["baseline_only_right"]) == (3, 2)
    assert (p["both_right"], p["both_wrong"]) == (5, 0)
    assert p["sign_test_p_exact_two_sided"] == pytest.approx(1.0)  # 2*(1+5+10)/32 capped
    assert p["accuracy_diff_agent_minus_baseline_points"] == 10.0
    lo, hi = p["accuracy_diff_bootstrap95_points"]
    assert lo < 0 < hi and p["bootstrap_range_excludes_zero"] is False
    r = s["agent_retry"]
    assert r["questions_retried"] == 3 and r["share_retried_pct"] == 30.0
    assert r["first_pass_would_have_refused"] == 3 and r["first_pass_would_have_answered"] == 0
    assert r["retry_failed_first_pass_stood"] == 1
    assert (r["after_retry_right"], r["after_retry_answered_wrong"]) == (2, 0)
    assert r["after_retry_still_refused"] == 0
    assert r["verdict_changes_first_draft_to_final"] == {"maybe -> no": 1, "yes -> no": 1}
    assert r["cost_usd_per_retried_question"] == 0.06
    d = s["decision_rules"]
    assert d["adr_045_agent_at_or_above_baseline"]["met"] is True
    assert d["adr_043_clearly_beats_preregistered"]["met"] is False
    printed = capsys.readouterr().out
    assert SECRET not in printed and "SECRET-CLAIM" not in printed


def test_nothing_but_numbers_is_written(env):
    score(env)
    (out,) = [d for d in env["results"].iterdir() if d.name.startswith("m15-test2-score-")]
    assert sorted(p.name for p in out.iterdir()) == ["run_config.json", "summary.json"]
    for p in out.iterdir():
        text = p.read_text(encoding="utf-8")
        assert SECRET not in text and "SECRET-CLAIM" not in text


def test_a_tie_meets_adr_045_but_not_adr_043(env):
    s = score(env, agent_rows=make_rows("agent", BASELINE))
    d = s["decision_rules"]
    assert s["agent_vs_baseline"]["accuracy_diff_agent_minus_baseline_points"] == 0.0
    assert d["adr_045_agent_at_or_above_baseline"]["met"] is True
    assert d["adr_043_clearly_beats_preregistered"]["met"] is False


def test_agent_below_baseline_meets_neither_rule(env):
    s = score(env, baseline_rows=make_rows("baseline", AGENT), agent_rows=make_rows(
        "agent", BASELINE))  # fmt: skip
    d = s["decision_rules"]
    assert s["systems"]["agent"]["correct"] == 7 and s["systems"]["baseline"]["correct"] == 8
    assert d["adr_045_agent_at_or_above_baseline"]["met"] is False
    assert d["adr_043_clearly_beats_preregistered"]["met"] is False


def test_a_clear_win_meets_adr_043(env):
    # 40 questions: both right on 30, agent-only right on 10 -> p = 2/1024, range above zero
    gold = {str(i): "yes" for i in range(1, 41)}
    items = [{"pmid": p, "question": "q", "final_decision": "yes"} for p in gold]
    body = json.dumps({"split": "test2", "items": items}).encode()
    sha = hashlib.sha256(body).hexdigest()
    manifests = score_test2.MANIFESTS
    (manifests / "test2.json").write_bytes(body)
    (manifests / "test2.sha256").write_text(f"{sha}  test2.json\n", encoding="utf-8")
    env["sha"] = sha
    baseline = {p: ("yes" if int(p) <= 30 else "no") for p in gold}
    agent = dict.fromkeys(gold, "yes")
    s = score(env, make_rows("baseline", baseline, gold, {}), make_rows("agent", agent, gold, {}))
    p = s["agent_vs_baseline"]
    assert (p["agent_only_right"], p["baseline_only_right"]) == (10, 0)
    assert p["sign_test_p_exact_two_sided"] == pytest.approx(2 / 1024)
    assert p["accuracy_diff_bootstrap95_points"][0] > 0
    assert s["decision_rules"]["adr_043_clearly_beats_preregistered"]["met"] is True
    assert s["agent_retry"]["questions_retried"] == 0
    assert s["agent_retry"]["cost_usd_per_retried_question"] is None


def test_a_retry_whose_first_pass_would_have_answered_is_counted_not_hidden(env):
    agent = make_rows("agent")
    seven = next(r for r in agent if r["pmid"] == "7")  # answered "yes" (wrong)
    seven |= {"retry_used": True, "first_draft_answer": draft("yes", "7"),
              "first_critic_results": critic("keep"), "first_answer_supported": True}  # fmt: skip
    r = score(env, agent_rows=agent)["agent_retry"]
    assert r["questions_retried"] == 4 and r["first_pass_would_have_answered"] == 1
    assert r["after_retry_answered_wrong"] == 1


def _bad_split(env, b, a):
    write_run(env["results"], "test2-baseline-x", "baseline", b, env["sha"], split="dev")
    write_run(env["results"], "test2-agent-x", "agent", a, env["sha"])


def _bad_hash(env, b, a):
    write_run(env["results"], "test2-baseline-x", "baseline", b, "0" * 64)
    write_run(env["results"], "test2-agent-x", "agent", a, env["sha"])


def _swapped_systems(env, b, a):
    write_run(env["results"], "test2-baseline-x", "agent", a, env["sha"])
    write_run(env["results"], "test2-agent-x", "baseline", b, env["sha"])


def _void(env, b, a):
    write_run(env["results"], "test2-baseline-x", "baseline", b, env["sha"])
    d = write_run(env["results"], "test2-agent-x", "agent", a, env["sha"])
    (d / run_system.VOID_FILE).write_text("{}", encoding="utf-8")


def _order(env, b, a):
    write_run(env["results"], "test2-baseline-x", "baseline", b[::-1], env["sha"])
    write_run(env["results"], "test2-agent-x", "agent", a, env["sha"])


def _subset(env, b, a):
    write_run(env["results"], "test2-baseline-x", "baseline", b[:9], env["sha"])
    write_run(env["results"], "test2-agent-x", "agent", a[:9], env["sha"])


def _gold(env, b, a):
    b[0] |= {"gold_label": "no", "correct": False}
    _ok(env, b, a)


def _correct_flag(env, b, a):
    a[4]["correct"] = True  # pmid 5 was refused, so it cannot be correct
    _ok(env, b, a)


def _summary(env, b, a):
    write_run(env["results"], "test2-baseline-x", "baseline", b, env["sha"])
    write_run(env["results"], "test2-agent-x", "agent", a, env["sha"],
              summary=stored_summary(a) | {"accuracy": 0.9})  # fmt: skip


def _ok(env, b, a):
    write_run(env["results"], "test2-baseline-x", "baseline", b, env["sha"])
    write_run(env["results"], "test2-agent-x", "agent", a, env["sha"])


@pytest.mark.parametrize(
    ("setup", "message"),
    [(_bad_split, "not a baseline test2 run"), (_bad_hash, "frozen test2.json"),
     (_swapped_systems, "not a baseline test2 run"), (_void, "void or unfinished"),
     (_order, "different question IDs"), (_subset, "exactly the test2 questions"),
     (_gold, "gold label differs"), (_correct_flag, "`correct` flag"),
     (_summary, "stored summary")],
)  # fmt: skip
def test_bad_inputs_are_refused_before_anything_is_written(env, setup, message):
    setup(env, make_rows("baseline"), make_rows("agent"))
    with pytest.raises(SystemExit, match=message):
        score_test2.main(["--baseline-run", "test2-baseline-x", "--agent-run", "test2-agent-x"])
    assert sorted(d.name for d in env["results"].iterdir()) == ["test2-agent-x", "test2-baseline-x"]


def test_a_dirty_tree_is_refused(env, monkeypatch):
    monkeypatch.setattr(score_test2, "git_state", lambda: CLEAN | {"git_dirty": True})
    with pytest.raises(SystemExit, match="clean commit"):
        score_test2.main(["--baseline-run", "a", "--agent-run", "b"])


def test_the_quoted_rules_are_word_for_word_in_decisions_md():
    decisions = ROOT / "DECISIONS.md"
    if not decisions.exists():  # git-ignored planning file: checked locally, not in CI
        pytest.skip("DECISIONS.md is not in this checkout")
    flat = " ".join(decisions.read_text(encoding="utf-8").split())
    assert score_test2.ADR_045_LINE in flat
    assert score_test2.ADR_043_LINE in flat


def test_a_clear_loss_is_not_a_clear_win(env):
    # 40 questions: both right on 30, baseline-only right on 10: range below 0, p = 2/1024
    gold = {str(i): "yes" for i in range(1, 41)}
    items = [{"pmid": p, "question": "q", "final_decision": "yes"} for p in gold]
    body = json.dumps({"split": "test2", "items": items}).encode()
    env["sha"] = hashlib.sha256(body).hexdigest()
    (score_test2.MANIFESTS / "test2.json").write_bytes(body)
    (score_test2.MANIFESTS / "test2.sha256").write_text(f"{env['sha']}  test2.json\n")
    agent = {p: ("yes" if int(p) <= 30 else "no") for p in gold}
    baseline = dict.fromkeys(gold, "yes")
    s = score(env, make_rows("baseline", baseline, gold, {}), make_rows("agent", agent, gold, {}))
    rule = s["decision_rules"]["adr_043_clearly_beats_preregistered"]
    assert s["agent_vs_baseline"]["accuracy_diff_bootstrap95_points"][1] < 0
    assert rule["parts"] == {"range_excludes_zero": True, "sign_test_p_below_0_05": True,
                             "agent_ahead": False}  # fmt: skip
    assert rule["met"] is False
    assert s["decision_rules"]["adr_045_agent_at_or_above_baseline"]["met"] is False


def test_changed_agent_rules_are_refused(env, monkeypatch):
    _ok(env, make_rows("baseline"), make_rows("agent"))
    monkeypatch.setattr(score_test2, "rules_sha256", lambda: "f" * 64)
    with pytest.raises(SystemExit, match="different agent rules"):
        score_test2.main(["--baseline-run", "test2-baseline-x", "--agent-run", "test2-agent-x"])


def test_a_second_completed_run_of_a_system_is_refused(env):
    _ok(env, make_rows("baseline"), make_rows("agent"))
    write_run(env["results"], "test2-agent-y", "agent", make_rows("agent"), env["sha"])
    with pytest.raises(SystemExit, match="Exactly one may exist"):
        score_test2.main(["--baseline-run", "test2-baseline-x", "--agent-run", "test2-agent-x"])
    assert not [d for d in env["results"].iterdir() if d.name.startswith("m15-")]


def test_an_unreadable_first_pass_fails_without_echoing_text(env, capsys):
    agent = make_rows("agent")
    agent[7]["first_draft_answer"]["verdict"] = "SECRET-CLAIM not a verdict"
    _ok(env, make_rows("baseline"), agent)
    with pytest.raises(SystemExit) as stop:
        score_test2.main(["--baseline-run", "test2-baseline-x", "--agent-run", "test2-agent-x"])
    assert "unreadable" in str(stop.value) and "SECRET" not in str(stop.value)
    assert "SECRET" not in capsys.readouterr().out
