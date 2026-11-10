"""M9-T2: the failure classifier (eval/runners/failure_analysis.py) on hand-made rows.

No real row is classified here: the one real run is M9-T3."""

import json
import re

import pytest

from eval.runners import failure_analysis as fa

IE = fa.IE


def row(pmid="1", gold="yes", verdict="no", **over):
    base = {
        "pmid": pmid,
        "gold_label": gold,
        "verdict": verdict,
        "correct": gold == verdict,
        "abstained": False,
        "abstain_reason": None,
        "error": None,
        "gold_final_rank": 1,
        "planner_added_evidence": 0,
        "answer_supported": True,
        "critic_results": [],
        "critic_removed": [],
        "validation": {
            "removed_uncited": [],
            "removed_invalid_citation": [],
            "downgraded_to_insufficient": False,
        },
    }
    return base | over


def refusal(reason, pmid="1", **over):
    return row(pmid, verdict=IE, abstained=True, abstain_reason=reason, **over)


def result(claim_id="c1", action="keep", supported=True, citation_valid=True):
    return {
        "claim_id": claim_id,
        "action": action,
        "supported": supported,
        "citation_valid": citation_valid,
        "reason": "x",
    }


def validation(downgraded=False, uncited=(), invalid=()):
    return {
        "removed_uncited": list(uncited),
        "removed_invalid_citation": list(invalid),
        "downgraded_to_insufficient": downgraded,
    }


def categorize(agent_row, baseline_correct=False):
    gold = agent_row["gold_label"]
    other = row(
        agent_row["pmid"], gold, gold if baseline_correct else "maybe" if gold != "maybe" else "no"
    )
    records = fa.classify([agent_row], [other])
    assert len(records) == 1
    return records[0]


NSC = "no_supported_claims"
CASES = {
    "refusal_evidence_gate": refusal("weak_evidence"),
    "refusal_model_abstained": refusal("model_abstained"),
    "refusal_citation_check_left_no_claims": refusal(NSC, validation=validation(downgraded=True)),
    "refusal_critic_left_no_claims": refusal(NSC, critic_results=[result(action="remove")]),
    "refusal_critic_rejected_answer": refusal(
        NSC, critic_results=[result()], answer_supported=False
    ),
    "answered_gold_not_rank_1": row(gold_final_rank=2),
    "answered_maybe_boundary": row(gold="maybe", verdict="yes"),
    "answered_yes_no_flip": row(gold="no", verdict="yes"),
}


@pytest.mark.parametrize("expected", CASES)
def test_each_category(expected):
    assert categorize(CASES[expected])["category"] == expected


def test_maybe_boundary_also_covers_a_maybe_verdict():
    assert categorize(row(gold="yes", verdict="maybe"))["category"] == "answered_maybe_boundary"


def test_planner_rule_needs_the_baseline_right():
    r = row(planner_added_evidence=2)
    assert (
        categorize(r, baseline_correct=True)["category"] == "answered_planner_added_baseline_right"
    )
    assert categorize(r, baseline_correct=False)["category"] == "answered_yes_no_flip"


def test_first_rule_wins_and_later_rules_are_recorded():
    r = row(gold="maybe", verdict="no", gold_final_rank=None, planner_added_evidence=1)
    record = categorize(r, baseline_correct=True)
    assert record["category"] == "answered_gold_not_rank_1"
    assert record["also_matched"] == [
        "answered_planner_added_baseline_right",
        "answered_maybe_boundary",
    ]
    assert record["group"] == "gap"


def test_refusal_ignores_rank_and_planner_fields():
    r = refusal("weak_evidence", gold_final_rank=None, planner_added_evidence=3)
    record = categorize(r, baseline_correct=True)
    assert record["category"] == "refusal_evidence_gate"
    assert record["also_matched"] == []


@pytest.mark.parametrize(
    ("results", "report", "survivors"),
    [
        ([result("c1")], validation(), {"c1"}),
        ([result("c1", action="remove"), result("c1")], validation(), set()),  # first result only
        ([result("c1"), result("c1", action="remove")], validation(), {"c1"}),
        ([result("l1")], validation(), set()),  # a limitation is not a claim
        ([result("c123")], validation(), set()),  # must fully match c + 1 or 2 digits
        ([result("c1", supported=False)], validation(), set()),
        ([result("c1", citation_valid=False)], validation(), set()),
        ([result("c1"), result("c2")], validation(uncited=["c1"], invalid=["c2"]), set()),
    ],
)
def test_surviving_claims(results, report, survivors):
    assert fa.surviving_claims(row(critic_results=results, validation=report)) == survivors


def test_rule_4_when_no_claim_survives_even_if_the_critic_rejected_the_answer():
    r = refusal(NSC, critic_results=[result(action="remove")], answer_supported=False)
    assert categorize(r)["category"] == "refusal_critic_left_no_claims"


def test_rule_4_when_answer_supported_is_not_false():
    r = refusal(NSC, critic_results=[result()], answer_supported=None)
    assert categorize(r)["category"] == "refusal_critic_left_no_claims"


def test_downgrade_is_rule_3_even_with_a_kept_claim():
    r = refusal(NSC, critic_results=[result()], answer_supported=False,
                validation=validation(downgraded=True))  # fmt: skip
    assert categorize(r)["category"] == "refusal_citation_check_left_no_claims"


@pytest.mark.parametrize(
    ("agent", "baseline", "message"),
    [
        ([row()], [row("2")], "same set of pmids"),
        ([row(), row()], [row()], "repeats"),
        ([row(correct=True)], [row()], "`correct`"),
        ([row(abstained=True)], [row()], "`abstained` disagrees with the verdict"),
        ([row(verdict=IE, abstained=True)], [row()], "`abstain_reason`"),
        ([row(error="boom")], [row()], "records an error"),
        ([refusal("error")], [row()], "records an error"),
    ],
)
def test_stop_conditions_2_to_5(agent, baseline, message):
    with pytest.raises(SystemExit, match=message):
        fa.check_rows(agent, baseline)


def test_stop_conditions_check_the_baseline_file_too():
    with pytest.raises(SystemExit, match="baseline pmid 1"):
        fa.check_rows([row()], [row(error="boom")])


def test_stop_when_a_field_is_missing():
    broken = row()
    del broken["error"]
    with pytest.raises(SystemExit, match="no field 'error'"):
        fa.check_rows([broken], [row()])


@pytest.mark.parametrize(
    "bad",
    [
        refusal("some_new_reason"),
        refusal(NSC, validation=validation() | {"downgraded_to_insufficient": None}),
        row(gold="yes", verdict="unknown_label"),
    ],
)
def test_stop_condition_6_no_rule_matches(bad):
    with pytest.raises(SystemExit, match="matches no rule"):
        categorize(bad)


@pytest.mark.parametrize(
    ("a", "b", "p"),
    [(5, 0, 0.0625), (0, 5, 0.0625), (0, 0, 1.0), (3, 1, 0.625), (1, 1, 1.0), (10, 0, 2 / 1024)],
)
def test_sign_test_hand_checked(a, b, p):
    assert fa.sign_test_two_sided(a, b) == pytest.approx(p)


def small_set():
    """Six questions, worked out by hand:
    q1 both right; q2 agent refused, baseline right (gap); q3 both wrong (shared flip);
    q4 agent answered wrong, baseline right (gap, rank 3); q5 agent right, baseline wrong;
    q6 both refused (shared, gate)."""
    agent = [
        row("1", "yes", "yes"),
        refusal(NSC, "2", critic_results=[result(action="remove"), result("c2")],
                critic_removed=["c1"], answer_supported=False),
        row("3", "no", "yes"),
        row("4", "yes", "no", gold_final_rank=3),
        row("5", "no", "no"),
        refusal("weak_evidence", "6", gold_final_rank=None),
    ]  # fmt: skip
    baseline = [
        row("1", "yes", "yes"),
        row("2", "yes", "yes"),
        row("3", "no", "maybe"),
        row("4", "yes", "yes"),
        row("5", "no", "yes"),
        refusal("weak_evidence", "6"),
    ]
    return agent, baseline


def test_summary_numbers_hand_checked():
    agent, baseline = small_set()
    records = fa.classify(agent, baseline)
    s = fa.summarize(agent, baseline, records)
    assert s["n_questions"] == 6
    assert (s["agent_correct"], s["baseline_correct"]) == (2, 3)
    assert s["agent_accuracy"] == pytest.approx(2 / 6)
    assert s["baseline_accuracy"] == pytest.approx(3 / 6)
    assert (s["agent_errors"], s["agent_answered_wrongly"]) == (4, 2)
    assert s["agent_refusals_scored_wrong"] == 2
    assert s["agent_refusals_by_reason"] == {NSC: 1, "weak_evidence": 1}
    assert s["answered_only"] == {
        "agent": {"answered": 4, "correct": 2, "accuracy": 0.5},
        "baseline": {"answered": 5, "correct": 3, "accuracy": 0.6},
    }
    assert s["paired"] == {
        "both_wrong": 2,
        "agent_wrong_baseline_right": 2,
        "baseline_wrong_agent_right": 1,
        "exact_two_sided_sign_test_p": 1.0,
    }
    assert sum(s["categories"].values()) == 4
    assert s["categories"]["refusal_critic_rejected_answer"] == 1
    assert s["categories"]["answered_gold_not_rank_1"] == 1
    assert s["categories"]["answered_yes_no_flip"] == 1
    assert s["categories"]["refusal_evidence_gate"] == 1
    assert s["categories"]["refusal_model_abstained"] == 0  # a zero is still listed
    assert sum(s["categories_gap"].values()) == 2
    assert sum(s["categories_shared"].values()) == 2
    assert s["overlaps_on_answered_errors"] == {"answered_yes_no_flip": 1}
    assert s["label_tables"]["shared_gold_by_agent_verdict"] == {"no": {"yes": 1}, "yes": {IE: 1}}
    assert s["label_tables"]["shared_gold_by_baseline_verdict"] == {
        "no": {"maybe": 1},
        "yes": {IE: 1},
    }
    assert s["label_tables"]["gap_gold_by_agent_verdict"] == {"yes": {IE: 1, "no": 1}}
    assert s["retrieval"] == {"agent_gold_in_final_evidence": 5, "errors_gold_not_rank_1": 2}
    assert s["critic"] == {
        "results_total": 2,
        "results_not_keep": 1,
        "questions_with_a_not_keep": 1,
        "item_ids_removed": 1,
        "questions_with_a_removal": 1,
    }


@pytest.fixture
def stored(tmp_path, monkeypatch):
    """Fake stored runs, pinned the way the real ones are."""
    agent, baseline = small_set()
    inputs = {}
    for system, rows in (("agent", agent), ("baseline", baseline)):
        path = tmp_path / f"{system}.jsonl"
        data = "".join(json.dumps(r) + "\n" for r in rows).encode()
        path.write_bytes(data)
        inputs[system] = (path.name, fa.sha256(data))
    monkeypatch.setattr(fa, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(fa, "INPUTS", inputs)
    comparison = {"accuracy": {"ci95": [-3, -1]}, "answered_only_accuracy": {"ci95": [-2, 2]}}
    scoring = json.dumps({"comparisons": {"agent_minus_baseline": comparison}}).encode()
    (tmp_path / "scoring.json").write_bytes(scoring)
    monkeypatch.setattr(fa, "STORED_SCORING", ("scoring.json", fa.sha256(scoring)))
    clean = {"git_commit": "abc1234def", "git_dirty": False, "git_diff_sha256": None}
    monkeypatch.setattr(fa, "git_state", lambda: clean)
    return tmp_path


def run_folders(path):
    return sorted(p.name for p in path.glob(f"{fa.RUN_PREFIX}*"))


def test_real_run_writes_one_new_folder_and_leaves_inputs_alone(stored):
    before = {p.name: p.read_bytes() for p in stored.glob("*.jsonl")}
    fa.main([])
    folders = run_folders(stored)
    assert len(folders) == 1 and folders[0].endswith("-abc1234")
    out = stored / folders[0]
    assert sorted(p.name for p in out.iterdir()) == ["categories.jsonl", "summary.json"]
    summary = json.loads((out / "summary.json").read_text())
    records = [json.loads(line) for line in (out / "categories.jsonl").read_text().splitlines()]
    assert summary["agent_errors"] == len(records) == 4
    assert summary["rubric"]["sha256"] == fa.RUBRIC_SHA256
    copied = summary["stored_paired_bootstrap"]
    assert copied["agent_minus_baseline_accuracy"] == {"ci95": [-3, -1]}
    assert copied["agent_minus_baseline_answered_only_accuracy"] == {"ci95": [-2, 2]}
    assert {p.name: p.read_bytes() for p in stored.glob("*.jsonl")} == before


def test_runs_only_once(stored):
    fa.main([])
    with pytest.raises(SystemExit, match="runs once"):
        fa.main([])
    assert len(run_folders(stored)) == 1


def test_refuses_a_dirty_tree(stored, monkeypatch):
    monkeypatch.setattr(fa, "git_state", lambda: {"git_commit": "abc", "git_dirty": True})
    with pytest.raises(SystemExit, match="clean tree"):
        fa.main([])
    assert run_folders(stored) == []


def test_tampered_input_is_refused_and_nothing_written(stored):
    path = stored / "agent.jsonl"
    path.write_bytes(path.read_bytes().replace(b'"verdict": "yes"', b'"verdict": "no"', 1))
    with pytest.raises(SystemExit, match="pinned SHA-256"):
        fa.main([])
    assert run_folders(stored) == []


def test_tampered_scoring_file_is_refused_and_nothing_written(stored):
    (stored / "scoring.json").write_bytes(b"{}")
    with pytest.raises(SystemExit, match="M4 scoring file"):
        fa.main([])
    assert run_folders(stored) == []


def test_main_applies_the_row_stop_conditions_and_writes_nothing(stored, monkeypatch):
    data = (json.dumps(row(error="boom")) + "\n").encode()
    (stored / "agent.jsonl").write_bytes(data)
    (stored / "baseline.jsonl").write_bytes(data)
    pins = {s: (f"{s}.jsonl", fa.sha256(data)) for s in ("agent", "baseline")}
    monkeypatch.setattr(fa, "INPUTS", pins)
    with pytest.raises(SystemExit, match="records an error"):
        fa.main([])
    assert run_folders(stored) == []


def test_a_row_matching_no_rule_writes_nothing(stored, monkeypatch):
    agent = [refusal("some_new_reason")]
    data = (json.dumps(agent[0]) + "\n").encode()
    (stored / "agent.jsonl").write_bytes(data)
    (stored / "baseline.jsonl").write_bytes(data)
    pins = {s: (f"{s}.jsonl", fa.sha256(data)) for s in ("agent", "baseline")}
    monkeypatch.setattr(fa, "INPUTS", pins)
    with pytest.raises(SystemExit, match="matches no rule"):
        fa.main([])
    assert run_folders(stored) == []


def test_changed_rubric_is_refused(stored, monkeypatch):
    monkeypatch.setattr(fa, "RUBRIC_SHA256", "0" * 64)
    with pytest.raises(SystemExit, match="rubric"):
        fa.main([])
    assert run_folders(stored) == []


def test_check_inputs_classifies_and_writes_nothing(stored, monkeypatch, capsys):
    def must_not_run(*_):
        raise AssertionError("--check-inputs must not classify")

    monkeypatch.setattr(fa, "classify", must_not_run)
    fa.main(["--check-inputs"])
    assert "Nothing classified" in capsys.readouterr().out
    assert run_folders(stored) == []


@pytest.mark.parametrize("arg", ["--agent-run=other", "--input=x.jsonl", "other.jsonl"])
def test_no_way_to_name_another_input(arg):
    with pytest.raises(SystemExit) as stop:
        fa.main([arg])
    assert stop.value.code == 2  # argparse refused it


def test_code_matches_the_frozen_rubric():
    text = fa.RUBRIC.read_text(encoding="utf-8")
    assert re.findall(r"^\| \d \| `(\w+)` \|", text, flags=re.M) == list(fa.CATEGORIES)
    for name, sha in fa.INPUTS.values():
        assert f"`eval/results/{name}` | `{sha}`" in text
    lock = fa.RUBRIC.with_suffix(".sha256").read_text(encoding="utf-8")
    assert lock.split()[0] == fa.RUBRIC_SHA256 == fa.sha256(fa.RUBRIC.read_bytes())


def test_real_stored_files_pass_the_input_checks(capsys):
    fa.main(["--check-inputs"])
    assert "Rows: agent 200, baseline 200." in capsys.readouterr().out
