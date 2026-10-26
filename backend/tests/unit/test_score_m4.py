"""M4-T4 scorer (eval/runners/score_m4.py) on four hand-made questions whose every metric is worked
out by hand below. Fake run folders in tmp_path; stored summaries come from the real
run_system.summarize, so the cross-check is tested against the code that wrote the real runs."""

import json

import pytest

from eval.runners import run_system, score_m4

IE = "insufficient_evidence"
GOLD = {"1": "yes", "2": "no", "3": "maybe", "4": "yes"}
# gold paper rank in the final evidence: q1, q2 rank 1; q3 not retrieved; q4 rank 3
EVIDENCE = {"1": ["1", "9"], "2": ["2", "9"], "3": ["8", "9"], "4": ["8", "9", "4"]}
VERDICTS = {
    "baseline": {"1": "yes", "2": "yes", "3": IE, "4": "yes"},  # acc 2/4, cov 3/4, answered 2/3
    "agent": {"1": "yes", "2": "no", "3": IE, "4": IE},  # acc 2/4, cov 2/4, answered 2/2
    "closed_book": {"1": "yes", "2": "yes", "3": "yes", "4": "yes"},  # acc 2/4, cov 4/4
}
ITEMS = {  # (pmid, item id) -> judge label; q3 has no items in either system
    "baseline": {("1", "c1"): "supported", ("1", "c2"): "partially_supported",
                 ("2", "c1"): "unsupported", ("2", "l1"): "supported",
                 ("4", "c1"): "supported"},
    "agent": {("1", "c1"): "supported", ("2", "c1"): "supported"},
}  # fmt: skip
RUNS = {"baseline": "test-baseline-x", "agent": "test-agent-x", "closed_book": "test-closed_book-x"}
JUDGE = "judge-test-x"


def make_row(system, pmid, total_ms, cost):
    verdict = VERDICTS[system][pmid]
    items = ITEMS.get(system, {})
    mk = lambda i: {"id": i, "text": f"{system} {pmid} {i}", "citation_ids": [f"PMID:{pmid}"]}  # noqa: E731
    evidence = [] if system == "closed_book" else EVIDENCE[pmid]
    extra = {"planner_added_evidence": int(pmid == "1")} if system == "agent" else {}
    return {
        "pmid": pmid, "question": f"Q{pmid}?", "gold_label": GOLD[pmid], "verdict": verdict,
        "correct": verdict == GOLD[pmid], "abstained": verdict == IE,
        "abstain_reason": "weak_evidence" if verdict == IE else None, "answer": "",
        "claims": [mk(i) for (p, i) in items if p == pmid and i.startswith("c")],
        "limitations": [mk(i) for (p, i) in items if p == pmid and i.startswith("l")],
        "validation": {"removed_uncited": [], "removed_invalid_citation": [],
                       "removed_limitations": [], "downgraded_to_insufficient": False},
        "evidence_pmids": evidence, "gate_passed": verdict != IE, "rerank_skipped": False,
        "gold_final_rank": evidence.index(pmid) + 1 if pmid in evidence else None,
        "error": None, "timings_ms": {"total": total_ms, "retrieval": 100.0, "gate": 0.0,
                                      "synthesis": 0.0 if verdict == IE else 500.0 + total_ms},
        "usage": [{"input_tokens": 10, "output_tokens": 5}], "cost_usd": cost, **extra,
    }  # fmt: skip


@pytest.fixture
def env(tmp_path, monkeypatch):
    results = tmp_path / "results"
    results.mkdir()
    monkeypatch.setattr(score_m4, "RESULTS_DIR", results)
    monkeypatch.setattr(score_m4, "git_state", lambda: {"git_commit": "abc1234def",
                                                        "git_dirty": False,
                                                        "git_diff_sha256": None})  # fmt: skip
    manifest = {"items": [{"pmid": p, "question": f"Q{p}?", "final_decision": g}
                          for p, g in GOLD.items()]}  # fmt: skip
    monkeypatch.setattr(score_m4, "load_manifest", lambda split: manifest)
    for system, run_id in RUNS.items():
        d = results / run_id
        d.mkdir()
        rows = [make_row(system, p, 1000.0 * int(p), 0.01 * int(p)) for p in GOLD]
        config = {"run_id": run_id, "system": system, "split": "test", "question_ids": list(GOLD)}
        (d / "run_config.json").write_text(json.dumps(config))
        (d / "raw.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (d / "summary.json").write_text(json.dumps(run_system.summarize(rows)))
    d = results / JUDGE
    d.mkdir()
    (d / "run_config.json").write_text(json.dumps({
        "baseline_run": RUNS["baseline"], "agent_run": RUNS["agent"], "setup_check": None,
        "question_ids": ["1", "2", "4"]}))  # fmt: skip
    letter = {"baseline": "A", "agent": "B"}
    judgments = [{"pmid": p, "item_id": i, "blind_id": f"{letter[s]}.{i}", "system": s, "kind":
                  "claim" if i.startswith("c") else "limitation", "text": f"{s} {p} {i}",
                  "citation_ids": [f"PMID:{p}"], "label": label}
                 for s, items in ITEMS.items() for (p, i), label in items.items()]  # fmt: skip
    key = {p: {"SYSTEM_A": "baseline", "SYSTEM_B": "agent"} for p in ("1", "2", "4")}
    (d / "blinding_key.json").write_text(json.dumps(key))
    (d / "judgments.jsonl").write_text("".join(json.dumps(j) + "\n" for j in judgments))
    (d / "summary.json").write_text("{}")
    return results


def run(env):
    score_m4.main(["--baseline-run", RUNS["baseline"], "--agent-run", RUNS["agent"],
                   "--closed-book-run", RUNS["closed_book"], "--judge-run", JUDGE])  # fmt: skip
    (out,) = [d for d in env.iterdir() if d.name.startswith("m4-final-")]
    return out


def test_answer_metrics_match_hand_computed_values(env):
    s = json.loads((run(env) / "summary.json").read_text())["systems"]
    b, a, c = s["baseline"], s["agent"], s["closed_book"]
    # refusals count as wrong in overall accuracy; coverage reported beside answered-only accuracy
    assert (b["accuracy"]["count"], b["accuracy"]["n"]) == (2, 4)
    assert b["coverage"]["value"] == 0.75
    assert b["answered_only_accuracy"]["value"] == pytest.approx(2 / 3)
    assert (a["accuracy"]["value"], a["coverage"]["value"]) == (0.5, 0.5)
    assert a["answered_only_accuracy"]["value"] == 1.0
    assert (c["accuracy"]["value"], c["coverage"]["value"]) == (0.5, 1.0)
    assert c["answered_only_accuracy"]["value"] == 0.5
    assert b["refusals_by_reason"] == {"weak_evidence": 1}
    assert a["refusals_by_reason"] == {"weak_evidence": 2}
    assert (b["hit_at_1"]["value"], b["hit_at_5"]["value"]) == (0.5, 0.75)
    assert c["hit_at_1"] == c["hit_at_5"] == "not applicable"
    # totals 1000..4000 ms: inclusive median 2500; costs 0.01..0.04: mean and median 0.025
    assert b["latency_ms"]["p50"]["estimate"] == 2500
    assert b["cost_usd_per_query"]["mean"]["estimate"] == pytest.approx(0.025)
    assert b["cost_usd_per_query"]["median"]["estimate"] == pytest.approx(0.025)
    assert b["accuracy"]["wilson95"] == pytest.approx([0.1500, 0.8500], abs=5e-4)
    # a stage that did not run (0.0 after a refusal) is left out of its latency figures
    assert b["latency_ms"]["stages"]["synthesis"]["n"] == 3  # q3 refused
    assert b["latency_ms"]["stages"]["synthesis"]["p50"] == 2500.0  # 1500, 2500, 4500
    assert b["latency_ms"]["stages"]["retrieval"]["n"] == 4
    assert score_m4.stage_latency([make_row("agent", p, 1.0, 0) for p in GOLD], "gate")["n"] == 4
    assert a["planner_added_evidence"] == {"questions_with_added_papers": 1, "of_questions": 4}


def test_comparisons_are_paired_and_cover_the_planned_pairs(env):
    comp = json.loads((run(env) / "summary.json").read_text())["comparisons"]
    assert comp["agent_minus_baseline"]["accuracy"]["estimate"] == 0
    assert comp["agent_minus_baseline"]["coverage"]["estimate"] == pytest.approx(-0.25)
    assert set(comp["baseline_minus_closed_book"]) == {"accuracy", "coverage",
                                                       "answered_only_accuracy"}  # fmt: skip
    assert comp["agent_minus_closed_book"]["coverage"]["estimate"] == pytest.approx(-0.5)
    assert set(comp["agent_minus_baseline"]) >= {"hit_at_1", "hit_at_5", "latency_p50_ms",
                                                 "cost_mean_usd"}  # fmt: skip


def test_claim_support_is_held_and_hand_checked(env):
    out = run(env)
    held = json.loads((out / score_m4.HELD_FILE).read_text())
    assert held["name"] == "Independent-judge estimated claim support"
    b, a = held["systems"]["baseline"], held["systems"]["agent"]
    # baseline claims: supported, partially, unsupported, supported -> strict 2/4, lenient 3/4
    assert (b["claim_strict"]["count"], b["claim_strict"]["n"]) == (2, 4)
    assert (b["claim_lenient"]["count"], b["claim_lenient"]["n"]) == (3, 4)
    assert (b["limitation_strict"]["count"], b["limitation_strict"]["n"]) == (1, 1)
    assert (a["claim_strict"]["count"], a["claim_strict"]["n"]) == (2, 2)
    assert b["questions_with_a_not_supported_item"]["count"] == 2  # q1 and q2
    assert held["agent_minus_baseline"]["claim_strict"]["estimate"] == pytest.approx(0.5)
    # nothing the owner might open (summary, raw, methodology) holds a judge label or figure
    summary_text = (out / "summary.json").read_text()
    assert "HELD" in summary_text and "claim_strict" not in summary_text
    for name in ("summary.json", "raw.jsonl", "methodology.md"):
        text = (out / name).read_text()
        assert "partially_supported" not in text and "unsupported" not in text, name
        assert '"label"' not in text, name


def test_refuses_when_a_figure_disagrees_with_the_stored_run_summary(env):
    path = env / RUNS["agent"] / "summary.json"
    stored = json.loads(path.read_text())
    path.write_text(json.dumps(stored | {"accuracy": 0.75}))
    with pytest.raises(SystemExit, match="Disagrees"):
        run(env)


def test_refuses_judgments_that_do_not_match_the_items_shown(env):
    path = env / JUDGE / "judgments.jsonl"
    path.write_text("".join(path.read_text().splitlines(keepends=True)[:-1]))
    with pytest.raises(SystemExit, match="judgments do not match"):
        run(env)
    assert not any(d.name.startswith("m4-final-") for d in env.iterdir())


def test_refuses_judgments_whose_blind_id_contradicts_the_blinding_key(env):
    path = env / JUDGE / "blinding_key.json"
    key = json.loads(path.read_text())
    key["2"] = {"SYSTEM_A": "agent", "SYSTEM_B": "baseline"}
    path.write_text(json.dumps(key))
    with pytest.raises(SystemExit, match="contradicts the blinding key"):
        run(env)


def test_refuses_a_judged_item_whose_citations_differ_from_the_run(env):
    path = env / JUDGE / "judgments.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["citation_ids"] = ["PMID:999"]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(SystemExit, match="differs from the run"):
        run(env)


def test_refuses_a_void_judge_run_a_dirty_tree_and_wrong_gold(env, monkeypatch):
    (env / JUDGE / "VOID.json").write_text("{}")
    with pytest.raises(SystemExit, match="void or unfinished"):
        run(env)
    (env / JUDGE / "VOID.json").unlink()
    monkeypatch.setattr(score_m4, "git_state", lambda: {"git_commit": "x", "git_dirty": True})
    with pytest.raises(SystemExit, match="clean commit"):
        run(env)
    clean = {"git_commit": "abc1234", "git_dirty": False}
    monkeypatch.setattr(score_m4, "git_state", lambda: clean)
    monkeypatch.setattr(score_m4, "load_manifest", lambda split: {"items": [
        {"pmid": p, "final_decision": "no"} for p in GOLD]})  # fmt: skip
    with pytest.raises(SystemExit, match="gold labels differ"):
        run(env)
