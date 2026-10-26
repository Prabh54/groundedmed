"""Second-AI cross-check (eval/runners/crosscheck.py, ADR-037): join on the committed blind sheet,
binary agreement per judge group, refusals. Fake judge run, sheet and answers in tmp_path."""

import csv
import json

import pytest

from eval.runners import audit_sheet, crosscheck, judge

from .test_audit_sheet import RUN, judgments

SCORING = "m4-final-x"


@pytest.fixture
def env(tmp_path, monkeypatch):
    results, audits = tmp_path / "results", tmp_path / "audits"
    (results / RUN).mkdir(parents=True)
    (results / RUN / "judgments.jsonl").write_text(
        "".join(json.dumps(j) + "\n" for j in judgments())
    )
    (results / RUN / "summary.json").write_text("{}")
    (results / SCORING).mkdir()
    (results / SCORING / "run_config.json").write_text(json.dumps({"inputs": {"judge": RUN}}))
    held = {"name": "Independent-judge estimated claim support", "caveat": "c", "systems": {
        s: {k: {"count": 1, "n": 2, "value": 0.5, "wilson95": [0.1, 0.9],
                "question_bootstrap95": [0.2, 0.8]}
            for k in ("claim_strict", "claim_lenient", "limitation_strict", "limitation_lenient",
                      "questions_with_a_not_supported_item")} | {"not_judged": 0}
        for s in ("baseline", "agent")},
        "agent_minus_baseline": {"claim_strict": {"estimate": 0.0,
                                                  "ci95": [-0.1, 0.1]}}}  # fmt: skip
    (results / SCORING / "claim_support.HELD-until-audit.json").write_text(json.dumps(held))
    corpus = tmp_path / "corpus.jsonl"
    section = lambda q: {"label": "RESULTS", "text": f"Results of {5000 + q}."}  # noqa: E731
    corpus.write_text("".join(json.dumps({"pmid": str(5000 + q), "sections": [section(q)]}) + "\n"
                              for q in range(30)))  # fmt: skip
    for mod in (audit_sheet, crosscheck):
        monkeypatch.setattr(mod, "RESULTS_DIR", results)
        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(audit_sheet, "AUDITS_DIR", audits)
    monkeypatch.setattr(judge, "CORPUS", corpus)
    monkeypatch.setattr(crosscheck, "git_state", lambda: {"git_commit": "abc1234",
                                                          "git_dirty": False})  # fmt: skip
    monkeypatch.setattr(crosscheck, "committed", lambda path: True)
    audit_sheet.main(["--judge-run", RUN])
    return {"results": results, "audits": audits}


def order():
    return audit_sheet.shuffled(audit_sheet.draw(judgments()))


def write_answers(env, answer_for):
    path = env["audits"] / f"{RUN}-second-ai-answers.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "answer", "note"])
        for i, item in enumerate(order(), start=1):
            w.writerow([i, answer_for(item), "n"])
    return path


def run():
    crosscheck.main(["--judge-run", RUN, "--scoring-run", SCORING])


def test_agreement_is_binary_per_group_and_hand_checked(env):
    # fake draw: baseline 5 rejected + 10 accepted, agent 2 rejected + 13 accepted.
    # Second AI: agrees on every accepted claim except the agent's first two; says Not supported
    # on every rejected claim except one baseline claim it marks Unsure.
    seen = {"agent_s": 0, "base_r": 0}

    def answer(item):
        if item["sample_group"] == "judge_supported":
            if item["system"] == "agent" and seen["agent_s"] < 2:
                seen["agent_s"] += 1
                return "Not supported"
            return "Supported"
        if item["system"] == "baseline" and seen["base_r"] == 0:
            seen["base_r"] += 1
            return "Unsure"
        return "Not supported"

    write_answers(env, answer)
    run()
    (out,) = [d for d in env["results"].iterdir() if d.name.startswith("m4-crosscheck-")]
    a = json.loads((out / "agreement.json").read_text())
    s, r = a["judge_supported"], a["judge_rejected"]
    assert (s["baseline"]["agree"], s["baseline"]["compared"]) == (10, 10)
    assert (s["agent"]["agree"], s["agent"]["compared"]) == (11, 13)
    assert (s["pooled"]["agree"], s["pooled"]["compared"]) == (21, 23)
    assert (r["baseline"]["agree"], r["baseline"]["compared"], r["baseline"]["unsure"]) == (4, 4, 1)
    assert (r["agent"]["agree"], r["agent"]["compared"]) == (2, 2)
    # "unsupported" judged claims answered Not supported count as agreement (binary rule)
    assert crosscheck.agrees("Not supported", "partially_supported")
    assert not crosscheck.agrees("Supported", "partially_supported")
    merged = list(csv.DictReader((env["audits"] / f"{RUN}-second-ai-crosscheck.csv").open(
        encoding="utf-8-sig")))  # fmt: skip
    assert list(merged[0]) == crosscheck.COLUMNS and len(merged) == 30
    assert all(m["sample_purpose"].startswith("SECOND-AI CROSS-CHECK ONLY") for m in merged)
    text = (out / "report.md").read_text()
    assert "second-AI cross-check" in text and "NOT a human audit" in text
    assert "manual audit" not in text.lower()


def test_refuses_if_the_committed_sheet_differs_from_the_rebuilt_draw(env):
    sheet = env["audits"] / f"{RUN}-owner-worksheet.csv"
    rows = list(csv.DictReader(sheet.open(encoding="utf-8-sig")))
    rows[0]["claim"], rows[1]["claim"] = rows[1]["claim"], rows[0]["claim"]
    with sheet.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    write_answers(env, lambda item: "Supported")
    with pytest.raises(SystemExit, match="differs from the rebuilt draw"):
        run()


@pytest.mark.parametrize("bad", ["", "supported", "Maybe"])
def test_refuses_missing_or_unknown_answers(env, bad):
    path = write_answers(env, lambda item: "Supported")
    lines = path.read_text().splitlines()
    lines[3] = f"3,{bad},n"
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(SystemExit, match="allowed answer"):
        run()


def test_refuses_duplicate_rows_uncommitted_answers_and_overwrite(env, monkeypatch):
    path = write_answers(env, lambda item: "Supported")
    good = path.read_text()
    path.write_text(good.replace("\n2,", "\n1,", 1))
    with pytest.raises(SystemExit, match="exactly once"):
        run()
    path.write_text(good)
    monkeypatch.setattr(crosscheck, "committed", lambda p: False)
    with pytest.raises(SystemExit, match="committed before the join"):
        run()
    monkeypatch.setattr(crosscheck, "committed", lambda p: True)
    run()
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        run()
