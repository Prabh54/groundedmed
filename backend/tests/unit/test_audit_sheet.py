"""M4-T5 blind audit sheet (eval/runners/audit_sheet.py): rubric v2 §7 draw and ADR-036 blinding.
Fake judgments and corpus in tmp_path; no network."""

import csv
import json

import pytest

from eval.runners import audit_sheet, judge

RUN = "judge-test-x"
# baseline: 8 judge-rejected claims (more than 5); agent: 2 (fallback fills from supported)
REJECTED = {"baseline": 8, "agent": 2}
LABEL_WORDS = ("partially_supported", "unsupported", "judge_rejected", "judge_supported")


def judgments():
    out = []
    for system in ("baseline", "agent"):
        for q in range(30):
            pmid = str(5000 + q)
            label = "unsupported" if q < REJECTED[system] else "supported"
            if q == 29:
                label = "not_judged"
            for item_id, kind in (("c1", "claim"), ("l1", "limitation")):
                out.append({"pmid": pmid, "item_id": item_id, "kind": kind, "system": system,
                            "blind_id": f"A.{item_id}", "citation_ids": [f"PMID:{pmid}"],
                            "text": f"Finding {item_id} about paper {pmid}, item {len(out)}.",
                            "label": label, "evidence_quote": f"QUOTE-{len(out)}",
                            "reason": f"REASON-{len(out)}"})  # fmt: skip
    return out


@pytest.fixture
def env(tmp_path, monkeypatch):
    results, audits = tmp_path / "results", tmp_path / "audits"
    (results / RUN).mkdir(parents=True)
    (results / RUN / "judgments.jsonl").write_text(
        "".join(json.dumps(j) + "\n" for j in judgments())
    )
    (results / RUN / "summary.json").write_text("{}")
    corpus = tmp_path / "corpus.jsonl"
    section = lambda q: {"label": "RESULTS", "text": f"Results of {5000 + q}."}  # noqa: E731
    docs = [{"pmid": str(5000 + q), "sections": [section(q)]} for q in range(30)]
    corpus.write_text("".join(json.dumps(d) + "\n" for d in docs))
    monkeypatch.setattr(audit_sheet, "RESULTS_DIR", results)
    monkeypatch.setattr(audit_sheet, "AUDITS_DIR", audits)
    monkeypatch.setattr(audit_sheet, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(judge, "CORPUS", corpus)
    return audits / f"{RUN}-owner-worksheet.csv"


def test_draw_follows_rubric_v2_section_7():
    sample = audit_sheet.draw(judgments())
    for system, want_r in (("baseline", 5), ("agent", 2)):  # agent: all 2 rejected, then fill
        mine = [s for s in sample if s["system"] == system]
        assert len(mine) == 15
        assert sum(s["sample_group"] == "judge_rejected" for s in mine) == want_r
        assert all(s["kind"] == "claim" and s["label"] != "not_judged" for s in mine)
    assert sample == audit_sheet.draw(judgments())  # seeded: same draw every time


def test_sheet_hides_system_and_judge_verdict(env):
    audit_sheet.main(["--judge-run", RUN])
    raw = env.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")  # UTF-8 BOM so Excel reads it correctly
    rows = list(csv.DictReader(env.open(encoding="utf-8-sig")))
    assert list(rows[0]) == audit_sheet.COLUMNS and len(rows) == 30
    assert all(r[audit_sheet.ANSWER] == "" and r["my note (optional)"] == "" for r in rows)
    text = raw.decode("utf-8-sig")
    for hidden in ("baseline", "agent", "QUOTE-", "REASON-", *LABEL_WORDS):
        assert hidden not in text, hidden
    assert "Results of" in text  # the cited passage text is there


def test_rows_are_shuffled_so_order_reveals_nothing(env):
    audit_sheet.main(["--judge-run", RUN])
    rows = list(csv.DictReader(env.open(encoding="utf-8-sig")))
    sample = audit_sheet.draw(judgments())
    by_text = {s["text"]: s for s in sample}
    order = [by_text[r["claim"]] for r in rows]
    systems = [s["system"] for s in order]
    rejected_rows = [i for i, s in enumerate(order) if s["sample_group"] == "judge_rejected"]
    assert systems != sorted(systems) and systems != sorted(systems, reverse=True)
    assert rejected_rows != list(range(len(rejected_rows)))  # not all at the top


def test_never_overwrites_the_owner_sheet_and_refuses_a_void_run(env):
    audit_sheet.main(["--judge-run", RUN])
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        audit_sheet.main(["--judge-run", RUN])
    env.unlink()
    (env.parent.parent / "results" / RUN / "VOID.json").write_text("{}")
    with pytest.raises(SystemExit, match="void or unfinished"):
        audit_sheet.main(["--judge-run", RUN])
