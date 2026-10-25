"""Independent judge rubric v1 is frozen and blind (M4-T1, eval/rubric/judge_v1.md). Offline."""

import re
import string

import pytest

from eval import judge_rubric as jr

# Frozen at M4-T1 (2026-10-24), before any test-split run. Never edit these to match changed files:
# a changed rubric or prompt is a new version (judge_v2) with a DECISIONS.md entry.
FROZEN_JUDGE_V1_SHA256 = {
    "judge_v1.md": "9954d19e1b91575899dd2c2d62e2b2f644a05768b264a87b43362694179d026b",
    "judge_v1_prompt.txt": "b240c38ad57deceb651f1357112ceb362848afd392a53d9932d473d537b27a95",
    "judge_v1_schema.json": "5d35d697c96b13a639920588ab37e5018c0bbbb66483e52dd9bdeeec71342915",
}

PLACEHOLDERS = {
    "SYSTEM_INSTRUCTION": set(),
    "USER_TEMPLATE": {"question", "item_ids", "system_blocks"},
    "SYSTEM_BLOCK_TEMPLATE": {"system_label", "items"},
    "EMPTY_SYSTEM_BLOCK_TEMPLATE": {"system_label"},
    "ITEM_TEMPLATE": {"item_id", "statement", "passages"},
    "PASSAGE_TEMPLATE": {"citation_id", "passage_text"},
}
# Words that would tell the judge which system is which, or who built it (rubric §3).
UNBLINDING_WORDS = (
    "agent", "baseline", "critic", "planner", "improved", "claude", "anthropic", "sonnet",
    "groundedmed", "pubmedqa", "gold", "reference", "final_decision", "long_answer",
)  # fmt: skip


def placeholders(template: str) -> set[str]:
    return {field for _, field, _, _ in string.Formatter().parse(template) if field is not None}


def test_files_match_committed_hashes():
    assert jr.file_hashes() == jr.frozen_hashes() == FROZEN_JUDGE_V1_SHA256
    jr.check_frozen()


def test_check_frozen_refuses_a_changed_file(monkeypatch):
    changed = {**jr.frozen_hashes(), "judge_v1.md": "0" * 64}
    monkeypatch.setattr(jr, "frozen_hashes", lambda: changed)
    with pytest.raises(SystemExit):
        jr.check_frozen()


def test_prompt_sections_and_placeholders():
    sections = jr.load_prompt()
    assert {name: placeholders(body) for name, body in sections.items()} == PLACEHOLDERS


def test_prompt_never_identifies_a_system():
    text = (jr.RUBRIC_DIR / "judge_v1_prompt.txt").read_text(encoding="utf-8").lower()
    found = [w for w in UNBLINDING_WORDS if re.search(rf"\b{re.escape(w)}\b", text)]
    assert found == []
    for label in jr.SYSTEM_LABELS:
        assert label in jr.load_prompt()["SYSTEM_INSTRUCTION"]


def test_labels_agree_across_schema_prompt_and_rubric():
    schema = jr.load_schema()
    item = schema["properties"]["judgments"]["items"]
    assert tuple(item["properties"]["label"]["enum"]) == jr.LABELS
    assert set(item["required"]) == {"item_id", "evidence_quote", "reason", "label"}
    instruction = jr.load_prompt()["SYSTEM_INSTRUCTION"]
    rubric = (jr.RUBRIC_DIR / "judge_v1.md").read_text(encoding="utf-8")
    for label in jr.LABELS:
        assert f'"{label}"' in instruction
        assert f"`{label}`" in rubric


def test_templates_render_without_leftover_braces():
    s = jr.load_prompt()
    passage = s["PASSAGE_TEMPLATE"].format(citation_id="PMID:1", passage_text="RESULTS: x {y}")
    item = s["ITEM_TEMPLATE"].format(item_id="A.c1", statement="A claim.", passages=passage)
    block = s["SYSTEM_BLOCK_TEMPLATE"].format(system_label="SYSTEM_A", items=item)
    empty = s["EMPTY_SYSTEM_BLOCK_TEMPLATE"].format(system_label="SYSTEM_B")
    user = s["USER_TEMPLATE"].format(
        question="Q?", item_ids="A.c1", system_blocks=block + "\n\n" + empty
    )
    assert '<ANSWER_SET name="SYSTEM_A">' in user and "(no items)" in user
    assert "{" not in user.replace("{y}", "")
