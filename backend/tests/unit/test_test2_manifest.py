"""Integrity tests for the second frozen test set, test2 (ADR-043). Offline.

These tests read only PMIDs, label counts and hashes; they never print a question or a label of
an item.
"""

import hashlib
import json
import re
from pathlib import Path

import pytest

from eval.runners import run_system
from scripts import lock_test2, prepare_dataset

ROOT = Path(__file__).resolve().parents[3]
MANIFESTS = ROOT / "data" / "manifests"

FRONTEND_SOURCE = (
    "frontend/app",
    "frontend/components",
    "frontend/lib",
    "frontend/scripts",
    "frontend/e2e",
    "frontend/e2e-live",
)
# M15-T10: the only test2 folders the /eval page may read (finished, scored runs).
TEST2_RESULT_DIRS = {
    "eval/results/m15-test2-score-20261110-225734-bc0d65f",
    "eval/results/test2-baseline-20261110-205943-9553791",
    "eval/results/test2-agent-20261110-213115-bd5773d",
}
SOURCE_SUFFIXES = {".py", ".ts", ".tsx", ".js", ".mjs"}
# Frozen on 2026-11-10 (M15-T1). Changing test2 needs a recorded reason; never edit this to match.
FROZEN_TEST2_SHA256 = "de7021939eefe70781bec68ec2b9738472fa7dfc00f5abc588161cbdbaa36f06"


def pmids(name: str) -> set[str]:
    return {item["pmid"] for item in json.loads((MANIFESTS / f"{name}.json").read_bytes())["items"]}


def test2_manifest() -> dict:
    return json.loads((MANIFESTS / "test2.json").read_bytes())


def test_test2_matches_its_frozen_hash():
    digest = hashlib.sha256((MANIFESTS / "test2.json").read_bytes()).hexdigest()
    assert (MANIFESTS / "test2.sha256").read_text(encoding="utf-8").split() == [
        digest,
        "test2.json",
    ]
    assert digest == FROZEN_TEST2_SHA256


def test_test2_rebuilds_byte_identical_from_the_reserve():
    rebuilt = prepare_dataset.to_json(
        lock_test2.build_test2((MANIFESTS / "reserve.json").read_bytes())
    )
    assert rebuilt == (MANIFESTS / "test2.json").read_text(encoding="utf-8")


def test_test2_records_the_reserve_it_came_from():
    source = test2_manifest()["source"]
    assert source["file"] == "reserve.json"
    assert source["sha256"] == hashlib.sha256((MANIFESTS / "reserve.json").read_bytes()).hexdigest()


def test_test2_is_inside_the_reserve_and_apart_from_test_and_dev():
    test2 = pmids("test2")
    assert len(test2) == 400
    assert test2 <= pmids("reserve")
    assert not test2 & pmids("test")
    assert not test2 & pmids("dev")


def test_test2_label_mix_and_exclusion():
    manifest = test2_manifest()
    counts = {label: 0 for label in prepare_dataset.LABEL_ORDER}
    for item in manifest["items"]:
        counts[item["final_decision"]] += 1
    assert counts == manifest["label_counts"] == {"yes": 220, "no": 136, "maybe": 44}
    assert (
        manifest["count"] == 400 and manifest["split"] == "test2" and manifest["seed"] == 20261110
    )
    assert "24519615" not in pmids("test2") and manifest["excluded"] == ["24519615"]


def test_lock_script_refuses_to_change_an_existing_test2(tmp_path, monkeypatch):
    (tmp_path / "reserve.json").write_bytes((MANIFESTS / "reserve.json").read_bytes())
    (tmp_path / "test2.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(lock_test2, "MANIFESTS", tmp_path)
    with pytest.raises(prepare_dataset.FrozenManifestError):
        lock_test2.main()


def test_only_the_lock_script_and_the_guarded_runner_read_test2():
    """M15-T9: the runner accepts test2 behind the final-test guards (test_test2_runner_guards.py)
    and score_test2 scores complete runs; nothing else (judge, app, scripts) loads it. M15-T10: the
    /eval page reads the finished test2 *result* files (never the manifest or its questions)."""
    with pytest.raises(SystemExit, match="without --i-understand-this-is-the-final-test"):
        run_system.main(["--system", "agent", "--split", "test2"])  # refused before any read
    readers = [
        p.relative_to(ROOT).as_posix()
        for folder in ("eval", "scripts", "backend/app", *FRONTEND_SOURCE)
        for p in (ROOT / folder).rglob("*")
        if p.suffix in SOURCE_SUFFIXES and "test2" in p.read_text(encoding="utf-8")
    ]
    assert sorted(readers) == [
        "eval/runners/run_system.py",
        "eval/runners/score_test2.py",
        "frontend/lib/checkEval.test.ts",
        "frontend/lib/checkEval.ts",
        "frontend/lib/evalFormat.ts",
        "scripts/lock_test2.py",
    ]
    for page_file in ("checkEval.test.ts", "checkEval.ts", "evalFormat.ts"):
        text = (ROOT / "frontend" / "lib" / page_file).read_text(encoding="utf-8")
        for banned in ("manifests/test2", "test2.json", "test2.sha256", ".jsonl", "raw."):
            assert banned not in text, (page_file, banned)  # summaries only, never questions
        for folder in re.findall(r"eval/results/[\w.-]*test2[\w.-]*", text):
            assert folder in TEST2_RESULT_DIRS, (page_file, folder)


def test_the_eval_page_reads_only_test2_summaries_and_run_configs():
    text = (ROOT / "frontend" / "lib" / "evalFormat.ts").read_text(encoding="utf-8")
    entries = re.findall(r"^\s*(t2_\w+): `\$\{T2_\w+\}/([^`]+)`,$", text, re.MULTILINE)
    assert len(entries) == 6
    assert {name for _, name in entries} == {"summary.json", "run_config.json"}
