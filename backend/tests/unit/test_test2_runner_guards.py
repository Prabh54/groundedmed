"""test2 (ADR-043, M15-T9) gets the same final-test guards as the original test split, and its
once-only rule is separate from the original test set's.

Uses fake manifests and a fake results folder in tmp_path; the real test2 questions are never
read."""

import hashlib
import json

import pytest

from eval.runners import run_system

CLEAN = {"git_commit": "a" * 40, "git_dirty": False, "git_diff_sha256": None}
APPROVAL = "Owner: yes, run test2 once for this system."


def fake_run(results, run_id, split, system="baseline", completed=True, **config):
    run_dir = results / run_id
    run_dir.mkdir()
    config = {"run_id": run_id, "split": split, "system": system, **config}
    (run_dir / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
    if completed:
        (run_dir / "summary.json").write_text("{}", encoding="utf-8")


@pytest.fixture
def results(tmp_path, monkeypatch):
    out = tmp_path / "results"
    out.mkdir()
    monkeypatch.setattr(run_system, "RESULTS_DIR", out)
    return out


def test_test2_is_a_frozen_final_split():
    assert run_system.FINAL_SPLITS == ("test", "test2")


def test_test2_needs_the_flag_a_clean_tree_and_no_limit():
    clean, dirty = {"git_dirty": False}, {"git_dirty": True}
    with pytest.raises(SystemExit, match="without"):
        run_system.check_allowed("test2", final_test_ok=False, git=clean)
    with pytest.raises(SystemExit, match="dirty"):
        run_system.check_allowed("test2", final_test_ok=True, git=dirty)
    with pytest.raises(SystemExit, match="partial"):
        run_system.check_allowed("test2", final_test_ok=True, git=clean, limit=5)
    run_system.check_allowed("test2", final_test_ok=True, git=clean)


def test_tampered_test2_manifest_is_refused(tmp_path, monkeypatch):
    (tmp_path / "test2.json").write_text('{"split": "test2", "items": []}', encoding="utf-8")
    (tmp_path / "test2.sha256").write_text("0" * 64 + "  test2.json\n", encoding="utf-8")
    monkeypatch.setattr(run_system, "MANIFESTS", tmp_path)
    with pytest.raises(SystemExit, match="test2.sha256"):
        run_system.load_manifest("test2")


def test_test2_hash_is_checked_against_test2_sha256_not_test_sha256(tmp_path, monkeypatch):
    body = b'{"split": "test2", "items": []}'
    (tmp_path / "test2.json").write_bytes(body)
    (tmp_path / "test2.sha256").write_text(hashlib.sha256(body).hexdigest() + "  test2.json\n")
    (tmp_path / "test.sha256").write_text("0" * 64 + "  test.json\n", encoding="utf-8")
    monkeypatch.setattr(run_system, "MANIFESTS", tmp_path)
    assert run_system.load_manifest("test2")["split"] == "test2"


def test_once_only_counts_each_frozen_split_separately(results):
    fake_run(results, "test-agent-v1", "test", system="agent")  # the v1.0.0 run
    assert run_system.check_once_only("test2", "agent", rerun=False, reason=None) is None
    fake_run(results, "test2-agent-1", "test2", system="agent")
    with pytest.raises(SystemExit, match="test2-agent-1"):
        run_system.check_once_only("test2", "agent", rerun=False, reason=None)
    assert run_system.completed_test_runs("agent", "test") == ["test-agent-v1"]
    assert run_system.completed_test_runs("agent", "test2") == ["test2-agent-1"]
    assert run_system.check_once_only("test2", "baseline", rerun=False, reason=None) is None


@pytest.mark.parametrize(
    "key",
    [
        "agent_synthesizer_effort",
        "retry_prompt",
        "retry_user_template_sha256",
        "veto_rule",
        "agent_rules_sha256",
    ],
)
def test_rerun_after_a_void_test2_run_cannot_change_the_adr044_settings(results, key):
    fake_run(results, "test2-agent-void", "test2", system="agent", completed=False,
             owner_approval="An earlier approval text", **{key: "first"})  # fmt: skip
    config = {"system": "agent", "split": "test2", "owner_approval": APPROVAL}
    with pytest.raises(SystemExit, match=f"Void test2 run test2-agent-void .*{key}"):
        run_system.check_same_identity_as_void_runs(config | {key: "changed"})
    assert run_system.check_same_identity_as_void_runs(config | {key: "first"}) == [
        "test2-agent-void"
    ]


def test_void_identity_check_stays_within_the_split(results):
    fake_run(results, "test-agent-void", "test", system="agent", completed=False,
             owner_approval=APPROVAL, llm_model="other-model")  # fmt: skip
    config = {"system": "agent", "split": "test2", "owner_approval": APPROVAL, "llm_model": "m"}
    assert run_system.check_same_identity_as_void_runs(config) == []
    fake_run(results, "test2-agent-void", "test2", system="agent", completed=False,
             owner_approval="An earlier approval text", llm_model="m",
             agent_synthesizer_effort="medium")  # fmt: skip
    with pytest.raises(SystemExit, match="agent_synthesizer_effort"):
        run_system.check_same_identity_as_void_runs(config | {"agent_synthesizer_effort": "high"})


def test_end_to_end_test2_runs_once_and_records_its_hash(results, tmp_path, monkeypatch):
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    items = [{"pmid": "1", "question": "Q?", "final_decision": "yes"}]
    body = json.dumps({"split": "test2", "dataset": "d", "revision": "r", "items": items}).encode()
    (manifests / "test2.json").write_bytes(body)
    (manifests / "test2.sha256").write_text(hashlib.sha256(body).hexdigest() + "  test2.json\n")
    monkeypatch.setattr(run_system, "MANIFESTS", manifests)
    monkeypatch.setattr(run_system, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(run_system, "git_state", lambda: CLEAN)
    monkeypatch.setattr(run_system, "voyage_client", lambda **kw: object())
    monkeypatch.setattr(run_system, "anthropic_client", lambda: object())
    monkeypatch.setattr(run_system, "connect", lambda: type("C", (), {"close": lambda s: None})())
    calls = []

    def answer(question, **kwargs):  # recorded; stored as an error row
        calls.append(question)
        raise RuntimeError("fake outage")

    monkeypatch.setattr(run_system, "answer_baseline", answer)
    fake_run(results, "test-baseline-v1", "test")  # a completed run of the OTHER frozen split
    base = ["--system", "baseline", "--split", "test2", run_system.FINAL_TEST_FLAG,
            "--max-cost", "6", "--approval", APPROVAL]  # fmt: skip

    with pytest.raises(SystemExit, match="partial"):
        run_system.main([*base, "--limit", "1"])
    monkeypatch.setitem(run_system.EST_COST_PER_QUESTION, "baseline", 5.0)  # a paid-size run
    with pytest.raises(SystemExit, match="owner's typed OK"):
        run_system.main(base[:-2])  # no --approval
    monkeypatch.setitem(run_system.EST_COST_PER_QUESTION, "baseline", 0.015)
    assert calls == [] and len(list(results.iterdir())) == 1  # refused before any call or folder
    run_system.main(base)
    (new,) = [d for d in results.iterdir() if d.name.startswith("test2-")]
    config = json.loads((new / "run_config.json").read_text(encoding="utf-8"))
    assert config["split"] == "test2" and config["question_ids"] == ["1"]
    assert config["manifest_sha256"] == hashlib.sha256(body).hexdigest()
    assert config["owner_approval"] == APPROVAL and config["max_cost_usd"] == 6.0
    assert (new / "summary.json").exists() and calls == ["Q?"]

    with pytest.raises(SystemExit, match="already exists"):  # once per system on test2
        run_system.main(base)
    assert calls == ["Q?"] and len(list(results.iterdir())) == 2
