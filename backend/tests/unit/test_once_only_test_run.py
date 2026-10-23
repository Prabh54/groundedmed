"""The frozen test split runs once per system unless overridden with a reason (ADR-031).

Uses a fake test manifest and results folder in tmp_path: the real test set is never read."""

import hashlib
import json

import pytest

from eval.runners import run_system

CLEAN = {"git_commit": "a" * 40, "git_dirty": False, "git_diff_sha256": None}
REASON = "Provider outage: 40% error rows in the first run"


def fake_run(results, run_id, split="test", system="baseline", completed=True):
    run_dir = results / run_id
    run_dir.mkdir()
    config = {"run_id": run_id, "split": split, "system": system}
    (run_dir / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
    if completed:
        (run_dir / "summary.json").write_text("{}", encoding="utf-8")


@pytest.fixture
def results(tmp_path, monkeypatch):
    out = tmp_path / "results"
    out.mkdir()
    monkeypatch.setattr(run_system, "RESULTS_DIR", out)
    return out


def test_first_test_run_is_allowed_without_override(results):
    assert run_system.check_once_only("test", "baseline", rerun=False, reason=None) is None


def test_completed_test_run_blocks_a_second_one(results):
    fake_run(results, "test-baseline-1")
    with pytest.raises(SystemExit, match="already exists"):
        run_system.check_once_only("test", "baseline", rerun=False, reason=None)


@pytest.mark.parametrize(
    "rerun,reason",
    [(True, None), (False, REASON), (True, "because"), (True, "                         ")],
)
def test_override_needs_both_the_flag_and_a_real_reason(results, rerun, reason):
    fake_run(results, "test-baseline-1")
    with pytest.raises(SystemExit):
        run_system.check_once_only("test", "baseline", rerun=rerun, reason=reason)


def test_override_with_flag_and_reason_is_allowed_and_recorded(results):
    fake_run(results, "test-baseline-1")
    override = run_system.check_once_only("test", "baseline", rerun=True, reason=REASON)
    assert override["reason"] == REASON
    assert override["previous_completed_test_runs"] == ["test-baseline-1"]


def test_only_completed_test_runs_of_the_same_system_count(results):
    fake_run(results, "test-baseline-crashed", completed=False)  # no summary.json
    fake_run(results, "test-agent-1", system="agent")
    fake_run(results, "dev-baseline-1", split="dev")
    assert run_system.check_once_only("test", "baseline", rerun=False, reason=None) is None


def test_override_flags_are_refused_where_they_mean_nothing(results):
    with pytest.raises(SystemExit):
        run_system.check_once_only("dev", "baseline", rerun=True, reason=REASON)
    with pytest.raises(SystemExit):  # nothing to override yet
        run_system.check_once_only("test", "baseline", rerun=True, reason=REASON)


def test_end_to_end_second_run_refused_then_override_written(results, tmp_path, monkeypatch):
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    items = [{"pmid": "1", "question": "Q?", "final_decision": "yes"}]
    body = json.dumps({"split": "test", "dataset": "d", "revision": "r", "items": items}).encode()
    (manifests / "test.json").write_bytes(body)
    (manifests / "test.sha256").write_text(hashlib.sha256(body).hexdigest() + "  test.json\n")
    monkeypatch.setattr(run_system, "MANIFESTS", manifests)
    monkeypatch.setattr(run_system, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(run_system, "git_state", lambda: CLEAN)
    monkeypatch.setattr(run_system, "voyage_client", lambda **kw: object())
    monkeypatch.setattr(run_system, "anthropic_client", lambda: object())
    monkeypatch.setattr(run_system, "connect", lambda: type("C", (), {"close": lambda s: None})())
    calls = []

    def answer(question, **kwargs):  # any call is recorded; the run stores it as an error row
        calls.append(question)
        raise RuntimeError("fake outage")

    monkeypatch.setattr(run_system, "answer_baseline", answer)
    fake_run(results, "test-baseline-first")
    base = ["--system", "baseline", "--split", "test", run_system.FINAL_TEST_FLAG]

    with pytest.raises(SystemExit, match="already exists"):
        run_system.main(base)
    assert calls == [] and len(list(results.iterdir())) == 1  # refused before any call or folder

    run_system.main([*base, run_system.RERUN_FLAG, "--reason", REASON])
    (new,) = [d for d in results.iterdir() if d.name != "test-baseline-first"]
    override = json.loads((new / "override.json").read_text(encoding="utf-8"))
    assert override["reason"] == REASON
    assert override["previous_completed_test_runs"] == ["test-baseline-first"]
    config = json.loads((new / "run_config.json").read_text(encoding="utf-8"))
    assert config["test_rerun_override"]["reason"] == REASON
