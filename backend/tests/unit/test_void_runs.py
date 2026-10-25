"""Runs that stop early are void (ADR-034): not scored, not counted as the test run, and the
rerun must test the identical system. Fake manifests and results in tmp_path; no network."""

import hashlib
import json
from datetime import UTC, datetime, timedelta

import pytest

from eval.runners import run_system

CLEAN = {"git_commit": "b" * 40, "git_dirty": False, "git_diff_sha256": None}
APPROVAL = ["--max-cost", "5", "--approval", "Owner: yes, run the test split once."]
FRESH = "Owner: the first run was void; yes, rerun it once."


class Clock:
    """datetime stand-in: each now() is one second later, so run IDs never collide."""

    t = datetime(2026, 10, 24, tzinfo=UTC)

    @classmethod
    def now(cls, tz=None):
        cls.t += timedelta(seconds=1)
        return cls.t


@pytest.fixture
def env(tmp_path, monkeypatch):
    results, manifests = tmp_path / "results", tmp_path / "manifests"
    results.mkdir()
    manifests.mkdir()
    items = [{"pmid": str(i), "question": f"Q{i}?", "final_decision": "yes"} for i in range(1, 11)]
    for split in ("dev", "test"):
        body = json.dumps({"split": split, "dataset": "d", "revision": "r", "items": items})
        (manifests / f"{split}.json").write_bytes(body.encode())
    digest = hashlib.sha256((manifests / "test.json").read_bytes()).hexdigest()
    (manifests / "test.sha256").write_text(f"{digest}  test.json\n")
    corpus = tmp_path / "corpus.json"
    corpus.write_text("{}")
    for name, value in [("RESULTS_DIR", results), ("MANIFESTS", manifests), ("REPO_ROOT", tmp_path),
                        ("CORPUS_MANIFEST", corpus), ("datetime", Clock)]:  # fmt: skip
        monkeypatch.setattr(run_system, name, value)
    monkeypatch.setattr(run_system, "git_state", lambda: CLEAN)
    monkeypatch.setattr(run_system, "voyage_client", lambda **kw: object())
    monkeypatch.setattr(run_system, "anthropic_client", lambda: object())
    monkeypatch.setattr(run_system, "connect", lambda: type("C", (), {"close": lambda s: None})())
    return results


def answers(monkeypatch, *outcomes):
    """Question n raises outcomes[n] (an exception class), or the last one for later questions."""
    calls = []

    def answer(question, **kwargs):
        calls.append(question)
        raise outcomes[min(len(calls), len(outcomes)) - 1]("fake")

    monkeypatch.setattr(run_system, "answer_baseline", answer)
    return calls


def run(split, *extra):
    flags = [run_system.FINAL_TEST_FLAG, *APPROVAL] if split == "test" else []
    run_system.main(["--system", "baseline", "--split", split, *flags, *extra])


def only(results, void):
    dirs = [d for d in sorted(results.iterdir()) if (d / run_system.VOID_FILE).exists() == void]
    return dirs


def test_five_failures_in_a_row_void_the_run(env, monkeypatch):
    calls = answers(monkeypatch, RuntimeError)
    with pytest.raises(SystemExit, match="VOID run"):
        run("dev")
    assert len(calls) == run_system.MAX_CONSECUTIVE_ERRORS  # stopped; no more questions sent
    (run_dir,) = only(env, void=True)
    void = json.loads((run_dir / run_system.VOID_FILE).read_text())
    assert "in a row failed" in void["reason"] and void["questions_done"] == 5
    assert not (run_dir / "summary.json").exists()


def test_failures_without_a_streak_are_error_rows_counted_incorrect(env, monkeypatch):
    answers(monkeypatch, RuntimeError)
    monkeypatch.setattr(run_system, "MAX_CONSECUTIVE_ERRORS", 11)  # streak rule out of reach
    run("dev")
    (run_dir,) = only(env, void=False)
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["complete"] is True and summary["errors"] == 10 and summary["accuracy"] == 0


def test_streak_counter_resets_after_a_success(env, monkeypatch):
    rows_seen = []

    def fake_row(item, answer):
        rows_seen.append(item["pmid"])
        return run_system.error_row(item, "ok-stub", 1.0) | {"error": None}

    calls = []

    def answer(question, **kwargs):
        calls.append(question)
        if len(calls) % 5 == 0:  # every 5th question succeeds: never 5 failures in a row
            return object()
        raise RuntimeError("fake")

    monkeypatch.setattr(run_system, "answer_baseline", answer)
    monkeypatch.setattr(run_system, "run_row", fake_row)
    run("dev")
    assert len(calls) == 10 and rows_seen == ["5", "10"]
    assert only(env, void=True) == []


def test_interrupt_mid_run_voids_and_reraises(env, monkeypatch):
    answers(monkeypatch, RuntimeError, KeyboardInterrupt)
    with pytest.raises(KeyboardInterrupt):
        run("dev")
    (run_dir,) = only(env, void=True)
    void = json.loads((run_dir / run_system.VOID_FILE).read_text())
    assert void["reason"] == "interrupted: KeyboardInterrupt" and void["questions_done"] == 1
    assert (run_dir / run_system.VOID_RAW).exists() and not (run_dir / "raw.jsonl").exists()


def test_void_test_run_does_not_use_up_the_test_set(env, monkeypatch):
    answers(monkeypatch, KeyboardInterrupt)
    with pytest.raises(KeyboardInterrupt):
        run("test")
    (void_dir,) = only(env, void=True)
    assert run_system.completed_test_runs("baseline") == []

    answers(monkeypatch, RuntimeError, ValueError)  # alternating -> no 5-failure streak
    monkeypatch.setattr(run_system, "MAX_CONSECUTIVE_ERRORS", 11)
    with pytest.raises(SystemExit, match="fresh typed approval"):
        run("test")  # same approval text as the void run: refused before any folder
    assert len(list(env.iterdir())) == 1
    run("test", "--approval", FRESH)  # allowed without --rerun-final-test
    (done,) = only(env, void=False)
    config = json.loads((done / "run_config.json").read_text())
    assert config["earlier_void_test_runs"] == [void_dir.name]
    assert run_system.completed_test_runs("baseline") == [done.name]
    with pytest.raises(SystemExit, match="already exists"):  # the once-only rule still holds
        run("test", "--approval", FRESH)


def test_rerun_after_a_void_run_must_test_the_identical_system(env, monkeypatch):
    answers(monkeypatch, KeyboardInterrupt)
    with pytest.raises(KeyboardInterrupt):
        run("test")
    monkeypatch.setattr(run_system, "SYSTEM_PROMPT", "a tweaked synthesizer prompt")
    answers(monkeypatch, RuntimeError)
    with pytest.raises(SystemExit, match="different system.*synthesizer_prompt_sha256"):
        run("test", "--approval", FRESH)
    assert len(list(env.iterdir())) == 1  # refused before creating a folder


def complete_run(results, name, pmids, question_ids=None):
    d = results / name
    d.mkdir()
    config = {"run_id": name, "question_ids": question_ids or pmids}
    (d / "run_config.json").write_text(json.dumps(config))
    (d / "raw.jsonl").write_text("".join(json.dumps({"pmid": p}) + "\n" for p in pmids))
    (d / "summary.json").write_text("{}")
    return d


def test_scorer_gate_accepts_only_complete_runs_on_identical_questions(env):
    a = complete_run(env, "a", ["1", "2", "3"])
    b = complete_run(env, "b", ["1", "2", "3"])
    assert run_system.require_same_questions(a, b) == ["1", "2", "3"]

    partial = complete_run(env, "partial", ["1", "2"], question_ids=["1", "2", "3"])
    other = complete_run(env, "other", ["1", "2", "4"])
    void = complete_run(env, "void", ["1", "2", "3"])
    (void / run_system.VOID_FILE).write_text("{}")
    unfinished = complete_run(env, "unfinished", ["1", "2", "3"])
    (unfinished / "summary.json").unlink()
    for bad, message in [(partial, "planned question IDs"), (void, "void or unfinished"),
                         (unfinished, "void or unfinished")]:  # fmt: skip
        with pytest.raises(SystemExit, match=message):
            run_system.require_complete_run(bad)
    with pytest.raises(SystemExit, match="different question IDs"):
        run_system.require_same_questions(a, other)
