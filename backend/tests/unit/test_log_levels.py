"""Per-request log lines carry an explicit `level` (stdout, JSON), so a normal request is "info"
and only real failures are "error" in a hosting log viewer."""

import json

from backend.app.main import log, log_level


def test_a_normal_answered_request_is_info():
    record = {
        "request_id": "r",
        "mode": "agent",
        "verdict": "yes",
        "abstained": False,
        "error": None,
    }
    assert log_level(record) == "info"


def test_refusals_and_cache_hits_are_info_not_errors():
    assert log_level({"event": "cache_hit", "verdict": "yes", "error": None}) == "info"
    assert (
        log_level({"abstained": True, "abstain_reason": "weak_evidence", "error": None}) == "info"
    )
    assert log_level({"event": "client", "peer": "10.0.0.1"}) == "info"


def test_failures_are_errors():
    assert log_level({"event": "error", "stage": "retrieval"}) == "error"
    assert log_level({"abstain_reason": "error", "error": "planner: APIConnectionError"}) == "error"


def test_rate_limits_and_abandoned_requests_are_warnings():
    assert log_level({"event": "rate_limited", "reason": "daily_cap"}) == "warning"
    assert log_level({"event": "cancelled", "stopped_before": "critic"}) == "warning"


def test_log_prints_one_json_line_to_stdout_with_the_level_first(capsys):
    log({"request_id": "r", "event": "error", "stage": "query"})
    log({"request_id": "r2", "verdict": "yes"})
    out, err = capsys.readouterr()
    assert err == ""
    first, second = (json.loads(line) for line in out.splitlines())
    assert list(first)[0] == "level" and first["level"] == "error" and first["stage"] == "query"
    assert second["level"] == "info" and second["verdict"] == "yes"


def test_an_explicit_level_in_the_record_is_not_overridden_by_the_default():
    # records never set one today; the record's own key wins if it ever does
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        log({"level": "debug", "event": "x"})
    assert json.loads(buf.getvalue())["level"] == "debug"
