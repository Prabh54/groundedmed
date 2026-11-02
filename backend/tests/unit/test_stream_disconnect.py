"""M6-F2: when the visitor leaves /query/stream mid-answer, no further model call starts; the step
already running finishes and its cost is still logged.

The app is driven at the ASGI level, as uvicorn drives it (spec 2.3: Starlette listens for the
disconnect) and as a spec 2.4 server would (the app must check for it itself). Fake model, Voyage
and database from test_api_agent.py; no network.
"""

import asyncio
import json
import threading
import time
from collections.abc import Callable

import pytest

from backend.app import main
from backend.app.agents import retriever as retriever_module
from backend.app.agents.pipeline import answer_agent
from backend.app.config import LLM_MODEL
from backend.app.guardrails.rate_limit import RateLimiter
from backend.app.llm.client import Cancelled, llm_cost
from backend.app.retrieval import retrieve as baseline_retrieve_module
from backend.tests.unit.test_api_agent import QUESTION, FakeConn, FakeLLM, FakeVoyage, dense_search

PLANNER_COST = llm_cost(LLM_MODEL, 1000, 100)  # FakeLLM reports 1,000 in / 100 out tokens


class SlowLLM(FakeLLM):
    """Like FakeLLM, but the call for `slow_stage` waits for `release`, like a slow model call."""

    def __init__(self, slow_stage: str | None):
        super().__init__()
        self.slow_stage = slow_stage
        self.running, self.release = threading.Event(), threading.Event()

    def create(self, **kwargs):
        if self.stage_of[kwargs["system"]] == self.slow_stage:
            self.running.set()
            assert self.release.wait(5), "test never released the slow call"
        return super().create(**kwargs)


class SlowVoyage(FakeVoyage):
    """Records every call; rerank waits for `release` when `slow_rerank` is set."""

    def __init__(self, slow_rerank: bool = False):
        super().__init__()
        self.slow_rerank, self.calls = slow_rerank, []
        self.running, self.release = threading.Event(), threading.Event()

    def embed(self, texts, **kwargs):
        self.calls.append("embed")
        return super().embed(texts, **kwargs)

    def rerank(self, query, documents, **kwargs):
        self.calls.append("rerank")
        if self.slow_rerank:
            self.running.set()
            assert self.release.wait(5), "test never released the slow call"
        return super().rerank(query, documents, **kwargs)


@pytest.fixture
def fakes(monkeypatch):
    def make(llm, voyage):
        monkeypatch.setattr(main, "connect", lambda: FakeConn())
        monkeypatch.setattr(main, "clients", lambda: (voyage, llm))
        return llm, voyage

    monkeypatch.setattr(retriever_module, "dense_search", dense_search)
    monkeypatch.setattr(baseline_retrieve_module, "dense_search", dense_search)
    monkeypatch.setattr(main, "limiter", RateLimiter(per_ip_per_hour=100, daily_cap=100))
    return make


def run_stream(mode: str, leave_when: Callable[[str], bool], spec: str) -> list[dict]:
    """POST /query/stream; the client disconnects once `leave_when(text received so far)` is true.
    Returns the sent messages.
    Fails (instead of hanging) if the app does not finish within 5 s of the disconnect."""
    body = json.dumps({"question": QUESTION, "mode": mode}).encode()
    scope = {
        "type": "http", "asgi": {"version": "3.0", "spec_version": spec}, "http_version": "1.1",
        "method": "POST", "scheme": "http", "path": "/query/stream", "raw_path": b"/query/stream",
        "root_path": "", "query_string": b"", "client": ("198.51.100.1", 50000),
        "server": ("testserver", 80),
        "headers": [(b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode())],
    }  # fmt: skip
    pending = [{"type": "http.request", "body": body, "more_body": False}]
    sent: list[dict] = []

    async def receive():
        if pending:
            return pending.pop()
        while not leave_when(body_text(sent)):
            await asyncio.sleep(0.005)
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    async def main_call():
        await asyncio.wait_for(main.app(scope, receive, send), timeout=10)

    asyncio.run(main_call())
    return sent


def wait_for_log(capsys, event: str) -> dict:
    out, deadline = "", time.monotonic() + 5
    while time.monotonic() < deadline:
        out += capsys.readouterr().out
        for line in out.splitlines():
            if line.startswith("{") and json.loads(line).get("event") == event:
                return json.loads(line)
        time.sleep(0.01)
    raise AssertionError(f"no {event!r} log line; output was:\n{out}")


def body_text(sent: list[dict]) -> str:
    return b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body").decode()


@pytest.mark.parametrize("spec", ["2.3", "2.4"])
def test_agent_stops_after_the_running_step_when_the_visitor_leaves(fakes, capsys, spec):
    llm, voyage = fakes(SlowLLM("planner"), SlowVoyage())
    # the visitor sees the first step start, then leaves while the planner call is running
    leave = lambda text: llm.running.is_set() and "Planning evidence needs" in text  # noqa: E731
    sent = run_stream("agent", leave, spec)
    llm.release.set()  # the planner call was already running, so it is allowed to finish
    line = wait_for_log(capsys, "cancelled")
    assert line["stopped_before"] == "retriever" and line["mode"] == "agent"
    # usage = the advice screen (free in unit tests) + the planner
    assert line["cost_usd"] == pytest.approx(PLANNER_COST) and len(line["usage"]) == 2
    time.sleep(0.2)  # give a wrongly continuing worker time to make its next call
    assert llm.calls == ["planner"] and voyage.calls == []  # no later model or Voyage call
    assert "Planning evidence needs" in body_text(sent) and "event: result" not in body_text(sent)


@pytest.mark.parametrize("spec", ["2.3", "2.4"])
def test_baseline_skips_the_model_call_when_the_visitor_leaves(fakes, capsys, spec):
    llm, voyage = fakes(SlowLLM(None), SlowVoyage(slow_rerank=True))
    run_stream("baseline", lambda text: voyage.running.is_set(), spec)  # leaves during retrieval
    voyage.release.set()
    line = wait_for_log(capsys, "cancelled")
    assert line["stopped_before"] == "synthesizer"
    assert line["usage"] and all(u["model"] != LLM_MODEL for u in line["usage"])  # Voyage only
    assert line["cost_usd"] == pytest.approx(sum(u["cost_usd"] for u in line["usage"]))
    time.sleep(0.2)
    assert llm.calls == []  # the paid synthesis call never started


def test_a_visitor_who_stays_gets_the_full_answer(fakes, capsys):
    llm, voyage = fakes(SlowLLM(None), SlowVoyage())
    sent = run_stream("agent", lambda text: False, "2.3")  # never leaves; the server finishes
    assert "event: result" in body_text(sent)
    assert llm.calls == ["planner", "synthesizer", "critic"]
    assert '"event": "cancelled"' not in capsys.readouterr().out


def test_answer_agent_raises_cancelled_before_the_next_node_with_cost_so_far():
    llm, events = SlowLLM(None), []
    stop_after_planner = lambda: any(e[:2] == ("planner", "done") for e in events)  # noqa: E731
    with pytest.raises(Cancelled) as info:
        answer_agent(QUESTION, conn=FakeConn(), voyage=SlowVoyage(), llm=llm,
                     emit=events.append, should_stop=stop_after_planner)  # fmt: skip
    assert info.value.before == "retriever" and llm.calls == ["planner"]
    assert [u.cost_usd for u in info.value.usage] == [pytest.approx(PLANNER_COST)]
    assert ("retriever", "started") not in [e[:2] for e in events]  # not even announced
