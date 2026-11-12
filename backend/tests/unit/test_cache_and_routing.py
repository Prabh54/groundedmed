"""M11 (ADR-046): prompt-cache cost accounting, the cache marker, and per-stage routing settings."""

from types import SimpleNamespace

import pytest

from backend.app import config
from backend.app.agents import critic, planner
from backend.app.config import ADVICE_MODEL, LLM_MODEL
from backend.app.llm.client import llm_cost, structured_call
from backend.app.models.answer import DraftAnswer, LLMUsage

VALID = '{"verdict": "yes", "claims": [], "limitations": []}'


class FakeClient:
    def __init__(self, usage):
        self.usage, self.kwargs = usage, None
        self.messages = SimpleNamespace(create=self.create)

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=VALID)], stop_reason="end_turn",
            stop_details=None, usage=self.usage,
        )  # fmt: skip


def test_llm_cost_hand_checked_with_cache_tokens():
    # Sonnet 5.5: 1000*$2 + 500*$10 + 4000*$0.20 + 2000*$2.50 per million = $0.0128
    assert llm_cost(LLM_MODEL, 1000, 500, 4000, 2000) == pytest.approx(0.0128)
    # Haiku 5.5: 1000*$0.10 + 500*$0.50 + 4000*$0.01 + 2000*$0.125 per million = $0.00064
    assert llm_cost(ADVICE_MODEL, 1000, 500, 4000, 2000) == pytest.approx(0.00064)


def test_llm_cost_without_cache_tokens_is_unchanged():
    assert llm_cost(LLM_MODEL, 3000, 500) == pytest.approx(0.011)


def test_cache_marker_only_when_asked():
    plain, cached = FakeClient(SimpleNamespace(input_tokens=1, output_tokens=1)), None
    structured_call(plain, system="S", user="u", schema=DraftAnswer, effort="low")
    assert plain.kwargs["system"] == "S"  # byte-identical request to v1.1.0
    cached = FakeClient(SimpleNamespace(input_tokens=1, output_tokens=1))
    structured_call(cached, system="S", user="u", schema=DraftAnswer, effort="low", cache=True)
    assert cached.kwargs["system"] == [
        {"type": "text", "text": "S", "cache_control": {"type": "ephemeral"}}
    ]


def test_usage_records_cache_tokens_and_costs_them():
    usage = SimpleNamespace(
        input_tokens=1000, output_tokens=500,
        cache_read_input_tokens=4000, cache_creation_input_tokens=2000,
    )  # fmt: skip
    _, got, _ = structured_call(
        FakeClient(usage), system="S", user="u", schema=DraftAnswer, effort="low", cache=True
    )
    assert got.cache_read_input_tokens == 4000 and got.cache_write_input_tokens == 2000
    assert got.cost_usd == pytest.approx(0.0128)


def test_old_usage_records_still_load_without_cache_fields():
    old = LLMUsage(model=LLM_MODEL, input_tokens=10, output_tokens=5, cost_usd=0.1)
    assert old.cache_read_input_tokens == 0 and old.cache_write_input_tokens == 0


def test_stage_settings_default_to_v110_behaviour(monkeypatch):
    for var in ("PLANNER_MODEL", "CRITIC_MODEL", "PROMPT_CACHE_STAGES"):
        monkeypatch.delenv(var, raising=False)
    assert config._stage_model("PLANNER_MODEL") == LLM_MODEL
    assert config._stage_model("CRITIC_MODEL") == LLM_MODEL
    assert config._cache_stages() == frozenset()


def test_stage_settings_reject_unknown_values(monkeypatch):
    monkeypatch.setenv("CRITIC_MODEL", "claude-made-up")
    with pytest.raises(RuntimeError):
        config._stage_model("CRITIC_MODEL")
    monkeypatch.setenv("PROMPT_CACHE_STAGES", "planner,bogus")
    with pytest.raises(RuntimeError):
        config._cache_stages()
    monkeypatch.setenv("CRITIC_MODEL", ADVICE_MODEL)
    assert config._stage_model("CRITIC_MODEL") == ADVICE_MODEL


def test_planner_and_critic_use_their_routed_model_and_cache_flag(monkeypatch):
    monkeypatch.setattr(planner, "PLANNER_MODEL", ADVICE_MODEL)
    monkeypatch.setattr(planner, "PROMPT_CACHE_STAGES", frozenset({"planner"}))
    seen = {}

    def fake_call(client, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop")

    monkeypatch.setattr(planner, "structured_call", fake_call)
    with pytest.raises(RuntimeError):
        planner.plan(None, "q?")
    assert seen["model"] == ADVICE_MODEL and seen["cache"] is True

    monkeypatch.setattr(critic, "CRITIC_MODEL", ADVICE_MODEL)
    monkeypatch.setattr(critic, "PROMPT_CACHE_STAGES", frozenset())
    seen.clear()
    monkeypatch.setattr(critic, "structured_call", fake_call)
    monkeypatch.setattr(critic, "render_user_message", lambda *a: "u")
    with pytest.raises(RuntimeError):
        critic.critique(None, "q?", None, [])
    assert seen["model"] == ADVICE_MODEL and seen["cache"] is False
