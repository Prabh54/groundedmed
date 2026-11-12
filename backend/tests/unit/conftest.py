"""Unit tests run the API with the personal-advice screen answering "not advice" for free,
so tests written before it keep their fake model and exact costs. Tests of the screen itself are
marked `real_advice_screen` and get the real function."""

import pytest

from backend.app.config import ADVICE_MODEL
from backend.app.models.answer import LLMUsage

FREE = LLMUsage(model=ADVICE_MODEL, input_tokens=0, output_tokens=0, cost_usd=0.0)
NO_ADVICE = (False, FREE, 0.0)


@pytest.fixture(autouse=True)
def advice_screen_passes(request, monkeypatch):
    if request.node.get_closest_marker("real_advice_screen"):
        return
    from backend.app import main

    monkeypatch.setattr(main, "screen", lambda question: NO_ADVICE)


@pytest.fixture(autouse=True)
def fresh_response_cache(monkeypatch):
    """The answer cache is process-wide; every test starts with an empty one."""
    from backend.app import main
    from backend.app.response_cache import ResponseCache

    monkeypatch.setattr(main, "response_cache", ResponseCache())
