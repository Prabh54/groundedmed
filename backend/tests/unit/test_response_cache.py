"""The exact-match response cache itself (M11-T4, ADR-046)."""

from backend.app.response_cache import ResponseCache, make_key, normalize_question, storable

ID = {"models": {"critic": "m"}, "gate_threshold": 0.72}


def test_key_is_exact_match_only():
    base = make_key("Does X reduce Y?", "agent", ID)
    assert make_key("  Does X reduce Y?\n", "agent", ID) == base  # outer whitespace trimmed
    assert make_key("Does X reduce Y", "agent", ID) != base  # one character different
    assert make_key("does X reduce Y?", "agent", ID) != base  # case matters
    assert make_key("Does X  reduce Y?", "agent", ID) != base  # inner whitespace matters
    assert make_key("Does X reduce Y?", "baseline", ID) != base


def test_unicode_forms_of_the_same_text_match():
    composed, decomposed = "café?", "café?"
    assert normalize_question(composed) == normalize_question(decomposed)
    assert make_key(composed, "agent", ID) == make_key(decomposed, "agent", ID)


def test_any_change_to_the_system_identity_changes_the_key():
    base = make_key("q?", "agent", ID)
    assert make_key("q?", "agent", {**ID, "gate_threshold": 0.7}) != base
    assert make_key("q?", "agent", {**ID, "models": {"critic": "other"}}) != base


def test_only_complete_answered_results_are_storable():
    assert storable(abstained=False, error=None, rerank_skipped=False)
    assert not storable(abstained=True, error=None, rerank_skipped=False)  # refusal
    assert not storable(abstained=False, error="boom", rerank_skipped=False)  # error
    assert not storable(abstained=False, error=None, rerank_skipped=True)  # fallback ranking


def test_lru_evicts_the_least_recently_used():
    cache = ResponseCache(max_entries=2)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == 1  # a is now the most recent
    cache.put("c", 3)
    assert cache.get("b") is None and cache.get("a") == 1 and cache.get("c") == 3


def test_entries_expire_after_the_ttl():
    now = [0.0]
    cache = ResponseCache(ttl_seconds=10, clock=lambda: now[0])
    cache.put("a", 1)
    now[0] = 9.0
    assert cache.get("a") == 1
    now[0] = 11.0
    assert cache.get("a") is None and len(cache) == 0


def test_the_cache_can_be_switched_off_with_zero_entries():
    cache = ResponseCache(max_entries=0)
    cache.put("a", 1)
    assert cache.get("a") is None
