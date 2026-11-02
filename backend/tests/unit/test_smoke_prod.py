"""M6-T5: the production smoke script against a fake API and site (httpx.MockTransport, no
network), and the rule that it and the live browser test are manual only (never CI or on push)."""

import json
import re

import httpx
import pytest

from backend.app.config import REPO_ROOT
from scripts import smoke_prod

COMMIT = "a" * 40
PMID = smoke_prod.EXAMPLE_PMID


def sse(events: list[tuple[str, dict]]) -> str:
    return "".join(f"event: {k}\ndata: {json.dumps(d)}\n\n" for k, d in events)


def answer_events(cite: str = PMID) -> list[tuple[str, dict]]:
    stages = [
        ("stage", {"name": n, "status": s, "message": "", "ms": 1.0})
        for n in smoke_prod.AGENT_STAGES
        for s in ("started", "done")
    ]
    result = {
        "verdict": "yes",
        "abstained": False,
        "abstain_reason": None,
        "stages": [],
        "claims": [{"id": "c1", "text": "t", "citation_ids": [f"PMID:{cite}"]}],
        "citations": [{"id": f"PMID:{cite}"}],
        "estimated_cost_usd": 0.026,
        "prompt_versions": {"advice_check": "v1"},
    }
    return [*stages, ("result", result)]


class FakeLive:
    """The live API and site as the smoke script expects them; attributes break one thing."""

    def __init__(self):
        self.version, self.cite, self.queries = COMMIT, PMID, []
        self.size_limit = True

    def __call__(self, request: httpx.Request) -> httpx.Response:
        url, method = str(request.url), request.method
        if url.startswith(smoke_prod.SITE):
            return httpx.Response(200, text="<title>GroundedMed</title>")
        path = request.url.path
        if path == "/health":
            return httpx.Response(200, json={"status": "ok", "version": self.version, "db": "ok"})
        if method == "OPTIONS":
            origin = request.headers["origin"]
            if origin != smoke_prod.SITE:
                return httpx.Response(400, text="Disallowed CORS origin")
            return httpx.Response(200, headers={"access-control-allow-origin": origin})
        if path.startswith("/sources/"):
            return httpx.Response(200, json={"id": f"PMID:{PMID}", "sections": [{"label": "R"}]})
        if path == "/query":
            if len(request.content) > 8192 and self.size_limit:
                return httpx.Response(413, json={"detail": "Request too large."})
            return httpx.Response(422, json={"detail": "too short"})
        if path == "/query/stream":
            question = json.loads(request.content)["question"]
            self.queries.append(question)
            if question == smoke_prod.PAID[0][1]:
                result = {
                    "abstained": True,
                    "abstain_reason": "personal_advice",
                    "claims": [],
                    "stages": [],
                    "citations": [],
                    "estimated_cost_usd": 0.0001,
                }
                return httpx.Response(200, text=sse([("result", result)]))
            if question == smoke_prod.PAID[1][1]:
                # as main.py: skipped steps are only in the result's stages, never streamed
                skipped = [{"name": n, "status": "skipped"} for n in ("synthesizer", "critic")]
                result = {
                    "abstained": True,
                    "abstain_reason": "weak_evidence",
                    "claims": [],
                    "stages": skipped,
                    "citations": [],
                    "estimated_cost_usd": 0.004,
                }
                done = [("stage", {"name": "gate", "status": "done"})]
                return httpx.Response(200, text=sse([*done, ("result", result)]))
            return httpx.Response(200, text=sse(answer_events(self.cite)))
        return httpx.Response(404)


def run(fake: FakeLive, tmp_path, *args: str) -> tuple[int, dict]:
    out, out_free = tmp_path / "smoke.json", tmp_path / "smoke-free.json"
    client = httpx.Client(transport=httpx.MockTransport(fake))
    code = smoke_prod.main(list(args), client=client, current_backend=lambda sha: sha == COMMIT,
                           out=out, out_free=out_free)  # fmt: skip
    return code, json.loads((out if "--paid" in args else out_free).read_text())


def test_all_checks_pass_and_the_paid_questions_go_cheapest_first(tmp_path):
    fake = FakeLive()
    code, record = run(fake, tmp_path, "--paid")
    assert code == 0 and record["passed"] and record["stopped"] is None
    assert len(record["checks"]) == 14 and all(c["ok"] for c in record["checks"])
    assert fake.queries == [q for _, q, _ in smoke_prod.PAID]
    assert record["spent_usd"] == pytest.approx(0.0301)


def test_without_paid_no_question_is_sent(tmp_path):
    fake = FakeLive()
    code, record = run(fake, tmp_path)
    assert code == 0 and fake.queries == [] and record["spent_usd"] == 0


def test_a_free_run_never_overwrites_the_paid_record(tmp_path):
    run(FakeLive(), tmp_path, "--paid")
    paid_before = (tmp_path / "smoke.json").read_text()
    code, record = run(FakeLive(), tmp_path)
    assert code == 0 and not record["paid"] and len(record["checks"]) == 11
    assert (tmp_path / "smoke.json").read_text() == paid_before
    free_before = (tmp_path / "smoke-free.json").read_text()
    run(FakeLive(), tmp_path, "--paid")
    assert (tmp_path / "smoke-free.json").read_text() == free_before


def test_an_api_without_the_size_limit_fails_the_free_check(tmp_path):
    fake = FakeLive()
    fake.size_limit = False  # the old API: a huge body is read, then refused as too long (422)
    code, record = run(fake, tmp_path)
    failed = [c["check"] for c in record["checks"] if not c["ok"]]
    assert code == 1 and failed == ["request over 8 KB refused (413, never read)"]


def test_a_failed_free_check_sends_no_paid_question(tmp_path):
    fake = FakeLive()
    fake.version = "b" * 40  # an old deploy
    code, record = run(fake, tmp_path, "--paid")
    assert code == 1 and fake.queries == [] and "no paid question" in record["stopped"]


def test_an_answer_that_does_not_cite_its_own_paper_fails(tmp_path):
    fake = FakeLive()
    fake.cite = "99999999"
    code, record = run(fake, tmp_path, "--paid")
    assert code == 1 and not record["checks"][-1]["ok"]


def test_the_cap_is_checked_before_each_question():
    fake = FakeLive()
    client = httpx.Client(transport=httpx.MockTransport(fake))
    rows, spent, stopped = smoke_prod.paid_checks(client, cap=0.04)
    assert fake.queries == [
        smoke_prod.PAID[0][1],
        smoke_prod.PAID[1][1],
    ]  # the answer would pass $0.04
    assert "cap" in stopped and spent == pytest.approx(0.0041)


def test_the_live_checks_are_manual_only():
    """Neither the smoke script nor the live browser test runs in CI, on push or on a schedule."""
    workflows = "".join(
        p.read_text(encoding="utf-8") for p in (REPO_ROOT / ".github/workflows").glob("*.y*ml")
    )
    assert not re.search(r"smoke_prod|playwright\.live|e2e-live|test:e2e:live", workflows)
    frontend = REPO_ROOT / "frontend"
    default_config = (frontend / "playwright.config.ts").read_text(encoding="utf-8")
    assert 'testDir: "e2e"' in default_config  # CI's `npm run test:e2e` never sees e2e-live/
    live_config = (frontend / "playwright.live.config.ts").read_text(encoding="utf-8")
    assert 'testDir: "e2e-live"' in live_config and "LIVE_E2E" in live_config
    assert "retries: 0" in live_config and "workers: 1" in live_config
