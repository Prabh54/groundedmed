"""Production smoke test (M6-T5): checks the live API and site, and saves what it saw.

MANUAL ONLY. Never run by CI or on push (a test checks the workflows); the paid part needs --paid.
Run: uv run python -m scripts.smoke_prod            free checks only ($0)
     uv run python -m scripts.smoke_prod --paid     + 3 agent questions (~$0.03; $0.05 cap)
Free: /health (ok, database ok, the latest commit Render deploys), /debug/client-ip off, CORS
refuses another site and allows ours, the site's / and /eval, an empty question refused, one source
abstract with no conclusion fields. Paid, in agent mode as the site sends them, cheapest first: an
advice question (refused before any search), a casual wording that must stop at the evidence gate,
and a home-page example (cited answer that cites its own paper, every step done). The cap is checked
BEFORE each question (spent + reserve <= cap), using the cost the API reports. No retries.
Writes docs/deploy/smoke-prod-results.json for a paid run (committed) and
smoke-prod-results-free.json for a free run (git-ignored), so a free run never overwrites the paid
record or dirties git. The paid file holds the latest paid attempt, passed or not (git keeps
earlier ones). No secrets; the API never returns any.
"""

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "docs/deploy/smoke-prod-results.json"
OUT_FREE = REPO_ROOT / "docs/deploy/smoke-prod-results-free.json"
API = "https://groundedmed-api.onrender.com"
SITE = "https://groundedmed.vercel.app"
OTHER_SITE = "https://example.com"
CAP_USD = 0.05
# Render redeploys only when one of these changes (build filters, ARCHITECTURE §11).
DEPLOY_PATHS = ["backend", "pyproject.toml", "uv.lock", ".python-version"]
FORBIDDEN_FIELDS = {
    "long_answer",
    "final_decision",
    "reasoning_required_pred",
    "reasoning_free_pred",
}
AGENT_STAGES = ["planner", "retriever", "gate", "synthesizer", "critic", "validator"]

# (name, question, reserve USD). Reserves: an advice refusal measured ~$0.0001, a gate refusal
# ~$0.004, a full answer $0.024-0.029 live (dev agent max $0.040).
EXAMPLE_PMID = "20121683"
PAID = [
    (
        "advice refusal",
        "My mum just had a stroke. Should she get the clot-busting drug rt-PA?",
        0.002,
    ),
    ("weak evidence refusal", "can teh brain right half side learn to read on its own", 0.01),
    (
        "cited answer",
        "Are rural Australian patients willing to help teach medical students?",
        0.045,
    ),
]


def runs_current_backend(deployed: str) -> bool:
    """Is the deployed commit's backend the same as this checkout's HEAD? Render deploys the newest
    commit of a push that changes its build-filter paths, so the deployed SHA can be a later docs
    commit; what matters is that no backend file differs."""
    diff = subprocess.run(["git", "diff", "--quiet", deployed, "HEAD", "--", *DEPLOY_PATHS],
                          cwd=REPO_ROOT, capture_output=True)  # fmt: skip
    return diff.returncode == 0


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def free_checks(client: httpx.Client, current_backend=runs_current_backend) -> list[dict]:
    checks = []

    def check(name: str, ok: bool, seen) -> None:
        checks.append({"check": name, "ok": bool(ok), "seen": seen})

    h = client.get(f"{API}/health")
    body = h.json() if h.status_code == 200 else {}
    check(
        "health ok, database ok",
        h.status_code == 200 and body.get("status") == "ok" and body.get("db") == "ok",
        {"status": h.status_code, **body},
    )
    check(
        "deployed commit has the same backend files as HEAD",
        bool(body.get("version")) and current_backend(body["version"]),
        {"deployed": body.get("version")},
    )
    d = client.get(f"{API}/debug/client-ip")
    check("debug page is off (404)", d.status_code == 404, d.status_code)

    def preflight(origin: str) -> httpx.Response:
        return client.options(
            f"{API}/query/stream",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )

    bad, good = preflight(OTHER_SITE), preflight(SITE)
    check(
        "CORS refuses another site",
        bad.status_code >= 400 and "access-control-allow-origin" not in bad.headers,
        bad.status_code,
    )
    check(
        "CORS allows the site",
        good.status_code == 200 and good.headers.get("access-control-allow-origin") == SITE,
        {
            "status": good.status_code,
            "allow_origin": good.headers.get("access-control-allow-origin"),
        },
    )
    for path in ["/", "/eval"]:
        r = client.get(f"{SITE}{path}")
        check(f"site {path} loads", r.status_code == 200 and "GroundedMed" in r.text, r.status_code)
    e = client.post(f"{API}/query", json={"question": "", "mode": "agent"})
    check("empty question refused (422, never answered)", e.status_code == 422, e.status_code)
    s = client.get(f"{API}/sources/{EXAMPLE_PMID}")
    src = s.json() if s.status_code == 200 else {}
    check(
        "source abstract loads with no conclusion fields",
        s.status_code == 200 and src.get("sections") and not FORBIDDEN_FIELDS & set(src),
        {
            "status": s.status_code,
            "fields": sorted(src),
            "sections": len(src.get("sections") or []),
        },
    )
    return checks


def judge(name: str, events: list[tuple[str, dict]]) -> tuple[bool, dict]:
    """Did this paid question come back as it must? Returns (ok, what was seen)."""
    kinds = [k for k, _ in events]
    result = events[-1][1] if kinds and kinds[-1] == "result" else {}
    stages = [(d["name"], d["status"]) for k, d in events if k == "stage"]
    seen = {
        "events": kinds.count("stage"),
        "last_event": kinds[-1] if kinds else None,
        "verdict": result.get("verdict"),
        "abstain_reason": result.get("abstain_reason"),
        "claims": len(result.get("claims") or []),
        "cited_pmids": sorted(
            {c.split(":")[1] for cl in result.get("claims") or [] for c in cl["citation_ids"]}
        ),
        "cost_usd": result.get("estimated_cost_usd"),
        "prompt_versions": result.get("prompt_versions"),
    }
    if not result:
        return False, seen | {"error": events[-1][1] if events else None}
    if name == "advice refusal":
        ok = (
            kinds == ["result"]
            and result["abstain_reason"] == "personal_advice"
            and not result["claims"]
            and not result["stages"]
        )
    elif name == "weak evidence refusal":
        status = {st["name"]: st["status"] for st in result["stages"]}  # skips are not streamed
        ok = (
            result["abstain_reason"] == "weak_evidence"
            and not result["claims"]
            and status.get("synthesizer") == status.get("critic") == "skipped"
        )
    else:
        cited = {c for cl in result["claims"] for c in cl["citation_ids"]}
        ok = (
            not result["abstained"]
            and result["verdict"] in {"yes", "no"}
            and result["claims"]
            and all(cl["citation_ids"] for cl in result["claims"])
            and cited <= {c["id"] for c in result["citations"]}
            and f"PMID:{EXAMPLE_PMID}" in cited
            and stages == [(n, s) for n in AGENT_STAGES for s in ("started", "done")]
        )
    return bool(ok), seen


def paid_checks(client: httpx.Client, cap: float = CAP_USD) -> tuple[list[dict], float, str | None]:
    rows, spent = [], 0.0
    for name, question, reserve in PAID:
        if spent + reserve > cap:
            return (
                rows,
                spent,
                f"cap ${cap:.2f} would be exceeded before '{name}' (spent ${spent:.4f})",
            )
        r = client.post(
            f"{API}/query/stream",
            json={"question": question, "mode": "agent"},
            headers={"Origin": SITE},
            timeout=120,
        )
        if r.status_code != 200:
            rows.append(
                {
                    "check": name,
                    "ok": False,
                    "seen": {"status": r.status_code, "body": r.text[:200]},
                }
            )
            return rows, spent, f"'{name}' got HTTP {r.status_code}"
        ok, seen = judge(name, parse_sse(r.text))
        spent += seen.get("cost_usd") or 0.0
        rows.append({"check": name, "ok": ok, "seen": seen})
        if not ok:
            return rows, spent, f"'{name}' did not come back as expected"
    return rows, spent, None


def main(
    argv: list[str] | None = None,
    client: httpx.Client | None = None,
    current_backend=runs_current_backend,
    out: Path = OUT,
    out_free: Path = OUT_FREE,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paid", action="store_true", help="also send the 3 paid questions")
    args = parser.parse_args(argv)
    client = client or httpx.Client(timeout=30, follow_redirects=True)

    checks = free_checks(client, current_backend)
    spent, stopped = 0.0, None
    if not all(c["ok"] for c in checks):
        stopped = "a free check failed, so no paid question was sent"
    elif args.paid:
        paid, spent, stopped = paid_checks(client)
        checks += paid
    for c in checks:
        print(f"{'PASS' if c['ok'] else 'FAIL'}  {c['check']}  {json.dumps(c['seen'])[:160]}")
    passed = all(c["ok"] for c in checks) and stopped is None
    record = {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "api": API,
        "site": SITE,
        "paid": args.paid,
        "cap_usd": CAP_USD,
        "spent_usd": round(spent, 6),
        "passed": passed,
        "stopped": stopped,
        "checks": checks,
    }
    out = out if args.paid else out_free
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(
        f"{'ALL PASSED' if passed else 'FAILED'}: "
        f"{sum(c['ok'] for c in checks)}/{len(checks)} checks; spent ${spent:.4f}"
        + (f"; STOPPED: {stopped}" if stopped else "")
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
