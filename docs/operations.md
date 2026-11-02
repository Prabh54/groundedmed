# Operations: live checks and upkeep

For the project owner and maintainers. Run everything from the repo root. Hosting settings are in
[ARCHITECTURE.md §11](../ARCHITECTURE.md); costs and dates to watch are in [PROGRESS.md](../PROGRESS.md).

## Live site

- Site: https://groundedmed.vercel.app (evaluation dashboard: `/eval`)
- API: https://groundedmed-api.onrender.com (`/health` returns the deployed commit and a database check)
- Spend protection: 10 questions per hour per visitor, 100 per day for the whole site, plus a monthly spend limit in
  the Anthropic console (set by the owner; not visible to the code). Each live answer costs about $0.03 in model calls.

## Live checks (manual only; never run by CI)

| Command | What it does | Cost |
|---|---|---|
| `uv run python -m scripts.smoke_prod` | Free checks: health, deployed commit, debug page off, CORS, site pages, input validation, one source abstract | $0 |
| `uv run python -m scripts.smoke_prod --paid` | The free checks plus 3 real questions (advice refusal, weak-evidence refusal, cited answer); writes `docs/deploy/smoke-prod-results.json` | about $0.03 (capped at $0.05) |
| `cd frontend && npm run test:e2e:live` | Browser tests against the live site, including 3 real questions | about $0.03 |

Paid checks need the owner's typed OK first. Push and wait for the deploy before running them: the deployed-commit
check fails while backend changes are unpushed or undeployed.

## Upkeep

- Database keep-alive: `uv run python -m scripts.keep_alive` (one read-only query). It also runs every 3 days from
  `.github/workflows/keepalive.yml`, so the free database is not paused for inactivity.
- Service check: `uv run python -m scripts.check_services` prints OK or FAIL for the database, Voyage and Anthropic,
  never a secret.
- The question limits live only in the API's memory (`backend/app/guardrails/rate_limit.py`; one process,
  `--workers 1`). Any redeploy, manual restart, crash or health-check restart sets them back to zero, and the daily
  count also starts again at 00:00 UTC (confirmed in the M7-T4 review, 2026-11-02; not fixed, listed in README
  Limitations). Before a live demo: if the site says "Demo limit reached", the daily cap may be full; a manual
  restart of the API service in the Render dashboard clears it.
- Every push that changes `backend/**`, `pyproject.toml`, `uv.lock` or `.python-version` (backend tests included)
  redeploys the API, which also resets the limits.
