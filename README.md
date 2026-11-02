# GroundedMed

Grounded biomedical research analyst over PubMedQA: answers cite retrieved passages, and the system
abstains when evidence is weak. Research demonstration only — not medical advice.

Status: early build. See SPEC.md (what), ARCHITECTURE.md (how), PLAN.md and PROGRESS.md (where we are).

## Live demo

https://groundedmed.vercel.app (API: https://groundedmed-api.onrender.com). Frontend on Vercel Hobby, FastAPI on Render
Starter ($7/month), Supabase Postgres + pgvector (free tier). Each answer costs about $0.03 in model calls; visitors get
10 questions per hour and the site 100 per day. Manual live checks (never run by CI): `uv run python -m
scripts.smoke_prod` (free; `--paid` sends 3 questions) and `npm run test:e2e:live` in `frontend/` (3 paid questions).

## Live site vs the benchmark

The public demo screens every question for personal medical advice before answering ("should I take ...",
"my mum has ...", doses, "my doctor said ...") and refuses those with a pointer to a doctor or pharmacist
(ADR-042). That screen was added after the evaluation and is **not part of the benchmarked system**: the
measured baseline and agent never run it, so the published results do not include it.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is installed by uv if missing.

```bash
uv sync
uv run ruff check .
uv run pytest
```

Run everything from the repo root; scripts run as `uv run python -m scripts.<name>` (ADR-024).

Secrets go in a local `.env` (never committed); `.env.example` lists the variable names.
