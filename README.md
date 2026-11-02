# GroundedMed

Grounded biomedical research analyst over PubMedQA: answers cite retrieved passages, and the system
abstains when evidence is weak. Research demonstration only — not medical advice.

Status: early build. See SPEC.md (what), ARCHITECTURE.md (how), PLAN.md and PROGRESS.md (where we are).

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
