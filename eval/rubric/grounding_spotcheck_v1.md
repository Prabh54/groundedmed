# Grounding spot-check — method v1 (frozen 2026-10-23)

Purpose: measure how often displayed claims/limitations state something their cited passage does not
support. The citation validator cannot catch this (it checks IDs only). Baseline figure set on the
synthesizer-v3 dev run; the same check is re-run unchanged on agent output after M3. **Do not change
this method.** A changed method needs a new version (v2) and must be re-run on both systems.

Not the M4 independent judge (Gemini, rubric judge_v1): this is a small, reviewer-read audit.

## Sample (fixed)
The 18 dev questions in `FROZEN_PMIDS` in `eval/grounding_spotcheck.py` (the rows the M2 final reviewer
chose by hand on 2026-10-23; no selection rule — the list itself is the sample). For any run, use that
run's rows for exactly these question PMIDs, whatever the system answered (answered or model-declined
rows; a row with no claims or limitations contributes 0 items).

## Items
Every claim and every limitation in those rows, as stored in the run's `raw.jsonl` (after validation).

## Judge
A fresh-context reviewer (Claude subagent) with no other context than this file and the packet produced by
`uv run python -m eval.grounding_spotcheck packet --run <run-id>`, which shows each item directly above the
full text of every passage it cites. The packet does not name the system (baseline/agent). The judge
labels every item and returns one JSON line per item into `eval/audits/grounding/<run-id>-judgments.jsonl`.

Original instruction given to the M2 reviewer (verbatim):
> for at least 15 answered rows of eval/results/dev-baseline-20261023-033422-9396a90/raw.jsonl, read each
> claim and the full text of the passage it cites (passages are in data/corpus.jsonl, field
> `passage`/`sections`, keyed by `pmid`) and report every claim whose numbers or statements are not
> supported by or misstate the cited passage. Include PMID 10490564's row. Quote the specific mismatching
> phrases (short quotes only).

## Labels
Written down from how the M2 reviewer applied them:
- `supported` — every number and statement is in, or follows directly from, the cited passage(s).
  A mild paraphrase or obvious inference counts as supported.
- `partly` — the core finding is supported, but the item adds a detail, qualifier, setting or conclusion
  that the cited passage does not contain (e.g. a country the passage never names), or overstates it.
- `unsupported` — the item's main statement is not in, or is contradicted by, the cited passage(s).
- `not_judged` — the judge did not assess the item (reported, excluded from the denominator).

## Number
`unsupported-detail rate = (partly + unsupported) / judged items`, plus the number of rows with at least
one such item. Computed by `uv run python -m eval.grounding_spotcheck score --run <run-id>`, which refuses
if the judgments do not cover exactly the run's items (matched by question, item ID and a hash of the
item text).

## Comparability
Small sample (18 questions, ~90 items) and one model judge: treat differences of a few items as noise.
For the M3 comparison, judge the agent packet with the same instruction and labels; optionally also
re-judge the baseline packet in the same session (blind) to see judge-to-judge variation.
