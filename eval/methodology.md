# M4 evaluation methodology (written before any test-split run)

Status: draft written 2026-10-24, before any test question was run or any result seen. Each final run
gets a copy at `eval/results/<run-id>/methodology.md` with its run IDs, dates and numbers filled in from
`summary.json`. Anything changed after test results are seen is marked there as a change, with its
reason; nothing in this file is edited to fit results.

## What is compared
- **Baseline**: retrieve → evidence gate → one cited Claude Sonnet 5.5 answer → citation validator.
- **Agent**: planner → pooled retriever → gate → synthesizer (same prompt as the baseline) → critic →
  validator. Same model, retriever, gate, evidence budget and validator as the baseline (ADR-015).
- **Closed-book control** (ADR-010, PLAN M4-T3b): same model, question only, verdict only, no retrieval.
  Pending the owner's confirmation at the M4-T3 cost gate.

## Test set
200 frozen PubMedQA expert-labelled questions (110 yes / 68 no / 22 maybe), `data/manifests/test.json`,
sha256 `41f10a29…224f2f`, frozen 2026-10-21 (commit e3c77f9). Never used to tune anything. Every system
gets the identical 200 IDs; the scorer refuses otherwise. Each system runs on it once (ADR-031).

## Per-question record (`raw.jsonl`)
Question ID, gold label, retrieved PMIDs and ranks, gold rank, system verdict, refused or not and why,
correct or not, claims and limitations with citations, judge label per claim (added by the judge run),
total and per-stage latency, token usage and cost, errors.

## Metrics
- **Overall accuracy** = correct / 200. **A refusal (`insufficient_evidence`) counts as incorrect.**
- **Coverage** = answered / 200. **Answered-only accuracy** = correct answered / answered. These two are
  always shown side by side: a system can raise answered-only accuracy just by refusing more.
- Majority-class reference ("always yes") = 110/200 = 0.55.
- Hit@1, Hit@5: share of questions whose source paper is in the top 1 / 5 passages.
- **Independent-judge estimated claim support** = claims the judge labels `supported` / judged claims
  (partially supported counts as not supported); also a lenient version and the same for limitations
  (rubric `eval/rubric/judge_v2.md` §6). Not a "hallucination rate".
- Latency p50 / p95 (total and per stage); cost per query, mean and median, from provider token usage at
  list prices.

## Uncertainty
- Proportions (accuracy, coverage, answered-only accuracy, Hit@k, claim support): Wilson 95% intervals.
  For claim support also a bootstrap that resamples questions, because claims from one answer are not
  independent; that is the more honest interval when they differ.
- Latency and cost: bootstrap 95% intervals (2,000 resamples, seed 20261024).
- Baseline vs agent (and closed-book vs baseline): paired bootstrap over the same question IDs, 2,000
  resamples, seed 20261024.

## Independent judge
Gemini (Google, `EVALUATOR_API_KEY`), a different provider from the runtime system and its critic
(Anthropic). Rubric v2, prompt and schema frozen and hash-checked before any test run. Blind: the systems
are shown as SYSTEM_A / SYSTEM_B, assigned by a seeded coin per question; the judge is never told which
is which, nor the verdict, gold label or reference conclusion. Each claim is judged only against the
passages it cites.

## Manual audit — limitation, stated up front
`eval/audits/<run-id>-manual-sample.csv` holds about 30 claims (15 per system) for the owner to label.
**It is a stratified sample** (up to 5 judge-rejected claims per system, the rest judge-supported), chosen
to test whether the judge's rejections and acceptances hold up. **The share of audit claims the owner
finds supported is NOT an estimate of claim support** and is never reported as one; every CSV row says
so in its `sample_purpose` column. If a system has fewer than 5 judge-rejected claims, all of them are
used and the rest are judge-supported; with 0, rejected-group agreement is "not measurable". Agreement
is reported per group and per system, with counts and Wilson intervals (ADR-033).

## Run integrity
- A run that stops before all planned questions, for any reason (cost cap, 5 failures in a row, crash,
  interruption), is **void**: not scored, no results computed or shown, rows renamed
  `raw.VOID-do-not-score.jsonl`, and a new run ID is used. It does not count as the system's one test run,
  because no results were seen; the rerun must test the identical system (checked by code). A complete
  run cannot be voided. Void runs stay on disk and are listed in the final methodology (ADR-034).
- Isolated failed questions are kept as error rows and count as incorrect.
- Spending: each test run has a soft cap of about twice its estimate (proposed: baseline $6, agent $14),
  approved by the owner in writing before the run.

## Known limitations (stated before results)
- Hit@5 is saturated on dev (1.00): questions are usually derived from the source paper's title and the
  distractors are not hard enough, so Hit@k barely separates the systems. Answer metrics matter more.
- PubMedQA is public; the model may know answers from training. The closed-book control measures this.
- The synthesizer prompt (v3) and the gate threshold were shaped on the 100 dev questions, so dev numbers
  are optimistic; only the test numbers are unbiased.
- One run per system: the runtime model has no fixed seed and Voyage embeddings drift slightly between
  calls, so a rerun would differ a little. Small differences between systems may be noise; the paired
  intervals show how much.
- One model judge, blinding imperfect (the systems differ in style). The manual audit checks it on ~30
  claims only.
- Latency measured sequentially on a developer machine, not under production load.
