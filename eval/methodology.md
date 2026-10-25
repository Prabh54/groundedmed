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
(Anthropic). Model `gemini-3.1-pro-preview`, pinned to Google's reported version
`3.1-pro-preview-01-2026` (ADR-035); every judge result file records the model string, the version
Google reports for each response, the model metadata and the rubric fingerprints. Rubric v2, prompt and schema frozen and hash-checked before any test run. Blind: the systems
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
  because no results were seen; the rerun must test the identical system and carry the owner's fresh
  typed approval (both checked by code, for system runs and for judge runs). A complete run cannot be
  voided. Void runs stay on disk and are listed in the final methodology (ADR-034).
- Isolated failed questions are kept as error rows and count as incorrect.
- Spending: each run has a soft cap, approved by the owner in writing before the run. Spent on M4
  dev work (owner-approved): judge test call $0.01493, judge setup check on 10 dev questions $0.5766,
  closed-book check on 5 dev questions $0.0053 — $0.5969 in all. Proposed for the test runs (revised
  2026-10-25 from measured data): baseline est $3.00 / cap $6, agent est $7.00 / cap $14, closed-book
  est $0.21 / cap $1, judge on all test claims est $10.68 / cap $20 — est $20.89, caps $41.00.
  **True ceiling including void reruns: $82.00** — every run voided once at its cap and rerun once to
  its cap. A second rerun is impossible without a new typed approval (the runner refuses a rerun that
  reuses the void run's approval text). The judge contingency below would add up to $21.55 (or $43.10
  if that re-judge were itself voided once), only with a new approval.

## Judge runs: checkpoint and resume (owner instruction 2026-10-25, before any test scoring)
- Each question's judge call and judgments are written to disk as soon as that question is done
  (`calls.jsonl`, `judgments.jsonl`), so a dropped connection does not lose finished work.
- A judge run that stops early is still **void** (ADR-034). It is never resumed automatically: a
  plain re-run always starts a new run ID, and the runner refuses a re-judge of the same pair that
  reuses an approval already spent on a void run.
- A void judge run may be **resumed only with the owner's fresh typed approval**
  (`--resume <run-id> --approval "<new text>" --max-cost <cap for the whole run>`). The runner refuses
  unless: the stop was not a model change (a changed judge means re-judging everything, below); the
  rubric files, rubric version, judge model and pinned versions equal the stopped run's; the blinding
  key rebuilds identically; and every already-judged question's request rebuilds byte-identically
  (same request hash). It then judges only the remaining questions, appending to the same files; each
  question is judged exactly once. The resume (time, approval, questions done, the void record, code
  commit) is recorded in `run_config.json` and `summary.json`; the cap covers the whole run.
- Void judgments are never read before scoring, whether or not the run is later resumed.
- **Daily request quota (owner instruction 2026-10-25):** Gemini counts requests per model per day,
  resetting at midnight Pacific time; Google offers no API for the remaining quota. Before starting
  (or resuming), the runner takes the limit and today's usage as shown in AI Studio
  (`--rpd-limit`, `--rpd-used-today`), uses our own logged requests for today instead if they are
  higher, and refuses unless the remaining requests cover every question still to judge plus 10%
  for re-asks and retries. The check is saved with the run.
- **Rate limits and stopping errors:** a per-minute rate limit (HTTP 429) is retried, waiting at
  least as long as Google asks (up to 5 minutes, 4 tries). A daily-quota 429, a billing or access
  error (HTTP 400/401/402/403) or an unavailable model (404) is not retried: the run stops at once
  and is void. After a quota or billing stop it may be resumed only with the owner's fresh typed
  approval; an unavailable model follows the contingency below (re-judge everything).
- Second test judge run `judge-test-20261025-225605-00b81f4`: void after 5 questions, every call
  refused with HTTP 402 (prepaid credit used up); $0 spent.
- Defects found in the first test judge run (`judge-test-20261025-213638-cc3f2f7`, void after 118/197
  questions, $4.8673): a dropped connection was not retried; model-check records were not saved on a
  crash; the fresh-approval rule existed for system runs but not for the judge. All three fixed with
  tests that failed on the old code. That run was not resumed (owner: re-judge in full, new run ID).

## If the judge model is retired or changes (decided 2026-10-25, before any judging)
Detection: a free check compares Google's model metadata version with the pinned
`3.1-pro-preview-01-2026` before every judge run, after every 25 questions, and at the end; every
response's reported `modelVersion` must equal `gemini-3.1-pro-preview`, the value seen at the M4-T0
test call. Either mismatch, or Google refusing the model, stops judging (`JudgeModelChanged` / API
error), which makes that judge run void. Limitation, measured at the test call: the per-response value
is only the undated model name, so the dated version is visible only in the metadata. A swap that
Google does not reflect in the metadata version cannot be detected.

- **Before any test judging:** pick a replacement by a DECISIONS.md entry (same rubric v2, prompt and
  schema, which are model-independent), run a fake-data test call and the 10-question dev setup check
  with it, and get the owner's OK. Nothing to re-judge.
- **During test judging:** the judge run is void (rubric v2 §5): its judgments are not read or scored.
  After the replacement steps above, both systems' claims are judged again in full under a new judge
  run ID. Baseline and agent are always judged by the same model version: never a mix.
- **After test judging has finished:** the reported results stand, labelled with the model and version
  that produced them. Nothing is re-judged unless the owner asks; any later re-judging covers both
  systems in full and is reported beside the original, never replacing it.
- **System runs are never re-run because of a judge change.** The baseline, agent and closed-book
  answers are fixed files; the judge only reads them, so the frozen test set is not used again.
- **Manual audit:** the owner's labels do not depend on the judge and are kept. If the judge changes,
  agreement is recomputed on the same audit claims against the new judge's labels, noting that the
  sample groups were drawn from the earlier judge's labels.
- **Cost of re-judging everything:** about $11.28 estimated (test call $0.02 + dev setup check $0.58 +
  all test claims $10.68, if the replacement is priced like gemini-3.1-pro-preview), cap $21.55.
  Figures measured by the dev setup check (judge-dev-20261025-192833-c72234e).
- **Time:** about 194 judge calls (one per test question with items), run one after another:
  measured 22 s mean / 38 s max per call on the tail-heavy dev sample, so about 70 minutes, at most
  about 2 hours, plus the owner's decision and approvals.

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
