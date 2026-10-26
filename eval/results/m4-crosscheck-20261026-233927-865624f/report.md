# M4 claim support and second-AI cross-check — `m4-crosscheck-20261026-233927-865624f`

Computed 2026-10-26T23:39:27+00:00 at commit `865624fbfee1428b8703f59f5b3df9e95e6ef8ed`. Scoring run `m4-final-20261026-222211-498b76a`; judge run `judge-test-20261026-091111-06b80ea`; second AI: ChatGPT (OpenAI), model as reported by the owner: GPT-6 Sol.

## Independent-judge estimated claim support (Gemini judge, test split)

| | Baseline | Agent |
|---|---|---|
| Claims, strict (partially = not supported) | 93.2% (90.7%–95.1%), 455/488 | 95.0% (92.6%–96.7%), 421/443 |
| Claims, lenient (partially counts) | 99.8% (98.8%–100.0%), 487/488 | 100.0% (99.1%–100.0%), 443/443 |
| Limitations, strict | 86.2% (82.5%–89.3%), 344/399 | 91.3% (87.8%–93.9%), 306/335 |
| Limitations, lenient | 99.5% (98.2%–99.9%), 397/399 | 99.7% (98.3%–99.9%), 334/335 |
| Questions with ≥1 not-fully-supported item | 35.5% (29.2%–42.4%), 70/197 | 20.8% (15.7%–27.1%), 40/192 |
| claim_strict question-level bootstrap 95% | 90.7%–95.5% | 92.7%–97.2% |
| claim_lenient question-level bootstrap 95% | 99.4%–100.0% | 100.0%–100.0% |
| Not judged | 0 | 0 |

Caveat: The two systems show different numbers of claims (the agent's critic removes some, and it abstains more), so the denominators differ.

Agent minus baseline (paired bootstrap 95%):

- claim_strict: +0.018 (-0.000 to +0.037)
- claim_lenient: +0.002 (+0.000 to +0.006)
- limitation_strict: +0.051 (+0.018 to +0.086)
- limitation_lenient: +0.002 (-0.006 to +0.010)

## Second-AI cross-check: does a second AI agree with the judge?

Stratified 30-claim sample (rubric v2 §7): per system up to 5 judge-rejected and the rest judge-accepted. Agreement is reported separately for each group; the share the second AI calls supported is NOT a claim-support estimate.

| Judge group | Baseline | Agent | Both systems |
|---|---|---|---|
| judge_supported | 80.0% (49.0%–94.3%), 8/10 | 60.0% (31.3%–83.2%), 6/10 | 70.0% (48.1%–85.5%), 14/20 |
| judge_rejected | 80.0% (37.6%–96.4%), 4/5 | 80.0% (37.6%–96.4%), 4/5 | 80.0% (49.0%–94.3%), 8/10 |

Unsure answers (excluded): 0. Rule: binary: judge 'supported' agrees with Supported; judge 'partially_supported' or 'unsupported' agrees with Not supported; Unsure excluded and counted.

Limitations: no person checked these claims. Agreement shows whether two AI models from different providers read the claims the same way; shared blind spots are possible, so agreement does not prove the judge is right. How the second AI was run is as reported by the owner (ADR-037).
