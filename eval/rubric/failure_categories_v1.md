# Failure categories: method v1 (frozen 2026-11-09)

Purpose: put every question the agent got wrong on the frozen 200-question test set into exactly one
category, using stored fields only. These rules were written before any classifier existed and
before any question was given a category. They were not written blind to the results: overall
scores, refusal-reason totals and the number of errors in each system had already been read.

Rules 1 to 5 record which step of the pipeline produced a refusal. Rules 6 and 7 record a signal
present on an answered row, not the step that caused the error. No category proves why an error
happened.

## Inputs (read-only)

| System | Stored file | SHA-256 |
|---|---|---|
| Agent | `eval/results/test-agent-20261025-200618-1f1e727/raw.jsonl` | `773b19a980f3782b5ad78d351bd7f76f6fdc17a0338e10c5683cc3966dee46bf` |
| Baseline | `eval/results/test-baseline-20261025-194639-4ebd38d/raw.jsonl` | `5b40cfb95d8dadb8b019594ef751e07ad64c34baafa68315cadafeb083eeaa57` |

Rows are joined on `pmid`. No model is called, nothing is re-judged and no stored file is changed.

## Terms

- **Error:** an agent row with `correct` false. A refusal is an error because every gold label is
  yes, no or maybe and the benchmark counts a refusal as incorrect. "Wrong refusal" below means
  scored wrong in that sense; it does not say that refusing was a bad decision.
- **Answered:** `abstained` is false.
- **Baseline right / wrong:** `correct` on the baseline row with the same `pmid`.
- **Gap question:** an error where the baseline is right.
- **Shared error:** an error where the baseline is wrong.
- **Surviving claims** (used only by rules 4 and 5): the set of `claim_id` values in `critic_results`
  that fully match `c\d{1,2}` and whose first result for that ID has `action` keep with `supported`
  and `citation_valid` both true, minus every ID in `validation.removed_uncited` and
  `validation.removed_invalid_citation`.

## Stop conditions (checked first, for every row)

The run stops with an error and writes no results if any of these is true. A stopped run is fixed
by a new rubric version, never by an exception made during the run.

1. A file's SHA-256 differs from the table above.
2. The two files do not hold the same set of `pmid` values, or a `pmid` repeats within a file.
3. `correct` differs from (`verdict` equals `gold_label`).
4. `abstained` differs from (`verdict` equals `insufficient_evidence`), or from (`abstain_reason` is
   not null).
5. `error` is not null, or `abstain_reason` is `error`.
6. An error row matches no rule below.

## Categories

Apply the rules in this order to each error. The first rule that matches is the row's only category.

| Order | Category | Rule |
|---|---|---|
| 1 | `refusal_evidence_gate` | `abstain_reason` is `weak_evidence`. |
| 2 | `refusal_model_abstained` | `abstain_reason` is `model_abstained`. |
| 3 | `refusal_citation_check_left_no_claims` | `abstain_reason` is `no_supported_claims` and `validation.downgraded_to_insufficient` is true. |
| 4 | `refusal_critic_left_no_claims` | `abstain_reason` is `no_supported_claims`, `validation.downgraded_to_insufficient` is false, and either surviving claims is empty or `answer_supported` is not false. |
| 5 | `refusal_critic_rejected_answer` | `abstain_reason` is `no_supported_claims`, `validation.downgraded_to_insufficient` is false, surviving claims is not empty, and `answer_supported` is false. |
| 6 | `answered_gold_not_rank_1` | Answered, and `gold_final_rank` is not 1 (null counts as not 1). |
| 7 | `answered_planner_added_baseline_right` | Answered, `planner_added_evidence` is above 0, and the baseline is right. |
| 8 | `answered_maybe_boundary` | Answered, and `gold_label` or `verdict` is `maybe`. |
| 9 | `answered_yes_no_flip` | Answered, and `gold_label` and `verdict` are `yes` and `no` in either order. |

## Reporting overlaps

For each answered error, the classifier records every later rule the row would also have matched. A
row still counts in one category only. Rules 6 to 9 apply only to answered rows, so a refusal's
`gold_final_rank` and `planner_added_evidence` play no part in its category.

## What these categories cannot show

- Rule 3 means the deterministic check left no claim; that also covers items dropped for a malformed
  ID and a draft with no claims. Rule 4 means no claim passed the critic; that also covers a claim the
  critic returned no result for, which was not judged. Rule 5 means at least one claim survived but
  the critic said the claims do not support the verdict.
- Whether a removed claim was actually correct cannot be measured here: the run stored the removed
  claim's ID and the critic's reason, not its text.
- Rules 4 and 5 rely on `critic_results`. The draft's claim list was not stored, so a kept critic
  result for an ID that is not a real validated claim cannot be detected. With `answer_supported`
  false, such a row is filed under rule 5 even if no claim really survived. Reading these rows in
  full can reveal it, but that does not change a category.
- Rule 6 records that the source paper was below first place or absent, not that this caused the error.
- Rule 7 records that the planner added papers, that the agent was wrong and that the baseline was
  right. It does not show that the added papers caused the error.
- Model answers vary between runs, and these test questions are not re-run.

## Changing these rules

Do not change this file. A changed method needs a new version (v2) and a new analysis run, and the
v1 file and its results stay as they are. Reading individual cases can add notes to a report but
never changes a category. This file holds no result counts.
