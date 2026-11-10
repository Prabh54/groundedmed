"""M9: sort the agent's errors on the frozen test set into the frozen failure categories.

    uv run python -m eval.runners.failure_analysis --check-inputs   # checks; classifies nothing
    uv run python -m eval.runners.failure_analysis                  # the one real run (M9-T3)

The method is eval/rubric/failure_categories_v1.md; that file is the authority and this script only
applies it. Read-only and $0: no model is called, nothing is re-judged, no stored file is changed.
The two inputs, and the M4 summary it copies intervals from, are fixed in code and pinned by
SHA-256, so there is no way to point it at anything else. A run that hits a stop condition exits
with an error and writes no results. A real run needs a clean tree, happens once, and writes
`eval/results/failure-analysis-<time>-<commit>/` (summary.json, categories.jsonl)."""

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from math import comb

from eval.runners.retrieval import REPO_ROOT, RESULTS_DIR, git_state

RUBRIC = REPO_ROOT / "eval" / "rubric" / "failure_categories_v1.md"
RUBRIC_SHA256 = "4af5a06a24e238484b4ecc899d44cccbe7a464324c3bba7878504bc25b94c551"
INPUTS = {
    "agent": (
        "test-agent-20261025-200618-1f1e727/raw.jsonl",
        "773b19a980f3782b5ad78d351bd7f76f6fdc17a0338e10c5683cc3966dee46bf",
    ),
    "baseline": (
        "test-baseline-20261025-194639-4ebd38d/raw.jsonl",
        "5b40cfb95d8dadb8b019594ef751e07ad64c34baafa68315cadafeb083eeaa57",
    ),
}
# Not used for any category: only its stored paired intervals are copied into the summary.
STORED_SCORING = (
    "m4-final-20261026-222211-498b76a/summary.json",
    "48be3903c63695733bcc54cbf486871c199c1957b11dc274927e8f277394c827",
)
RUN_PREFIX = "failure-analysis-"

IE = "insufficient_evidence"
CLAIM_ID = re.compile(r"c\d{1,2}")
CATEGORIES = (  # the rubric's order; the first rule that matches is the row's only category
    "refusal_evidence_gate",
    "refusal_model_abstained",
    "refusal_citation_check_left_no_claims",
    "refusal_critic_left_no_claims",
    "refusal_critic_rejected_answer",
    "answered_gold_not_rank_1",
    "answered_planner_added_baseline_right",
    "answered_maybe_boundary",
    "answered_yes_no_flip",
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_input(system: str) -> list[dict]:
    """Stop condition 1. The hash and the rows come from the same bytes."""
    name, expected = INPUTS[system]
    data = (RESULTS_DIR / name).read_bytes()
    if sha256(data) != expected:
        raise SystemExit(
            f"Stop: {system} file differs from its pinned SHA-256; no results written."
        )
    return [json.loads(line) for line in data.split(b"\n") if line.strip()]


def load_stored_intervals() -> dict:
    """The M4 paired bootstrap intervals, copied unchanged for the report; never recomputed."""
    name, expected = STORED_SCORING
    data = (RESULTS_DIR / name).read_bytes()
    if sha256(data) != expected:
        raise SystemExit("Stop: the M4 scoring file differs from its pinned SHA-256.")
    comparison = json.loads(data)["comparisons"]["agent_minus_baseline"]
    return {
        "file": name,
        "sha256": expected,
        "agent_minus_baseline_accuracy": comparison["accuracy"],
        "agent_minus_baseline_answered_only_accuracy": comparison["answered_only_accuracy"],
    }


def check_row(system: str, row: dict) -> None:
    """Stop conditions 3 to 5, for one row."""
    label = f"{system} pmid {row['pmid']}"
    if row["correct"] != (row["verdict"] == row["gold_label"]):
        raise SystemExit(f"Stop: {label}: `correct` disagrees with verdict and gold label.")
    if row["abstained"] != (row["verdict"] == IE):
        raise SystemExit(f"Stop: {label}: `abstained` disagrees with the verdict.")
    if row["abstained"] != (row["abstain_reason"] is not None):
        raise SystemExit(f"Stop: {label}: `abstained` disagrees with `abstain_reason`.")
    if row["error"] is not None or row["abstain_reason"] == "error":
        raise SystemExit(f"Stop: {label}: the row records an error.")


def check_rows(agent: list[dict], baseline: list[dict]) -> None:
    """Stop conditions 2 to 5, for every row of both files."""
    try:
        for system, rows in (("agent", agent), ("baseline", baseline)):
            pmids = [r["pmid"] for r in rows]
            if len(set(pmids)) != len(pmids):
                raise SystemExit(f"Stop: a pmid repeats within the {system} file.")
            for row in rows:
                check_row(system, row)
    except KeyError as missing:
        raise SystemExit(f"Stop: a row has no field {missing}.") from None
    if {r["pmid"] for r in agent} != {r["pmid"] for r in baseline}:
        raise SystemExit("Stop: the two files do not hold the same set of pmids.")


def surviving_claims(row: dict) -> set[str]:
    """The rubric's "surviving claims" term, used only by rules 4 and 5."""
    first: dict[str, dict] = {}
    for result in row["critic_results"] or []:
        first.setdefault(result["claim_id"], result)
    kept = {
        claim_id
        for claim_id, r in first.items()
        if CLAIM_ID.fullmatch(claim_id)
        and r["action"] == "keep"
        and r["supported"] is True
        and r["citation_valid"] is True
    }
    report = row["validation"]
    return kept - set(report["removed_uncited"]) - set(report["removed_invalid_citation"])


def matching_rules(row: dict, baseline_right: bool) -> list[str]:
    """Every category this error matches, in rubric order."""
    reason, gold, verdict = row["abstain_reason"], row["gold_label"], row["verdict"]
    matched = []
    if reason == "weak_evidence":
        matched.append("refusal_evidence_gate")
    if reason == "model_abstained":
        matched.append("refusal_model_abstained")
    if reason == "no_supported_claims":
        downgraded = row["validation"]["downgraded_to_insufficient"]
        if downgraded is True:
            matched.append("refusal_citation_check_left_no_claims")
        elif downgraded is False:
            if not surviving_claims(row) or row["answer_supported"] is not False:
                matched.append("refusal_critic_left_no_claims")
            else:
                matched.append("refusal_critic_rejected_answer")
    if row["abstained"] is False:
        added = row["planner_added_evidence"]
        if row["gold_final_rank"] != 1:
            matched.append("answered_gold_not_rank_1")
        if isinstance(added, int) and added > 0 and baseline_right:
            matched.append("answered_planner_added_baseline_right")
        if "maybe" in (gold, verdict):
            matched.append("answered_maybe_boundary")
        if {gold, verdict} == {"yes", "no"}:
            matched.append("answered_yes_no_flip")
    return matched


def classify(agent: list[dict], baseline: list[dict]) -> list[dict]:
    """One record per agent error, in file order. Stop condition 6 applies here."""
    baseline_by_pmid = {r["pmid"]: r for r in baseline}
    records = []
    for row in agent:
        if row["correct"] is not False:
            continue
        other = baseline_by_pmid[row["pmid"]]
        baseline_right = other["correct"] is True
        try:
            matched = matching_rules(row, baseline_right)
        except KeyError as missing:
            raise SystemExit(f"Stop: pmid {row['pmid']} has no field {missing}.") from None
        if not matched:
            raise SystemExit(f"Stop: pmid {row['pmid']} matches no rule; no results written.")
        records.append(
            {
                "pmid": row["pmid"],
                "category": matched[0],
                "also_matched": matched[1:],
                "group": "gap" if baseline_right else "shared",
                "gold_label": row["gold_label"],
                "agent_verdict": row["verdict"],
                "baseline_verdict": other["verdict"],
                "abstain_reason": row["abstain_reason"],
                "gold_final_rank": row["gold_final_rank"],
                "planner_added_evidence": row["planner_added_evidence"],
            }
        )
    return records


def sign_test_two_sided(a: int, b: int) -> float:
    """Exact two-sided sign test on the discordant pairs (ties are not counted)."""
    n = a + b
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(a, b) + 1)) / 2**n
    return min(1.0, 2 * tail)


def category_counts(records: list[dict]) -> dict[str, int]:
    counts = Counter(r["category"] for r in records)
    return {category: counts[category] for category in CATEGORIES}


def label_table(records: list[dict], verdict_field: str) -> dict[str, dict[str, int]]:
    table: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        table[r["gold_label"]][r[verdict_field]] += 1
    return {gold: dict(sorted(table[gold].items())) for gold in sorted(table)}


def answered_only(rows: list[dict]) -> dict:
    answered = [r for r in rows if r["abstained"] is False]
    correct = sum(r["correct"] for r in answered)
    return {
        "answered": len(answered),
        "correct": correct,
        "accuracy": correct / len(answered) if answered else None,
    }


def summarize(agent: list[dict], baseline: list[dict], records: list[dict]) -> dict:
    """Every count and rate the report needs, computed from the two pinned files."""
    gap = [r for r in records if r["group"] == "gap"]
    shared = [r for r in records if r["group"] == "shared"]
    agent_right = {r["pmid"] for r in agent if r["correct"]}
    baseline_only_wrong = sum(1 for r in baseline if not r["correct"] and r["pmid"] in agent_right)
    categories = category_counts(records)
    if sum(categories.values()) != len(records) or len(gap) + len(shared) != len(records):
        raise SystemExit("Stop: the category counts do not add up; no results written.")
    critic = [r["critic_results"] or [] for r in agent]
    return {
        "n_questions": len(agent),
        "agent_correct": len(agent_right),
        "baseline_correct": sum(r["correct"] for r in baseline),
        "agent_accuracy": len(agent_right) / len(agent),
        "baseline_accuracy": sum(r["correct"] for r in baseline) / len(baseline),
        "agent_errors": len(records),
        "agent_answered_wrongly": sum(r["abstain_reason"] is None for r in records),
        "agent_refusals_scored_wrong": sum(r["abstain_reason"] is not None for r in records),
        "agent_refusals_by_reason": dict(
            sorted(Counter(r["abstain_reason"] for r in records if r["abstain_reason"]).items())
        ),
        "answered_only": {"agent": answered_only(agent), "baseline": answered_only(baseline)},
        "paired": {
            "both_wrong": len(shared),
            "agent_wrong_baseline_right": len(gap),
            "baseline_wrong_agent_right": baseline_only_wrong,
            "exact_two_sided_sign_test_p": sign_test_two_sided(len(gap), baseline_only_wrong),
        },
        "categories": categories,
        "categories_gap": category_counts(gap),
        "categories_shared": category_counts(shared),
        "overlaps_on_answered_errors": dict(
            sorted(Counter(m for r in records for m in r["also_matched"]).items())
        ),
        "label_tables": {
            "shared_gold_by_agent_verdict": label_table(shared, "agent_verdict"),
            "shared_gold_by_baseline_verdict": label_table(shared, "baseline_verdict"),
            "gap_gold_by_agent_verdict": label_table(gap, "agent_verdict"),
        },
        "retrieval": {
            "agent_gold_in_final_evidence": sum(r["gold_final_rank"] is not None for r in agent),
            "errors_gold_not_rank_1": sum(r["gold_final_rank"] != 1 for r in records),
        },
        "critic": {
            "results_total": sum(map(len, critic)),
            "results_not_keep": sum(x["action"] != "keep" for c in critic for x in c),
            "questions_with_a_not_keep": sum(any(x["action"] != "keep" for x in c) for c in critic),
            "item_ids_removed": sum(len(r["critic_removed"]) for r in agent),
            "questions_with_a_removal": sum(bool(r["critic_removed"]) for r in agent),
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Sort the agent's test errors (rubric v1).")
    parser.add_argument(
        "--check-inputs",
        action="store_true",
        help="only check the pinned files and stop conditions 1 to 5; classifies nothing",
    )
    args = parser.parse_args(argv)

    if sha256(RUBRIC.read_bytes()) != RUBRIC_SHA256:
        raise SystemExit("Stop: the rubric differs from its locked SHA-256; no results written.")
    agent, baseline = load_input("agent"), load_input("baseline")
    stored_intervals = load_stored_intervals()
    check_rows(agent, baseline)
    if args.check_inputs:
        print(
            f"Rubric hash OK. Input hashes OK. Rows: agent {len(agent)}, baseline {len(baseline)}."
        )
        print("Stop conditions 1 to 5 clear. Nothing classified, nothing written.")
        return

    if earlier := sorted(p.name for p in RESULTS_DIR.glob(f"{RUN_PREFIX}*")):
        raise SystemExit(f"Stop: this analysis runs once and already exists: {earlier}.")
    git = git_state()
    if git["git_dirty"]:
        raise SystemExit("Stop: commit first; a real run needs a clean tree.")

    records = classify(agent, baseline)
    now = datetime.now(UTC)
    run_id = f"{RUN_PREFIX}{now:%Y%m%d-%H%M%S}-{git['git_commit'][:7]}"
    summary = {
        "run_id": run_id,
        "created_at": now.isoformat(),
        "git_commit": git["git_commit"],
        "rubric": {"file": RUBRIC.relative_to(REPO_ROOT).as_posix(), "sha256": RUBRIC_SHA256},
        "inputs": {system: {"file": name, "sha256": h} for system, (name, h) in INPUTS.items()},
        **summarize(agent, baseline, records),
        "stored_paired_bootstrap": stored_intervals,
    }
    out = RESULTS_DIR / run_id
    out.mkdir(parents=True, exist_ok=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (out / "categories.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    print(f"Errors categorized: {len(records)}. Saved to {out}")
    for category, count in summary["categories"].items():
        print(f"{count:4d}  {category}")


if __name__ == "__main__":
    main()
