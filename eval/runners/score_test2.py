"""M15-T9 scorer: baseline vs agent on the second frozen test set, test2 (ADR-043, ADR-045). $0.

    uv run python -m eval.runners.score_test2 --baseline-run <test2-baseline-...> \
        --agent-run <test2-agent-...>

No model or API is called. Before computing anything it refuses unless: the Git tree is clean; both
runs are complete (not void) test2 runs of the named systems on the frozen test2.json (hash checked
against test2.sha256); they cover every test2 question in the same order; every stored gold label
equals the manifest and every stored `correct` flag equals verdict == gold; and the stored run
summaries agree with the recomputed accuracy, refusals, errors and cost.

Reports, from the stored rows only: correct and refused per system, accuracy (Wilson 95%), cost
per question, median time per answer; agent-only vs baseline-only right; the exact two-sided sign
test; the paired bootstrap 95% range on agent minus baseline accuracy; how many questions used the
agent's retry and what it changed. Then both written decision rules, each labelled with its source:
  - ADR-045 (owner): agent at or above the baseline (correct counts compared);
  - ADR-043 (pre-registered): "clearly beats" = bootstrap range excludes 0 AND sign test p < 0.05,
    with the agent ahead (a clear loss also excludes 0; it is not "beats").
Also refuses when the agent run's rules.py hash differs from the checked-out one (the retry replay
would use other rules) and when a system has more than one completed test2 run (no picking).
Prints and stores numbers and counts only: never a question, answer or claim text.
Writes eval/results/m15-test2-score-<time>-<commit>/ (run_config.json, summary.json).
"""

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter
from datetime import UTC, datetime

from backend.app.agents.rules import refusal_reason
from backend.app.models.agent import CriticResult
from backend.app.models.answer import DraftAnswer
from eval.metrics.intervals import N_RESAMPLES, SEED, paired_bootstrap_ci, wilson
from eval.runners.failure_analysis import sign_test_two_sided
from eval.runners.retrieval import RESULTS_DIR, git_state
from eval.runners.run_system import (
    AGENT_RULES_FILE,
    MANIFESTS,
    completed_test_runs,
    load_manifest,
    require_complete_run,
    require_same_questions,
)

SPLIT = "test2"
IE = "insufficient_evidence"
TOLERANCE = 1e-9
# The decision rules, quoted from DECISIONS.md (never paraphrased into a different rule).
ADR_045_LINE = (
    'Supersedes for M15 the ADR-043 go/no-go "DEV at least +3" with the owner\'s "at or above the '
    'same-day DEV baseline".'
)
ADR_043_LINE = (
    '"clearly beats" = on test2, the paired bootstrap 95% range of agent minus baseline accuracy '
    "excludes zero AND the exact two-sided sign test p < 0.05."
)


def rules_sha256() -> str:
    return hashlib.sha256(AGENT_RULES_FILE.read_bytes()).hexdigest()


def is_correct(r: dict) -> bool:
    return r["verdict"] == r["gold_label"]  # a refusal or an error row is never correct


def acc(rows: list[dict]) -> float:
    return sum(map(is_correct, rows)) / len(rows)


def system_metrics(rows: list[dict]) -> dict:
    n = len(rows)
    correct = sum(map(is_correct, rows))
    refused = [r for r in rows if r["verdict"] == IE]
    lo, hi = wilson(correct, n)
    return {
        "n_questions": n,
        "correct": correct,
        "refused": len(refused),
        "refusal_reasons": dict(
            sorted(Counter(r["abstain_reason"] or "unspecified" for r in refused).items())
        ),  # fmt: skip
        "errors": sum(r["error"] is not None for r in rows),
        "accuracy_pct": round(100 * correct / n, 2),
        "accuracy_wilson95_pct": [round(100 * lo, 2), round(100 * hi, 2)],
        "cost_usd_total": round(sum(r["cost_usd"] for r in rows), 4),
        "cost_usd_per_question": round(sum(r["cost_usd"] for r in rows) / n, 5),
        "median_time_per_answer_ms": round(
            statistics.median(r["timings_ms"]["total"] for r in rows), 1
        ),
    }


def paired(baseline: list[dict], agent: list[dict]) -> dict:
    """Same question in row i of both lists (require_same_questions checked the order)."""
    pairs = list(zip(baseline, agent, strict=True))
    agent_only = sum(is_correct(a) and not is_correct(b) for b, a in pairs)
    baseline_only = sum(is_correct(b) and not is_correct(a) for b, a in pairs)
    diff = paired_bootstrap_ci(baseline, agent, acc)
    return {
        "agent_only_right": agent_only,
        "baseline_only_right": baseline_only,
        "both_right": sum(is_correct(a) and is_correct(b) for b, a in pairs),
        "both_wrong": sum(not is_correct(a) and not is_correct(b) for b, a in pairs),
        "sign_test_p_exact_two_sided": sign_test_two_sided(agent_only, baseline_only),
        "accuracy_diff_agent_minus_baseline_points": round(100 * diff["estimate"], 4),
        "accuracy_diff_bootstrap95_points": [round(100 * v, 4) for v in diff["ci95"]],
        "bootstrap_range_excludes_zero": not diff["includes_zero"],
        "bootstrap_range_entirely_above_zero": diff["ci95"][0] > 0,
        "bootstrap": {"resamples": N_RESAMPLES, "seed": SEED, "kind": "paired, by question"},
    }


def first_pass_reason(row: dict) -> str | None:
    """Why the agent's first pass would have been refused under the final rules, or None."""
    try:
        draft = DraftAnswer.model_validate(row["first_draft_answer"])
        results = [CriticResult.model_validate(r) for r in row["first_critic_results"]]
    except (KeyError, TypeError, ValueError):  # no traceback: it would echo the claim text
        raise SystemExit(f"Question {row['pmid']}: stored first pass is unreadable.") from None
    evidence_ids = [f"PMID:{p}" for p in row["evidence_pmids"]]
    return refusal_reason(draft, evidence_ids, results, row["first_answer_supported"])[0]


def retry_report(agent: list[dict]) -> dict:
    """What the retry changed. Under ADR-045 it runs only when the first pass would be refused,
    so "the first pass alone" = a refusal (counted wrong) unless the check below disagrees."""
    retried = [r for r in agent if r.get("retry_used")]
    reasons = [first_pass_reason(r) for r in retried]
    failed = [r for r in retried if r.get("retry_error")]
    ok = [r for r in retried if not r.get("retry_error")]
    return {
        "questions_retried": len(retried),
        "share_retried_pct": round(100 * len(retried) / len(agent), 2),
        "first_pass_would_have_refused": sum(x is not None for x in reasons),
        "first_pass_would_have_answered": sum(x is None for x in reasons),  # expected 0
        "retry_failed_first_pass_stood": len(failed),
        "after_retry_right": sum(map(is_correct, ok)),
        "after_retry_answered_wrong": sum(r["verdict"] != IE and not is_correct(r) for r in ok),
        "after_retry_still_refused": sum(r["verdict"] == IE for r in ok),
        "verdict_changes_first_draft_to_final": dict(
            sorted(
                Counter(
                    f"{r['first_draft_answer']['verdict']} -> {r['verdict']}"
                    for r in ok
                    if r["first_draft_answer"]["verdict"] != r["verdict"]
                ).items()
            )
        ),  # fmt: skip
        "cost_usd_per_retried_question": (
            round(sum(r["cost_usd"] for r in retried) / len(retried), 5) if retried else None
        ),
    }


def decisions(baseline: dict, agent: dict, pair: dict) -> dict:
    at_or_above = agent["correct"] >= baseline["correct"]
    agent_ahead = pair["agent_only_right"] > pair["baseline_only_right"]
    clearly = (
        pair["bootstrap_range_excludes_zero"]
        and pair["sign_test_p_exact_two_sided"] < 0.05
        and agent_ahead
    )
    return {
        "adr_045_agent_at_or_above_baseline": {
            "source": "DECISIONS.md ADR-045", "quoted": ADR_045_LINE,
            "applied_as": "agent correct >= baseline correct on the same test2 questions",
            "note": "ADR-045 wrote this as the DEV go/no-go; applied to test2 at the owner's "
                    "request (2026-11-10).",
            "met": at_or_above,
        },
        "adr_043_clearly_beats_preregistered": {
            "source": "DECISIONS.md ADR-043 (not superseded)", "quoted": ADR_043_LINE,
            "applied_as": "bootstrap range excludes 0 AND sign test p < 0.05, with the agent "
                          "ahead (\"beats\": a clear loss is not a win)",
            "parts": {"range_excludes_zero": pair["bootstrap_range_excludes_zero"],
                      "sign_test_p_below_0_05": pair["sign_test_p_exact_two_sided"] < 0.05,
                      "agent_ahead": agent_ahead},
            "met": clearly,
        },
    }  # fmt: skip


def check_inputs(names: dict) -> dict:
    """Every refusal happens here, before anything is computed or written. Returns rows."""
    frozen = (MANIFESTS / f"{SPLIT}.sha256").read_text(encoding="utf-8").split()[0]
    dirs = {s: RESULTS_DIR / n for s, n in names.items()}
    for system, d in dirs.items():
        config = json.loads((d / "run_config.json").read_text(encoding="utf-8"))
        if config.get("system") != system or config.get("split") != SPLIT:
            raise SystemExit(f"{d.name} is not a {system} {SPLIT} run; refusing.")
        if config.get("manifest_sha256") != frozen:
            raise SystemExit(f"{d.name} did not run on the frozen {SPLIT}.json; refusing.")
        if system == "agent" and config.get("agent_rules_sha256") != rules_sha256():
            raise SystemExit(f"{d.name} ran with different agent rules than {AGENT_RULES_FILE.name}"
                             " now; check out the run's commit to score it.")  # fmt: skip
    ids = require_same_questions(*dirs.values())
    rows = {s: require_complete_run(d) for s, d in dirs.items()}
    for system, d in dirs.items():
        completed = completed_test_runs(system, SPLIT)
        if completed != [d.name]:
            raise SystemExit(
                f"Completed {system} {SPLIT} runs: {completed}. Exactly one may exist "
                "(ADR-031); the owner decides which counts. Refusing."
            )
    gold = {it["pmid"]: it["final_decision"] for it in load_manifest(SPLIT)["items"]}
    if sorted(ids) != sorted(gold) or len(ids) != len(gold):
        raise SystemExit(f"The runs do not cover exactly the {SPLIT} questions; refusing.")
    for system, rs in rows.items():
        if any(r["gold_label"] != gold[r["pmid"]] for r in rs):
            raise SystemExit(f"{system}: a stored gold label differs from {SPLIT}.json; refusing.")
        if any(r["correct"] != is_correct(r) for r in rs):
            raise SystemExit(f"{system}: a stored `correct` flag disagrees with verdict == gold.")
        stored = json.loads((dirs[system] / "summary.json").read_text(encoding="utf-8"))
        ours = system_metrics(rs)
        checks = [
            (stored["n_questions"], ours["n_questions"]),
            (stored["accuracy"], ours["correct"] / ours["n_questions"]),
            (stored["verdict_counts"].get(IE, 0), ours["refused"]),
            (stored["errors"], ours["errors"]),
            (stored["cost_usd"]["total"], ours["cost_usd_total"]),
        ]
        if any(abs(a - b) > 1e-4 for a, b in checks):
            raise SystemExit(f"{system}: recomputed numbers disagree with its stored summary.")
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--agent-run", required=True)
    args = parser.parse_args(argv)

    git = git_state()
    if git["git_dirty"]:
        raise SystemExit("Uncommitted changes: score only from a clean commit.")
    names = {"baseline": args.baseline_run, "agent": args.agent_run}
    rows = check_inputs(names)
    systems = {s: system_metrics(rs) for s, rs in rows.items()}
    pair = paired(rows["baseline"], rows["agent"])
    summary = {
        "split": SPLIT,
        "n_questions": systems["baseline"]["n_questions"],
        "rules": {"accuracy": "refusals and error rows count as incorrect",
                  "accuracy_interval": "Wilson 95%",
                  "difference": "paired bootstrap 95% over the same question IDs",
                  "sign_test": "exact two-sided, discordant pairs only",
                  "cost": "Anthropic LLM tokens at list price; Voyage not included"},
        "systems": systems,
        "agent_vs_baseline": pair,
        "agent_retry": retry_report(rows["agent"]),
        "decision_rules": decisions(systems["baseline"], systems["agent"], pair),
    }  # fmt: skip

    created = datetime.now(UTC)
    run_id = f"m15-{SPLIT}-score-{created:%Y%m%d-%H%M%S}-{git['git_commit'][:7]}"
    out = RESULTS_DIR / run_id
    out.mkdir(parents=True, exist_ok=False)
    config = {"run_id": run_id, "created_utc": created.isoformat(timespec="seconds"), **git,
              "scored_split": SPLIT, "inputs": names,
              "manifest_sha256": hashlib.sha256((MANIFESTS / f"{SPLIT}.json").read_bytes())
              .hexdigest()}  # fmt: skip
    (out / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved to eval/results/{run_id}")


if __name__ == "__main__":
    main(sys.argv[1:])
