"""M4-T4 scorer: metrics and intervals for the frozen test runs (docs/reviews/m4-t4-compute-plan.md,
owner-approved 2026-10-26; eval/methodology.md; rubric judge_v2.md §6).

Reads the three complete system test runs and the complete judge run, never edits them, and writes
ONE new folder `eval/results/m4-final-<time>-<commit>/`: raw.jsonl, summary.json, run_config.json,
methodology.md. Judge-estimated claim support is computed but HELD (owner instruction 2026-10-26:
not shown until the owner has filled in the audit worksheet): it goes only to
`claim_support.HELD-until-audit.json`, and nothing else written or printed contains judge labels.

    uv run python -m eval.runners.score_m4 --baseline-run <id> --agent-run <id> \
        --closed-book-run <id> --judge-run <id>

Refuses: a dirty git tree, a void or unfinished run, runs over different question IDs, IDs or gold
labels that differ from the frozen test manifest, a judge run for another pair, judgments that do
not match the items the systems showed, or any figure that disagrees with the value run_system.py
already stored in that run's summary.json."""

import argparse
import json
import statistics
import sys
from collections import Counter
from datetime import UTC, datetime

from eval.metrics.answers import ABSTAIN
from eval.metrics.intervals import N_RESAMPLES, SEED, bootstrap_ci, paired_bootstrap_ci, wilson
from eval.runners.judge import VOID_FILE, items_of
from eval.runners.retrieval import REPO_ROOT, RESULTS_DIR, git_state
from eval.runners.run_system import load_manifest, require_complete_run, require_same_questions

SYSTEMS = ("baseline", "agent", "closed_book")
JUDGED = ("baseline", "agent")
STAGES = {
    "baseline": ("retrieval", "synthesis"),
    "agent": ("planner", "retriever", "gate", "synthesizer", "critic", "validator"),
    "closed_book": (),
}
POST_GATE = {"synthesis", "synthesizer", "critic"}
HELD_FILE = "claim_support.HELD-until-audit.json"
CLAIM_SUPPORT_NAME = "Independent-judge estimated claim support"
NOT_JUDGED_LIMIT = 0.02  # rubric v2 §5: above this share a system's claim support is incomplete
COST_NOTE = "Anthropic LLM tokens at list price; Voyage embedding and rerank not included " \
            "(inside the free allocation)"  # fmt: skip
TOLERANCE = 1e-9


def quantile(values: list[float], q: int) -> float | None:
    """Same percentile rule as run_system.summarize (inclusive), so the two can be cross-checked."""
    if len(values) < 2:
        return values[0] if values else None
    return statistics.quantiles(values, n=100, method="inclusive")[q - 1]


def prop(successes: int, n: int) -> dict:
    ci = wilson(successes, n)
    return {"count": successes, "n": n, "value": successes / n if n else None,
            "wilson95": list(ci) if ci else None}  # fmt: skip


# Per-row facts. A refusal (verdict insufficient_evidence) or an error row is never correct.
def is_correct(r: dict) -> bool:
    return r["verdict"] == r["gold_label"]


def is_answered(r: dict) -> bool:
    return r["verdict"] != ABSTAIN


def hit(r: dict, k: int) -> bool:
    return r["pmid"] in r["evidence_pmids"][:k]


# Statistics over a list of rows (used for point estimates and every bootstrap resample).
def acc(rs):
    return sum(map(is_correct, rs)) / len(rs)


def cov(rs):
    return sum(map(is_answered, rs)) / len(rs)


def answered_acc(rs):
    answered = [r for r in rs if is_answered(r)]
    return sum(map(is_correct, answered)) / len(answered) if answered else None


def hit1(rs):
    return sum(hit(r, 1) for r in rs) / len(rs)


def hit5(rs):
    return sum(hit(r, 5) for r in rs) / len(rs)


def latency_p50(rs):
    return quantile([r["timings_ms"]["total"] for r in rs], 50)


def latency_p95(rs):
    return quantile([r["timings_ms"]["total"] for r in rs], 95)


def cost_mean(rs):
    return statistics.mean(r["cost_usd"] for r in rs)


def cost_median(rs):
    return statistics.median(r["cost_usd"] for r in rs)


def answer_metrics(system: str, rows: list[dict]) -> dict:
    n = len(rows)
    answered = [r for r in rows if is_answered(r)]
    refusals = Counter(r["abstain_reason"] or "unspecified" for r in rows if not is_answered(r))
    retrieval = system != "closed_book"
    out = {
        "n_questions": n,
        "accuracy": prop(sum(map(is_correct, rows)), n),
        "coverage": prop(len(answered), n),
        "answered_only_accuracy": prop(sum(map(is_correct, answered)), len(answered)),
        "refusals_by_reason": dict(sorted(refusals.items())),
        "errors": sum(r["error"] is not None for r in rows),
        "hit_at_1": prop(sum(hit(r, 1) for r in rows), n) if retrieval else "not applicable",
        "hit_at_5": prop(sum(hit(r, 5) for r in rows), n) if retrieval else "not applicable",
        "latency_ms": {
            "p50": bootstrap_ci(rows, latency_p50),
            "p95": bootstrap_ci(rows, latency_p95),
            "stages": {s: stage_latency(rows, s) for s in STAGES[system]},
            "measured_from": "developer machine, one question at a time, not production load",
        },
        "llm_cost_only_usd_per_query": {
            "mean": bootstrap_ci(rows, cost_mean),
            "median": bootstrap_ci(rows, cost_median),
            "total": sum(r["cost_usd"] for r in rows),
            "includes": COST_NOTE,
        },
    }
    if system == "agent":
        ok = [r for r in rows if r["error"] is None]
        out["planner_added_evidence"] = {
            "questions_with_added_papers": sum(r["planner_added_evidence"] > 0 for r in ok),
            "of_questions": len(ok),
        }
    return out


def stage_latency(rows: list[dict], stage: str) -> dict:
    """p50/p95 over the questions where the stage ran (gate refusals skip later stages)."""
    # stages after the evidence gate do not run on a gate refusal or error row (stored as 0.0);
    # fast stages that did run may also round to 0.0, so the rule is the gate, not the value
    values = [r["timings_ms"][stage] for r in rows if stage in r["timings_ms"]
              and (stage not in POST_GATE or r["gate_passed"])]  # fmt: skip
    return {"n": len(values), "p50": quantile(values, 50), "p95": quantile(values, 95)}


def cross_check(system: str, metrics: dict, stored: dict) -> list[str]:
    """Figures run_system.py already wrote for this run must be reproduced exactly."""
    pairs = [
        ("accuracy", metrics["accuracy"]["value"], stored["accuracy"]),
        ("coverage", metrics["coverage"]["value"], stored["coverage"]),
        ("answered_only_accuracy", metrics["answered_only_accuracy"]["value"],
         stored["answered_only_accuracy"]),
        ("latency p50", metrics["latency_ms"]["p50"]["estimate"], stored["latency_ms"]["p50"]),
        ("latency p95", metrics["latency_ms"]["p95"]["estimate"], stored["latency_ms"]["p95"]),
        ("cost total", round(metrics["llm_cost_only_usd_per_query"]["total"], 4),
         stored["cost_usd"]["total"]),
        ("errors", metrics["errors"], stored["errors"]),
    ]  # fmt: skip
    if system != "closed_book":
        pairs += [("hit_at_1", metrics["hit_at_1"]["value"], stored["hit_at_1"]),
                  ("hit_at_5", metrics["hit_at_5"]["value"], stored["hit_at_5"])]  # fmt: skip
    return [f"{system} {name}" for name, ours, theirs in pairs
            if ours is None or theirs is None or abs(ours - theirs) > TOLERANCE]  # fmt: skip


def comparisons(rows: dict) -> dict:
    """Paired bootstrap differences, `later minus earlier` system in each name."""
    answer_stats = {"accuracy": acc, "coverage": cov, "answered_only_accuracy": answered_acc}
    retrieval_stats = {"hit_at_1": hit1, "hit_at_5": hit5, "latency_p50_ms": latency_p50,
                       "llm_cost_only_mean_usd": cost_mean}  # fmt: skip
    b, a, c = rows["baseline"], rows["agent"], rows["closed_book"]
    return {
        "agent_minus_baseline": {name: paired_bootstrap_ci(b, a, f)
                                 for name, f in (answer_stats | retrieval_stats).items()},
        "baseline_minus_closed_book": {name: paired_bootstrap_ci(c, b, f)
                                       for name, f in answer_stats.items()},
        "agent_minus_closed_book": {name: paired_bootstrap_ci(c, a, f)
                                    for name, f in answer_stats.items()},
        "how_to_read": "an interval that includes 0 means no clear difference",
    }  # fmt: skip


def join_judgments(rows: dict, judgments: list[dict], blinding: dict) -> dict:
    """judgments per system, checked one-to-one against the items each system showed."""
    by_system = {}
    for system in JUDGED:
        shown = {(r["pmid"], it["item_id"]): it for r in rows[system] for it in items_of(r)}
        judged = [j for j in judgments if j["system"] == system]
        keys = [(j["pmid"], j["item_id"]) for j in judged]
        if len(keys) != len(set(keys)) or set(keys) != set(shown):
            raise SystemExit(f"{system}: judgments do not match the items shown; refusing.")
        for j in judged:
            it = shown[(j["pmid"], j["item_id"])]
            if (j["text"], j["kind"], j["citation_ids"]) != (it["text"], it["kind"],
                                                             it["citation_ids"]):  # fmt: skip
                raise SystemExit(f"{system}: a judged item differs from the run; refusing.")
            if blinding[j["pmid"]][f"SYSTEM_{j['blind_id'][0]}"] != system:
                raise SystemExit(f"{system}: blind ID {j['blind_id']} contradicts the blinding "
                                 "key; refusing.")  # fmt: skip
        by_system[system] = judged
    return by_system


def per_question(judged: list[dict], pmids: list[str], kind: str) -> list[tuple[int, int, int]]:
    """(supported, supported or partially, judged) per question, in `pmids` order."""
    out = []
    for p in pmids:
        labels = [
            j["label"]
            for j in judged
            if j["pmid"] == p and j["kind"] == kind and j["label"] != "not_judged"
        ]
        out.append((labels.count("supported"),
                    labels.count("supported") + labels.count("partially_supported"),
                    len(labels)))  # fmt: skip
    return out


def ratio(index: int):
    def stat(rs):
        n = sum(r[2] for r in rs)
        return sum(r[index] for r in rs) / n if n else None

    return stat


def claim_support(by_system: dict, judged_pmids: list[str]) -> dict:
    """Rubric v2 §6. HELD: written only to HELD_FILE."""
    out = {"name": CLAIM_SUPPORT_NAME,
           "caveat": "The two systems show different numbers of claims (the agent's critic removes "
                     "some, and it abstains more), so the denominators differ.",
           "systems": {}}  # fmt: skip
    for system, judged in by_system.items():
        result = {"label_counts": {}}
        for kind in ("claim", "limitation"):
            mine = [j for j in judged if j["kind"] == kind]
            result["label_counts"][kind] = dict(sorted(Counter(j["label"] for j in mine).items()))
            rows = [q for q in per_question(judged, judged_pmids, kind) if q[2]]
            n = sum(q[2] for q in rows)
            for name, index in (("strict", 0), ("lenient", 1)):
                cluster = bootstrap_ci(rows, ratio(index))["ci95"] if rows else None
                result[f"{kind}_{name}"] = prop(sum(q[index] for q in rows), n) | {
                    "question_bootstrap95": cluster
                }
        with_items = {j["pmid"] for j in judged if j["label"] != "not_judged"}
        not_ok = ("partially_supported", "unsupported")
        flagged = {j["pmid"] for j in judged if j["label"] in not_ok}
        result["questions_with_a_not_supported_item"] = prop(len(flagged), len(with_items))
        not_judged = sum(j["label"] == "not_judged" for j in judged)
        result["not_judged"] = not_judged
        result["incomplete"] = not_judged / len(judged) > NOT_JUDGED_LIMIT if judged else True
        out["systems"][system] = result
    deltas = {}
    for kind in ("claim", "limitation"):
        base = per_question(by_system["baseline"], judged_pmids, kind)
        agent = per_question(by_system["agent"], judged_pmids, kind)
        for name, index in (("strict", 0), ("lenient", 1)):
            deltas[f"{kind}_{name}"] = paired_bootstrap_ci(base, agent, ratio(index))
    out["agent_minus_baseline"] = deltas
    return out


def check_judge_run(judge_dir, args, ids: list[str], rows: dict) -> list[str]:
    """The judge run must be complete and judge exactly this pair. Its summary.json is never
    opened here (it holds label counts)."""
    if (judge_dir / VOID_FILE).exists() or not (judge_dir / "summary.json").exists():
        raise SystemExit(f"{judge_dir.name} is void or unfinished; it must not be scored.")
    config = json.loads((judge_dir / "run_config.json").read_text(encoding="utf-8"))
    if (config["baseline_run"], config["agent_run"]) != (args.baseline_run, args.agent_run):
        raise SystemExit(f"{judge_dir.name} judged a different pair of runs; refusing.")
    with_items = [p for i, p in enumerate(ids)
                  if any(items_of(rows[s][i]) for s in JUDGED)]  # fmt: skip
    if config["question_ids"] != with_items or config.get("setup_check"):
        raise SystemExit(f"{judge_dir.name} did not judge every test question with items.")
    return with_items


def fmt_pct(p: dict | str) -> str:
    if isinstance(p, str):
        return p
    if p["value"] is None:
        return "n/a"
    lo, hi = p["wilson95"]
    return f"{p['value']:.1%} ({lo:.1%}–{hi:.1%}), {p['count']}/{p['n']}"


def fmt_ci(b: dict, scale: float = 1, digits: int = 2) -> str:
    est, (lo, hi) = b["estimate"] * scale, b["ci95"]
    return f"{est:.{digits}f} ({lo * scale:.{digits}f}–{hi * scale:.{digits}f})"


def results_section(summary: dict) -> str:
    s = summary["systems"]
    lines = ["| Metric | Baseline | Agent | Closed-book |", "|---|---|---|---|"]
    for label, key in (("Overall accuracy (refusal = wrong)", "accuracy"),
                       ("Coverage (answered / 200)", "coverage"),
                       ("Answered-only accuracy", "answered_only_accuracy"),
                       ("Hit@1", "hit_at_1"), ("Hit@5", "hit_at_5")):  # fmt: skip
        lines.append(f"| {label} | " + " | ".join(fmt_pct(s[x][key]) for x in SYSTEMS) + " |")
    for label, key in (("Latency p50, s", "p50"), ("Latency p95, s", "p95")):
        cells = [fmt_ci(s[x]["latency_ms"][key], 1 / 1000, digits=2) for x in SYSTEMS]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    for label, key in (("LLM cost only, per query, mean $", "mean"),
                       ("LLM cost only, per query, median $", "median")):  # fmt: skip
        cells = [fmt_ci(s[x]["llm_cost_only_usd_per_query"][key], digits=5) for x in SYSTEMS]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    majority = fmt_pct(summary["majority_class_reference"])
    lines.append(f"| Majority-class reference (always yes) | {majority} | | |")
    lines += ["", "Main comparison — paired differences (the planned paired bootstrap, 95%; an "
              "interval that includes 0 = no clear difference):", "",
              "| Comparison | Metric | Difference | 95% interval |",
              "|---|---|---|---|"]  # fmt: skip
    for comp, metrics in summary["comparisons"].items():
        if not isinstance(metrics, dict):
            continue
        for name, d in metrics.items():
            if d["estimate"] is None or d["ci95"] is None:
                lines.append(f"| {comp} | {name} | n/a | n/a |")
                continue
            lines.append(f"| {comp} | {name} | {d['estimate']:+.4f} | "
                         f"{d['ci95'][0]:+.4f} to {d['ci95'][1]:+.4f} |")  # fmt: skip
    return "\n".join(lines)


def write_methodology(out, summary: dict, config: dict) -> None:
    base = (REPO_ROOT / "eval" / "methodology.md").read_text(encoding="utf-8")
    runs = "\n".join(f"- {k}: `{v}`" for k, v in config["inputs"].items())
    text = (
        f"# M4 final results and methodology — `{config['run_id']}`\n\n"
        f"Computed {config['created_utc']} at commit `{config['git_commit']}` by "
        "`eval/runners/score_m4.py`, from the plan approved by the owner on 2026-10-26 "
        "(`docs/reviews/m4-t4-compute-plan.md`).\n\nInputs (complete runs, never edited):\n"
        f"{runs}\n\n## Results (test split, 200 questions per system)\n\n"
        f"{results_section(summary)}\n\n"
        f"LLM cost only: {COST_NOTE}. Latency was {summary['systems']['baseline']['latency_ms']['measured_from']}.\n\n"  # noqa: E501
        f"## {CLAIM_SUPPORT_NAME}\n\nHELD until the owner has filled in the audit worksheet "
        "(owner instruction 2026-10-26). Computed by this run and stored separately; reported here "
        "after the audit.\n\n## Changes made after test results were seen\n\nNone.\n\n---\n\n"
        "The methodology as written before any test run follows, unchanged.\n\n" + base
    )
    (out / "methodology.md").write_text(text, encoding="utf-8")


def raw_row(system: str, r: dict) -> dict:
    """One joined per-question record. Judge labels are held, so none are written here."""
    retrieval = system != "closed_book"
    return {"system": system, "pmid": r["pmid"], "question": r["question"],
            "gold_label": r["gold_label"], "verdict": r["verdict"], "correct": is_correct(r),
            "answered": is_answered(r), "abstain_reason": r["abstain_reason"], "error": r["error"],
            "gold_final_rank": r["gold_final_rank"] if retrieval else None,
            "hit_at_1": hit(r, 1) if retrieval else None,
            "hit_at_5": hit(r, 5) if retrieval else None,
            "evidence_pmids": r["evidence_pmids"],
            "claims": [{"id": c["id"], "text": c["text"], "citation_ids": c["citation_ids"]}
                       for c in r["claims"]],
            "limitations": [{"id": c["id"], "text": c["text"], "citation_ids": c["citation_ids"]}
                            for c in r["limitations"]],
            "timings_ms": r["timings_ms"], "usage": r["usage"],
            "cost_usd": r["cost_usd"]}  # fmt: skip


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    for name in ("--baseline-run", "--agent-run", "--closed-book-run", "--judge-run"):
        parser.add_argument(name, required=True)
    args = parser.parse_args(argv)

    git = git_state()
    if git["git_dirty"]:
        raise SystemExit("Uncommitted changes: score only from a clean commit.")
    names = {"baseline": args.baseline_run, "agent": args.agent_run,
             "closed_book": args.closed_book_run}  # fmt: skip
    dirs = {s: RESULTS_DIR / n for s, n in names.items()}
    for system, d in dirs.items():
        config = json.loads((d / "run_config.json").read_text(encoding="utf-8"))
        if config["system"] != system or config["split"] != "test":
            raise SystemExit(f"{d.name} is not a {system} test run; refusing.")
    ids = require_same_questions(*dirs.values())
    rows = {s: require_complete_run(d) for s, d in dirs.items()}
    gold = {it["pmid"]: it["final_decision"] for it in load_manifest("test")["items"]}
    if sorted(ids) != sorted(gold) or any(
        r["gold_label"] != gold[r["pmid"]] for s in SYSTEMS for r in rows[s]
    ):
        raise SystemExit("Question IDs or gold labels differ from the frozen test manifest.")
    if any(r["correct"] != is_correct(r) for s in SYSTEMS for r in rows[s]):
        raise SystemExit("A stored `correct` flag disagrees with verdict == gold; refusing.")
    judge_dir = RESULTS_DIR / args.judge_run
    judged_pmids = check_judge_run(judge_dir, args, ids, rows)
    judgments = [json.loads(line)
                 for line in (judge_dir / "judgments.jsonl").open(encoding="utf-8")]  # fmt: skip
    blinding = json.loads((judge_dir / "blinding_key.json").read_text(encoding="utf-8"))
    by_system = join_judgments(rows, judgments, blinding)

    systems = {s: answer_metrics(s, rows[s]) for s in SYSTEMS}
    mismatches = [
        m
        for s in SYSTEMS
        for m in cross_check(
            s, systems[s], json.loads((dirs[s] / "summary.json").read_text(encoding="utf-8"))
        )
    ]
    if mismatches:
        raise SystemExit(f"Disagrees with the stored run summaries: {mismatches}; refusing.")
    yes = sum(g == "yes" for g in gold.values())
    summary = {
        "split": "test", "n_questions": len(ids),
        "rules": {"overall_accuracy": "refusals and error rows count as incorrect",
                  "coverage": "always shown beside answered-only accuracy",
                  "proportions": "Wilson 95%", "latency_cost": "bootstrap 95%",
                  "differences": "paired bootstrap 95% over the same question IDs",
                  "resamples": N_RESAMPLES, "seed": SEED},
        "majority_class_reference": prop(yes, len(gold)),
        "systems": systems,
        "comparisons": comparisons(rows),
        "claim_support": f"HELD until the owner's audit worksheet is filled in (owner instruction "
                         f"2026-10-26); computed in {HELD_FILE}",
    }  # fmt: skip
    held = claim_support(by_system, judged_pmids)

    created = datetime.now(UTC)
    run_id = f"m4-final-{created:%Y%m%d-%H%M%S}-{git['git_commit'][:7]}"
    out = RESULTS_DIR / run_id
    out.mkdir(parents=True, exist_ok=False)
    config = {"run_id": run_id, "created_utc": created.isoformat(timespec="seconds"), **git,
              "inputs": names | {"judge": args.judge_run}, "judged_questions": len(judged_pmids),
              "plan": "docs/reviews/m4-t4-compute-plan.md (owner OK 2026-10-26)",
              "resamples": N_RESAMPLES, "seed": SEED, "held_files": [HELD_FILE]}  # fmt: skip
    (out / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    with (out / "raw.jsonl").open("w", encoding="utf-8") as f:
        for i in range(len(ids)):
            for s in SYSTEMS:
                f.write(json.dumps(raw_row(s, rows[s][i])) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out / HELD_FILE).write_text(json.dumps(held, indent=2) + "\n", encoding="utf-8")
    write_methodology(out, summary, config)
    print(f"Saved to eval/results/{run_id} (claim support held in {HELD_FILE}; not printed)")


if __name__ == "__main__":
    main(sys.argv[1:])
