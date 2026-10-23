"""Run a system over a frozen split; store raw + summary results (ARCHITECTURE §9, PLAN M2-T5).

Run: uv run python -m eval.runners.run_system --system baseline|agent --split dev [--limit N]
     [--max-cost USD]

--max-cost is a hard spending stop: before each question the runner checks that the money spent so
far plus the most expensive question seen (at least PER_QUESTION_ALLOWANCE) stays within the cap;
otherwise it stops, and the summary says `stopped_at_cost_cap` with the number of questions run.

Calls the same Python function the API uses (not HTTP), sequentially. The gold label is read here
only, after the answer is produced; the pipeline never sees it. The test split additionally needs
--i-understand-this-is-the-final-test, a clean Git tree, the manifest matching test.sha256, and no
--limit. Each system runs on the test split once (ADR-031): if a completed test run of that system
exists, the script refuses unless given both --rerun-final-test and --reason "<why>"; the reason and
the earlier run IDs are written to override.json in the new run directory. A question that fails
(provider or database outage) is stored as an error row, counted as incorrect; the run continues.
Results go to eval/results/<run-id>/ (raw.jsonl, summary.json, run_config.json); a run is never
overwritten.
"""

import argparse
import hashlib
import json
import statistics
import sys
import time
from datetime import UTC, datetime

from backend.app.agents import critic as agent_critic
from backend.app.agents import planner as agent_planner
from backend.app.agents.pipeline import answer_agent
from backend.app.baseline.pipeline import answer_baseline
from backend.app.baseline.synthesizer import (
    SYNTHESIZER_EFFORT,
    SYNTHESIZER_VERSION,
    SYSTEM_PROMPT,
    USER_TEMPLATE,
)
from backend.app.config import (
    EMBEDDING_MODEL,
    EVIDENCE_GATE_DEV_RUN,
    EVIDENCE_GATE_THRESHOLD,
    LLM_MODEL,
    REPO_ROOT,
    RERANK_MODEL,
    redact,
)
from backend.app.llm.client import anthropic_client
from backend.app.retrieval.embed import voyage_client
from backend.app.retrieval.vector_search import connect
from eval.metrics.answers import (
    accuracy,
    answered_only_accuracy,
    coverage,
    majority_class_accuracy,
)
from eval.metrics.retrieval import gold_rank, mean_hit_at_k
from eval.runners.retrieval import CORPUS_MANIFEST, RESULTS_DIR, git_state

MANIFESTS = REPO_ROOT / "data" / "manifests"
FINAL_TEST_FLAG = "--i-understand-this-is-the-final-test"
RERUN_FLAG = "--rerun-final-test"
MIN_REASON_CHARS = 20
PER_QUESTION_ALLOWANCE = 0.05  # USD; agent questions measured $0.024-0.034 (M3-T7)


def load_manifest(split: str) -> dict:
    data = (MANIFESTS / f"{split}.json").read_bytes()
    if split == "test":
        frozen = (MANIFESTS / "test.sha256").read_text(encoding="utf-8").split()[0]
        if hashlib.sha256(data).hexdigest() != frozen:
            raise SystemExit("test.json does not match the frozen test.sha256; refusing to run.")
    manifest = json.loads(data)
    if manifest["split"] != split:
        raise RuntimeError(f"{split}.json does not declare split {split!r}")
    return manifest


def check_allowed(split: str, final_test_ok: bool, git: dict, limit: int | None = None) -> None:
    if split == "test" and not final_test_ok:
        raise SystemExit(f"Refusing the frozen test split without {FINAL_TEST_FLAG}.")
    if split == "test" and git["git_dirty"]:
        raise SystemExit("Refusing the frozen test split from a dirty Git tree; commit first.")
    if split == "test" and limit is not None:
        raise SystemExit("Refusing a partial run of the frozen test split (--limit).")
    if limit is not None and limit < 1:
        raise SystemExit("--limit must be at least 1.")


def completed_test_runs(system: str) -> list[str]:
    """Run IDs of finished (summary.json written) test-split runs of `system`."""
    done = []
    for config_path in sorted(RESULTS_DIR.glob("*/run_config.json")):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("split") == "test" and config.get("system") == system:
            if (config_path.parent / "summary.json").exists():
                done.append(config["run_id"])
    return done


def check_once_only(split: str, system: str, rerun: bool, reason: str | None) -> dict | None:
    """The frozen test split runs once per system (ADR-031). Returns the override record, if any."""
    if split != "test":
        if rerun or reason is not None:
            raise SystemExit(f"{RERUN_FLAG} and --reason apply only to the test split.")
        return None
    previous = completed_test_runs(system)
    if not previous:
        if rerun or reason is not None:
            raise SystemExit(f"No completed {system} test run exists; drop {RERUN_FLAG}/--reason.")
        return None
    if not rerun or reason is None:
        raise SystemExit(
            f"A completed {system} test run already exists ({', '.join(previous)}). The frozen "
            f'test set runs once; to override, pass {RERUN_FLAG} and --reason "<why>".'
        )
    if len(reason.strip()) < MIN_REASON_CHARS:
        raise SystemExit(
            f"--reason must explain the rerun (at least {MIN_REASON_CHARS} characters)."
        )
    return {
        "reason": reason.strip(),
        "previous_completed_test_runs": previous,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def create_run_dir(run_id: str):
    out = RESULTS_DIR / run_id
    out.mkdir(parents=True, exist_ok=False)  # never overwrite an earlier run
    return out


def error_row(item: dict, error: str, elapsed_ms: float) -> dict:
    """A question the system could not answer because a provider or the database failed."""
    return {
        "pmid": item["pmid"], "question": item["question"], "gold_label": item["final_decision"],
        "verdict": "insufficient_evidence", "correct": False, "abstained": True,
        "abstain_reason": "error", "answer": "", "claims": [], "limitations": [],
        "validation": {"removed_uncited": [], "removed_invalid_citation": [],
                       "removed_limitations": [], "downgraded_to_insufficient": False},
        "evidence_pmids": [], "rerank_scores": [], "top_rerank_score": None,
        "gate_passed": False, "gold_final_rank": None, "rerank_skipped": False,
        "rerank_error": None, "error": error, "timings_ms": {"total": round(elapsed_ms, 1)},
        "usage": [], "cost_usd": 0.0, "prompt_versions": {},
    }  # fmt: skip


def run_row(item: dict, answer) -> dict:
    final_pmids = [c.pmid for c in answer.evidence]
    return {
        "pmid": item["pmid"],
        "question": item["question"],
        "gold_label": item["final_decision"],
        "verdict": answer.verdict,
        "correct": answer.verdict == item["final_decision"],
        "abstained": answer.abstained,
        "abstain_reason": answer.abstain_reason,
        "answer": answer.answer,
        "claims": [c.model_dump() for c in answer.claims],
        "limitations": [c.model_dump() for c in answer.limitations],
        "validation": answer.validation.model_dump(),
        "evidence_pmids": final_pmids,
        "rerank_scores": [c.rerank_score for c in answer.evidence],
        "top_rerank_score": answer.top_rerank_score,
        "gate_passed": answer.gate_passed,
        "gold_final_rank": gold_rank(final_pmids, item["pmid"]),
        "rerank_skipped": answer.rerank_skipped,
        "rerank_error": answer.rerank_error,
        "error": answer.error,
        "timings_ms": {k: round(v, 1) for k, v in answer.timing_ms.items()},
        "usage": [u.model_dump() for u in answer.usage],
        "cost_usd": answer.estimated_cost_usd,
        "prompt_versions": answer.prompt_versions,
    }


def agent_fields(state) -> dict:
    """Agent-only diagnostics stored with each row (ADR-023 planner effect, critic decisions)."""
    question_top20 = {c.pmid for c in state["retrieved_candidates"][:20]}
    return {
        "subqueries": [s.query for s in state["subquestions"]],
        "n_candidates": len(state["retrieved_candidates"]),
        "planner_added_evidence": state["planner_added_evidence"],
        "gold_in_question_top20": None,  # filled by the caller (needs the gold PMID)
        "critic_results": [r.model_dump() for r in state["critic_results"]],
        "critic_removed": state["critic_removed"],
        "answer_supported": state["answer_supported"],
        "_question_top20": sorted(question_top20),
    }


def agent_row(item: dict, answer, state) -> dict:
    row = run_row(item, answer) | agent_fields(state)
    top20 = row.pop("_question_top20")
    pool = {c.pmid for c in state["retrieved_candidates"]}
    row["gold_in_question_top20"] = item["pmid"] in top20
    row["gold_in_pool"] = item["pmid"] in pool
    return row


def summarize_agent(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["error"] is None and "planner_added_evidence" in r]
    added = [r["planner_added_evidence"] for r in ok]
    critic_checked = [r for r in ok if r["critic_results"]]
    removed = [len(r["critic_removed"]) for r in ok]
    critic_abstained = [
        r for r in ok
        if r["abstain_reason"] == "no_supported_claims"
        and not r["validation"]["downgraded_to_insufficient"]
    ]  # fmt: skip

    def stage_p50(name):
        values = [r["timings_ms"][name] for r in ok if r["timings_ms"].get(name)]
        return statistics.median(values) if values else None

    return {
        "planner_added_evidence": {
            "mean": sum(added) / len(added) if added else None,
            "share_questions_gt0": sum(a > 0 for a in added) / len(added) if added else None,
            "questions_gt0": sum(a > 0 for a in added),
            "gold_found_only_via_subqueries": sum(
                (not r["gold_in_question_top20"]) and r["gold_in_pool"] for r in ok
            ),
        },
        "mean_candidates": sum(r["n_candidates"] for r in ok) / len(ok) if ok else None,
        "critic": {
            "answers_checked": len(critic_checked),
            "items_checked": sum(len(r["critic_results"]) for r in ok),
            "items_removed": sum(removed),
            "answers_with_removals": sum(n > 0 for n in removed),
            "answer_supported_false": sum(r["answer_supported"] is False for r in ok),
            "answers_abstained_by_critic": len(critic_abstained),
        },
        "stage_latency_p50_ms": {
            n: stage_p50(n) for n in ("planner", "retriever", "synthesizer", "critic")
        },
    }


def summarize(rows: list[dict]) -> dict:
    verdicts = [r["verdict"] for r in rows]
    gold = [r["gold_label"] for r in rows]
    totals = [r["timings_ms"]["total"] for r in rows]
    answered_totals = [r["timings_ms"]["total"] for r in rows if r["gate_passed"]]
    llm_in = sum(u.get("input_tokens", 0) for r in rows for u in r["usage"])
    llm_out = sum(u.get("output_tokens", 0) for r in rows for u in r["usage"])
    cost = sum(r["cost_usd"] for r in rows)
    reasons: dict[str, int] = {}
    for r in rows:
        if r["abstain_reason"]:
            reasons[r["abstain_reason"]] = reasons.get(r["abstain_reason"], 0) + 1

    def pct(values, q):
        if len(values) < 2:
            return values[0] if values else None
        return statistics.quantiles(values, n=100, method="inclusive")[q - 1]

    return {
        "n_questions": len(rows),
        "accuracy": accuracy(verdicts, gold),
        "coverage": coverage(verdicts),
        "answered_only_accuracy": answered_only_accuracy(verdicts, gold),
        "majority_class_reference": majority_class_accuracy(gold),
        "verdict_counts": {v: verdicts.count(v) for v in sorted(set(verdicts))},
        "gold_counts": {g: gold.count(g) for g in sorted(set(gold))},
        "abstain_reasons": reasons,
        "hit_at_1": mean_hit_at_k(
            [r["evidence_pmids"] for r in rows], [r["pmid"] for r in rows], 1
        ),
        "hit_at_5": mean_hit_at_k(
            [r["evidence_pmids"] for r in rows], [r["pmid"] for r in rows], 5
        ),
        "citations_removed": {
            "uncited": sum(len(r["validation"]["removed_uncited"]) for r in rows),
            "invalid": sum(len(r["validation"]["removed_invalid_citation"]) for r in rows),
            "limitations": sum(len(r["validation"].get("removed_limitations", [])) for r in rows),
            "answers_downgraded": sum(r["validation"]["downgraded_to_insufficient"] for r in rows),
        },
        "errors": sum(r["error"] is not None for r in rows),
        "rerank_skipped": sum(r["rerank_skipped"] for r in rows),
        "latency_ms": {
            "p50": pct(totals, 50),
            "p95": pct(totals, 95),
            "p50_when_llm_called": pct(answered_totals, 50),
            "measured_from": "developer machine, sequential, one reused DB connection",
        },
        "llm_tokens": {"input": llm_in, "output_incl_thinking": llm_out},
        "cost_usd": {"total": round(cost, 4), "per_query": round(cost / len(rows), 5)},
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=["baseline", "agent"], required=True)
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--limit", type=int, help="first N dev questions only (smoke runs)")
    parser.add_argument(FINAL_TEST_FLAG, dest="final_test_ok", action="store_true")
    parser.add_argument(
        RERUN_FLAG,
        dest="rerun",
        action="store_true",
        help="override the once-only rule for the test split (needs --reason)",
    )
    parser.add_argument(
        "--reason", help="why the test split is being run again (saved with the run)"
    )
    parser.add_argument("--max-cost", type=float, help="hard spending stop for this run (USD)")
    args = parser.parse_args(argv)
    if args.max_cost is not None and args.max_cost <= 0:
        raise SystemExit("--max-cost must be positive.")

    git = git_state()
    check_allowed(args.split, args.final_test_ok, git, args.limit)
    override = check_once_only(args.split, args.system, args.rerun, args.reason)
    manifest = load_manifest(args.split)
    items = manifest["items"][: args.limit]
    run_id = f"{args.split}-{args.system}-{datetime.now(UTC):%Y%m%d-%H%M%S}-{git['git_commit'][:7]}"
    out = create_run_dir(run_id)
    if override:
        (out / "override.json").write_text(json.dumps(override, indent=2) + "\n", encoding="utf-8")
    config = {
        "run_id": run_id,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        **git,
        "system": args.system,
        "split": args.split,
        "limit": args.limit,
        "question_ids": [i["pmid"] for i in items],
        "manifest_sha256": hashlib.sha256(
            (MANIFESTS / f"{args.split}.json").read_bytes()
        ).hexdigest(),
        "corpus_manifest_sha256": hashlib.sha256(CORPUS_MANIFEST.read_bytes()).hexdigest(),
        "dataset": manifest["dataset"],
        "dataset_revision": manifest["revision"],
        "llm_model": LLM_MODEL,
        "synthesizer_prompt": SYNTHESIZER_VERSION,
        "synthesizer_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "synthesizer_user_template_sha256": hashlib.sha256(USER_TEMPLATE.encode()).hexdigest(),
        "synthesizer_effort": SYNTHESIZER_EFFORT,
        "evidence_gate_threshold": EVIDENCE_GATE_THRESHOLD,
        "evidence_gate_dev_run": EVIDENCE_GATE_DEV_RUN,
        "embedding_model": EMBEDDING_MODEL,
        "rerank_model": RERANK_MODEL,
        "max_cost_usd": args.max_cost,
        "test_rerun_override": override,
    }
    if args.system == "agent":
        config |= {
            "planner_prompt": agent_planner.PLANNER_VERSION,
            "planner_prompt_sha256": hashlib.sha256(
                agent_planner.SYSTEM_PROMPT.encode()
            ).hexdigest(),
            "planner_user_template_sha256": hashlib.sha256(
                agent_planner.USER_TEMPLATE.encode()
            ).hexdigest(),
            "planner_effort": agent_planner.PLANNER_EFFORT,
            "critic_prompt": agent_critic.CRITIC_VERSION,
            "critic_prompt_sha256": hashlib.sha256(agent_critic.SYSTEM_PROMPT.encode()).hexdigest(),
            "critic_user_template_sha256": hashlib.sha256(
                agent_critic.USER_TEMPLATE.encode()
            ).hexdigest(),
            "critic_effort": agent_critic.CRITIC_EFFORT,
        }
    (out / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    rows = []
    stopped_at_cap = False
    voyage, llm = voyage_client(max_retries=3, timeout=60), anthropic_client()
    conn = None
    with (out / "raw.jsonl").open("w", encoding="utf-8") as raw:
        for i, item in enumerate(items, start=1):
            if args.max_cost is not None:
                spent = sum(r["cost_usd"] for r in rows)
                worst = max([PER_QUESTION_ALLOWANCE] + [r["cost_usd"] for r in rows])
                if spent + worst > args.max_cost:
                    stopped_at_cap = True
                    print(f"  COST CAP: stopping before question {i} (spent ${spent:.3f})")
                    break
            start = time.perf_counter()
            try:
                conn = conn or connect()
                if args.system == "agent":
                    answer, state = answer_agent(item["question"], conn=conn, voyage=voyage,
                                                 llm=llm)  # fmt: skip
                    row = agent_row(item, answer, state)
                else:
                    answer = answer_baseline(item["question"], conn=conn, voyage=voyage, llm=llm)
                    row = run_row(item, answer)
            except Exception as exc:  # outage: record it, reconnect, keep going
                error = redact(f"{type(exc).__name__}: {exc}")[:300]
                row = error_row(item, error, (time.perf_counter() - start) * 1000)
                print(f"  ERROR on {item['pmid']}: {error}", flush=True)
                if conn is not None:
                    conn.close()
                conn = None
            rows.append(row)
            raw.write(json.dumps(row) + "\n")
            raw.flush()
            if i % 10 == 0:
                spent = sum(r["cost_usd"] for r in rows)
                print(f"  {i}/{len(items)}  spent ${spent:.3f}", flush=True)

    if conn is not None:
        conn.close()
    summary = {"run_id": run_id, "system": args.system, "split": args.split, **summarize(rows)}
    if args.system == "agent":
        summary["agent"] = summarize_agent(rows)
    if args.max_cost is not None:
        summary["stopped_at_cost_cap"] = stopped_at_cap
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved to {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main(sys.argv[1:])
