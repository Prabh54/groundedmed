"""Run a system over a frozen split; store raw + summary results (ARCHITECTURE §9, PLAN M2-T5).

Run: uv run python -m eval.runners.run_system --system baseline|agent --split dev \
         --max-cost USD --approval "<the owner's approval, word for word>"
     (small smoke runs, --limit N, estimated at or below APPROVAL_THRESHOLD_USD need neither)

Approval guard (owner instructions 2026-10-24): the run's cost is estimated before anything starts
(questions x EST_COST_PER_QUESTION for the system). Any run estimated above APPROVAL_THRESHOLD_USD
is refused unless it has both --max-cost and --approval with the owner's own approval text (at
least MIN_APPROVAL_CHARS characters); `--limit 100` is treated like a full run. The estimate and the
approval text are saved in run_config.json and summary.json.

--max-cost is a soft spending stop: before each question the runner checks that the money spent so
far plus a margin (twice the most expensive question so far, at least PER_QUESTION_ALLOWANCE) stays
within the cap; otherwise it stops. It can only be exceeded if one question costs more than that
margin. Failed questions count the calls they made.

Void runs (owner instruction 2026-10-24, ADR-034): a run that stops before all of its planned
questions (cost cap, MAX_CONSECUTIVE_ERRORS failures in a row, crash, Ctrl-C) is void. It writes
VOID.json, renames its rows to raw.VOID-do-not-score.jsonl, computes and prints no results, and
exits with an error; a new run (new run ID) is needed. A void test run does not count as the
system's one test run. The next test run of that system refuses unless its frozen system identity
(model, prompts, efforts, gate threshold) equals the void run's, so nothing about the system can
change between the two. `require_complete_run` and `require_same_questions` are the checks the
scorer runs before computing anything.

Calls the same Python function the API uses (not HTTP), sequentially. The gold label is read here
only, after the answer is produced; the pipeline never sees it. The frozen test splits
(FINAL_SPLITS: `test`, and `test2` from M15-T9, ADR-043) additionally need
--i-understand-this-is-the-final-test, a clean Git tree, the manifest matching its frozen
<split>.sha256, and no --limit. Each system runs on each frozen test split once (ADR-031): if a
completed run of that system on that split exists, the script refuses unless given both
--rerun-final-test and --reason "<why>"; the reason and the earlier run IDs are written to
override.json in the new run directory. Runs of one split never count for the other. A question
that fails (provider or database outage) is stored as an error row, counted as incorrect; the run
continues unless MAX_CONSECUTIVE_ERRORS questions fail in a row (an outage), which makes the run
void.
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
from backend.app.agents import reviser as agent_reviser
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
from eval import closed_book
from eval.metrics.answers import (
    accuracy,
    answered_only_accuracy,
    coverage,
    majority_class_accuracy,
)
from eval.metrics.retrieval import gold_rank, mean_hit_at_k
from eval.runners.retrieval import CORPUS_MANIFEST, RESULTS_DIR, git_state

MANIFESTS = REPO_ROOT / "data" / "manifests"
AGENT_RULES_FILE = REPO_ROOT / "backend" / "app" / "agents" / "rules.py"
# Frozen test splits: guarded, once per system each. test2 = ADR-043 (M15-T9).
FINAL_SPLITS = ("test", "test2")
FINAL_TEST_FLAG = "--i-understand-this-is-the-final-test"
RERUN_FLAG = "--rerun-final-test"
MIN_REASON_CHARS = 20
MIN_APPROVAL_CHARS = 10
APPROVAL_THRESHOLD_USD = 0.50
# Conservative per-question estimates from measured dev runs (baseline v3 $0.013, agent $0.028;
# agent questions up to ~$0.035 in M3-T7; closed-book $0.00105 on 5 dev questions, 2026-10-25).
# Used only to decide whether approval is needed.
EST_COST_PER_QUESTION = {"baseline": 0.015, "agent": 0.035, "closed_book": 0.005}
PER_QUESTION_ALLOWANCE = 0.05  # USD; agent questions measured $0.024-0.034 (M3-T7)
MAX_CONSECUTIVE_ERRORS = 5  # this many failed questions in a row = an outage -> void run (ADR-034)
VOID_FILE = "VOID.json"
VOID_RAW = "raw.VOID-do-not-score.jsonl"
# run_config keys that define the system being tested; must match across a void run and its rerun
IDENTITY_KEYS = (
    "system", "split", "question_ids", "manifest_sha256", "corpus_manifest_sha256", "llm_model",
    "synthesizer_prompt_sha256", "synthesizer_user_template_sha256", "synthesizer_effort",
    "evidence_gate_threshold", "embedding_model", "rerank_model", "planner_prompt_sha256",
    "planner_user_template_sha256", "planner_effort", "critic_prompt_sha256",
    "critic_user_template_sha256", "critic_effort", "closed_book_prompt_sha256",
    "closed_book_user_template_sha256", "closed_book_effort", "agent_synthesizer_effort",
    "retry_prompt", "retry_user_template_sha256", "veto_rule", "agent_rules_sha256",
)  # fmt: skip


def load_manifest(split: str) -> dict:
    data = (MANIFESTS / f"{split}.json").read_bytes()
    if split in FINAL_SPLITS:
        frozen = (MANIFESTS / f"{split}.sha256").read_text(encoding="utf-8").split()[0]
        if hashlib.sha256(data).hexdigest() != frozen:
            raise SystemExit(
                f"{split}.json does not match the frozen {split}.sha256; refusing to run."
            )
    manifest = json.loads(data)
    if manifest["split"] != split:
        raise RuntimeError(f"{split}.json does not declare split {split!r}")
    return manifest


def check_allowed(split: str, final_test_ok: bool, git: dict, limit: int | None = None) -> None:
    final = split in FINAL_SPLITS
    if final and not final_test_ok:
        raise SystemExit(f"Refusing the frozen {split} split without {FINAL_TEST_FLAG}.")
    if final and git["git_dirty"]:
        raise SystemExit(f"Refusing the frozen {split} split from a dirty Git tree; commit first.")
    if final and limit is not None:
        raise SystemExit(f"Refusing a partial run of the frozen {split} split (--limit).")
    if limit is not None and limit < 1:
        raise SystemExit("--limit must be at least 1.")


def estimated_cost(system: str, n_questions: int) -> float:
    return round(n_questions * EST_COST_PER_QUESTION[system], 4)


def check_approval(estimate: float, max_cost: float | None, approval: str | None) -> None:
    """A run estimated above APPROVAL_THRESHOLD_USD needs a cost cap and the owner's approval text,
    whatever --limit says (owner instruction 2026-10-24)."""
    if max_cost is not None and max_cost <= 0:
        raise SystemExit("--max-cost must be positive.")
    if approval is not None and len(approval.strip()) < MIN_APPROVAL_CHARS:
        raise SystemExit(
            f"--approval must quote the owner's approval (>= {MIN_APPROVAL_CHARS} chars)."
        )
    if estimate > APPROVAL_THRESHOLD_USD and (max_cost is None or approval is None):
        raise SystemExit(
            f"Refusing: this run is estimated at ${estimate:.2f} "
            f"(> ${APPROVAL_THRESHOLD_USD:.2f}). It needs both --max-cost USD and "
            "--approval \"<the owner's approval, word for word>\". Get the owner's typed OK first."
        )


def test_run_configs(system: str, split: str = "test") -> list[tuple[dict, bool]]:
    """(run_config, completed) for every run directory of `system` on the frozen `split`. Completed
    = it has summary.json and no VOID.json; anything else (void, crashed) is not a completed run."""
    found = []
    for config_path in sorted(RESULTS_DIR.glob("*/run_config.json")):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("split") == split and config.get("system") == system:
            run_dir = config_path.parent
            done = (run_dir / "summary.json").exists() and not (run_dir / VOID_FILE).exists()
            found.append((config, done))
    return found


def completed_test_runs(system: str, split: str = "test") -> list[str]:
    """Run IDs of finished runs of `system` on the frozen `split` (void and crashed excluded)."""
    return [config["run_id"] for config, done in test_run_configs(system, split) if done]


def check_same_identity_as_void_runs(config: dict) -> list[str]:
    """A rerun after a void test run must test the identical system (ADR-034) and carry a fresh
    owner approval, not the one the void run spent. Returns the void run IDs (recorded in the new
    run's config)."""
    void = [c for c, done in test_run_configs(config["system"], config["split"]) if not done]
    for earlier in void:
        if earlier.get("owner_approval") == config.get("owner_approval"):
            raise SystemExit(
                f"Void {config['split']} run {earlier['run_id']} used this same approval. A rerun "
                "after a void run needs the owner's fresh typed approval (ADR-034); refusing."
            )
        changed = [k for k in IDENTITY_KEYS if earlier.get(k) != config.get(k)]
        if changed:
            raise SystemExit(
                f"Void {config['split']} run {earlier['run_id']} tested a different system "
                f"({', '.join(changed)}). A rerun after a void run must not change the system; "
                "refusing."
            )
    return [c["run_id"] for c in void]


def write_void(out, run_id: str, reason: str, done: int, planned: int) -> None:
    """Mark a run void (ADR-034): no summary, rows renamed so nothing scores them by accident."""
    record = {
        "run_id": run_id,
        "void": True,
        "reason": reason,
        "questions_done": done,
        "questions_planned": planned,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "rule": "Stopped before all planned questions: void. Not scored, no results computed or "
        "shown. Does not count as the system's test run (ADR-034). Rerun under a new run ID with "
        "the identical system.",
    }
    (out / VOID_FILE).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if (out / "raw.jsonl").exists():
        (out / "raw.jsonl").rename(out / VOID_RAW)


def require_complete_run(run_dir) -> list[dict]:
    """The scorer's gate: refuse a void, crashed or partial run. Returns its rows."""
    if (run_dir / VOID_FILE).exists() or not (run_dir / "summary.json").exists():
        raise SystemExit(f"{run_dir.name} is void or unfinished; it must not be scored.")
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (run_dir / "raw.jsonl").open(encoding="utf-8")]
    if [r["pmid"] for r in rows] != config["question_ids"]:
        raise SystemExit(f"{run_dir.name}: rows do not match its planned question IDs; refusing.")
    return rows


def require_same_questions(*run_dirs) -> list[str]:
    """Every compared run must be complete and cover the identical question IDs, in order."""
    ids = [[r["pmid"] for r in require_complete_run(d)] for d in run_dirs]
    if any(i != ids[0] for i in ids[1:]):
        raise SystemExit("Runs cover different question IDs; refusing to compare them.")
    return ids[0]


def check_once_only(split: str, system: str, rerun: bool, reason: str | None) -> dict | None:
    """Each frozen test split runs once per system (ADR-031). Returns the override record, if any.
    Completed runs of one frozen split never count for another."""
    if split not in FINAL_SPLITS:
        if rerun or reason is not None:
            raise SystemExit(f"{RERUN_FLAG} and --reason apply only to the frozen test splits.")
        return None
    previous = completed_test_runs(system, split)
    if not previous:
        if rerun or reason is not None:
            raise SystemExit(
                f"No completed {system} {split} run exists; drop {RERUN_FLAG}/--reason."
            )
        return None
    if not rerun or reason is None:
        raise SystemExit(
            f"A completed {system} {split} run already exists ({', '.join(previous)}). A frozen "
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


def error_row(item: dict, error: str, elapsed_ms: float, usage: list | None = None) -> dict:
    """A question the system could not answer because a provider or the database failed. `usage`
    holds the calls made before the failure (e.g. the agent's planner), so their cost counts."""
    usage = usage or []
    return {
        "pmid": item["pmid"], "question": item["question"], "gold_label": item["final_decision"],
        "verdict": "insufficient_evidence", "correct": False, "abstained": True,
        "abstain_reason": "error", "answer": "", "claims": [], "limitations": [],
        "validation": {"removed_uncited": [], "removed_invalid_citation": [],
                       "removed_limitations": [], "downgraded_to_insufficient": False},
        "evidence_pmids": [], "rerank_scores": [], "top_rerank_score": None,
        "gate_passed": False, "gold_final_rank": None, "rerank_skipped": False,
        "rerank_error": None, "error": error, "timings_ms": {"total": round(elapsed_ms, 1)},
        "usage": [u.model_dump() for u in usage], "cost_usd": sum(u.cost_usd for u in usage),
        "prompt_versions": {},
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


def closed_book_row(item: dict, verdict: str, usage, elapsed_ms: float) -> dict:
    """A closed-book answer (ADR-010): verdict only; retrieval and claim fields do not apply."""
    row = error_row(item, "", elapsed_ms, [usage])
    return row | {
        "verdict": verdict, "correct": verdict == item["final_decision"], "abstained": False,
        "abstain_reason": None, "error": None, "gate_passed": None,
        "prompt_versions": {"closed_book": closed_book.CLOSED_BOOK_VERSION},
    }  # fmt: skip


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
        # M15-T4: the synthesizer's draft (verdict, claims, limitations) before the critic and
        # validator, so rule-only changes can be replayed on stored runs. Eval records only.
        "draft_answer": state["draft_answer"].model_dump() if state["draft_answer"] else None,
        # ADR-044 retry diagnostics (first pass kept only when a retry ran)
        "retry_used": state["retry_used"],
        "retry_error": state["retry_error"],
        "first_draft_answer": (
            state["first_draft_answer"].model_dump() if state["first_draft_answer"] else None
        ),
        "first_critic_results": [r.model_dump() for r in state["first_critic_results"]],
        "first_answer_supported": state["first_answer_supported"],
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
    parser.add_argument("--system", choices=["baseline", "agent", "closed_book"], required=True)
    parser.add_argument("--split", choices=["dev", *FINAL_SPLITS], required=True)
    parser.add_argument("--limit", type=int, help="first N dev questions only (smoke runs)")
    parser.add_argument(FINAL_TEST_FLAG, dest="final_test_ok", action="store_true")
    parser.add_argument(
        RERUN_FLAG,
        dest="rerun",
        action="store_true",
        help="override the once-only rule for a frozen test split (needs --reason)",
    )
    parser.add_argument(
        "--reason", help="why a frozen test split is being run again (saved with the run)"
    )
    parser.add_argument("--max-cost", type=float, help="soft spending stop for this run (USD)")
    parser.add_argument("--approval", help="the owner's approval text, word for word (saved)")
    args = parser.parse_args(argv)

    git = git_state()
    check_allowed(args.split, args.final_test_ok, git, args.limit)
    override = check_once_only(args.split, args.system, args.rerun, args.reason)
    manifest = load_manifest(args.split)
    items = manifest["items"][: args.limit]
    estimate = estimated_cost(args.system, len(items))
    check_approval(estimate, args.max_cost, args.approval)  # before any folder or paid call
    run_id = f"{args.split}-{args.system}-{datetime.now(UTC):%Y%m%d-%H%M%S}-{git['git_commit'][:7]}"
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
        "owner_approval": args.approval.strip() if args.approval else None,
        "estimated_cost_usd": estimate,
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
            # ADR-044: the agent's synthesizer effort (first pass and retry) and the retry turn
            "agent_synthesizer_effort": agent_reviser.AGENT_SYNTHESIZER_EFFORT,
            "retry_prompt": agent_reviser.RETRY_VERSION,
            "retry_user_template_sha256": hashlib.sha256(
                agent_reviser.RETRY_TEMPLATE.encode()
            ).hexdigest(),
            "veto_rule": "fix B: no veto when every citation-valid claim was kept",
            # the deterministic keep/refuse rules themselves (ADR-045), not just their label
            "agent_rules_sha256": hashlib.sha256(AGENT_RULES_FILE.read_bytes()).hexdigest(),
        }
    if args.system == "closed_book":
        config |= {
            "closed_book_prompt": closed_book.CLOSED_BOOK_VERSION,
            "closed_book_prompt_sha256": hashlib.sha256(
                closed_book.SYSTEM_PROMPT.encode()
            ).hexdigest(),
            "closed_book_user_template_sha256": hashlib.sha256(
                closed_book.USER_TEMPLATE.encode()
            ).hexdigest(),
            "closed_book_effort": closed_book.CLOSED_BOOK_EFFORT,
        }
    if args.split in FINAL_SPLITS:  # before the new folder exists, so not compared with itself
        config["earlier_void_test_runs"] = check_same_identity_as_void_runs(config)
    out = create_run_dir(run_id)
    if override:
        (out / "override.json").write_text(json.dumps(override, indent=2) + "\n", encoding="utf-8")
    (out / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    rows = []
    stop_reason = None
    consecutive_errors = 0
    needs_retrieval = args.system != "closed_book"
    voyage = voyage_client(max_retries=3, timeout=60) if needs_retrieval else None
    llm = anthropic_client()
    conn = None
    try:
        with (out / "raw.jsonl").open("w", encoding="utf-8") as raw:
            for i, item in enumerate(items, start=1):
                if args.max_cost is not None:
                    spent = sum(r["cost_usd"] for r in rows)
                    margin = max([PER_QUESTION_ALLOWANCE] + [2 * r["cost_usd"] for r in rows])
                    if spent + margin > args.max_cost:
                        stop_reason = f"cost cap ${args.max_cost} reached before question {i} "
                        stop_reason += f"(spent ${spent:.3f})"
                        break
                start = time.perf_counter()
                try:
                    conn = conn or (connect() if needs_retrieval else None)
                    if args.system == "closed_book":
                        verdict, usage, ms = closed_book.answer_closed_book(llm, item["question"])
                        row = closed_book_row(item, verdict, usage, ms)
                    elif args.system == "agent":
                        answer, state = answer_agent(item["question"], conn=conn, voyage=voyage,
                                                     llm=llm)  # fmt: skip
                        row = agent_row(item, answer, state)
                    else:
                        answer = answer_baseline(item["question"], conn=conn, voyage=voyage,
                                                 llm=llm)  # fmt: skip
                        row = run_row(item, answer)
                    consecutive_errors = 0
                except Exception as exc:  # outage: record it, reconnect, keep going
                    error = redact(f"{type(exc).__name__}: {exc}")[:300]
                    row = error_row(item, error, (time.perf_counter() - start) * 1000,
                                    getattr(exc, "usage", None))  # fmt: skip
                    print(f"  ERROR on {item['pmid']}: {error}", flush=True)
                    if conn is not None:
                        conn.close()
                    conn = None
                    consecutive_errors += 1
                rows.append(row)
                raw.write(json.dumps(row) + "\n")
                raw.flush()
                if i % 10 == 0:
                    spent = sum(r["cost_usd"] for r in rows)
                    print(f"  {i}/{len(items)}  spent ${spent:.3f}", flush=True)
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS and i < len(items):
                    stop_reason = f"{consecutive_errors} questions in a row failed (outage?)"
                    break
    except BaseException as exc:  # crash or Ctrl-C mid-run: the run is void
        write_void(out, run_id, f"interrupted: {type(exc).__name__}", len(rows), len(items))
        raise
    finally:
        if conn is not None:
            conn.close()
    if stop_reason is None and len(rows) != len(items):
        stop_reason = "fewer rows than planned questions"
    if stop_reason is not None:
        write_void(out, run_id, stop_reason, len(rows), len(items))
        spent = sum(r["cost_usd"] for r in rows)
        raise SystemExit(
            f"VOID run {run_id}: {stop_reason}; {len(rows)}/{len(items)} questions, ${spent:.3f} "
            "spent. Not scored and no results computed (ADR-034). Start a new run."
        )
    summary = {"run_id": run_id, "system": args.system, "split": args.split, **summarize(rows)}
    if args.system == "agent":
        summary["agent"] = summarize_agent(rows)
    if args.system == "closed_book":  # no retrieval, no claims (ADR-010)
        summary["latency_ms"]["measured_from"] = "developer machine, sequential, no retrieval"
        summary |= {"hit_at_1": "not applicable", "hit_at_5": "not applicable",
                    "citations_removed": "not applicable",
                    "claim_support": "not applicable"}  # fmt: skip
    summary["complete"] = True
    summary["owner_approval"] = config["owner_approval"]
    summary["estimated_cost_usd"] = estimate
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved to {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main(sys.argv[1:])
