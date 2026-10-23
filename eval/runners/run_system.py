"""Run a system over a frozen split; store raw + summary results (ARCHITECTURE §9, PLAN M2-T5).

Run: uv run python -m eval.runners.run_system --system baseline --split dev [--limit N]

Calls the same Python function the API uses (not HTTP), sequentially. The gold label is read here
only, after the answer is produced; the pipeline never sees it. The test split additionally needs
--i-understand-this-is-the-final-test, a clean Git tree, the manifest matching test.sha256, and no
--limit. A question that fails (provider or database outage) is stored as an error row, counted as
incorrect, and the run continues. Results go to eval/results/<run-id>/ (raw.jsonl, summary.json,
run_config.json); a run is never overwritten.
"""

import argparse
import hashlib
import json
import statistics
import sys
import time
from datetime import UTC, datetime

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
    parser.add_argument("--system", choices=["baseline"], required=True)
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--limit", type=int, help="first N dev questions only (smoke runs)")
    parser.add_argument(FINAL_TEST_FLAG, dest="final_test_ok", action="store_true")
    args = parser.parse_args(argv)

    git = git_state()
    check_allowed(args.split, args.final_test_ok, git, args.limit)
    manifest = load_manifest(args.split)
    items = manifest["items"][: args.limit]
    run_id = f"{args.split}-{args.system}-{datetime.now(UTC):%Y%m%d-%H%M%S}-{git['git_commit'][:7]}"
    out = create_run_dir(run_id)
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
    }
    (out / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    rows = []
    voyage, llm = voyage_client(max_retries=3, timeout=60), anthropic_client()
    conn = None
    with (out / "raw.jsonl").open("w", encoding="utf-8") as raw:
        for i, item in enumerate(items, start=1):
            start = time.perf_counter()
            try:
                conn = conn or connect()
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
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved to {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main(sys.argv[1:])
