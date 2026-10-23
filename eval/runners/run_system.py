"""Run a system over a frozen split; store raw + summary results (ARCHITECTURE §9, PLAN M2-T5).

Run: uv run python -m eval.runners.run_system --system baseline --split dev [--limit N]

Calls the same Python function the API uses (not HTTP), sequentially. The gold label is read here
only, after the answer is produced; the pipeline never sees it. The test split additionally needs
--i-understand-this-is-the-final-test and a clean Git tree. Results go to
eval/results/<run-id>/ (raw.jsonl, summary.json, run_config.json); a run is never overwritten.
"""

import argparse
import hashlib
import json
import statistics
import sys
from datetime import UTC, datetime

from backend.app.baseline.pipeline import answer_baseline
from backend.app.baseline.synthesizer import SYNTHESIZER_EFFORT, SYNTHESIZER_VERSION, SYSTEM_PROMPT
from backend.app.config import (
    EMBEDDING_MODEL,
    EVIDENCE_GATE_DEV_RUN,
    EVIDENCE_GATE_THRESHOLD,
    LLM_MODEL,
    REPO_ROOT,
    RERANK_MODEL,
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


def load_split(split: str) -> list[dict]:
    manifest = json.loads((MANIFESTS / f"{split}.json").read_text(encoding="utf-8"))
    if manifest["split"] != split:
        raise RuntimeError(f"{split}.json does not declare split {split!r}")
    return manifest["items"]


def check_allowed(split: str, final_test_ok: bool, git: dict) -> None:
    if split == "test" and not final_test_ok:
        raise SystemExit(f"Refusing the frozen test split without {FINAL_TEST_FLAG}.")
    if split == "test" and git["git_dirty"]:
        raise SystemExit("Refusing the frozen test split from a dirty Git tree; commit first.")


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
        "uncertainty": answer.uncertainty,
        "validation": answer.validation.model_dump(),
        "evidence_pmids": final_pmids,
        "rerank_scores": [c.rerank_score for c in answer.evidence],
        "top_rerank_score": answer.top_rerank_score,
        "gate_passed": answer.gate_passed,
        "gold_final_rank": gold_rank(final_pmids, item["pmid"]),
        "rerank_skipped": answer.rerank_skipped,
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
        return statistics.quantiles(values, n=100, method="inclusive")[q - 1] if values else None

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
    parser.add_argument("--limit", type=int, help="first N questions only (smoke runs)")
    parser.add_argument(FINAL_TEST_FLAG, dest="final_test_ok", action="store_true")
    args = parser.parse_args(argv)

    git = git_state()
    check_allowed(args.split, args.final_test_ok, git)
    items = load_split(args.split)[: args.limit]
    run_id = f"{args.split}-{args.system}-{datetime.now(UTC):%Y%m%d-%H%M%S}-{git['git_commit'][:7]}"
    out = RESULTS_DIR / run_id
    out.mkdir(parents=True, exist_ok=False)  # never overwrite an earlier run
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
        "llm_model": LLM_MODEL,
        "synthesizer_prompt": SYNTHESIZER_VERSION,
        "synthesizer_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "synthesizer_effort": SYNTHESIZER_EFFORT,
        "evidence_gate_threshold": EVIDENCE_GATE_THRESHOLD,
        "evidence_gate_dev_run": EVIDENCE_GATE_DEV_RUN,
        "embedding_model": EMBEDDING_MODEL,
        "rerank_model": RERANK_MODEL,
    }
    (out / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    rows = []
    voyage, llm = voyage_client(), anthropic_client()
    with connect() as conn, (out / "raw.jsonl").open("w", encoding="utf-8") as raw:
        for i, item in enumerate(items, start=1):
            answer = answer_baseline(item["question"], conn=conn, voyage=voyage, llm=llm)
            row = run_row(item, answer)
            rows.append(row)
            raw.write(json.dumps(row) + "\n")
            raw.flush()
            if i % 10 == 0:
                spent = sum(r["cost_usd"] for r in rows)
                print(f"  {i}/{len(items)}  spent ${spent:.3f}", flush=True)

    summary = {"run_id": run_id, "system": args.system, "split": args.split, **summarize(rows)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved to {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main(sys.argv[1:])
