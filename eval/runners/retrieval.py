"""Dev retrieval evaluation: Hit@1 / Hit@5, dense and reranked (PLAN.md M1-T5).

Run: uv run python -m eval.runners.retrieval

Dev split only. Each question's gold paper is the paper it was written from (its PMID); only
`question` and `pmid` are read from the manifest, never the yes/no/maybe label. Results go to
eval/results/<run-id>/ (raw.jsonl, summary.json, run_config.json); an existing run is never
overwritten.
"""

import hashlib
import json
import statistics
import subprocess
from datetime import UTC, datetime

from backend.app.config import EMBEDDING_MODEL, REPO_ROOT, RERANK_MODEL
from backend.app.retrieval.embed import voyage_client
from backend.app.retrieval.retrieve import K_CANDIDATES, K_FINAL, retrieve
from backend.app.retrieval.vector_search import connect
from eval.metrics.retrieval import gold_rank, mean_hit_at_k

DEV_MANIFEST = REPO_ROOT / "data" / "manifests" / "dev.json"
CORPUS_MANIFEST = REPO_ROOT / "data" / "manifests" / "corpus.json"
RESULTS_DIR = REPO_ROOT / "eval" / "results"


def load_dev_questions() -> list[dict]:
    manifest = json.loads(DEV_MANIFEST.read_text(encoding="utf-8"))
    if manifest["split"] != "dev":
        raise RuntimeError("retrieval runner only evaluates the dev split")
    return [{"pmid": item["pmid"], "question": item["question"]} for item in manifest["items"]]


def git_state() -> dict:
    """Commit, dirty flag, and a hash of uncommitted changes so a dirty run can still be matched."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    status = git("status", "--porcelain")
    diff = git("diff", "HEAD") + "\n" + git("ls-files", "--others", "--exclude-standard")
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(status),
        "git_diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if status else None,
    }


def summarize(rows: list[dict]) -> dict:
    gold = [r["gold_pmid"] for r in rows]
    dense = [r["dense_pmids"] for r in rows]
    final = [r["final_pmids"] for r in rows]
    totals = [r["timings_ms"]["total"] for r in rows]
    tokens: dict[str, int] = {}
    cost = 0.0
    for r in rows:
        for u in r["usage"]:
            tokens[u["model"]] = tokens.get(u["model"], 0) + u["tokens"]
            cost += u["cost_usd"]
    return {
        "n_questions": len(rows),
        "dense": {
            "hit_at_1": mean_hit_at_k(dense, gold, 1),
            "hit_at_5": mean_hit_at_k(dense, gold, 5),
            "hit_at_20": mean_hit_at_k(dense, gold, K_CANDIDATES),
        },
        "reranked": {
            "hit_at_1": mean_hit_at_k(final, gold, 1),
            "hit_at_5": mean_hit_at_k(final, gold, K_FINAL),
        },
        "rerank_skipped": sum(r["rerank_skipped"] for r in rows),
        "latency_ms": {
            "p50": statistics.median(totals),
            "p95": statistics.quantiles(totals, n=20, method="inclusive")[18],
            "measured_from": "developer machine, one reused DB connection",
        },
        "tokens": tokens,
        "list_price_usd": round(cost, 6),
    }


def main() -> None:
    questions = load_dev_questions()
    git = git_state()
    run_id = f"dev-retrieval-{datetime.now(UTC):%Y%m%d-%H%M%S}-{git['git_commit'][:7]}"
    out = RESULTS_DIR / run_id
    out.mkdir(parents=True, exist_ok=False)  # never overwrite an earlier run

    rows = []
    client = voyage_client()
    with connect() as conn:
        for i, q in enumerate(questions, start=1):
            result = retrieve(q["question"], conn=conn, client=client)
            dense_pmids = [c.pmid for c in result.candidates]
            final_pmids = [c.pmid for c in sorted(result.candidates, key=_final_order)]
            if final_pmids[:K_FINAL] != [c.pmid for c in result.evidence]:
                raise RuntimeError(f"final ranking disagrees with retrieve() evidence: {q['pmid']}")
            rows.append(
                {
                    "gold_pmid": q["pmid"],
                    "dense_pmids": dense_pmids,
                    "dense_scores": [round(c.dense_score, 6) for c in result.candidates],
                    "final_pmids": final_pmids,
                    "rerank_scores": [
                        None if c.rerank_score is None else round(c.rerank_score, 6)
                        for c in sorted(result.candidates, key=_final_order)
                    ],
                    "gold_dense_rank": gold_rank(dense_pmids, q["pmid"]),
                    "gold_final_rank": gold_rank(final_pmids, q["pmid"]),
                    "rerank_skipped": result.rerank_skipped,
                    "rerank_error": result.rerank_error,
                    "timings_ms": {k: round(v, 1) for k, v in result.timings_ms.items()},
                    "usage": [u.model_dump() for u in result.usage],
                }
            )
            if i % 20 == 0:
                print(f"  {i}/{len(questions)}", flush=True)

    summary = {"run_id": run_id, "split": "dev", **summarize(rows)}
    config = {
        "run_id": run_id,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        **git,
        "split": "dev",
        "dev_manifest_sha256": hashlib.sha256(DEV_MANIFEST.read_bytes()).hexdigest(),
        "corpus_manifest_sha256": hashlib.sha256(CORPUS_MANIFEST.read_bytes()).hexdigest(),
        "embedding_model": EMBEDDING_MODEL,
        "rerank_model": RERANK_MODEL,
        "k_candidates": K_CANDIDATES,
        "k_final": K_FINAL,
        "gold_definition": "the paper the dev question was written from (its PMID)",
    }
    with (out / "raw.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved to {out.relative_to(REPO_ROOT)}")


def _final_order(c) -> tuple[int, int]:
    """Final ranking: rerank order, or dense order when reranking was skipped."""
    return (c.rerank_rank if c.rerank_rank is not None else c.dense_rank, c.dense_rank)


if __name__ == "__main__":
    main()
