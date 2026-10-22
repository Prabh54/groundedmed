"""Print what retrieval returns for a question: PMIDs, scores, ranks, text (PLAN.md M1-T3).

Run:
  uv run python -m scripts.retrieve_debug "Does aspirin reduce ...?"
  uv run python -m scripts.retrieve_debug --dev 3     # first 3 dev questions, gold marked

Only the dev manifest is ever read here; test questions are never used for debugging.
"""

import argparse
import json

from backend.app.config import REPO_ROOT
from backend.app.retrieval.retrieve import retrieve

DEV_MANIFEST = REPO_ROOT / "data" / "manifests" / "dev.json"


def show(question: str, gold: str | None = None) -> None:
    result = retrieve(question)
    print(f"\nQ: {question}")
    if gold:
        print(f"   gold PMID: {gold}")
    status = f"SKIPPED ({result.rerank_error})" if result.rerank_skipped else "ok"
    timings = ", ".join(f"{k} {v:.0f} ms" for k, v in result.timings_ms.items())
    print(f"   rerank: {status}; {timings}")
    print("   final evidence (top 5):")
    for c in result.evidence:
        mark = "  <== gold" if c.pmid == gold else ""
        rerank = f"{c.rerank_score:.3f}" if c.rerank_score is not None else "-"
        print(
            f"   PMID:{c.pmid}  rerank {rerank} (#{c.rerank_rank})  "
            f"dense {c.dense_score:.3f} (#{c.dense_rank}){mark}"
        )
        print(f"      {c.passage[:160].replace(chr(10), ' ')}...")
    others = [c for c in result.candidates if c not in result.evidence]
    print("   other dense candidates: " + ", ".join(f"{c.pmid}(#{c.dense_rank})" for c in others))
    if gold and all(c.pmid != gold for c in result.candidates):
        print(f"   gold not in dense top {len(result.candidates)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?")
    parser.add_argument("--dev", type=int, metavar="N", help="use the first N dev questions")
    args = parser.parse_args()
    if args.dev:
        items = json.loads(DEV_MANIFEST.read_text(encoding="utf-8"))["items"][: args.dev]
        for item in items:
            show(item["question"], gold=item["pmid"])
    elif args.question:
        show(args.question)
    else:
        parser.error("give a question or --dev N")


if __name__ == "__main__":
    main()
