"""Print an agent retrieval trace: planner sub-queries -> per-query hits -> pooled, reranked top 5
(smoke test).

Run: uv run python -m scripts.agent_retrieve_debug --dev 3      # first N dev questions
     uv run python -m scripts.agent_retrieve_debug "question"   # any question
Each question costs one planner call (well under $0.01); Voyage use is inside the free allocation.
Dev questions only: reads `question` and the source `pmid`, never the yes/no/maybe label.
"""

import argparse
import textwrap

from backend.app.agents.planner import plan
from backend.app.agents.retriever import retrieve_pooled
from backend.app.llm.client import anthropic_client
from backend.app.retrieval.embed import voyage_client
from backend.app.retrieval.vector_search import connect
from eval.runners.retrieval import load_dev_questions


def show(question: str, gold: str | None, llm, voyage, conn) -> float:
    subquestions, _, planner_usage, planner_ms = plan(llm, question)
    result = retrieve_pooled(question, subquestions, conn=conn, client=voyage)
    question_top20 = result.query_hits["q0"]
    found_by = {}
    for qid, pmids in result.query_hits.items():
        for p in pmids:
            found_by.setdefault(p, []).append(qid)
    # Rerank scores are per paper (query = the question), so the baseline's top 5 is the best 5
    # of the question's own top 20 under the same scores.
    scored = {c.pmid: c.rerank_score for c in result.candidates}
    baseline_top5 = sorted(question_top20, key=lambda p: -(scored[p] or 0))[:5]

    print("=" * 100)
    print(f"QUESTION: {question}" + (f"   (source paper: PMID {gold})" if gold else ""))
    print(f"PLANNER ({planner_ms:.0f} ms, ${planner_usage.cost_usd:.4f}):")
    for q in result.queries:
        hits = result.query_hits[q.id]
        new = sum(p not in question_top20 for p in hits)
        label = "the question itself" if q.id == "q0" else q.query
        print(f"  {q.id}: {textwrap.shorten(label, 85)}")
        print(f"      dense top {len(hits)}: {new} papers not in q0's top 20")
    skipped = f"  ** RERANK SKIPPED: {result.rerank_error}" if result.rerank_skipped else ""
    print(
        f"POOL: {len(result.candidates)} candidate passages (deduped by PMID), "
        f"reranked against the question{skipped}"
    )
    print("SELECTED EVIDENCE (top 5):")
    for c in result.evidence:
        origin = (
            "q0 top 20" if c.pmid in question_top20 else "NEW via " + "+".join(found_by[c.pmid])
        )
        flag = "  <- source paper" if c.pmid == gold else ""
        score = f"{c.rerank_score:.3f}" if c.rerank_score is not None else "n/a"
        print(f"  #{c.rerank_rank or '-'} PMID {c.pmid}  rerank {score}  [{origin}]{flag}")
    print(f"planner_added_evidence: {result.planner_added_evidence}")
    same = baseline_top5 == [c.pmid for c in result.evidence]
    print(f"baseline top 5 under the same scores: {baseline_top5}  (same as agent: {same})")
    print(f"timings_ms: { {k: round(v) for k, v in result.timings_ms.items()} }")
    print(f"voyage usage: {[(u.model, u.tokens) for u in result.usage]}")
    return planner_usage.cost_usd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?")
    parser.add_argument("--dev", type=int, default=0, help="first N dev questions")
    args = parser.parse_args()
    items = [(args.question, None)] if args.question else []
    items += [(q["question"], q["pmid"]) for q in load_dev_questions()[: args.dev]]

    llm, voyage, total = anthropic_client(), voyage_client(), 0.0
    with connect() as conn:
        for question, gold in items:
            total += show(question, gold, llm, voyage, conn)
    print("=" * 100)
    print(f"total Anthropic cost: ${total:.4f}")


if __name__ == "__main__":
    main()
