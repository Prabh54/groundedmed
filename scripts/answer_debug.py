"""Print baseline answers with their citations for a few questions (smoke test, PLAN.md M2-T1).

Run: uv run python -m scripts.answer_debug --dev 5      # first N dev questions
     uv run python -m scripts.answer_debug "question"   # any question
Each answer costs roughly $0.01-0.02. Dev questions only: the yes/no/maybe label is never read.
"""

import argparse
import json
import textwrap

from backend.app.baseline.pipeline import answer_baseline
from backend.app.llm.client import anthropic_client
from backend.app.retrieval.embed import voyage_client
from backend.app.retrieval.vector_search import connect
from eval.runners.retrieval import load_dev_questions


def show(question: str, answer) -> None:
    print("=" * 100)
    print(f"QUESTION: {question}")
    print(f"VERDICT:  {answer.verdict}   abstained={answer.abstained} ({answer.abstain_reason})")
    print(f"SUMMARY:  {textwrap.fill(answer.answer, 100, subsequent_indent='          ')}")
    for claim in answer.claims:
        print(f"  [{claim.id}] {claim.text}  <- {', '.join(claim.citation_ids)}")
    print(f"UNCERTAINTY: {answer.uncertainty}")
    print(f"validation: {answer.validation.model_dump()}")
    print(f"timing_ms: {json.dumps({k: round(v) for k, v in answer.timing_ms.items()})}")
    print(f"cost_usd: {answer.estimated_cost_usd:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?")
    parser.add_argument("--dev", type=int, default=0, help="first N dev questions")
    args = parser.parse_args()
    questions = [args.question] if args.question else []
    questions += [q["question"] for q in load_dev_questions()[: args.dev]]

    llm, voyage = anthropic_client(), voyage_client()
    with connect() as conn:
        for q in questions:
            show(q, answer_baseline(q, conn=conn, voyage=voyage, llm=llm))


if __name__ == "__main__":
    main()
