"""Print the planner's real output for a few questions (smoke test, PLAN.md M3-T2).

Run: uv run python -m scripts.plan_debug --dev 3      # first N dev questions
     uv run python -m scripts.plan_debug "question"   # any question
Each call costs well under $0.01. Dev questions only: the yes/no/maybe label is never read.
"""

import argparse

from backend.app.agents.planner import PLANNER_EFFORT, PLANNER_VERSION, plan
from backend.app.llm.client import anthropic_client
from eval.runners.retrieval import load_dev_questions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?")
    parser.add_argument("--dev", type=int, default=0, help="first N dev questions")
    args = parser.parse_args()
    questions = [args.question] if args.question else []
    questions += [q["question"] for q in load_dev_questions()[: args.dev]]

    client, total = anthropic_client(), 0.0
    print(f"planner {PLANNER_VERSION}, effort {PLANNER_EFFORT}")
    for q in questions:
        cleaned, raw, usage, ms = plan(client, q)
        total += usage.cost_usd
        print("=" * 100)
        print(f"QUESTION: {q}")
        print("MODEL OUTPUT (raw):")
        for s in raw:
            print(f"  {s.id}: {s.query}")
        print("AFTER CLEANUP (used for retrieval, plus the question itself):")
        for s in cleaned:
            print(f"  {s.id}: {s.query}")
        print(
            f"tokens in/out: {usage.input_tokens}/{usage.output_tokens}  "
            f"cost ${usage.cost_usd:.4f}  time {ms:.0f} ms"
        )
    print("=" * 100)
    print(f"total cost: ${total:.4f}")


if __name__ == "__main__":
    main()
