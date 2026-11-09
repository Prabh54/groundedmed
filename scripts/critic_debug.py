"""Print the agent chain up to the critic for a few questions: planner -> retriever -> gate ->
synthesizer -> critic, with each claim's keep/remove decision (smoke test).

Run: uv run python -m scripts.critic_debug --dev 3 [--budget 0.09]
Roughly $0.02-0.03 per question; stops before a question that could exceed --budget.
Dev questions only: reads `question`, never the yes/no/maybe label.
"""

import argparse
import textwrap

from backend.app.agents.critic import apply_critic, critique
from backend.app.agents.planner import plan
from backend.app.agents.retriever import retrieve_pooled
from backend.app.baseline.synthesizer import synthesize
from backend.app.config import EVIDENCE_GATE_DENSE_THRESHOLD, EVIDENCE_GATE_THRESHOLD
from backend.app.guardrails.evidence_gate import gate_passes
from backend.app.llm.client import anthropic_client
from backend.app.retrieval.embed import voyage_client
from backend.app.retrieval.vector_search import connect
from eval.runners.retrieval import load_dev_questions

PER_QUESTION_MAX = 0.04


def wrap(text: str) -> str:
    return textwrap.fill(text, 96, initial_indent="      ", subsequent_indent="      ")


def show(question: str, llm, voyage, conn) -> float:
    print("=" * 100)
    print(f"QUESTION: {question}")
    subs, _, plan_usage, _ = plan(llm, question)
    cost = plan_usage.cost_usd
    retrieval = retrieve_pooled(question, subs, conn=conn, client=voyage)
    pmids = [c.pmid for c in retrieval.evidence]
    print(f"planner: {len(subs)} sub-queries; retriever: {len(retrieval.candidates)} candidates "
          f"-> {pmids} (planner added {retrieval.planner_added_evidence})")  # fmt: skip
    if not gate_passes(retrieval.evidence, EVIDENCE_GATE_THRESHOLD, EVIDENCE_GATE_DENSE_THRESHOLD):
        print("gate: FAILED -> abstain (weak_evidence); no synthesis or critic call")
        return cost
    draft, synth_usage, _ = synthesize(llm, question, retrieval.evidence)
    cost += synth_usage.cost_usd
    print(f"synthesizer: verdict {draft.verdict}, {len(draft.claims)} claims, "
          f"{len(draft.limitations)} limitations (${synth_usage.cost_usd:.4f})")  # fmt: skip
    if not (draft.claims or draft.limitations):
        print("critic: nothing to check, no call")
        return cost
    output, critic_usage, critic_ms = critique(llm, question, draft, retrieval.evidence)
    cost += critic_usage.cost_usd
    cleaned, removed = apply_critic(draft, output.results)
    results = {r.claim_id: r for r in output.results}
    print(f"CRITIC ({critic_ms:.0f} ms, ${critic_usage.cost_usd:.4f}, "
          f"tokens {critic_usage.input_tokens}/{critic_usage.output_tokens}):")  # fmt: skip
    for item in draft.claims + draft.limitations:
        r = results.get(item.id)
        decision = "REMOVED" if item.id in removed else "kept"
        detail = (f"action={r.action} supported={r.supported} citation_valid={r.citation_valid}"
                  if r else "no result")  # fmt: skip
        print(f"  [{item.id}] {decision:7}  {detail}  <- {', '.join(item.citation_ids)}")
        print(wrap(f"text:   {item.text}"))
        if r:
            print(wrap(f"reason: {r.reason}"))
    print(f"answer_supported: {output.answer_supported}")
    print(f"after critic: {len(cleaned.claims)} claims, {len(cleaned.limitations)} limitations; "
          f"question cost ${cost:.4f}")  # fmt: skip
    return cost


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", type=int, default=3, help="first N dev questions")
    parser.add_argument("--budget", type=float, default=0.09, help="stop before exceeding (USD)")
    args = parser.parse_args()
    questions = [q["question"] for q in load_dev_questions()[: args.dev]]

    llm, voyage, total = anthropic_client(), voyage_client(), 0.0
    with connect() as conn:
        for q in questions:
            if total + PER_QUESTION_MAX > args.budget:
                print(f"budget stop: ${total:.4f} spent, next question could exceed ${args.budget}")
                break
            total += show(q, llm, voyage, conn)
    print("=" * 100)
    print(f"total Anthropic cost: ${total:.4f}")


if __name__ == "__main__":
    main()
