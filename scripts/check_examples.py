"""Check candidate demo example questions by running each through the agent (M5 example picking).

Run: uv run python -m scripts.check_examples
Each candidate is a short rewording of a DEV question (never test). A candidate is kept only if
the agent answers yes/no without abstaining, matches the dev gold label, makes at least 2 cited
claims and cites the source paper. Stops before going over the owner's $1 cap.
Writes docs/design/example-questions-check.json.
"""

import json
from pathlib import Path

from backend.app.agents.pipeline import answer_agent
from backend.app.llm.client import anthropic_client
from backend.app.retrieval.embed import voyage_client
from backend.app.retrieval.vector_search import connect

REPO_ROOT = Path(__file__).resolve().parents[1]
CAP_USD = 1.00
PER_QUESTION_RESERVE_USD = 0.08  # dev agent run: mean $0.028, so this leaves a wide margin
OUT = REPO_ROOT / "docs/design/example-questions-check.json"

# (dev PMID, short wording shown to visitors). Gold labels are read from the dev agent run below.
CANDIDATES = [
    ("26370095", "Is paying pregnant smokers to quit worth the cost?"),
    ("21123461", "Are body measurements like waist size linked to height?"),
    (
        "17910536",
        "Can adults with mild intellectual disabilities get better at reading comprehension?",
    ),
    ("25819796", "Can the brain's right half learn to read on its own?"),
    ("21194998", "Does keyhole surgery hurt newborns less than open surgery?"),
    ("16809243", "Does a baby's sex affect asthma ER visits during pregnancy?"),
    ("28027677", "Do recorded lectures make medical students skip class?"),
    ("24669960", "Does a stroke patient's sex change how well the clot-busting drug rt-PA works?"),
]


def main() -> None:
    dev = json.loads((REPO_ROOT / "data/manifests/dev.json").read_text(encoding="utf-8"))
    test = json.loads((REPO_ROOT / "data/manifests/test.json").read_text(encoding="utf-8"))
    dev_pmids = {item["pmid"] for item in dev["items"]}
    test_pmids = {item["pmid"] for item in test["items"]}
    gold = {}
    raw = REPO_ROOT / "eval/results/dev-agent-20261023-222333-871b22d/raw.jsonl"
    for line in raw.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        gold[row["pmid"]] = row["gold_label"]
    for pmid, _ in CANDIDATES:
        if pmid not in dev_pmids or pmid in test_pmids:
            raise SystemExit(f"{pmid} is not a dev-only question")

    results, spent = [], 0.0
    llm, voyage = anthropic_client(), voyage_client()
    with connect() as conn:
        for pmid, question in CANDIDATES:
            if spent + PER_QUESTION_RESERVE_USD > CAP_USD:
                print(f"STOP: cap ${CAP_USD:.2f} would be exceeded (spent ${spent:.4f})")
                break
            answer, _ = answer_agent(question, conn=conn, voyage=voyage, llm=llm)
            spent += answer.estimated_cost_usd
            cited = sorted({c for claim in answer.claims for c in claim.citation_ids})
            keep = (
                not answer.abstained
                and answer.verdict in ("yes", "no")
                and answer.verdict == gold[pmid]
                and len(answer.claims) >= 2
                and f"PMID:{pmid}" in cited
            )
            results.append(
                {
                    "pmid": pmid,
                    "question": question,
                    "gold_label": gold[pmid],
                    "verdict": answer.verdict,
                    "abstained": answer.abstained,
                    "abstain_reason": answer.abstain_reason,
                    "n_claims": len(answer.claims),
                    "cited": cited,
                    "claims": [c.text for c in answer.claims],
                    "cost_usd": round(answer.estimated_cost_usd, 5),
                    "total_ms": round(answer.timing_ms["total"]),
                    "keep": keep,
                }
            )
            print(
                f"{'KEEP' if keep else 'DROP'} {pmid} verdict={answer.verdict} gold={gold[pmid]} "
                f"claims={len(answer.claims)} ${answer.estimated_cost_usd:.4f}  {question}"
            )
    OUT.write_text(
        json.dumps({"spent_usd": round(spent, 5), "cap_usd": CAP_USD, "results": results}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(f"spent ${spent:.4f} of ${CAP_USD:.2f}; wrote {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
