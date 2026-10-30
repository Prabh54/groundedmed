"""Check candidate demo example questions by running each through the agent (M5 example picking).

Run: uv run python -m scripts.check_examples                     (all candidates, $1 cap)
     uv run python -m scripts.check_examples --only PMID --cap 0.05  (one candidate)
Each candidate is a short rewording of a DEV question (never test). A candidate is kept only if
the agent answers yes/no without abstaining, matches the dev gold label, makes at least 2 cited
claims and cites the source paper.
The cap is checked BEFORE each question (spent + reserve <= cap); a question already started is
never cut off, so with --only the only protection is that one dev question has never cost more
than $0.0402 (dev agent run). Results are merged into docs/design/example-questions-check.json:
entries for PMIDs not re-checked are kept unchanged, and spent_usd is the total of all runs.
"""

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from backend.app.agents.pipeline import answer_agent
from backend.app.config import LLM_MODEL
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
    # 24669960 (rt-PA stroke drug) was checked on 2026-10-28; the owner removed it from the page on
    # 2026-10-29 (reads closest to treatment advice). Its recorded result stays in the file.
    ("11138995", "Is trouble putting feelings into words linked to unexplained physical symptoms?"),
    # Owner 2026-10-30 UTC: replace keyhole surgery (a treatment comparison); fill the rt-PA slot.
    (
        "22680064",
        "Can third-trimester ultrasound predict the position of the first twin at delivery?",
    ),
    ("17598882", "Is breast cancer prognosis inherited?"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="check just this candidate PMID and merge it into the file")
    parser.add_argument("--cap", type=float, default=CAP_USD, help="owner-approved USD cap")
    args = parser.parse_args()
    candidates = [c for c in CANDIDATES if args.only in (None, c[0])]
    if not candidates:
        raise SystemExit(f"{args.only} is not a candidate")
    if args.cap <= 0:
        raise SystemExit("--cap must be a positive USD amount")
    reserve = min(PER_QUESTION_RESERVE_USD, args.cap) if args.only else PER_QUESTION_RESERVE_USD

    def git(*cmd: str) -> str:
        return subprocess.run(["git", *cmd], cwd=REPO_ROOT, capture_output=True,
                              text=True).stdout.strip()  # fmt: skip

    commit = git("rev-parse", "--short", "HEAD") + (
        "-dirty" if git("status", "--porcelain") else ""
    )
    dev = json.loads((REPO_ROOT / "data/manifests/dev.json").read_text(encoding="utf-8"))
    test = json.loads((REPO_ROOT / "data/manifests/test.json").read_text(encoding="utf-8"))
    dev_pmids = {item["pmid"] for item in dev["items"]}
    test_pmids = {item["pmid"] for item in test["items"]}
    gold = {}
    raw = REPO_ROOT / "eval/results/dev-agent-20261023-222333-871b22d/raw.jsonl"
    for line in raw.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        gold[row["pmid"]] = row["gold_label"]
    for pmid, _ in candidates:
        if pmid not in dev_pmids or pmid in test_pmids:
            raise SystemExit(f"{pmid} is not a dev-only question")

    results, spent = [], 0.0
    llm, voyage = anthropic_client(), voyage_client()
    with connect() as conn:
        for pmid, question in candidates:
            if spent + reserve > args.cap:
                print(f"STOP: cap ${args.cap:.2f} would be exceeded (spent ${spent:.4f})")
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
                | {
                    "checked_at": datetime.now(UTC).isoformat(timespec="seconds"),
                    "model": LLM_MODEL,
                    "commit": commit,
                    "cap_usd": args.cap,
                }
            )
            print(
                f"{'KEEP' if keep else 'DROP'} {pmid} verdict={answer.verdict} gold={gold[pmid]} "
                f"claims={len(answer.claims)} ${answer.estimated_cost_usd:.4f}  {question}"
            )
    if not results:
        print("nothing checked; file unchanged")
        return
    if OUT.exists():  # keep every entry not re-checked now; spent_usd stays the total of all runs
        record = json.loads(OUT.read_text(encoding="utf-8"))
        checked = {r["pmid"] for r in results}
        record["results"] = [r for r in record["results"] if r["pmid"] not in checked] + results
        record["spent_usd"] = round(record["spent_usd"] + spent, 5)
    else:
        record = {"spent_usd": round(spent, 5), "cap_usd": args.cap, "results": results}
    OUT.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"spent ${spent:.4f} of ${args.cap:.2f}; wrote {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
