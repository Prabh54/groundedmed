"""Choose the evidence-gate threshold T on dev data only (PLAN.md M2-T3, ARCHITECTURE §7).

Run: uv run python -m eval.runners.gate_calibration <dev-retrieval-run-id>

Inputs: a stored dev retrieval run (rerank scores of each question's final top 5, plus its gold
PMID). No answer labels are read. Dev Hit@5 is 100%, so "gold in top 5 vs not" cannot be used
directly; instead each question gives two observations:
  - answerable:   top rerank score with the source paper present (the real result);
  - unanswerable: top rerank score among the other papers, i.e. what retrieval would return if
                  the corpus did not contain the source paper (a leave-one-out simulation).
Decision rule, fixed before looking at the result: pick T maximising Youden's J
(share of answerable kept + share of unanswerable abstained - 1) over a 0.01 grid; ties go to the
lower T (fewer refusals). If J < 0.5 the scores do not separate and the gate is turned off (T = 0).

The same rule on dense (cosine) scores gives the fallback threshold used when reranking fails, so
the gate never opens fully during a reranker outage.

It also retrieves a few hand-written out-of-scope probe questions (Voyage only, ~free) and reports
whether T would abstain on them; probes are a check, not used to choose T.
Writes eval/results/dev-gate-<timestamp>-<commit>/analysis.json; never overwrites a run.
"""

import json
import sys
from datetime import UTC, datetime

from backend.app.retrieval.retrieve import retrieve
from backend.app.retrieval.vector_search import connect
from eval.runners.retrieval import RESULTS_DIR, git_state

MIN_J = 0.5
OUT_OF_SCOPE_PROBES = [
    "What is the boiling point of tungsten?",
    "Who won the 2018 FIFA World Cup?",
    "How do I reset my router password?",
    "Does the Mars Perseverance rover use solar panels?",
    "What is the best recipe for sourdough bread?",
    "Which programming language is fastest for web servers?",
    "Does quantum entanglement allow faster-than-light communication?",
    "What is the tax rate on capital gains in Canada?",
]


def dense_observations(raw_rows: list[dict]) -> tuple[list[float], list[float]]:
    answerable = [r["dense_scores"][0] for r in raw_rows]
    unanswerable = [
        max(
            s
            for p, s in zip(r["dense_pmids"], r["dense_scores"], strict=True)
            if p != r["gold_pmid"]
        )
        for r in raw_rows
    ]
    return answerable, unanswerable


def observations(raw_rows: list[dict]) -> tuple[list[float], list[float]]:
    answerable, unanswerable = [], []
    for r in raw_rows:
        scored = [(p, s) for p, s in zip(r["final_pmids"], r["rerank_scores"], strict=True)]
        if any(s is None for _, s in scored):
            raise RuntimeError(f"rerank skipped for {r['gold_pmid']}; rerun retrieval first")
        answerable.append(scored[0][1])
        unanswerable.append(max(s for p, s in scored if p != r["gold_pmid"]))
    return answerable, unanswerable


def choose_threshold(answerable: list[float], unanswerable: list[float]) -> dict:
    best = None
    for i in range(0, 101):
        t = i / 100
        kept = sum(s >= t for s in answerable) / len(answerable)
        abstained = sum(s < t for s in unanswerable) / len(unanswerable)
        j = kept + abstained - 1
        if best is None or j > best["youden_j"] + 1e-12:
            best = {"threshold": t, "youden_j": round(j, 4), "answerable_kept": kept,
                    "unanswerable_abstained": abstained}  # fmt: skip
    if best["youden_j"] < MIN_J:
        best = {**best, "threshold": 0.0, "gate": "off (scores do not separate)"}
    else:
        best["gate"] = "on"
    return best


def main(dev_run_id: str) -> None:
    run_dir = RESULTS_DIR / dev_run_id
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    if config["split"] != "dev":
        raise RuntimeError("gate calibration only uses dev runs")
    rows = [json.loads(line) for line in (run_dir / "raw.jsonl").open(encoding="utf-8")]
    answerable, unanswerable = observations(rows)
    choice = choose_threshold(answerable, unanswerable)
    dense_answerable, dense_unanswerable = dense_observations(rows)
    dense_choice = choose_threshold(dense_answerable, dense_unanswerable)

    probes = []
    with connect() as conn:
        for q in OUT_OF_SCOPE_PROBES:
            result = retrieve(q, conn=conn)
            top = max(c.rerank_score for c in result.evidence)
            dense_top = result.candidates[0].dense_score
            probes.append(
                {
                    "question": q,
                    "top_rerank_score": round(top, 6),
                    "abstains": top < choice["threshold"],
                    "top_dense_score": round(dense_top, 6),
                    "dense_fallback_abstains": dense_top < dense_choice["threshold"],
                }
            )

    git = git_state()
    run_id = f"dev-gate-{datetime.now(UTC):%Y%m%d-%H%M%S}-{git['git_commit'][:7]}"
    out = RESULTS_DIR / run_id
    out.mkdir(parents=True, exist_ok=False)
    analysis = {
        "run_id": run_id,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        **git,
        "split": "dev",
        "source_dev_run": dev_run_id,
        "rule": "max Youden J on a 0.01 grid, ties -> lower T; J < 0.5 -> gate off",
        "choice": choice,
        "dense_fallback_choice": dense_choice,
        "n_questions": len(rows),
        "answerable_top_scores": sorted(round(s, 6) for s in answerable),
        "unanswerable_top_scores": sorted(round(s, 6) for s in unanswerable),
        "dense_answerable_top_scores": sorted(round(s, 6) for s in dense_answerable),
        "dense_unanswerable_top_scores": sorted(round(s, 6) for s in dense_unanswerable),
        "out_of_scope_probes": probes,
    }
    (out / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    keys = ("choice", "dense_fallback_choice", "out_of_scope_probes")
    print(json.dumps({k: analysis[k] for k in keys}, indent=2))
    print(f"Saved to {out.relative_to(RESULTS_DIR.parent.parent)}")


if __name__ == "__main__":
    main(sys.argv[1])
