"""M15-T6: replay post-critic rule changes on a stored agent DEV run, for $0.

    uv run python -m eval.runners.replay_fixes <dev-agent-run-id>

Reads the run's raw.jsonl (which must store `draft_answer`, M15-T4) and re-applies the agent's final
validator with the real citation check and critic results, under four rule sets:
  none  = version 1 as deployed ("revise" counts as remove; the critic can veto the verdict)
  A     = a "revise" item the critic marked supported with a valid citation is kept
  B     = no veto when every validated claim was kept by the critic
  A+B   = both
"none" must reproduce every stored verdict exactly, or the script stops and writes nothing.
No model is called; DEV runs only (any other split is refused). Writes
eval/results/dev-replay-<time>-<commit>/summary.json and rows.jsonl.
"""

import argparse
import json
from collections.abc import Callable
from datetime import UTC, datetime
from math import comb

from backend.app.agents.critic import _kept as kept_v1
from backend.app.guardrails.citation_validator import validate
from backend.app.models.agent import CriticResult
from backend.app.models.answer import DraftAnswer
from eval.runners.retrieval import RESULTS_DIR, git_state

IE = "insufficient_evidence"
BASELINE_RUN = "dev-baseline-20261023-033422-9396a90"  # the DEV baseline reference (76/100)
RULES = ("none", "A", "B", "A+B")


def kept_fix_a(result: CriticResult | None) -> bool:
    """Fix A: "soften this wording" no longer deletes an item the critic itself called supported."""
    return (
        result is not None
        and result.action in ("keep", "revise")
        and result.supported
        and result.citation_valid
    )


def replay_row(row: dict, rule: str) -> str:
    """The final verdict this row would get under `rule` (mirrors validator.finalize)."""
    if row["abstain_reason"] in ("error", "weak_evidence"):
        return row["verdict"]  # decided before the synthesizer; no rule here can change it
    draft = DraftAnswer.model_validate(row["draft_answer"])
    validated, report = validate(draft, [f"PMID:{p}" for p in row["evidence_pmids"]])
    kept: Callable = kept_fix_a if "A" in rule else kept_v1
    by_id: dict[str, CriticResult] = {}
    for r in row["critic_results"]:
        by_id.setdefault(r["claim_id"], CriticResult.model_validate(r))
    claims = [c for c in validated.claims if kept(by_id.get(c.id))]
    all_claims_kept = len(claims) == len(validated.claims)

    if report.downgraded_to_insufficient:
        return IE
    if validated.verdict != IE and not claims:
        return IE
    if validated.verdict != IE and row["answer_supported"] is False:
        if not ("B" in rule and all_claims_kept):
            return IE
    return validated.verdict


def published_revise_items(row: dict, rule: str) -> int:
    """Items published although the critic asked for softer wording ("revise"); only fix A
    publishes these. Items the critic marked unsupported are never published under any rule."""
    if replay_row(row, rule) == IE or row["abstain_reason"] in ("error", "weak_evidence"):
        return 0
    kept: Callable = kept_fix_a if "A" in rule else kept_v1
    results = [CriticResult.model_validate(r) for r in row["critic_results"]]
    return sum(1 for r in results if kept(r) and r.action == "revise")


def sign_test(a: int, b: int) -> float:
    n = a + b
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(a, b) + 1)) / 2**n)


def load_rows(run_id: str) -> tuple[dict, list[dict]]:
    config = json.loads((RESULTS_DIR / run_id / "run_config.json").read_text(encoding="utf-8"))
    if config.get("split") != "dev":
        raise SystemExit(f"Refusing {run_id}: replay runs on DEV runs only.")
    lines = (RESULTS_DIR / run_id / "raw.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    return config, rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_id", help="a completed DEV agent run under eval/results/")
    args = parser.parse_args(argv)

    config, rows = load_rows(args.run_id)
    if config.get("system") != "agent":
        raise SystemExit("Refusing: the run must be an agent run.")
    if any("draft_answer" not in r for r in rows):
        raise SystemExit("Refusing: this run does not store draft answers (needs M15-T4).")
    _, baseline = load_rows(BASELINE_RUN)
    base = {r["pmid"]: r for r in baseline}
    if set(base) != {r["pmid"] for r in rows}:
        raise SystemExit("Refusing: the agent and baseline runs hold different questions.")

    mismatched = [r["pmid"] for r in rows if replay_row(r, "none") != r["verdict"]]
    if mismatched:
        raise SystemExit(f"Stop: 'none' does not reproduce the stored verdicts: {mismatched}")
    git = git_state()
    if git["git_dirty"]:
        raise SystemExit("Stop: commit first; results must name a clean commit.")

    out_rows, results = [], {}
    for rule in RULES:
        verdicts = {r["pmid"]: replay_row(r, rule) for r in rows}
        right = {p for p, v in verdicts.items() if v == base[p]["gold_label"]}
        base_right = {p for p, b in base.items() if b["correct"]}
        results[rule] = {
            "correct": len(right),
            "n": len(rows),
            "accuracy": len(right) / len(rows),
            "refused": sum(v == IE for v in verdicts.values()),
            "vs_baseline": {
                "agent_only_right": len(right - base_right),
                "baseline_only_right": len(base_right - right),
                "sign_test_p": sign_test(len(right - base_right), len(base_right - right)),
            },
            "published_items_marked_revise": sum(published_revise_items(r, rule) for r in rows),
        }
        out_rows += [{"pmid": p, "rule": rule, "verdict": v, "gold": base[p]["gold_label"]}
                     for p, v in verdicts.items()]  # fmt: skip

    now = datetime.now(UTC)
    run_id = f"dev-replay-{now:%Y%m%d-%H%M%S}-{git['git_commit'][:7]}"
    summary = {
        "run_id": run_id,
        "created_at": now.isoformat(),
        "git_commit": git["git_commit"],
        "agent_run": args.run_id,
        "agent_run_cost_usd": sum(r["cost_usd"] for r in rows),
        "baseline_run": BASELINE_RUN,
        "baseline_correct": sum(r["correct"] for r in baseline),
        "model_calls": 0,
        "none_reproduces_stored_verdicts": True,
        "rules": results,
    }
    out = RESULTS_DIR / run_id
    out.mkdir(parents=True, exist_ok=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out / "rows.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in out_rows), encoding="utf-8"
    )
    print(f"Saved {out}")
    for rule, r in results.items():
        vs = r["vs_baseline"]
        print(f"{rule:5s} {r['correct']}/{r['n']} refused {r['refused']} | vs baseline "
              f"+{vs['agent_only_right']} -{vs['baseline_only_right']} p={vs['sign_test_p']:.3f} | "
              f"published revise {r['published_items_marked_revise']}")  # fmt: skip


if __name__ == "__main__":
    main()
