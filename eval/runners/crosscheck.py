"""M4-T6 (ADR-037): second-AI cross-check of the Gemini judge; reveals the held claim support.

    uv run python -m eval.runners.crosscheck --judge-run <judge-run> --scoring-run <m4-final-id>

The second AI (ChatGPT, OpenAI) answered the blind 30-row sheet (ADR-036). This script:
1. refuses unless the tree is clean and the second AI's answer file is committed (it must be on
   record before any join with judge labels);
2. rebuilds the seeded rubric v2 §7 draw and shuffle, and refuses unless the rebuilt sheet equals
   committed sheet row for row (so `row` points at the claim that was actually shown);
3. checks every row has exactly one allowed answer;
4. writes the full rubric §7 CSV (owner columns renamed second_ai_*) to
   `eval/audits/<judge-run>-second-ai-crosscheck.csv`, never overwriting;
5. computes binary agreement (ADR-036 point 4) per judge group (accepted / rejected), per system and
   pooled within each group, with Wilson 95% intervals; Unsure excluded and counted;
6. writes `eval/results/m4-crosscheck-<time>-<commit>/` with agreement.json, claim_support.json
   (copied unchanged from the scoring run's held file) and report.md.
It is always a "second-AI cross-check", never a manual or human audit."""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime

from eval.metrics.intervals import wilson
from eval.runners import audit_sheet
from eval.runners.judge import VOID_FILE, load_corpus
from eval.runners.retrieval import REPO_ROOT, RESULTS_DIR, git_state
from eval.runners.score_m4 import HELD_FILE

ANSWERS = ("Supported", "Not supported", "Unsure")
GROUPS = ("judge_supported", "judge_rejected")
SECOND_AI = "ChatGPT (OpenAI), model as reported by the owner: GPT-6 Sol"
NAME = "second-AI cross-check"
PURPOSE = (
    "SECOND-AI CROSS-CHECK ONLY - stratified sample, NOT an estimate of claim support, "
    "NOT a human audit"
)
COLUMNS = ["audit_row", "sample_purpose", "sample_group", "question_pmid", "claim_id",
           "claim_text", "cited_passages", "second_ai_label", "second_ai_note", "system",
           "judge_label", "judge_quote", "judge_reason"]  # fmt: skip


def committed(path) -> bool:
    """True when `path` is tracked by git (the clean-tree check covers 'unchanged since')."""
    out = subprocess.run(["git", "log", "--format=%H", "--", str(path)], cwd=REPO_ROOT,
                         capture_output=True, text=True, check=True).stdout  # fmt: skip
    return bool(out.strip())


def read_csv(path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def check_sheet(sheet: list[dict], rebuilt: list[dict]) -> None:
    keys = ("row", "claim", "cited passage")
    if [[str(r[k]) for k in keys] for r in rebuilt] != [[r[k] for k in keys] for r in sheet]:
        raise SystemExit("The committed sheet differs from the rebuilt draw; refusing.")


def check_answers(answers: list[dict], n: int) -> dict[int, dict]:
    rows = [int(a["row"]) for a in answers]
    if sorted(rows) != list(range(1, n + 1)):
        raise SystemExit(f"Answers must cover rows 1..{n} exactly once; refusing.")
    bad = [a["row"] for a in answers if a["answer"] not in ANSWERS]
    if bad:
        raise SystemExit(f"Rows without an allowed answer {ANSWERS}: {bad}; refusing.")
    return {int(a["row"]): a for a in answers}


def agrees(answer: str, judge_label: str) -> bool:
    """Binary rule (ADR-036): judge `supported` <-> Supported; partially/unsupported <-> Not
    supported."""
    return (answer == "Supported") == (judge_label == "supported")


def agreement(merged: list[dict]) -> dict:
    out = {}
    for group in GROUPS:
        out[group] = {}
        for system in ("baseline", "agent", "pooled"):
            rows = [r for r in merged if r["sample_group"] == group
                    and (system == "pooled" or r["system"] == system)]  # fmt: skip
            compared = [r for r in rows if r["second_ai_label"] != "Unsure"]
            agree = sum(agrees(r["second_ai_label"], r["judge_label"]) for r in compared)
            ci = wilson(agree, len(compared))
            out[group][system] = {
                "rows": len(rows), "unsure": len(rows) - len(compared),
                "compared": len(compared), "agree": agree,
                "agreement": agree / len(compared) if compared else None,
                "wilson95": list(ci) if ci else None,
                "measurable": bool(compared),
            }  # fmt: skip
    table = {}
    for r in merged:
        cell = table.setdefault(r["system"], {}).setdefault(r["judge_label"], {})
        cell[r["second_ai_label"]] = cell.get(r["second_ai_label"], 0) + 1
    out["judge_label_by_second_ai_answer"] = table
    out["rule"] = (
        "binary: judge 'supported' agrees with Supported; judge 'partially_supported' or "
        "'unsupported' agrees with Not supported; Unsure excluded and counted"
    )
    return out


def pct(p: dict) -> str:
    if p.get("value", p.get("agreement")) is None:
        return "not measurable"
    v = p.get("value", p.get("agreement"))
    lo, hi = p["wilson95"]
    n = p.get("n", p.get("compared"))
    k = p.get("count", p.get("agree"))
    return f"{v:.1%} ({lo:.1%}–{hi:.1%}), {k}/{n}"


def report(claims: dict, agree: dict, config: dict) -> str:
    s = claims["systems"]
    lines = [
        f"# M4 claim support and {NAME} — `{config['run_id']}`", "",
        f"Computed {config['created_utc']} at commit `{config['git_commit']}`. Scoring run "
        f"`{config['inputs']['scoring_run']}`; judge run `{config['inputs']['judge_run']}`; "
        f"second AI: {SECOND_AI}.", "",
        f"## {claims['name']} (Gemini judge, test split)", "",
        "| | Baseline | Agent |", "|---|---|---|",
    ]  # fmt: skip
    rows = (("Claims, strict (partially = not supported)", "claim_strict"),
            ("Claims, lenient (partially counts)", "claim_lenient"),
            ("Limitations, strict", "limitation_strict"),
            ("Limitations, lenient", "limitation_lenient"),
            ("Questions with ≥1 not-fully-supported item",
             "questions_with_a_not_supported_item"))  # fmt: skip
    for label, key in rows:
        lines.append(f"| {label} | {pct(s['baseline'][key])} | {pct(s['agent'][key])} |")
    for key in ("claim_strict", "claim_lenient"):
        cb = [s[x][key]["question_bootstrap95"] for x in ("baseline", "agent")]
        lines.append(f"| {key} question-level bootstrap 95% | {cb[0][0]:.1%}–{cb[0][1]:.1%} | "
                     f"{cb[1][0]:.1%}–{cb[1][1]:.1%} |")  # fmt: skip
    lines.append(f"| Not judged | {s['baseline']['not_judged']} | {s['agent']['not_judged']} |")
    lines += [
        "",
        f"Caveat: {claims['caveat']}",
        "",
        "Agent minus baseline (paired bootstrap 95%):",
        "",
    ]
    for key, d in claims["agent_minus_baseline"].items():
        ci = "n/a" if d["ci95"] is None else f"{d['ci95'][0]:+.3f} to {d['ci95'][1]:+.3f}"
        lines.append(f"- {key}: {d['estimate']:+.3f} ({ci})")
    lines += ["", "## Second-AI cross-check: does a second AI agree with the judge?", "",
              "Stratified 30-claim sample (rubric v2 §7): per system up to 5 judge-rejected and "
              "the rest judge-accepted. Agreement is reported separately for each group; the "
              "share the second AI calls supported is NOT a claim-support estimate.", "",
              "| Judge group | Baseline | Agent | Both systems |", "|---|---|---|---|"]  # fmt: skip
    for group in GROUPS:
        g = agree[group]
        lines.append(f"| {group} | {pct(g['baseline'])} | {pct(g['agent'])} | {pct(g['pooled'])} |")
    unsure = sum(agree[g]["pooled"]["unsure"] for g in GROUPS)
    lines += ["", f"Unsure answers (excluded): {unsure}. Rule: {agree['rule']}.", "",
              "Limitations: this is NOT a human audit. Agreement shows whether two AI models from "
              "different providers read the claims the same way; shared blind spots are possible, "
              "so agreement does not prove the judge is right. How the second AI was run is as "
              "reported by the owner (ADR-037)."]  # fmt: skip
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-run", required=True)
    parser.add_argument("--scoring-run", required=True)
    args = parser.parse_args(argv)
    git = git_state()
    if git["git_dirty"]:
        raise SystemExit("Uncommitted changes: run only from a clean commit.")
    judge_dir, scoring_dir = RESULTS_DIR / args.judge_run, RESULTS_DIR / args.scoring_run
    if (judge_dir / VOID_FILE).exists() or not (judge_dir / "summary.json").exists():
        raise SystemExit(f"{args.judge_run} is void or unfinished.")
    scoring_config = json.loads((scoring_dir / "run_config.json").read_text(encoding="utf-8"))
    if scoring_config["inputs"]["judge"] != args.judge_run:
        raise SystemExit("The scoring run used a different judge run; refusing.")
    sheet_path = audit_sheet.AUDITS_DIR / f"{args.judge_run}-owner-worksheet.csv"
    answers_path = audit_sheet.AUDITS_DIR / f"{args.judge_run}-second-ai-answers.csv"
    out_csv = audit_sheet.AUDITS_DIR / f"{args.judge_run}-second-ai-crosscheck.csv"
    if not committed(answers_path):
        raise SystemExit("The second AI's answers must be committed before the join; refusing.")
    if out_csv.exists():
        raise SystemExit(f"{out_csv.name} already exists; refusing to overwrite.")

    judgments = [json.loads(line)
                 for line in (judge_dir / "judgments.jsonl").open(encoding="utf-8")]  # fmt: skip
    sample = audit_sheet.draw(judgments)
    rebuilt = audit_sheet.worksheet_rows(sample, load_corpus())
    check_sheet(read_csv(sheet_path), rebuilt)
    answers = check_answers(read_csv(answers_path), len(rebuilt))
    merged = []
    for row, item in zip(rebuilt, audit_sheet.shuffled(sample), strict=True):
        a = answers[row["row"]]
        merged.append({
            "audit_row": row["row"], "sample_purpose": PURPOSE,
            "sample_group": item["sample_group"],
            "question_pmid": item["pmid"], "claim_id": item["item_id"], "claim_text": item["text"],
            "cited_passages": row["cited passage"], "second_ai_label": a["answer"],
            "second_ai_note": a["note"], "system": item["system"], "judge_label": item["label"],
            "judge_quote": item["evidence_quote"], "judge_reason": item["reason"],
        })  # fmt: skip
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(merged)

    agree = agreement(merged)
    claims = json.loads((scoring_dir / HELD_FILE).read_text(encoding="utf-8"))
    created = datetime.now(UTC)
    run_id = f"m4-crosscheck-{created:%Y%m%d-%H%M%S}-{git['git_commit'][:7]}"
    out = RESULTS_DIR / run_id
    out.mkdir(parents=True, exist_ok=False)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    config = {"run_id": run_id, "created_utc": created.isoformat(timespec="seconds"), **git,
              "name": NAME, "second_ai": SECOND_AI, "decision": "ADR-037",
              "inputs": {"judge_run": args.judge_run, "scoring_run": args.scoring_run,
                         "sheet": sheet_path.name, "answers": answers_path.name},
              "answers_sha256": sha(answers_path), "sheet_sha256": sha(sheet_path),
              "merged_csv": out_csv.name}  # fmt: skip
    (out / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (out / "agreement.json").write_text(json.dumps(agree, indent=2) + "\n", encoding="utf-8")
    (out / "claim_support.json").write_text(json.dumps(claims, indent=2) + "\n", encoding="utf-8")
    (out / "report.md").write_text(report(claims, agree, config), encoding="utf-8")
    print(f"Wrote {out_csv.relative_to(REPO_ROOT)} and eval/results/{run_id}/")


if __name__ == "__main__":
    main(sys.argv[1:])
