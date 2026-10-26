"""M4-T5: the owner's blind audit sheet (rubric v2 §7 draw; ADR-036 blind two-step handling).

    uv run python -m eval.runners.audit_sheet --judge-run <judge-test-run-id>

Draws 15 claims per system exactly as rubric v2 §7 says and writes ONLY the owner's worksheet:
`eval/audits/<judge-run-id>-owner-worksheet.csv` (UTF-8 with BOM, for Excel and Google Sheets), rows
in a seeded shuffled order, with no system, sample group, judge label, quote or reason. Nothing that
reveals the judge's verdict is written or printed; the seeded draw is rebuilt at merge time (M4-T6).
Never overwrites an existing worksheet (the owner may be filling it in)."""

import argparse
import csv
import json
import random
import sys

from eval.runners.judge import VOID_FILE, load_corpus, render_passage
from eval.runners.retrieval import REPO_ROOT, RESULTS_DIR

AUDITS_DIR = REPO_ROOT / "eval" / "audits"
SYSTEMS = ("baseline", "agent")
PER_SYSTEM = 15
MAX_REJECTED = 5
REJECTED = ("partially_supported", "unsupported")
SHUFFLE_SEED = "judge_v2-audit-worksheet|20261024"
ANSWER = "my answer (Supported / Not supported / Unsure)"
COLUMNS = ["row", "claim", "cited passage", ANSWER, "my note (optional)"]


def item_number(item_id: str) -> int:
    return int(item_id[1:])


def draw(judgments: list[dict]) -> list[dict]:
    """Rubric v2 §7, per system: claims only, not_judged excluded; group R (judge-rejected) and
    group S (judge-supported), each ordered by question PMID as a number, then item ID; one
    generator per system; min(5, |R|) from R first, then fill to 15 from S (all of S if smaller)."""
    sample = []
    for system in SYSTEMS:
        claims = [j for j in judgments if j["system"] == system and j["kind"] == "claim"
                  and j["label"] != "not_judged"]  # fmt: skip
        key = lambda j: (int(j["pmid"]), item_number(j["item_id"]))  # noqa: E731
        group_r = sorted((j for j in claims if j["label"] in REJECTED), key=key)
        group_s = sorted((j for j in claims if j["label"] == "supported"), key=key)
        rng = random.Random(f"judge_v2-audit|20261024|{system}")
        from_r = rng.sample(group_r, min(MAX_REJECTED, len(group_r)))
        from_s = rng.sample(group_s, min(PER_SYSTEM - len(from_r), len(group_s)))
        sample += [j | {"sample_group": "judge_rejected"} for j in from_r]
        sample += [j | {"sample_group": "judge_supported"} for j in from_s]
    return sample


def shuffled(sample: list[dict]) -> list[dict]:
    """Worksheet order: seeded shuffle, so neither the group nor the system shows in the order."""
    order = list(sample)
    random.Random(SHUFFLE_SEED).shuffle(order)
    return order


def cited_passages(item: dict, corpus: dict) -> str:
    """The cited passages exactly as the judge saw them, one block per cited paper."""
    blocks = []
    for cid in item["citation_ids"]:
        pmid = cid.split(":", 1)[1]
        blocks.append(f"[{cid}]\n{render_passage(corpus[pmid])}")
    return "\n\n".join(blocks)


def worksheet_rows(sample: list[dict], corpus: dict) -> list[dict]:
    return [{"row": i, "claim": item["text"], "cited passage": cited_passages(item, corpus),
             ANSWER: "", "my note (optional)": ""}
            for i, item in enumerate(shuffled(sample), start=1)]  # fmt: skip


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-run", required=True)
    args = parser.parse_args(argv)
    judge_dir = RESULTS_DIR / args.judge_run
    if (judge_dir / VOID_FILE).exists() or not (judge_dir / "summary.json").exists():
        raise SystemExit(f"{args.judge_run} is void or unfinished; no audit from it.")
    out = AUDITS_DIR / f"{args.judge_run}-owner-worksheet.csv"
    if out.exists():
        raise SystemExit(f"{out.name} already exists; refusing to overwrite the owner's sheet.")
    judgments = [json.loads(line)
                 for line in (judge_dir / "judgments.jsonl").open(encoding="utf-8")]  # fmt: skip
    sample = draw(judgments)
    rows = worksheet_rows(sample, load_corpus())
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    per_system = {s: sum(item["system"] == s for item in sample) for s in SYSTEMS}
    print(f"Wrote {out.relative_to(REPO_ROOT)}: {len(rows)} rows ({per_system['baseline']} "
          f"baseline, {per_system['agent']} agent, order shuffled; no judge columns)")  # fmt: skip


if __name__ == "__main__":
    main(sys.argv[1:])
