"""Grounding spot-check, method v1 (frozen; see eval/rubric/grounding_spotcheck_v1.md).

  uv run python -m eval.grounding_spotcheck packet --run <run-id>   # review packet for the judge
  uv run python -m eval.grounding_spotcheck score --run <run-id>    # rate from stored judgments

`packet` writes local-reports/grounding-packet-<run-id>.md (git-ignored: it holds PubMed abstracts)
with every item of the fixed sample directly above the full text of each passage it cites, without
naming the system. `score` reads eval/audits/grounding/<run-id>-judgments.jsonl, checks it covers
exactly the run's items, and writes eval/audits/grounding/<run-id>-score.json. No API calls.
"""

import argparse
import hashlib
import json
import sys

from backend.app.config import REPO_ROOT
from eval.runners.retrieval import RESULTS_DIR

METHOD = "grounding_spotcheck_v1"
# The 18 dev questions the M2 final reviewer read (2026-10-23). Fixed: the list is the sample.
FROZEN_PMIDS = (
    "10490564", "2503176", "7664228", "9582182", "11053064", "11411430", "12595848", "12970636",
    "15095519", "16296668", "19430778", "19925761", "21123461", "22954812", "24139705", "24172579",
    "25735444", "16776337",
)  # fmt: skip
LABELS = ("supported", "partly", "unsupported", "not_judged")
AUDIT_DIR = REPO_ROOT / "eval" / "audits" / "grounding"
CORPUS = REPO_ROOT / "data" / "corpus.jsonl"


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def sample_items(run_id: str) -> list[dict]:
    """Every claim and limitation of the run's rows for FROZEN_PMIDS, in a fixed order."""
    rows = {}
    for line in (RESULTS_DIR / run_id / "raw.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        rows[row["pmid"]] = row
    missing = [p for p in FROZEN_PMIDS if p not in rows]
    if missing:
        raise SystemExit(f"run {run_id} lacks sample questions {missing}")
    items = []
    for pmid in FROZEN_PMIDS:
        row = rows[pmid]
        for kind in ("claims", "limitations"):
            for item in row.get(kind, []):
                items.append({"pmid": pmid, "question": row["question"], "kind": kind,
                              "id": item["id"], "text": item["text"],
                              "citation_ids": item["citation_ids"],
                              "text_sha": text_hash(item["text"])})  # fmt: skip
    return items


def write_packet(run_id: str) -> None:
    corpus = {}
    for line in CORPUS.open(encoding="utf-8"):
        doc = json.loads(line)
        corpus[doc["pmid"]] = doc
    items = sample_items(run_id)
    lines = [
        f"# Grounding spot-check packet ({METHOD})",
        "",
        "Label every item: supported | partly | unsupported (see the rubric). Each item is",
        "shown above the full text of every passage it cites, exactly as the system saw it.",
        "",
    ]
    current = None
    for it in items:
        if it["pmid"] != current:
            current = it["pmid"]
            lines += ["=" * 100, f"## Question {current}: {it['question']}", ""]
        header = f"ITEM {it['pmid']} {it['id']} ({it['kind'][:-1]}, sha {it['text_sha']}):"
        lines += ["-" * 100, header, it["text"], ""]
        for cid in it["citation_ids"]:
            doc = corpus[cid.split(":", 1)[1]]
            lines.append(f"  PASSAGE {cid}:")
            lines += [f"    {s['label']}: {s['text']}" for s in doc["sections"]]
            lines.append("")
    out = REPO_ROOT / "local-reports" / f"grounding-packet-{run_id}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(items)} items from {len(FROZEN_PMIDS)} questions -> {out.relative_to(REPO_ROOT)}")
    print('Judge returns one JSON line per item: {"pmid", "id", "text_sha", "label", "note"}')


def score(run_id: str, judgments: list[dict]) -> dict:
    items = sample_items(run_id)
    expected = {(i["pmid"], i["id"], i["text_sha"]) for i in items}
    got = [(j["pmid"], j["id"], j["text_sha"]) for j in judgments]
    if len(got) != len(set(got)) or set(got) != expected:
        raise SystemExit(
            f"judgments do not match the run's items: missing {sorted(expected - set(got))}, "
            f"unexpected {sorted(set(got) - expected)}, duplicates {len(got) - len(set(got))}"
        )
    bad_labels = [j for j in judgments if j["label"] not in LABELS]
    if bad_labels:
        raise SystemExit(f"unknown labels: {sorted({j['label'] for j in bad_labels})}")
    judged = [j for j in judgments if j["label"] != "not_judged"]
    flagged = [j for j in judged if j["label"] in ("partly", "unsupported")]
    return {
        "method": METHOD,
        "run_id": run_id,
        "questions": len(FROZEN_PMIDS),
        "items": len(items),
        "judged_items": len(judged),
        "not_judged": len(judgments) - len(judged),
        "counts": {label: sum(j["label"] == label for j in judgments) for label in LABELS},
        "flagged_items": len(flagged),
        "unsupported_detail_rate": len(flagged) / len(judged) if judged else None,
        "rows_with_flagged_items": len({j["pmid"] for j in flagged}),
        "flagged": [
            {"pmid": j["pmid"], "id": j["id"], "label": j["label"], "note": j.get("note", "")}
            for j in flagged
        ],  # fmt: skip
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["packet", "score"])
    parser.add_argument("--run", required=True)
    args = parser.parse_args(argv)
    if args.command == "packet":
        write_packet(args.run)
        return
    path = AUDIT_DIR / f"{args.run}-judgments.jsonl"
    judgments = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    result = {"judgments_file": str(path.relative_to(REPO_ROOT)), **score(args.run, judgments)}
    (AUDIT_DIR / f"{args.run}-score.json").write_text(json.dumps(result, indent=2) + "\n",
                                                     encoding="utf-8")  # fmt: skip
    print(json.dumps({k: v for k, v in result.items() if k != "flagged"}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
