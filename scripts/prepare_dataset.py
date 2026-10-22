"""Build the PubMedQA manifests and corpus (ADR-008, ADR-009).

Run from the repo root: uv run python -m scripts.prepare_dataset

Answer fields are dropped at load time: `long_answer` and `reasoning_*` are never read into
memory, and `final_decision` only lives on a `BenchmarkItem` (eval manifests), never on a
`Document` (index).
"""

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

DATASET_REPO = "qiaojin/PubMedQA"
DATASET_REVISION = "9001f2853fb87cab8d220904e0de81ac6973b318"
LABELED, UNLABELED = "pqa_labeled", "pqa_unlabeled"
SOURCE_TAG = {LABELED: "pqa_l", UNLABELED: "pqa_u"}

# Fields that carry the answer. Never indexed or shown to a runtime model (CLAUDE.md, ADR-009).
ANSWER_FIELDS = ("long_answer", "final_decision", "reasoning_required_pred", "reasoning_free_pred")
# The only context keys a Document is built from.
CONTEXT_KEYS = ("contexts", "labels", "meshes")
LABEL_ORDER = ("yes", "no", "maybe")

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Section:
    label: str
    text: str


@dataclass(frozen=True)
class Document:
    """What may be indexed: conclusion-free context only (ARCHITECTURE §2)."""

    pmid: str
    source: str
    sections: tuple[Section, ...]
    meshes: tuple[str, ...]


@dataclass(frozen=True)
class BenchmarkItem:
    """A labeled question for eval manifests. Never indexed."""

    pmid: str
    question: str
    final_decision: str


@dataclass(frozen=True)
class SourceRow:
    """One dataset row after load-time filtering. `final_decision` is None for unlabeled rows."""

    document: Document
    question: str
    final_decision: str | None


def dataset_path(config: str) -> Path:
    filename = f"{config}/train-00000-of-00001.parquet"
    return Path(
        hf_hub_download(DATASET_REPO, filename, repo_type="dataset", revision=DATASET_REVISION)
    )


def read_rows(path: Path, config: str) -> list[SourceRow]:
    """Read one parquet file, keeping only allowed fields."""
    columns = ["pubid", "question", "context"]
    if config == LABELED:
        columns.append("final_decision")
    rows = []
    for raw in pq.read_table(path, columns=columns).to_pylist():
        context = raw["context"]
        # Copy allowed keys only; `reasoning_*` inside `context` are discarded here.
        contexts, labels, meshes = (list(context[key]) for key in CONTEXT_KEYS)
        document = Document(
            pmid=str(raw["pubid"]),
            source=SOURCE_TAG[config],
            sections=tuple(
                Section(label, text) for label, text in zip(labels, contexts, strict=True)
            ),
            meshes=tuple(meshes),
        )
        rows.append(SourceRow(document, raw["question"], raw.get("final_decision")))
    return rows


def benchmark_item(row: SourceRow) -> BenchmarkItem:
    if row.final_decision not in LABEL_ORDER:
        raise ValueError(f"PMID {row.document.pmid}: unexpected label {row.final_decision!r}")
    return BenchmarkItem(row.document.pmid, row.question, row.final_decision)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args()

    labeled = read_rows(dataset_path(LABELED), LABELED)
    unlabeled = read_rows(dataset_path(UNLABELED), UNLABELED)
    items = [benchmark_item(row) for row in labeled]
    labels = Counter(item.final_decision for item in items)

    print(f"Dataset {DATASET_REPO} @ {DATASET_REVISION}")
    print(f"{LABELED}: {len(labeled):,} rows; {UNLABELED}: {len(unlabeled):,} rows")
    print("Label distribution: " + ", ".join(f"{lab} {labels[lab]}" for lab in LABEL_ORDER))


if __name__ == "__main__":
    main()
