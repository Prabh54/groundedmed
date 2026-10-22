"""Build the PubMedQA manifests and corpus (ADR-008, ADR-009).

Run from the repo root: uv run python -m scripts.prepare_dataset

Answer fields are dropped at load time: `long_answer` and `reasoning_*` are never read into
memory, and `final_decision` only lives on a `BenchmarkItem` (eval manifests), never on a
`Document` (index).
"""

import argparse
import hashlib
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
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

# Split (ADR-008): per label, 20% test, next 10% dev, rest reserve.
SPLIT_SEED = 20261021
TEST_FRACTION, DEV_FRACTION = 0.2, 0.1
SPLITS = ("test", "dev", "reserve")

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"


class FrozenManifestError(RuntimeError):
    """A committed manifest would change. Frozen files are never overwritten (ADR-008)."""


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


def pmid_key(obj) -> int:
    return int(obj.pmid)


def split_labeled(items: list[BenchmarkItem], seed: int = SPLIT_SEED) -> dict[str, list]:
    """Stratified split. Each label group is sorted by PMID, then shuffled by a fresh
    `random.Random(seed)`, in LABEL_ORDER; each split is returned sorted by PMID."""
    splits = {name: [] for name in SPLITS}
    for label in LABEL_ORDER:
        group = sorted((item for item in items if item.final_decision == label), key=pmid_key)
        random.Random(seed).shuffle(group)
        n_test = round(len(group) * TEST_FRACTION)
        n_dev = round(len(group) * DEV_FRACTION)
        splits["test"] += group[:n_test]
        splits["dev"] += group[n_test : n_test + n_dev]
        splits["reserve"] += group[n_test + n_dev :]
    return {name: sorted(members, key=pmid_key) for name, members in splits.items()}


def label_counts(items: list[BenchmarkItem]) -> dict[str, int]:
    counts = Counter(item.final_decision for item in items)
    return {label: counts[label] for label in LABEL_ORDER}


def split_manifest(name: str, items: list[BenchmarkItem]) -> dict:
    return {
        "split": name,
        "dataset": DATASET_REPO,
        "revision": DATASET_REVISION,
        "seed": SPLIT_SEED,
        "count": len(items),
        "label_counts": label_counts(items),
        "items": [asdict(item) for item in items],
    }


def to_json(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False) + "\n"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_frozen(files: dict[Path, str]) -> None:
    """Write manifests, refusing to change any that already exist. Checks all before writing any."""
    changed = [path for path, text in files.items() if path.exists() and path.read_text() != text]
    if changed:
        names = ", ".join(str(path) for path in changed)
        raise FrozenManifestError(f"refusing to overwrite frozen manifest(s): {names}")
    for path, text in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR, help="default: data/")
    args = parser.parse_args()
    manifest_dir = args.out_dir / "manifests"

    labeled = read_rows(dataset_path(LABELED), LABELED)
    unlabeled = read_rows(dataset_path(UNLABELED), UNLABELED)
    items = [benchmark_item(row) for row in labeled]

    print(f"Dataset {DATASET_REPO} @ {DATASET_REVISION}")
    print(f"{LABELED}: {len(labeled):,} rows; {UNLABELED}: {len(unlabeled):,} rows")
    print(f"Label distribution: {label_counts(items)}")

    splits = split_labeled(items)
    files = {
        manifest_dir / f"{name}.json": to_json(split_manifest(name, splits[name]))
        for name in SPLITS
    }
    test_json = files[manifest_dir / "test.json"]
    files[manifest_dir / "test.sha256"] = f"{sha256_text(test_json)}  test.json\n"
    write_frozen(files)

    for name in SPLITS:
        print(f"{name}: {len(splits[name])} questions {label_counts(splits[name])}")
    print(f"test.json sha256: {sha256_text(test_json)}")
    print(f"Manifests written to {manifest_dir} (unchanged if they already existed)")


if __name__ == "__main__":
    try:
        main()
    except FrozenManifestError as error:
        raise SystemExit(f"ERROR: {error}") from None
