"""Build the PubMedQA manifests and corpus.

Run from the repo root: uv run python -m scripts.prepare_dataset

Answer fields are dropped at load time: `long_answer` and `reasoning_*` are never read into
memory, and `final_decision` only lives on a `BenchmarkItem` (eval manifests), never on a
`Document` (index).
"""

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

DATASET_REPO = "qiaojin/PubMedQA"
DATASET_REVISION = "9001f2853fb87cab8d220904e0de81ac6973b318"
LABELED, UNLABELED = "pqa_labeled", "pqa_unlabeled"
SOURCE_TAG = {LABELED: "pqa_l", UNLABELED: "pqa_u"}

# Fields that carry the answer. Never indexed or shown to a runtime model.
ANSWER_FIELDS = ("long_answer", "final_decision", "reasoning_required_pred", "reasoning_free_pred")
# The only context keys a Document is built from.
CONTEXT_KEYS = ("contexts", "labels", "meshes")
LABEL_ORDER = ("yes", "no", "maybe")

# Split: per label, 20% test, next 10% dev, rest reserve.
SPLIT_SEED = 20261021
TEST_FRACTION, DEV_FRACTION = 0.2, 0.1
SPLITS = ("test", "dev", "reserve")

# Corpus: all labeled contexts + MeSH-similar unlabeled distractors.
CORPUS_VERSION = "v1-seed20261021"
N_DISTRACTORS = 1500
# A MeSH term tagging more than this share of unlabeled papers is generic (Humans, Female,
# Retrospective Studies, ...) and does not count as topic overlap.
GENERIC_MESH_MAX_SHARE = 0.03
# Unlabeled papers whose normalized question, or first 200 normalized passage characters,
# equal a labeled paper's are copies of it (e.g. 18270496, 17491578 copy 18019905).
DUPLICATE_PREFIX_CHARS = 200
# Each labeled paper keeps this many best candidates; far more than the ~2 rounds needed.
CANDIDATES_PER_PAPER = 50

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"


class FrozenManifestError(RuntimeError):
    """A committed manifest would change. Frozen files are never overwritten."""


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


def normalize(text: str) -> str:
    """Lowercase, punctuation and whitespace collapsed to single spaces (for matching only)."""
    return re.sub(r"\W+", " ", text.lower()).strip()


def passage_text(document: Document) -> str:
    """Exactly what gets embedded: section texts in order, no labels, no question."""
    return "\n\n".join(section.text for section in document.sections)


def generic_mesh_terms(unlabeled: list[Document]) -> set[str]:
    counts = Counter(term for doc in unlabeled for term in set(doc.meshes))
    return {term for term, n in counts.items() if n / len(unlabeled) > GENERIC_MESH_MAX_SHARE}


def labeled_duplicates(labeled: list[SourceRow], unlabeled: list[SourceRow]) -> set[str]:
    """PMIDs of unlabeled rows that copy a labeled paper (same question or same opening text)."""

    def prefix(row: SourceRow) -> str:
        return normalize(passage_text(row.document))[:DUPLICATE_PREFIX_CHARS]

    questions = {normalize(row.question) for row in labeled}
    prefixes = {prefix(row) for row in labeled}
    return {
        row.document.pmid
        for row in unlabeled
        if normalize(row.question) in questions or prefix(row) in prefixes
    }


def select_distractors(
    labeled: list[Document], candidates: list[Document], generic: set[str], n: int
) -> list[str]:
    """Round-robin over labeled papers (PMID order): each takes its best not-yet-chosen candidate,
    ranked by number of shared non-generic MeSH terms, ties by PMID. Returns PMIDs sorted."""
    by_term = defaultdict(list)
    for doc in candidates:
        for term in set(doc.meshes) - generic:
            by_term[term].append(doc.pmid)

    ranked = []
    for doc in sorted(labeled, key=pmid_key):
        shared = Counter(pmid for term in set(doc.meshes) - generic for pmid in by_term[term])
        best = sorted(shared.items(), key=lambda kv: (-kv[1], int(kv[0])))
        ranked.append([pmid for pmid, _ in best[:CANDIDATES_PER_PAPER]])

    chosen: set[str] = set()
    positions = [0] * len(ranked)
    while len(chosen) < n:
        progressed = False
        for i, options in enumerate(ranked):
            while positions[i] < len(options) and options[positions[i]] in chosen:
                positions[i] += 1
            if positions[i] < len(options):
                chosen.add(options[positions[i]])
                progressed = True
                if len(chosen) == n:
                    break
        if not progressed:
            raise ValueError(f"only {len(chosen)} distractor candidates available, need {n}")
    return sorted(chosen, key=int)


def corpus_record(document: Document) -> dict:
    passage = passage_text(document)
    return {
        "pmid": document.pmid,
        "source": document.source,
        "sections": [asdict(section) for section in document.sections],
        "passage": passage,
        "meshes": list(document.meshes),
        "content_sha256": sha256_text(passage),
        "corpus_version": CORPUS_VERSION,
    }


def build_corpus(labeled: list[SourceRow], unlabeled: list[SourceRow]) -> tuple[list[dict], dict]:
    """Returns (corpus records sorted by PMID, corpus manifest)."""
    duplicates = labeled_duplicates(labeled, unlabeled)
    unlabeled_docs = [row.document for row in unlabeled]
    generic = generic_mesh_terms(unlabeled_docs)
    candidates = [doc for doc in unlabeled_docs if doc.pmid not in duplicates]
    labeled_docs = [row.document for row in labeled]
    chosen = set(select_distractors(labeled_docs, candidates, generic, N_DISTRACTORS))

    documents = labeled_docs + [doc for doc in candidates if doc.pmid in chosen]
    records = [corpus_record(doc) for doc in sorted(documents, key=pmid_key)]
    manifest = {
        "corpus_version": CORPUS_VERSION,
        "dataset": DATASET_REPO,
        "revision": DATASET_REVISION,
        "count": len(records),
        "source_counts": dict(sorted(Counter(r["source"] for r in records).items())),
        "distractor_rule": {
            "n_distractors": N_DISTRACTORS,
            "generic_mesh_max_share": GENERIC_MESH_MAX_SHARE,
            "generic_mesh_terms": sorted(generic),
            "excluded_duplicates": sorted(duplicates, key=int),
        },
        "documents": [
            {"pmid": r["pmid"], "source": r["source"], "content_sha256": r["content_sha256"]}
            for r in records
        ],
    }
    return records, manifest


def to_json(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False) + "\n"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_frozen(files: dict[Path, str]) -> None:
    """Write manifests, refusing to change any that already exist. Checks all before writing any."""
    changed = [
        path
        for path, text in files.items()
        if path.exists() and path.read_text(encoding="utf-8") != text
    ]
    if changed:
        names = ", ".join(str(path) for path in changed)
        raise FrozenManifestError(f"refusing to overwrite frozen manifest(s): {names}")
    for path, text in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


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

    records, corpus_manifest = build_corpus(labeled, unlabeled)
    files[manifest_dir / "corpus.json"] = to_json(corpus_manifest)
    write_frozen(files)
    corpus_path = args.out_dir / "corpus.jsonl"  # regenerated every run; not committed
    corpus_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
    )

    for name in SPLITS:
        print(f"{name}: {len(splits[name])} questions {label_counts(splits[name])}")
    print(f"test.json sha256: {sha256_text(test_json)}")
    rule = corpus_manifest["distractor_rule"]
    print(
        f"corpus: {corpus_manifest['count']} documents {corpus_manifest['source_counts']}; "
        f"{len(rule['generic_mesh_terms'])} generic MeSH terms; "
        f"excluded duplicates {rule['excluded_duplicates']}"
    )
    print(
        f"Manifests in {manifest_dir} (unchanged if they already existed); corpus at {corpus_path}"
    )


if __name__ == "__main__":
    try:
        main()
    except FrozenManifestError as error:
        raise SystemExit(f"ERROR: {error}") from None
