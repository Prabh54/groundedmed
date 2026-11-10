"""Derive the second frozen test set (test2) from the reserve questions (ADR-043).

Run once, from the repo root:  uv run python -m scripts.lock_test2

Rule: drop the ADR-025 leakage exception; per label (yes, no, maybe), sort by PMID as a number,
shuffle with a fresh random.Random(TEST2_SEED), take TEST2_COUNTS. Writes data/manifests/test2.json
and test2.sha256, refusing to change existing files (a re-run is a reproducibility check). Prints
counts and the hash only, never a question or a label of any item: nobody looks at test2 until its
one final run.
"""

import hashlib
import json
import random
from pathlib import Path

from scripts.prepare_dataset import (
    DATASET_REPO,
    DATASET_REVISION,
    LABEL_ORDER,
    sha256_text,
    to_json,
    write_frozen,
)

MANIFESTS = Path(__file__).resolve().parents[1] / "data" / "manifests"
TEST2_SEED = 20261110
TEST2_COUNTS = {"yes": 220, "no": 136, "maybe": 44}  # 2 x the test set's 110 / 68 / 22
# ADR-025 leakage exception 1; ADR-026 relied on it staying out of test.
EXCLUDED = ("24519615",)


def build_test2(reserve_bytes: bytes) -> dict:
    reserve = json.loads(reserve_bytes)
    if reserve["split"] != "reserve":
        raise ValueError("source is not the reserve manifest")
    items = [item for item in reserve["items"] if item["pmid"] not in EXCLUDED]
    chosen = []
    for label in LABEL_ORDER:
        group = sorted(
            (i for i in items if i["final_decision"] == label), key=lambda i: int(i["pmid"])
        )
        random.Random(TEST2_SEED).shuffle(group)
        if len(group) < TEST2_COUNTS[label]:
            raise ValueError(f"not enough reserve items labelled {label}")
        chosen += group[: TEST2_COUNTS[label]]
    return {
        "split": "test2",
        "dataset": DATASET_REPO,
        "revision": DATASET_REVISION,
        "seed": TEST2_SEED,
        "source": {"file": "reserve.json", "sha256": hashlib.sha256(reserve_bytes).hexdigest()},
        "excluded": list(EXCLUDED),
        "count": len(chosen),
        "label_counts": dict(TEST2_COUNTS),
        "items": sorted(chosen, key=lambda i: int(i["pmid"])),
    }


def main() -> None:
    text = to_json(build_test2((MANIFESTS / "reserve.json").read_bytes()))
    digest = sha256_text(text)
    write_frozen(
        {MANIFESTS / "test2.json": text, MANIFESTS / "test2.sha256": f"{digest}  test2.json\n"}
    )
    print(f"test2: {json.loads(text)['count']} questions {TEST2_COUNTS}; sha256 {digest}")


if __name__ == "__main__":
    main()
