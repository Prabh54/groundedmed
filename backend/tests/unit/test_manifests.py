"""Integrity tests on the committed, frozen manifests (M0-T3, ADR-008). Offline."""

import hashlib
import json
from pathlib import Path

import pytest

from scripts import prepare_dataset as pd

MANIFESTS = Path(__file__).resolve().parents[3] / "data" / "manifests"

# Frozen at M0-T3 (2026-10-21). Changing the test set requires a new ADR; never edit this to match.
FROZEN_TEST_SHA256 = "41f10a29ddc5126c62dcfe8e6d1141979e35022be246dca8c51eaa51ae224f2f"


def load(name):
    return json.loads((MANIFESTS / f"{name}.json").read_text(encoding="utf-8"))


def test_frozen_test_ids_match_manifest_hash():
    digest = hashlib.sha256((MANIFESTS / "test.json").read_bytes()).hexdigest()
    recorded = (MANIFESTS / "test.sha256").read_text(encoding="utf-8").split()
    assert recorded == [digest, "test.json"]
    assert digest == FROZEN_TEST_SHA256


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("test", {"yes": 110, "no": 68, "maybe": 22}),
        ("dev", {"yes": 55, "no": 34, "maybe": 11}),
        ("reserve", {"yes": 387, "no": 236, "maybe": 77}),
    ],
)
def test_label_mix(name, expected):
    manifest = load(name)
    actual = {label: 0 for label in expected}
    for item in manifest["items"]:
        actual[item["final_decision"]] += 1
    assert actual == expected == manifest["label_counts"]
    assert manifest["count"] == len(manifest["items"]) == sum(expected.values())


def test_splits_do_not_overlap_and_cover_all_labeled():
    ids = {name: [item["pmid"] for item in load(name)["items"]] for name in pd.SPLITS}
    for members in ids.values():
        assert len(members) == len(set(members))
    test, dev, reserve = (set(ids[name]) for name in pd.SPLITS)
    assert not test & dev
    assert not test & reserve
    assert not dev & reserve
    assert len(test | dev | reserve) == 1000


def test_manifests_record_provenance_and_carry_no_conclusions():
    for name in pd.SPLITS:
        manifest = load(name)
        assert manifest["split"] == name
        assert manifest["dataset"] == pd.DATASET_REPO
        assert manifest["revision"] == pd.DATASET_REVISION
        assert manifest["seed"] == pd.SPLIT_SEED
        for item in manifest["items"]:
            assert set(item) == {"pmid", "question", "final_decision"}
