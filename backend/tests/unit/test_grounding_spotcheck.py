"""Grounding spot-check v1 is frozen and reproduces the baseline (grounding_spotcheck_v1.md)."""

import hashlib
import json

import pytest

from backend.app.config import REPO_ROOT
from eval import grounding_spotcheck as gs

BASELINE_RUN = "dev-baseline-20261023-033422-9396a90"
JUDGMENTS = gs.AUDIT_DIR / f"{BASELINE_RUN}-judgments.jsonl"


def load_judgments():
    return [json.loads(line) for line in JUDGMENTS.open(encoding="utf-8")]


def test_method_v1_is_frozen():
    """Changing the sample or the rubric must be a new method version, not an edit to v1."""
    sample = hashlib.sha256(",".join(gs.FROZEN_PMIDS).encode()).hexdigest()
    rubric = hashlib.sha256((REPO_ROOT / "eval/rubric/grounding_spotcheck_v1.md").read_bytes())
    assert gs.METHOD == "grounding_spotcheck_v1"
    assert sample == "dbbb71c86e0556cd9f4a175bd0a927368dcb40a8ba48927997c37cb5c9007ec8"
    assert rubric.hexdigest() == "c1f30a622e14ae36ea5bcaa66fc92a62f1e08e4935a62251e5149bcdef9c31a8"


def test_baseline_reproduces_8_of_88():
    result = gs.score(BASELINE_RUN, load_judgments())
    assert result["items"] == 89 and result["judged_items"] == 88 and result["not_judged"] == 1
    assert result["flagged_items"] == 8 and result["rows_with_flagged_items"] == 6
    assert result["counts"] == {"supported": 80, "partly": 5, "unsupported": 3, "not_judged": 1}
    stored = json.loads((gs.AUDIT_DIR / f"{BASELINE_RUN}-score.json").read_text(encoding="utf-8"))
    assert stored["flagged_items"] == 8 and stored["judged_items"] == 88


def test_score_refuses_judgments_that_do_not_match_the_items():
    judgments = load_judgments()
    with pytest.raises(SystemExit):
        gs.score(BASELINE_RUN, judgments[:-1])  # one item missing
    with pytest.raises(SystemExit):
        gs.score(BASELINE_RUN, judgments + judgments[:1])  # duplicate
    edited = [dict(j) for j in judgments]
    edited[0]["text_sha"] = "000000000000"  # judged a different text than the run stored
    with pytest.raises(SystemExit):
        gs.score(BASELINE_RUN, edited)
    relabeled = [dict(j) for j in judgments]
    relabeled[0]["label"] = "mostly"
    with pytest.raises(SystemExit):
        gs.score(BASELINE_RUN, relabeled)
