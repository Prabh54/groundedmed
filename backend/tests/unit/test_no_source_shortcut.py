"""ADR-026: runtime code must not know which papers are labeled (only `pqa_l` papers can be gold).

`source` stays in corpus.jsonl / corpus.json for auditing; it is not loaded into the runtime
database and no runtime code may refer to it. This scans `backend/app/` (runtime code and prompts).
"""

from pathlib import Path

APP = Path(__file__).resolve().parents[2] / "app"
FORBIDDEN = ("pqa_l", "pqa_u", "pqa_labeled", "pqa_unlabeled")


def test_runtime_code_has_no_labeled_unlabeled_marker():
    offenders = [
        f"{path.relative_to(APP)}: {marker}"
        for path in sorted(APP.rglob("*"))
        if path.is_file() and path.suffix in {".py", ".txt", ".md", ".sql", ".json"}
        for marker in FORBIDDEN
        if marker in path.read_text(encoding="utf-8")
    ]
    assert offenders == []
