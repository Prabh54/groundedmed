"""Frozen independent-judge rubric v1 (eval/rubric/judge_v1.md): files, hashes, prompt sections.

The judge runner (M4-T2) loads its system instruction, templates and response schema from here, and
records `file_hashes()` in each judge run's config. No API calls.
"""

import hashlib
import json
import re

from backend.app.config import REPO_ROOT

JUDGE_VERSION = "judge_v1"
RUBRIC_DIR = REPO_ROOT / "eval" / "rubric"
JUDGE_FILES = ("judge_v1.md", "judge_v1_prompt.txt", "judge_v1_schema.json")
HASH_FILE = RUBRIC_DIR / "judge_v1.sha256"
LABELS = ("supported", "partially_supported", "unsupported")
SYSTEM_LABELS = ("SYSTEM_A", "SYSTEM_B")
SEED = 20261024
_SECTION = re.compile(r"^=== ([A-Z_]+) ===\n", re.MULTILINE)


def file_hashes() -> dict[str, str]:
    """SHA-256 of each frozen file as it is on disk now."""
    return {
        name: hashlib.sha256((RUBRIC_DIR / name).read_bytes()).hexdigest() for name in JUDGE_FILES
    }


def frozen_hashes() -> dict[str, str]:
    """The hashes committed in judge_v1.sha256 (`shasum -a 256` format)."""
    lines = HASH_FILE.read_text(encoding="utf-8").splitlines()
    return {name: digest for digest, name in (line.split() for line in lines if line.strip())}


def check_frozen() -> None:
    if file_hashes() != frozen_hashes():
        raise SystemExit("Judge rubric v1 files do not match judge_v1.sha256; refusing to judge.")


def load_prompt() -> dict[str, str]:
    """Sections of judge_v1_prompt.txt by name; each without its final newline."""
    text = (RUBRIC_DIR / "judge_v1_prompt.txt").read_text(encoding="utf-8")
    parts = _SECTION.split(text)
    if parts[0].strip():
        raise ValueError("text before the first section header")
    return {
        name: body.removesuffix("\n") for name, body in zip(parts[1::2], parts[2::2], strict=True)
    }


def load_schema() -> dict:
    return json.loads((RUBRIC_DIR / "judge_v1_schema.json").read_text(encoding="utf-8"))
