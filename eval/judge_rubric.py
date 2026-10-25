"""Frozen independent-judge rubric (current: v2, eval/rubric/judge_v2.md): files, hashes, prompt.

v2 replaced v1 before any judging (ADR-033/034); both stay frozen. The prompt and schema files are
the v1 files in both versions. The judge runner (M4-T2) loads its system instruction, templates and
response schema from here, and records `file_hashes()` in each judge run's config. No API calls.
"""

import hashlib
import json
import re

from backend.app.config import REPO_ROOT

RUBRIC_DIR = REPO_ROOT / "eval" / "rubric"
VERSIONS = {
    "judge_v1": ("judge_v1.md", "judge_v1_prompt.txt", "judge_v1_schema.json"),
    "judge_v2": ("judge_v2.md", "judge_v1_prompt.txt", "judge_v1_schema.json"),
}
JUDGE_VERSION = "judge_v2"  # the version the judge runner uses
LABELS = ("supported", "partially_supported", "unsupported")
SYSTEM_LABELS = ("SYSTEM_A", "SYSTEM_B")
SEED = 20261024
_SECTION = re.compile(r"^=== ([A-Z_]+) ===\n", re.MULTILINE)


def file_hashes(version: str = JUDGE_VERSION) -> dict[str, str]:
    """SHA-256 of each of the version's frozen files as it is on disk now."""
    return {
        name: hashlib.sha256((RUBRIC_DIR / name).read_bytes()).hexdigest()
        for name in VERSIONS[version]
    }


def frozen_hashes(version: str = JUDGE_VERSION) -> dict[str, str]:
    """The hashes committed in <version>.sha256 (`shasum -a 256` format)."""
    lines = (RUBRIC_DIR / f"{version}.sha256").read_text(encoding="utf-8").splitlines()
    return {name: digest for digest, name in (line.split() for line in lines if line.strip())}


def check_frozen(version: str = JUDGE_VERSION) -> None:
    if file_hashes(version) != frozen_hashes(version):
        raise SystemExit(f"{version} files do not match {version}.sha256; refusing to judge.")


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
