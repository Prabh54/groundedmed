"""`.env.example` must list variable names only, never values."""

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
ENV_EXAMPLE = REPO_ROOT / ".env.example"


def test_env_example_has_names_only():
    lines = ENV_EXAMPLE.read_text(encoding="utf-8").splitlines()
    assignments = [line for line in lines if line.strip() and not line.lstrip().startswith("#")]
    assert assignments, ".env.example lists no variables"
    for line in assignments:
        name, sep, value = line.partition("=")
        assert sep == "=", f"not a NAME= line: {line!r}"
        assert name.strip().isidentifier(), f"bad variable name: {name!r}"
        assert value.strip() == "", f"{name.strip()} has a value in .env.example"


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)


@pytest.mark.parametrize(
    "name", [".env", ".env.local", ".env.save", ".env.production", "frontend/.env.local"]
)
def test_env_files_are_git_ignored(name):
    assert git("check-ignore", "-q", name).returncode == 0, f"{name} would not be ignored"


def test_env_example_is_not_ignored_and_no_other_env_file_is_tracked():
    assert git("check-ignore", "-q", ".env.example").returncode == 1
    tracked = [f for f in git("ls-files").stdout.splitlines() if Path(f).name.startswith(".env")]
    assert tracked == [".env.example"]
