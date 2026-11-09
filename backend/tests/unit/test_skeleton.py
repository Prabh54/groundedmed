"""The package layout from ARCHITECTURE.md §1 imports from the repo root."""

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

PACKAGES = [
    "backend",
    "backend.app",
    "backend.app.models",
    "backend.app.retrieval",
    "backend.app.llm",
    "backend.app.agents",
    "backend.app.baseline",
    "backend.app.guardrails",
    "eval",
    "eval.runners",
    "eval.metrics",
    "scripts",
]


@pytest.mark.parametrize("name", PACKAGES)
def test_package_imports(name):
    module = importlib.import_module(name)
    # A folder without __init__.py still imports (namespace package) but has no __file__.
    assert module.__file__ and module.__file__.endswith("__init__.py"), f"{name} lacks __init__.py"


def test_packages_import_outside_pytest():
    """Scripts and uvicorn run as plain Python from the repo root, without pytest's pythonpath."""
    code = "import " + ", ".join(PACKAGES)
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
