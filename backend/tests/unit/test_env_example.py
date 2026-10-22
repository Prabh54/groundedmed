"""`.env.example` must list variable names only, never values (CLAUDE.md: never commit secrets)."""

from pathlib import Path

ENV_EXAMPLE = Path(__file__).resolve().parents[3] / ".env.example"


def test_env_example_has_names_only():
    lines = ENV_EXAMPLE.read_text(encoding="utf-8").splitlines()
    assignments = [line for line in lines if line.strip() and not line.lstrip().startswith("#")]
    assert assignments, ".env.example lists no variables"
    for line in assignments:
        name, sep, value = line.partition("=")
        assert sep == "=", f"not a NAME= line: {line!r}"
        assert name.strip().isidentifier(), f"bad variable name: {name!r}"
        assert value.strip() == "", f"{name.strip()} has a value in .env.example"
