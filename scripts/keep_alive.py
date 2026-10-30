"""Supabase Free keep-alive: one read-only query so the project is not paused for inactivity.

Run: uv run python -m scripts.keep_alive  (also every 3 days: .github/workflows/keepalive.yml)
Uses the read-only runtime role (DATABASE_URL); prints the row count only. Errors are redacted.
"""

import sys

from backend.app.config import redact
from backend.app.retrieval.vector_search import connect


def main() -> int:
    try:
        with connect() as conn:
            rows = conn.execute("select count(*) from documents").fetchone()[0]
    except Exception as exc:  # never print a connection string
        print(f"keep-alive FAILED: {redact(f'{type(exc).__name__}: {exc}')[:300]}")
        return 1
    print(f"keep-alive OK: documents has {rows} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
