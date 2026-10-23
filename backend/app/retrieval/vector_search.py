"""Exact cosine search in pgvector (ARCHITECTURE §4: no ANN index at 2,500 rows)."""

import psycopg

from backend.app.config import env, redact
from backend.app.models.retrieval import Candidate

SEARCH_SQL = """
select pmid, passage, sections, 1 - (embedding <=> %(q)s::vector) as score
from documents
order by embedding <=> %(q)s::vector, pmid
limit %(k)s
"""


def open_connection(url: str, **kwargs) -> psycopg.Connection:
    """Connect; driver errors are re-raised redacted (libpq can echo the URL with its password)."""
    try:
        return psycopg.connect(url, connect_timeout=15, sslmode="require", **kwargs)
    except psycopg.Error as exc:
        raise RuntimeError(f"database connection failed: {redact(str(exc))}") from None


def connect() -> psycopg.Connection:
    """Runtime connection: always the read-only role (DATABASE_URL)."""
    return open_connection(env("DATABASE_URL"), autocommit=True)


def dense_search(conn: psycopg.Connection, query_vector: list[float], k: int) -> list[Candidate]:
    literal = "[" + ",".join(repr(float(x)) for x in query_vector) + "]"
    rows = conn.execute(SEARCH_SQL, {"q": literal, "k": k}).fetchall()
    return [
        Candidate(pmid=pmid, passage=passage, sections=sections, dense_score=score, dense_rank=i)
        for i, (pmid, passage, sections, score) in enumerate(rows, start=1)
    ]


def get_document(conn: psycopg.Connection, pmid: str) -> dict | None:
    row = conn.execute(
        "select pmid, passage, sections, meshes from documents where pmid = %s", (pmid,)
    ).fetchone()
    if row is None:
        return None
    return {"pmid": row[0], "passage": row[1], "sections": row[2], "meshes": row[3]}
