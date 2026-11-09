"""Live checks of the loaded `documents` table and the read-only runtime role.

Needs DATABASE_URL (read-only runtime role): local .env, and a GitHub Actions secret so CI runs
them too.
The one test that inspects other roles also needs DATABASE_ADMIN_URL, which is never given to CI.
"""

import json
import os

import psycopg
import pytest

from backend.app.retrieval.vector_search import open_connection
from scripts.ingest import CORPUS_MANIFEST, DOCUMENT_COLUMNS, READER_ROLE, SUPABASE_API_ROLES

pytestmark = pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="DATABASE_URL not set")
needs_admin = pytest.mark.skipif(
    not os.environ.get("DATABASE_ADMIN_URL"), reason="DATABASE_ADMIN_URL not set (never set in CI)"
)


@pytest.fixture(scope="module")
def reader():
    with open_connection(os.environ["DATABASE_URL"]) as conn:
        yield conn


@pytest.fixture(scope="module")
def admin():
    with open_connection(os.environ["DATABASE_ADMIN_URL"]) as conn:
        yield conn


def manifest():
    return json.loads(CORPUS_MANIFEST.read_text(encoding="utf-8"))


def test_runtime_connection_is_the_read_only_role(reader):
    assert reader.execute("select current_user").fetchone()[0] == READER_ROLE
    assert reader.execute("show default_transaction_read_only").fetchone()[0] == "on"


def test_row_count_and_passages_match_frozen_manifest(reader):
    # The hash is computed by Postgres from the stored passage, so this proves the DB text itself.
    rows = reader.execute(
        "select pmid, encode(sha256(convert_to(passage, 'UTF8')), 'hex'), content_sha256 "
        "from documents"
    ).fetchall()
    expected = {d["pmid"]: d["content_sha256"] for d in manifest()["documents"]}
    assert len(rows) == manifest()["count"] == len(expected)
    assert {pmid: computed for pmid, computed, _ in rows} == expected
    assert all(computed == stored for _, computed, stored in rows)


def test_table_has_only_the_allowed_columns(reader):
    columns = reader.execute(
        "select column_name from information_schema.columns "
        "where table_schema = 'public' and table_name = 'documents' order by ordinal_position"
    ).fetchall()
    names = tuple(c[0] for c in columns)
    assert names == DOCUMENT_COLUMNS
    forbidden = {"source", "question", "title", "final_decision", "long_answer", "label", "split"}
    assert forbidden.isdisjoint(names) and not any(n.startswith("reasoning") for n in names)


def test_embeddings_are_complete(reader):
    models, dims, missing = reader.execute(
        "select array_agg(distinct embedding_model), array_agg(distinct vector_dims(embedding)), "
        "count(*) filter (where embedding is null) from documents"
    ).fetchone()
    assert models == ["voyage-4"] and dims == [1024] and missing == 0


@pytest.mark.parametrize(
    "statement",
    [
        "insert into documents (pmid, sections, passage, meshes, content_sha256, corpus_version, "
        "embedding_model, embedding) select 'x', sections, passage, meshes, content_sha256, "
        "corpus_version, embedding_model, embedding from documents limit 1",
        "update documents set passage = 'tampered' where pmid = (select min(pmid) from documents)",
        "delete from documents where pmid = (select min(pmid) from documents)",
        "create table reader_probe (id int)",
    ],
)
def test_runtime_role_cannot_write(statement):
    with open_connection(os.environ["DATABASE_URL"]) as conn:
        with pytest.raises(
            (psycopg.errors.InsufficientPrivilege, psycopg.errors.ReadOnlySqlTransaction)
        ):
            conn.execute(statement)


def test_runtime_role_cannot_write_even_outside_read_only_mode():
    """Privileges alone block INSERT, not just the read-only default."""
    with open_connection(os.environ["DATABASE_URL"]) as conn:
        conn.execute("set transaction read write")
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            conn.execute(
                "insert into documents (pmid, sections, passage, meshes, content_sha256, "
                "corpus_version, embedding_model, embedding) select 'x', sections, passage, "
                "meshes, content_sha256, corpus_version, embedding_model, embedding "
                "from documents limit 1"
            )


@needs_admin
def test_supabase_api_roles_cannot_read_documents(admin):
    for role in SUPABASE_API_ROLES:
        if admin.execute("select 1 from pg_roles where rolname = %s", (role,)).fetchone():
            assert not admin.execute(
                "select has_table_privilege(%s, 'public.documents', 'SELECT')", (role,)
            ).fetchone()[0]


def test_runtime_role_has_no_write_privileges(reader):
    """Grants alone forbid writes (row-level security also blocks them, but must not have to)."""
    granted = reader.execute(
        "select p from unnest(array['INSERT','UPDATE','DELETE','TRUNCATE']) p "
        "where has_table_privilege(current_user, 'public.documents', p)"
    ).fetchall()
    assert granted == []
    assert not reader.execute(
        "select has_schema_privilege(current_user, 'public', 'CREATE')"
    ).fetchone()[0]
