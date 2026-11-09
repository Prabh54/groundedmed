"""Embed the frozen corpus and load it into Postgres + pgvector.

Run: uv run python -m scripts.ingest

The only text sent to Voyage is each record's `passage` (conclusion-free section texts). PMIDs,
sources, MeSH terms and section labels are never embedded. Embeddings are cached in
data/cache/embeddings/<model>.jsonl keyed by content SHA-256, so a re-run embeds 0 new tokens.

Loading uses DATABASE_ADMIN_URL (the only write connection). It creates the `documents` table
(ARCHITECTURE §2; no `source` column) and a read-only login role for the runtime. The
first time, that role gets a random password and DATABASE_URL is written to .env, never printed.
"""

import hashlib
import json
import os
import secrets
from pathlib import Path
from urllib.parse import quote, urlsplit

import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

from backend.app.config import (
    EMBEDDING_DIMS,
    EMBEDDING_MODEL,
    PRICE_PER_M_TOKENS,
    REPO_ROOT,
    env,
    redact,
)
from backend.app.retrieval.embed import embed_texts, voyage_client
from backend.app.retrieval.vector_search import open_connection
from scripts.set_env import ENV_FILE, set_value

CORPUS_FILE = REPO_ROOT / "data" / "corpus.jsonl"
CORPUS_MANIFEST = REPO_ROOT / "data" / "manifests" / "corpus.json"
CACHE_DIR = REPO_ROOT / "data" / "cache" / "embeddings"
BATCH_SIZE = 128  # ~41k tokens per request at our ~320-token average passage

READER_ROLE = "groundedmed_reader"
# Supabase's API roles get table access by default; the documents table is not for them.
SUPABASE_API_ROLES = ("anon", "authenticated")
# Exactly the columns retrieval may see (ARCHITECTURE §2). Tests compare the live table to this.
DOCUMENT_COLUMNS = (
    "pmid",
    "sections",
    "passage",
    "meshes",
    "content_sha256",
    "corpus_version",
    "embedding_model",
    "embedding",
)
SCHEMA_SQL = f"""
create extension if not exists vector;
create table if not exists documents (
  pmid            text primary key,
  sections        jsonb not null,
  passage         text not null,
  meshes          text[] not null,
  content_sha256  text not null,
  corpus_version  text not null,
  embedding_model text not null,
  embedding       vector({EMBEDDING_DIMS}) not null
);
alter table documents enable row level security;
"""
UPSERT_SQL = """
insert into documents (pmid, sections, passage, meshes, content_sha256, corpus_version,
                       embedding_model, embedding)
values (%s, %s, %s, %s, %s, %s, %s, %s::vector)
on conflict (pmid) do update set
  sections = excluded.sections, passage = excluded.passage, meshes = excluded.meshes,
  content_sha256 = excluded.content_sha256, corpus_version = excluded.corpus_version,
  embedding_model = excluded.embedding_model, embedding = excluded.embedding
"""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_corpus(corpus_file: Path = CORPUS_FILE, manifest_file: Path = CORPUS_MANIFEST) -> list:
    """Read corpus.jsonl and check it is exactly the frozen manifest (same PMIDs and hashes)."""
    with corpus_file.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    expected = {d["pmid"]: d["content_sha256"] for d in manifest["documents"]}
    actual = {r["pmid"]: sha256_text(r["passage"]) for r in records}
    if len(records) != manifest["count"] or actual != expected:
        raise RuntimeError("data/corpus.jsonl does not match the frozen corpus manifest")
    return records


def texts_to_embed(records: list[dict]) -> dict[str, str]:
    """content_sha256 -> exactly the text that is embedded (the passage, nothing else)."""
    return {sha256_text(r["passage"]): r["passage"] for r in records}


def cache_file(model: str = EMBEDDING_MODEL, cache_dir: Path = CACHE_DIR) -> Path:
    return cache_dir / f"{model}.jsonl"


def load_cache(path: Path) -> dict[str, list[float]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return {row["sha256"]: row["embedding"] for row in map(json.loads, f)}


def embed_missing(texts: dict[str, str], path: Path, embed=None) -> tuple[dict, int]:
    """Embed texts not yet in the cache, appending each batch as it completes.

    Returns (sha256 -> embedding for every text, new tokens billed). `embed` is injectable for
    tests; it defaults to voyage-4 with input_type="document".
    """
    if embed is None:
        client = voyage_client(max_retries=3, timeout=60)

        def embed(batch: list[str]) -> tuple[list[list[float]], int]:
            return embed_texts(client, batch, input_type="document")

    cache = load_cache(path)
    missing = [sha for sha in texts if sha not in cache]
    tokens = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(missing), BATCH_SIZE):
        batch = missing[start : start + BATCH_SIZE]
        vectors, used = embed([texts[sha] for sha in batch])
        tokens += used
        with path.open("a", encoding="utf-8") as f:
            for sha, vector in zip(batch, vectors, strict=True):
                f.write(json.dumps({"sha256": sha, "embedding": vector}) + "\n")
                cache[sha] = vector
        print(f"  embedded {min(start + BATCH_SIZE, len(missing))}/{len(missing)}", flush=True)
    return {sha: cache[sha] for sha in texts}, tokens


def vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"


def document_rows(records: list[dict], embeddings: dict[str, list[float]]) -> list[tuple]:
    """One row per record, in DOCUMENT_COLUMNS order. `source` is deliberately not a column."""
    rows = []
    for r in records:
        sha = sha256_text(r["passage"])
        rows.append(
            (
                r["pmid"],
                Jsonb(r["sections"]),
                r["passage"],
                r["meshes"],
                sha,
                r["corpus_version"],
                EMBEDDING_MODEL,
                vector_literal(embeddings[sha]),
            )
        )
    return rows


def reader_url(admin_url: str, password: str) -> str:
    """The admin URL with the read-only role and its password (Supabase pooler user: role.ref)."""
    parts = urlsplit(admin_url)
    project_ref = parts.username.partition(".")[2]
    pooled = (parts.hostname or "").endswith(".pooler.supabase.com") and project_ref
    user = f"{READER_ROLE}.{project_ref}" if pooled else READER_ROLE
    host = parts.netloc.rpartition("@")[2]  # keeps the port and any IPv6 brackets as written
    netloc = f"{user}:{quote(password, safe='')}@{host}"
    return parts._replace(netloc=netloc).geturl()


def create_schema(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA_SQL)
    present = {row[0] for row in conn.execute("select rolname from pg_roles").fetchall()}
    for role in SUPABASE_API_ROLES:
        if role in present:
            conn.execute(sql.SQL("revoke all on documents from {}").format(sql.Identifier(role)))


def ensure_reader_role(conn: psycopg.Connection, new_password: str | None) -> None:
    """Create/refresh the read-only role: SELECT on documents only, read-only transactions.

    `new_password` is set when given (first setup); otherwise the existing password is kept.
    """
    role = sql.Identifier(READER_ROLE)
    exists = conn.execute("select 1 from pg_roles where rolname = %s", (READER_ROLE,)).fetchone()
    if not exists:
        if new_password is None:
            raise RuntimeError(f"role {READER_ROLE} does not exist and no password was given")
        conn.execute(sql.SQL("create role {} login").format(role))
    if new_password is not None:
        conn.execute(
            sql.SQL("alter role {} with login password {}").format(role, sql.Literal(new_password))
        )
    for statement in (
        "alter role {} set default_transaction_read_only = on",
        "grant usage on schema public to {}",
        "grant select on documents to {}",
        "drop policy if exists reader_select on documents",
        "create policy reader_select on documents for select to {} using (true)",
    ):
        conn.execute(sql.SQL(statement).format(role))


def load_documents(conn: psycopg.Connection, rows: list[tuple]) -> int:
    """Upsert all rows; refuse (without deleting anything) if the table holds unknown PMIDs."""
    with conn.cursor() as cur:
        cur.executemany(UPSERT_SQL, rows)
    pmids = [row[0] for row in rows]
    extra = conn.execute(
        "select count(*) from documents where pmid <> all(%s)", (pmids,)
    ).fetchone()[0]
    if extra:
        raise RuntimeError(f"documents holds {extra} rows not in the frozen corpus manifest")
    return conn.execute("select count(*) from documents").fetchone()[0]


def main() -> None:
    records = load_corpus()
    texts = texts_to_embed(records)
    path = cache_file()
    print(f"Corpus: {len(records)} documents; cached embeddings: {len(load_cache(path))}")
    embeddings, tokens = embed_missing(texts, path)
    cost = tokens * PRICE_PER_M_TOKENS[EMBEDDING_MODEL] / 1e6
    print(f"New tokens embedded: {tokens} (list price ${cost:.4f}; free allocation applies)")
    if len(embeddings) != len(records):
        raise RuntimeError(f"{len(embeddings)} embeddings for {len(records)} documents")
    print(f"Embeddings: {len(embeddings)} == corpus count {len(records)}")

    admin_url = env("DATABASE_ADMIN_URL")
    new_password = None if os.environ.get("DATABASE_URL", "").strip() else secrets.token_urlsafe(32)
    try:
        with open_connection(admin_url) as conn:  # one transaction
            create_schema(conn)
            ensure_reader_role(conn, new_password)
            count = load_documents(conn, document_rows(records, embeddings))
    except psycopg.Error as exc:  # driver errors can echo connection strings
        raise SystemExit(f"Database error: {redact(str(exc))}") from None
    if count != len(records):
        raise SystemExit(f"documents has {count} rows, manifest has {len(records)}")
    print(f"documents table: {count} rows == manifest count {len(records)}")
    if new_password is not None:
        text = ENV_FILE.read_text(encoding="utf-8")
        ENV_FILE.write_text(
            set_value(text, "DATABASE_URL", reader_url(admin_url, new_password)), "utf-8"
        )
        print(f"Read-only role {READER_ROLE} created; DATABASE_URL written to .env (not shown)")


if __name__ == "__main__":
    main()
