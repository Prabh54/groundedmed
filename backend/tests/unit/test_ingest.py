"""Embedding cache and embedded-text rules for scripts/ingest.py (PLAN.md M1-T1)."""

import json

import pytest

from scripts.ingest import CORPUS_FILE, embed_missing, load_corpus, sha256_text, texts_to_embed


def fake_records():
    return [
        {
            "pmid": "111",
            "source": "pqa_l",
            "sections": [
                {"label": "BACKGROUND", "text": "Alpha."},
                {"label": "RESULTS", "text": "B."},
            ],
            "passage": "Alpha.\n\nB.",
            "meshes": ["Humans"],
        },
        {
            "pmid": "222",
            "source": "pqa_u",
            "sections": [{"label": "METHODS", "text": "Gamma."}],
            "passage": "Gamma.",
            "meshes": [],
        },
    ]


class FakeEmbedder:
    def __init__(self):
        self.calls: list[list[str]] = []

    def __call__(self, batch):
        self.calls.append(batch)
        return [[float(len(t))] for t in batch], 10 * len(batch)


def test_only_the_passage_is_embedded():
    texts = texts_to_embed(fake_records())
    assert set(texts.values()) == {"Alpha.\n\nB.", "Gamma."}
    for text in texts.values():
        for forbidden in ("111", "222", "pqa_l", "pqa_u", "BACKGROUND", "RESULTS", "Humans"):
            assert forbidden not in text


def test_rerun_embeds_zero_new_tokens(tmp_path):
    path = tmp_path / "voyage-4.jsonl"
    texts = texts_to_embed(fake_records())
    first = FakeEmbedder()
    embeddings, tokens = embed_missing(texts, path, embed=first)
    assert tokens == 20 and len(embeddings) == 2 and len(first.calls) == 1

    second = FakeEmbedder()
    again, tokens = embed_missing(texts, path, embed=second)
    assert tokens == 0 and second.calls == [] and again == embeddings


def test_cache_is_keyed_by_content_hash(tmp_path):
    path = tmp_path / "voyage-4.jsonl"
    embed_missing(texts_to_embed(fake_records()), path, embed=FakeEmbedder())
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert {r["sha256"] for r in rows} == {sha256_text("Alpha.\n\nB."), sha256_text("Gamma.")}

    changed = fake_records()
    changed[1]["passage"] = "Gamma, edited."
    embedder = FakeEmbedder()
    _, tokens = embed_missing(texts_to_embed(changed), path, embed=embedder)
    assert embedder.calls == [["Gamma, edited."]] and tokens == 10


@pytest.mark.skipif(not CORPUS_FILE.exists(), reason="run scripts.prepare_dataset first")
def test_real_corpus_embedded_text_has_no_ids_or_source_labels():
    records = load_corpus()
    texts = texts_to_embed(records)
    assert len(texts) == len(records)  # one embedding per document, no shared passages
    offenders = [
        r["pmid"]
        for r in records
        if r["pmid"] in texts[sha256_text(r["passage"])]
        or any(tag in r["passage"].lower() for tag in ("pqa_l", "pqa_u", "pubid", "pmid"))
    ]
    assert offenders == []


def test_reader_url_supabase_pooler_and_plain_hosts():
    from urllib.parse import urlsplit

    from scripts.ingest import READER_ROLE, reader_url

    pooled = reader_url(
        "postgresql://postgres.abcref:pw@aws-0-us-east-1.pooler.supabase.com:5432/postgres", "n/ew"
    )
    parts = urlsplit(pooled)
    assert parts.username == f"{READER_ROLE}.abcref"
    assert parts.password == "n%2Few"  # escaped
    assert (parts.hostname, parts.port, parts.path) == (
        "aws-0-us-east-1.pooler.supabase.com",
        5432,
        "/postgres",
    )
    # Not a Supabase pooler: a dotted username is just a name, not role.projectref.
    plain = urlsplit(reader_url("postgresql://first.last:pw@localhost:5433/db", "x"))
    assert (plain.username, plain.port) == (READER_ROLE, 5433)
    assert urlsplit(reader_url("postgresql://u:pw@[::1]:5432/db", "x")).hostname == "::1"


def test_document_rows_never_include_source():
    from scripts.ingest import DOCUMENT_COLUMNS, document_rows

    records = fake_records()
    for r in records:
        r["corpus_version"] = "v1"
    embeddings = {sha256_text(r["passage"]): [0.5, 0.25] for r in records}
    rows = document_rows(records, embeddings)
    assert all(len(row) == len(DOCUMENT_COLUMNS) for row in rows)
    flat = [str(v) for row in rows for v in row]
    assert not any(v in ("pqa_l", "pqa_u") for v in flat)
    assert rows[0][7] == "[0.5,0.25]"
