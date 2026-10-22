"""Secrets never reach printed output, and the .env helper rejects malformed values."""

import pytest

from backend.app.config import redact
from scripts.set_env import clean_database_url, fill_password, problem, set_value

FAKE_URL = (
    "postgresql://postgres.abcref:S3cretPw99@aws-0-us-east-1.pooler.supabase.com:5432/postgres"
)


def test_redact_masks_env_values_and_url_passwords(monkeypatch):
    monkeypatch.setenv("DATABASE_ADMIN_URL", FAKE_URL)
    monkeypatch.setenv("VOYAGE_API_KEY", "pa-fakekey123")
    # The real driver error that leaked: a malformed value echoed back inside a longer message.
    message = (
        f'missing "=" after "postgresql:{FAKE_URL}" in connection info string; key pa-fakekey123'
    )
    out = redact(message)
    assert "S3cretPw99" not in out
    assert "pa-fakekey123" not in out
    # Also masks a password that appears on its own, e.g. "password authentication failed".
    assert "S3cretPw99" not in redact("bad password S3cretPw99")


def test_redact_masks_unknown_url_passwords(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert redact("postgresql://reader:hunter2@host/db") == "postgresql://reader:***@host/db"


def test_problem_rejects_malformed_database_urls():
    assert problem("DATABASE_ADMIN_URL", FAKE_URL) is None
    assert problem("DATABASE_ADMIN_URL", "postgresql:" + FAKE_URL) is not None  # doubled prefix
    assert problem("DATABASE_ADMIN_URL", FAKE_URL.replace("S3cretPw99", "[YOUR-PASSWORD]"))
    assert problem("DATABASE_ADMIN_URL", FAKE_URL.replace("S3cretPw99", "a@b")) is not None
    assert problem("VOYAGE_API_KEY", '"pa-key"') is not None
    assert problem("VOYAGE_API_KEY", "") is not None


def test_set_value_replaces_only_the_named_line():
    text = "# names only\nDATABASE_URL=\nVOYAGE_API_KEY=old\n"
    assert (
        set_value(text, "VOYAGE_API_KEY", "new")
        == "# names only\nDATABASE_URL=\nVOYAGE_API_KEY=new\n"
    )
    assert set_value("A=1\n", "DATABASE_URL", "x") == "A=1\nDATABASE_URL=x\n"


def test_fill_password_escapes_special_characters():
    copied = FAKE_URL.replace("S3cretPw99", "[YOUR-PASSWORD]")
    filled = fill_password(copied, "p@ss#w/rd")
    assert problem("DATABASE_ADMIN_URL", filled) is None
    assert filled == FAKE_URL.replace("S3cretPw99", "p%40ss%23w%2Frd")


def test_doubled_scheme_is_cleaned_and_errors_show_only_the_scheme():
    assert clean_database_url("postgresql:" + FAKE_URL) == FAKE_URL
    assert clean_database_url("postgresql:postgres:" + FAKE_URL) == FAKE_URL
    assert clean_database_url(FAKE_URL) == FAKE_URL
    reason = problem("DATABASE_ADMIN_URL", "postgresql:" + FAKE_URL)
    assert "'postgresql:postgresql://'" in reason and "S3cretPw99" not in reason
    assert "S3cretPw99" not in problem("DATABASE_ADMIN_URL", "S3cretPw99//x")


def test_problem_never_echoes_a_letters_only_secret():
    # Review finding: a value without '//' used to be shown if it looked like a scheme.
    for pasted in ("LettersOnlySecretPassword", "Secret:Word//rest"):
        assert "Secret" not in problem("DATABASE_ADMIN_URL", pasted)


def test_problem_rejects_unescaped_special_characters_in_password():
    for bad in ("ab/cd", "ab?cd", "ab#cd", "ab:cd"):
        assert problem("DATABASE_ADMIN_URL", FAKE_URL.replace("S3cretPw99", bad)) is not None


def test_redact_masks_keyword_and_percent_decoded_passwords(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", FAKE_URL.replace("S3cretPw99", "p%40ss%23word"))
    assert "p@ss#word" not in redact("auth failed for password p@ss#word")
    assert redact("host=h password=hunter2 user=u") == "host=h password=*** user=u"
    assert "hunter2" not in redact("dbname=x password='hunter2'")


def test_open_connection_redacts_driver_errors():
    """The real leak: libpq echoes a malformed conninfo string, password included."""
    from backend.app.retrieval.vector_search import open_connection

    with pytest.raises(RuntimeError) as excinfo:
        open_connection("postgresql:" + FAKE_URL)
    assert "S3cretPw99" not in str(excinfo.value)
    assert excinfo.value.__cause__ is None and excinfo.value.__suppress_context__
