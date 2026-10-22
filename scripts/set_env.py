"""Set one value in the local .env without an editor; the pasted value is never shown.

Run: uv run python -m scripts.set_env DATABASE_ADMIN_URL
"""

import getpass
import re
import sys
from urllib.parse import quote

from backend.app.config import REPO_ROOT, SECRET_ENV_VARS

ENV_FILE = REPO_ROOT / ".env"
PLACEHOLDER = "[YOUR-PASSWORD]"
# Copies of the Supabase URI twice arrived as "postgresql:postgresql://..." (2026-10-22).
_DOUBLED_SCHEME = re.compile(r"^(?:postgres(?:ql)?:)+(?=postgres(?:ql)?://)")


def clean_database_url(url: str) -> str:
    """Drop a repeated 'postgresql:' in front of the real scheme."""
    return _DOUBLED_SCHEME.sub("", url)


def url_start(value: str) -> str:
    """The part before '//', shown in errors only when it is a scheme (never a secret)."""
    start, sep, _ = value.partition("//")
    if sep and re.fullmatch(r"(?:postgres(?:ql)?:)+", start):
        return repr(start + "//")
    return "something else"


def fill_password(url: str, password: str) -> str:
    """Put the password into Supabase's copied URI, escaping characters like '@', '#' or '/'."""
    return url.replace(PLACEHOLDER, quote(password, safe=""))


def problem(name: str, value: str) -> str | None:
    """Return why the value looks wrong, without echoing it; None if it looks fine."""
    if not value:
        return "nothing was pasted"
    if any(ch.isspace() for ch in value) or value[0] in "'\"":
        return "it contains spaces or quotes"
    if name.startswith("DATABASE_"):
        if not value.startswith(("postgresql://", "postgres://")) or value.count("://") != 1:
            return (
                f"it must start with postgresql:// exactly once, but starts with {url_start(value)}"
            )
        if "[" in value or "YOUR-PASSWORD" in value:
            return "the [YOUR-PASSWORD] placeholder is still there"
        userinfo, _, _ = value.split("://", 1)[1].rpartition("@")
        password = userinfo.partition(":")[2]
        if (
            not userinfo
            or value.split("://", 1)[1].count("@") != 1
            or any(ch in password for ch in "/?#:[]")
        ):
            return (
                "the password has a special character (@ / ? # :) that breaks the URL; "
                f"paste the URI with {PLACEHOLDER} left in instead"
            )
    return None


def set_value(text: str, name: str, value: str) -> str:
    """Replace the NAME= line (or append one), keeping every other line as it was."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.split("=", 1)[0].strip() == name:
            lines[i] = f"{name}={value}"
            break
    else:
        lines.append(f"{name}={value}")
    return "\n".join(lines) + "\n"


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in SECRET_ENV_VARS:
        print(f"usage: uv run python -m scripts.set_env <{'|'.join(SECRET_ENV_VARS)}>")
        return 2
    name = sys.argv[1]
    value = getpass.getpass(f"Paste {name} (hidden), then press Enter: ").strip()
    if name.startswith("DATABASE_"):
        value = clean_database_url(value)
    if name.startswith("DATABASE_") and PLACEHOLDER in value:
        password = getpass.getpass("Paste the database password (hidden), then press Enter: ")
        value = fill_password(value, password.strip())
    reason = problem(name, value)
    if reason:
        print(f"Not saved: {reason}.")
        return 1
    ENV_FILE.write_text(set_value(ENV_FILE.read_text(encoding="utf-8"), name, value), "utf-8")
    print(f"Saved {name} to {ENV_FILE} ({len(value)} characters).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
