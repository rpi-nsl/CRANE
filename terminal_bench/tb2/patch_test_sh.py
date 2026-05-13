#!/usr/bin/env python3
"""Rewrite tests/test.sh in each task to install uv via pypi instead of
curl|sh from releases.astral.sh (which Daytona blocks as a non-whitelisted
subdomain of astral.sh).

Idempotent: keeps a .orig backup on first run and skips already-patched files.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parent / "tb2-zai"
MARKER = "# TB2_PIP_UV_PATCH"
# Matches `curl -LsSf https://astral.sh/uv/<version>/install.sh | sh`
# and the following `source $HOME/.local/bin/env` line.
CURL_LINE = re.compile(
    r"^curl\s+-LsSf\s+https://astral\.sh/uv/([\d.]+)/install\.sh\s*\|\s*sh\s*$",
    re.MULTILINE,
)
SOURCE_LINE = re.compile(
    r"^\s*source\s+\$HOME/\.local/bin/env\s*$",
    re.MULTILINE,
)


def patch(f: Path) -> tuple[bool, str]:
    """Returns (changed, reason)."""
    if not f.is_file():
        return False, "no file"
    text = f.read_text()
    if MARKER in text:
        return False, "already patched"
    if not CURL_LINE.search(text):
        return False, "no astral.sh curl line"

    backup = f.with_suffix(f.suffix + ".orig")
    if not backup.exists():
        backup.write_text(text)

    def _replace_curl(m: re.Match) -> str:
        version = m.group(1)
        # py3.11+ supports --break-system-packages; py3.10 does not.
        # Fallback to --user; then prepend $HOME/.local/bin to PATH so uvx is found.
        return (
            f"{MARKER}\n"
            f"python3 -m pip install --break-system-packages 'uv=={version}' 2>/dev/null \\\n"
            f"  || python3 -m pip install --user --no-warn-script-location 'uv=={version}' \\\n"
            f"  || python3 -m pip install 'uv=={version}'\n"
            'export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"\n'
            "hash -r"
        )

    text2 = CURL_LINE.sub(_replace_curl, text)
    # Drop the `source $HOME/.local/bin/env` line too — not needed when uv is pip-installed
    text2 = SOURCE_LINE.sub("", text2)
    f.write_text(text2)
    return True, "patched"


def main() -> None:
    # First positional arg is dataset root (tb2-zai / tb2-official). Optional.
    # Remaining positional args filter to specific task names.
    args = sys.argv[1:]
    root = DEFAULT_ROOT
    if args and (Path(args[0]).is_dir() or args[0].startswith("tb2-")):
        root = Path(args[0]).resolve() if Path(args[0]).is_absolute() else \
               (Path(__file__).resolve().parent / args[0])
        args = args[1:]
    only = set(args)
    changed = skipped = 0
    for td in sorted(root.iterdir()):
        if not td.is_dir() or td.name.startswith("."):
            continue
        if only and td.name not in only:
            continue
        for ts in td.glob("tests/test.sh"):
            ok, reason = patch(ts)
            if ok:
                changed += 1
                print(f"[+] {td.name}: {reason}")
            else:
                skipped += 1
    print(f"\nDataset: {root}")
    print(f"Patched {changed}, skipped {skipped}.")


if __name__ == "__main__":
    main()
