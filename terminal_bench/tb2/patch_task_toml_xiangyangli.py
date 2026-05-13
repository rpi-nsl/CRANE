#!/usr/bin/env python3
"""Swap task.toml docker_image from alexgshaw/<task>:20251031 to
xiangyangli/<task>:20260204 (zai-org verified, has env fixes).

Idempotent. Keeps .orig backup on first run.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

TB2_OFFICIAL = Path(__file__).resolve().parent / "tb2-official"
# Match any `docker_image = "alexgshaw/<name>:<tag>"`
LINE_RE = re.compile(
    r'^(\s*docker_image\s*=\s*")alexgshaw/([^:"]+):([^"]+)(".*)$',
    re.MULTILINE,
)


def patch(toml: Path) -> tuple[bool, str]:
    if not toml.is_file():
        return False, "no file"
    text = toml.read_text()
    m = LINE_RE.search(text)
    if not m:
        # Already xiangyangli (or something else) — skip
        if "xiangyangli/" in text:
            return False, "already xiangyangli"
        return False, "no alexgshaw docker_image line"

    backup = toml.with_suffix(toml.suffix + ".orig")
    if not backup.exists():
        backup.write_text(text)

    def _sub(mm: re.Match) -> str:
        return f"{mm.group(1)}xiangyangli/{mm.group(2)}:20260204{mm.group(4)}"

    text2 = LINE_RE.sub(_sub, text)
    toml.write_text(text2)
    return True, "patched"


def main() -> None:
    only = set(sys.argv[1:])
    changed = skipped = 0
    skipped_names = []
    for td in sorted(TB2_OFFICIAL.iterdir()):
        if not td.is_dir():
            continue
        if only and td.name not in only:
            continue
        toml = td / "task.toml"
        ok, reason = patch(toml)
        if ok:
            changed += 1
        else:
            skipped += 1
            skipped_names.append((td.name, reason))
    print(f"Patched {changed}, skipped {skipped}.")
    if skipped_names:
        print("\nSkipped:")
        for n, r in skipped_names:
            print(f"  {n}: {r}")


if __name__ == "__main__":
    main()
