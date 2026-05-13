#!/usr/bin/env python3
"""Inject an APT-sandbox-disable RUN step near the top of each task's
Dockerfile so `apt-get install` works under single-uid rootless podman.

Idempotent: marks files with a fingerprint comment, skips if already patched.
Keeps a .orig backup on first patch.
"""
from __future__ import annotations

import sys
from pathlib import Path

TASKS_DIR = Path(__file__).resolve().parent / "tasks-v0.2" / "tasks"
FINGERPRINT = "# TB2_ROOTLESS_APT_PATCH"

INJECTION = f"""{FINGERPRINT}
RUN printf 'APT::Sandbox::User "root";\\nAPT::Sandbox::Verify "false";\\n' \\
    > /etc/apt/apt.conf.d/99no-sandbox \\
 && mkdir -p /etc/dpkg/dpkg.cfg.d \\
 && printf 'force-unsafe-io\\n' > /etc/dpkg/dpkg.cfg.d/99unsafe-io
"""


def patch(df: Path) -> bool:
    if not df.is_file():
        return False
    text = df.read_text()
    if FINGERPRINT in text:
        return False  # already patched
    backup = df.with_suffix(df.suffix + ".orig")
    if not backup.exists():
        backup.write_text(text)

    lines = text.splitlines(keepends=True)
    # Find the first non-comment, non-blank line that is a FROM.
    # Insert our patch block right after the FIRST `FROM` directive.
    out = []
    inserted = False
    for line in lines:
        out.append(line)
        if not inserted and line.strip().upper().startswith("FROM "):
            if not out[-1].endswith("\n"):
                out[-1] += "\n"
            out.append("\n" + INJECTION + "\n")
            inserted = True
    if not inserted:
        print(f"WARN: no FROM in {df}", file=sys.stderr)
        return False
    df.write_text("".join(out))
    return True


def main():
    only = set(sys.argv[1:])
    count = 0
    for td in sorted(TASKS_DIR.iterdir()):
        if not td.is_dir():
            continue
        if only and td.name not in only:
            continue
        # Dockerfiles may live at task root OR in service subdirs (client/, server/, api/ ...)
        for df in sorted(td.rglob("Dockerfile")):
            if patch(df):
                count += 1
    print(f"Patched {count} Dockerfiles under {TASKS_DIR}")


if __name__ == "__main__":
    main()
