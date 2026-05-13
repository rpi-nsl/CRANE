#!/usr/bin/env python3
"""Apply the rootless-podman lchown fix to swebench's docker_utils.copy_to_container.

Background
----------
SWE-bench's harness tars files for `put_archive` with the host user's UID. On
clusters without `/etc/subuid` entries (e.g. shared HPC nodes), rootless
podman maps host_uid <-> container_uid 0 and rejects any other UID with
``lchown ... invalid argument``. Forcing ``uid=gid=0`` in the tar filter
puts the entry in the only valid bucket.

This is the same patch documented in the paper (Appendix §A.1.2,
"Container backend: podman replacing Docker"). It is needed only when
running the SWE-bench harness on rootless podman without subuid entries.

Apply once after ``pip install swebench``::

    python -m swe_bench.swebench_lchown_patch

The patch is idempotent — running it twice does nothing.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PATCH_MARKER = "# CRANE-PATCH: rootless-podman lchown fix"

PATCHED_FUNCTION = '''
def copy_to_container(container: Container, src: Path, dst: Path):
    """Copy a file from local to a docker/podman container.

    Forces uid=gid=0 in the tarinfo so rootless podman without subuid
    entries accepts the put_archive call.
    """
    if os.path.dirname(dst) == "":
        raise ValueError(
            f"Destination path parent directory cannot be empty!, dst: {dst}"
        )

    tar_path = src.with_suffix(".tar")

    def _root_owner(info):  # ''' + PATCH_MARKER.lstrip("#").strip() + '''
        info.uid = 0
        info.gid = 0
        info.uname = "root"
        info.gname = "root"
        return info

    with tarfile.open(tar_path, "w") as tar:
        tar.add(src, arcname=dst.name, filter=_root_owner)

    with open(tar_path, "rb") as tar_file:
        data = tar_file.read()

    container.put_archive(os.path.dirname(dst), data)
    tar_path.unlink()
'''


def _locate_docker_utils() -> Path:
    try:
        spec = importlib.util.find_spec("swebench.harness.docker_utils")
    except ModuleNotFoundError:
        spec = None
    if spec is None or spec.origin is None:
        sys.exit(
            "swebench is not importable from this venv. "
            "Activate the venv that contains swebench (e.g. .venv-openhands) and retry."
        )
    return Path(spec.origin)


def main() -> int:
    target = _locate_docker_utils()
    text = target.read_text()
    if PATCH_MARKER in text:
        print(f"[skip] {target} already patched.")
        return 0

    needle = "def copy_to_container(container: Container, src: Path, dst: Path):"
    start = text.find(needle)
    if start < 0:
        sys.exit(f"could not find copy_to_container() in {target}")

    # Find the next top-level def or end-of-file to delimit the original.
    end = text.find("\ndef ", start + 1)
    if end < 0:
        end = len(text)
    new_text = text[:start] + PATCHED_FUNCTION.lstrip("\n") + text[end:]
    target.write_text(new_text)
    print(f"[ok] patched {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
