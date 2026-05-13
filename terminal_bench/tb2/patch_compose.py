#!/usr/bin/env python3
"""Patch each v0.2.x task's docker-compose.yaml + Dockerfile so it runs
under rootless podman on this cluster.

Changes per task:
  * compose `client` service gets `network_mode: host`  (no aardvark-dns)
  * compose `client` command wrapped with an APT sandbox disable preamble
    so `apt-get install` works under a single-uid rootless user
  * sidecar services (if any) also get `network_mode: host` + ports stripped

Backs up each file once (.orig). Idempotent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

TASKS_DIR = Path(__file__).resolve().parent / "tasks-v0.2" / "tasks"

APT_PREAMBLE = (
    "echo 'APT::Sandbox::User \"root\";' > /etc/apt/apt.conf.d/99no-sandbox; "
    "mkdir -p /etc/dpkg/dpkg.cfg.d; "
    "echo 'force-unsafe-io' > /etc/dpkg/dpkg.cfg.d/99unsafe-io; "
)
DEFAULT_TAIL = "exec sleep infinity"


def _cmd_body(cmd) -> tuple[str, bool]:
    if cmd is None:
        return DEFAULT_TAIL, True
    if isinstance(cmd, str):
        s = cmd.strip()
        if s.startswith("sh -c "):
            return s[len("sh -c "):].strip().strip("'\""), True
        return s, False
    if isinstance(cmd, list):
        if len(cmd) >= 3 and cmd[0] in ("sh", "bash", "/bin/sh", "/bin/bash") and cmd[1] == "-c":
            return " ".join(cmd[2:]) if len(cmd) > 3 else cmd[2], True
        return " ".join(str(x) for x in cmd), False
    return str(cmd), False


def patch_compose(path: Path) -> bool:
    if not path.is_file():
        return False
    backup = path.with_suffix(".yaml.orig")
    if not backup.exists():
        backup.write_bytes(path.read_bytes())
    data = yaml.safe_load(backup.read_text()) or {}
    services = data.get("services", {}) or {}
    if "client" not in services:
        return False

    for name, svc in services.items():
        if not isinstance(svc, dict):
            continue
        svc["network_mode"] = "host"
        svc.pop("ports", None)
        svc.pop("networks", None)

    client = services["client"]
    body, _ = _cmd_body(client.get("command"))
    if "sleep infinity" not in body and "exec " not in body:
        body = f"{body}; {DEFAULT_TAIL}"
    client["command"] = ["sh", "-c", APT_PREAMBLE + body]

    # Let tar skip chown under single-uid rootless (uv/pip installers untar)
    env = client.get("environment") or []
    if isinstance(env, dict):
        env = [f"{k}={v}" for k, v in env.items()]
    if not any(str(e).startswith("TAR_OPTIONS=") for e in env):
        env.append("TAR_OPTIONS=--no-same-owner")
    client["environment"] = env

    path.write_text(yaml.safe_dump(data, sort_keys=False, default_flow_style=False))
    return True


def main():
    only = set(sys.argv[1:])
    count = 0
    for td in sorted(TASKS_DIR.iterdir()):
        if not td.is_dir():
            continue
        if only and td.name not in only:
            continue
        if patch_compose(td / "docker-compose.yaml"):
            count += 1
    print(f"Patched {count} docker-compose.yaml under {TASKS_DIR}")


if __name__ == "__main__":
    main()
