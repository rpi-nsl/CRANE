"""Sandbox runtime abstraction for openhands_swebench.

Two backends:
  - PodmanRuntime  (local, default): shells out to `podman`, current behavior
  - DaytonaRuntime (remote)       : creates Daytona sandboxes via daytona-sdk

Selected by env var SWE_RUNTIME ∈ {"podman", "daytona"} (default "podman").

Container handles are opaque strings. Each runtime knows how to interpret
its own handles, so call sites continue to pass `str` around unchanged.

Surface (kept narrow on purpose — only what openhands_swebench.py uses):
    start(instance_id) -> handle
    stop(handle)
    exec(handle, command, cwd, timeout) -> (exit_code, output)
    exists(handle) -> bool
    read_file(handle, path) -> (rc, text)
    write_file(handle, path, text) -> (rc, err)
    stat(handle, path) -> dict
    listdir(handle, path, maxdepth) -> str
    is_dead_error(exit_code, output) -> bool
"""
from __future__ import annotations

import logging
import os
import shlex
import subprocess
import uuid
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

DEFAULT_CMD_TIMEOUT = 300.0
_OUTPUT_TRUNC = 20000


def sweb_canon_tag(instance_id: str) -> str:
    return (
        f"docker.io/swebench/sweb.eval.x86_64.{instance_id.replace('__', '_1776_')}:latest"
    ).lower()


def sweb_ghcr_tag(instance_id: str) -> str:
    """Match env.sh::sweb_ghcr_tag (Epoch AI mirror)."""
    return f"ghcr.io/epoch-research/swe-bench.eval.x86_64.{instance_id}:latest".lower()


def resolve_image(instance_id: str) -> str:
    """Image to use for this instance, honoring SWE_IMAGE_SOURCE.

        canon (default) → docker.io/swebench/...
        ghcr            → ghcr.io/epoch-research/...
    """
    src = os.environ.get("SWE_IMAGE_SOURCE", "canon").lower().strip()
    if src == "ghcr":
        return sweb_ghcr_tag(instance_id)
    return sweb_canon_tag(instance_id)


class InstanceRuntime(ABC):
    name: str = "abstract"

    @abstractmethod
    def start(self, instance_id: str) -> str: ...

    @abstractmethod
    def stop(self, handle: str) -> None: ...

    @abstractmethod
    def exec(self, handle: str, command: str, cwd: str = "/testbed",
             timeout: float = DEFAULT_CMD_TIMEOUT) -> tuple[int, str]: ...

    @abstractmethod
    def exists(self, handle: str) -> bool: ...

    @abstractmethod
    def read_file(self, handle: str, path: str) -> tuple[int, str]: ...

    @abstractmethod
    def write_file(self, handle: str, path: str, text: str) -> tuple[int, str]: ...

    @abstractmethod
    def stat(self, handle: str, path: str) -> dict: ...

    @abstractmethod
    def listdir(self, handle: str, path: str, maxdepth: int = 2) -> str: ...

    def is_dead_error(self, exit_code: int, output: str) -> bool:
        return False


# ──────────────────────────────────────────────────────────────────────────
# Podman (local) — extracted from openhands_swebench.py
# ──────────────────────────────────────────────────────────────────────────

_DEAD_PODMAN_PATTERNS = (
    "no such container",
    "no container with name",
    "container not running",
    "container is not running",
    "does not exist",
)


def _podman(*args: str, capture: bool = True, timeout: float | None = None,
            check: bool = True, input: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["podman", *args],
        capture_output=capture,
        text=True,
        timeout=timeout,
        check=check,
        input=input,
    )


class PodmanRuntime(InstanceRuntime):
    name = "podman"

    def start(self, instance_id: str) -> str:
        image = resolve_image(instance_id)
        check = _podman("image", "exists", image, check=False, timeout=60)
        if check.returncode != 0:
            logger.warning("Image %s not local; pulling…", image)
            _podman("pull", image, check=True, timeout=900)

        cname = f"oh-{instance_id.replace('/', '_').replace('_', '-').lower()}-{uuid.uuid4().hex[:6]}"
        for attempt in range(2):
            try:
                _podman(
                    "run", "-d",
                    "--name", cname,
                    "--rm",
                    "--entrypoint", "sleep",
                    image, "infinity",
                    check=True, timeout=300,
                )
                return cname
            except subprocess.TimeoutExpired:
                logger.warning("podman run -d timed out for %s (attempt %d)", instance_id, attempt + 1)
                _podman("rm", "-f", cname, check=False, timeout=30)
                if attempt == 1:
                    raise
        return cname

    def stop(self, handle: str) -> None:
        _podman("rm", "-f", handle, check=False, timeout=30)

    def exec(self, handle: str, command: str, cwd: str = "/testbed",
             timeout: float = DEFAULT_CMD_TIMEOUT) -> tuple[int, str]:
        full = f"cd {shlex.quote(cwd)} && {command}"
        try:
            proc = subprocess.run(
                ["podman", "exec", "-i", handle, "bash", "-lc", full],
                capture_output=True, text=True, timeout=timeout,
            )
            out = (proc.stdout or "") + (proc.stderr or "")
            if len(out) > _OUTPUT_TRUNC:
                out = out[:_OUTPUT_TRUNC] + "\n\n[...output truncated at 20000 chars]"
            return proc.returncode, out
        except subprocess.TimeoutExpired as e:
            partial = ((e.stdout or b"").decode(errors="replace")
                       + (e.stderr or b"").decode(errors="replace"))
            return -1, partial + f"\n\n[command timed out after {timeout}s]"

    def exists(self, handle: str) -> bool:
        try:
            rc = subprocess.run(
                ["podman", "container", "exists", handle],
                capture_output=True, timeout=10,
            ).returncode
            return rc == 0
        except Exception:
            return False

    def read_file(self, handle: str, path: str) -> tuple[int, str]:
        try:
            proc = subprocess.run(
                ["podman", "exec", "-i", handle, "cat", path],
                capture_output=True, timeout=60,
            )
            return proc.returncode, proc.stdout.decode("utf-8", errors="replace")
        except subprocess.TimeoutExpired:
            return -1, "[read timed out]"

    def write_file(self, handle: str, path: str, text: str) -> tuple[int, str]:
        parent = os.path.dirname(path) or "/"
        mk = subprocess.run(
            ["podman", "exec", "-i", handle, "mkdir", "-p", parent],
            capture_output=True, text=True, timeout=30,
        )
        if mk.returncode != 0:
            return mk.returncode, mk.stderr or "mkdir failed"
        proc = subprocess.run(
            ["podman", "exec", "-i", handle, "bash", "-c", f"tee {shlex.quote(path)} > /dev/null"],
            input=text.encode("utf-8"),
            capture_output=True, timeout=120,
        )
        return proc.returncode, (proc.stderr or b"").decode("utf-8", errors="replace")

    def stat(self, handle: str, path: str) -> dict:
        proc = subprocess.run(
            ["podman", "exec", "-i", handle, "stat", "-c", "%F|%s", path],
            capture_output=True, text=True, timeout=30,
        )
        if proc.returncode != 0:
            return {"exists": False}
        kind, size = proc.stdout.strip().split("|", 1)
        return {"exists": True, "is_dir": "directory" in kind, "size": int(size)}

    def listdir(self, handle: str, path: str, maxdepth: int = 2) -> str:
        proc = subprocess.run(
            ["podman", "exec", "-i", handle,
             "find", path, "-maxdepth", str(maxdepth), "-not", "-path", "*/\\.*"],
            capture_output=True, text=True, timeout=30,
        )
        if proc.returncode != 0:
            return f"[find {path} failed: {proc.stderr.strip()}]"
        return proc.stdout

    def is_dead_error(self, exit_code: int, output: str) -> bool:
        if exit_code not in (125, 126, 127):
            return False
        low = (output or "").lower()
        return any(m in low for m in _DEAD_PODMAN_PATTERNS)


# ──────────────────────────────────────────────────────────────────────────
# Daytona (remote)
# ──────────────────────────────────────────────────────────────────────────

class DaytonaRuntime(InstanceRuntime):
    """Each handle is a Daytona sandbox id; the Sandbox object is cached.

    Config (env vars):
        DAYTONA_API_KEY        required
        DAYTONA_API_URL        optional (defaults to SDK's default)
        DAYTONA_TARGET         optional region/target
        DAYTONA_IMAGE_PREFIX   optional override for image registry
                               (default: docker.io/swebench)

    The SWE-bench eval images are pulled from Docker Hub by Daytona on
    sandbox creation. First-time pulls are slow; subsequent runs reuse
    Daytona snapshots automatically.
    """

    name = "daytona"

    def __init__(self) -> None:
        # Daytona's current SDK is published as `daytona` (the older
        # `daytona_sdk` is deprecated). We try the new package first and fall
        # back so users on the old package still work.
        try:
            from daytona import (  # type: ignore
                CreateSandboxFromImageParams,
                Daytona,
                DaytonaConfig,
                Resources,
            )
            self._FromImage = CreateSandboxFromImageParams
            self._Resources = Resources
        except ImportError:
            try:
                from daytona_sdk import (  # type: ignore
                    CreateSandboxParams as _LegacyParams,
                    Daytona,
                    DaytonaConfig,
                )
                self._FromImage = _LegacyParams
                self._Resources = None
            except ImportError as e:
                raise RuntimeError(
                    "Daytona SDK not installed. Run: pip install daytona"
                ) from e

        api_key = os.environ.get("DAYTONA_API_KEY")
        if not api_key:
            raise RuntimeError("DAYTONA_API_KEY not set")
        cfg_kwargs: dict[str, str] = {"api_key": api_key}
        if url := os.environ.get("DAYTONA_API_URL"):
            cfg_kwargs["api_url"] = url
        if target := os.environ.get("DAYTONA_TARGET"):
            cfg_kwargs["target"] = target
        self._client = Daytona(DaytonaConfig(**cfg_kwargs))
        self._sandboxes: dict[str, object] = {}

    def _get(self, handle: str):
        sb = self._sandboxes.get(handle)
        if sb is None:
            sb = self._client.get(handle)
            self._sandboxes[handle] = sb
        return sb

    def start(self, instance_id: str) -> str:
        image = resolve_image(instance_id)
        # `auto_stop_interval=0` disables Daytona's idle auto-stop — we manage
        # lifecycle ourselves via stop()/deadline timer.
        cpu = int(os.environ.get("DAYTONA_SANDBOX_CPU", "1"))
        mem = int(os.environ.get("DAYTONA_SANDBOX_MEM_GB", "1"))
        disk = int(os.environ.get("DAYTONA_SANDBOX_DISK_GB", "3"))
        kwargs: dict = {"image": image, "auto_stop_interval": 0}
        if self._Resources is not None:
            kwargs["resources"] = self._Resources(cpu=cpu, memory=mem, disk=disk)
        try:
            params = self._FromImage(**kwargs)
        except TypeError:
            kwargs.pop("auto_stop_interval", None)
            params = self._FromImage(**kwargs)
        # Retry on transient quota errors. Concurrent workers race for the
        # global CPU pool — if N+1 workers all call create() simultaneously,
        # the (N+1)-th gets "Total CPU limit exceeded". By the time we retry
        # 5-15s later, an earlier sandbox has finished and freed its slot.
        max_retries = int(os.environ.get("DAYTONA_START_RETRIES", "10"))
        backoff = float(os.environ.get("DAYTONA_START_BACKOFF", "10"))
        last_exc: Exception | None = None
        import time as _time
        for attempt in range(max_retries + 1):
            try:
                sb = self._client.create(params)
                break
            except Exception as exc:  # noqa: BLE001
                msg = str(exc).lower()
                transient = ("limit exceeded" in msg or "quota" in msg
                             or "rate" in msg or "too many" in msg)
                if not transient or attempt == max_retries:
                    raise
                last_exc = exc
                wait = backoff * (1 + attempt * 0.5)
                logger.info("daytona quota busy (attempt %d/%d) — sleeping %.0fs",
                            attempt + 1, max_retries, wait)
                _time.sleep(wait)
        else:  # pragma: no cover
            raise last_exc  # type: ignore
        sid = getattr(sb, "id", None) or getattr(sb, "sandbox_id", None)
        if not sid:
            raise RuntimeError(f"Daytona create() returned object with no id: {sb!r}")
        self._sandboxes[sid] = sb
        # Sleep-infinity equivalent: Daytona keeps the sandbox alive on its own,
        # we just need the FS + exec endpoint.
        return sid

    def stop(self, handle: str) -> None:
        sb = self._sandboxes.pop(handle, None)
        if sb is None:
            try:
                sb = self._client.get(handle)
            except Exception:
                return
        try:
            self._client.delete(sb)
        except Exception as exc:  # noqa: BLE001
            logger.warning("daytona delete(%s) failed: %s", handle, exc)

    def exec(self, handle: str, command: str, cwd: str = "/testbed",
             timeout: float = DEFAULT_CMD_TIMEOUT) -> tuple[int, str]:
        sb = self._get(handle)
        full = f"cd {shlex.quote(cwd)} && {command}"
        try:
            resp = sb.process.exec(full, timeout=int(timeout))
            out = getattr(resp, "result", "") or ""
            rc = getattr(resp, "exit_code", 0)
            if rc is None:
                rc = 0
            if len(out) > _OUTPUT_TRUNC:
                out = out[:_OUTPUT_TRUNC] + "\n\n[...output truncated at 20000 chars]"
            return int(rc), out
        except Exception as e:  # noqa: BLE001
            return -1, f"[daytona exec error: {e}]"

    def exists(self, handle: str) -> bool:
        try:
            sb = self._client.get(handle)
            state = (getattr(sb, "state", "") or getattr(sb, "status", "") or "").lower()
            return state not in {"stopped", "destroyed", "error", "deleted"}
        except Exception:
            return False

    def read_file(self, handle: str, path: str) -> tuple[int, str]:
        sb = self._get(handle)
        try:
            data = sb.fs.download_file(path)
            text = data.decode("utf-8", errors="replace") if isinstance(data, (bytes, bytearray)) else str(data)
            return 0, text
        except Exception:
            return self.exec(handle, f"cat {shlex.quote(path)}", cwd="/", timeout=60)

    def write_file(self, handle: str, path: str, text: str) -> tuple[int, str]:
        sb = self._get(handle)
        parent = os.path.dirname(path) or "/"
        try:
            sb.fs.create_folder(parent, "755")
        except Exception:
            # create_folder errors on already-existing dirs; tolerate.
            pass
        try:
            sb.fs.upload_file(text.encode("utf-8"), path)
            return 0, ""
        except Exception as e:
            # Heredoc fallback. The shell heredoc ALWAYS appends one trailing
            # newline — if `text` already ends with \n we'd write \n\n. Strip
            # the trailing newline to compensate; if `text` has no trailing
            # newline at all we still gain one, but that matches typical text
            # files and SWE-bench patches.
            body = text[:-1] if text.endswith("\n") else text
            quoted = shlex.quote(path)
            rc2, out2 = self.exec(
                handle,
                f"cat > {quoted} <<'__SWEBENCH_EOF__'\n{body}\n__SWEBENCH_EOF__",
                cwd="/", timeout=120,
            )
            return rc2, out2 or str(e)

    def stat(self, handle: str, path: str) -> dict:
        rc, out = self.exec(
            handle, f"stat -c '%F|%s' {shlex.quote(path)}",
            cwd="/", timeout=30,
        )
        if rc != 0:
            return {"exists": False}
        try:
            kind, size = out.strip().split("|", 1)
            return {"exists": True, "is_dir": "directory" in kind, "size": int(size)}
        except ValueError:
            return {"exists": False}

    def listdir(self, handle: str, path: str, maxdepth: int = 2) -> str:
        rc, out = self.exec(
            handle,
            f"find {shlex.quote(path)} -maxdepth {maxdepth} -not -path '*/\\.*'",
            cwd="/", timeout=30,
        )
        if rc != 0:
            return f"[find {path} failed: {out.strip()}]"
        return out

    def is_dead_error(self, exit_code: int, output: str) -> bool:
        if exit_code >= 0:
            return False
        low = (output or "").lower()
        return "daytona exec error" in low and any(
            k in low for k in ("not found", "stopped", "destroyed", "404")
        )


# ──────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────

_RUNTIME: InstanceRuntime | None = None


def get_runtime() -> InstanceRuntime:
    """Return the process-wide runtime singleton.

    Selected once via SWE_RUNTIME env var (default "podman"). Re-importing or
    changing the env var mid-run does not switch backends — restart the process.
    """
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME
    kind = os.environ.get("SWE_RUNTIME", "podman").lower().strip()
    if kind == "podman":
        _RUNTIME = PodmanRuntime()
    elif kind == "daytona":
        _RUNTIME = DaytonaRuntime()
    else:
        raise ValueError(f"unknown SWE_RUNTIME={kind!r} (expected 'podman' or 'daytona')")
    logger.info("instance runtime = %s", _RUNTIME.name)
    return _RUNTIME
