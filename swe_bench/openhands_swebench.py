#!/usr/bin/env python3
"""
Custom OpenHands SWE-bench driver for the podman + vLLM HPC setup.

We can't use the upstream `openhands-benchmarks` repo directly because it
requires docker buildx (for wrapping SWE-bench eval images with an
agent-server layer), which is unusable on this cluster — no subuid
entries, buildx can't set up user namespaces. Instead we:

  1. Reuse the `openhands-sdk` Agent/Conversation primitives and system
     prompt unchanged.
  2. Swap out the default TerminalTool executor so every bash command the
     agent issues is forwarded into a per-instance SWE-bench eval
     container via `podman exec`. This is the same container model that
     mini-swe-agent uses here.
  3. Skip FileEditorTool — its executor is tightly coupled to the local
     filesystem. The agent edits files via bash (sed/cat/patch/etc.),
     which is the pattern mini-swe-agent's config already assumes works.
  4. Extract the patch via `git diff <base_commit>` after the agent
     finishes, and dump `preds.json` in swebench harness format so the
     existing harness step in run_swebench.sh can score it.

Usage is intentionally close to mini-extra swebench / sweagent run-batch
so run_swebench.sh can drop it in as a third `AGENT=openhands` branch.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
import traceback
import uuid
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datasets import load_dataset

from openhands.sdk import LLM, Agent, Conversation, Tool, get_logger
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.event import ActionEvent, MessageEvent
from openhands.sdk.tool import ToolAnnotations, ToolExecutor, register_tool
from openhands.sdk.tool.builtins.finish import FinishAction
from openhands.sdk.workspace import LocalWorkspace
from openhands.tools.file_editor import (
    FileEditorAction,
    FileEditorObservation,
    FileEditorTool,
)
from openhands.tools.terminal import (
    TerminalAction,
    TerminalObservation,
    TerminalTool,
)
from openhands.tools.terminal.constants import CMD_OUTPUT_PS1_END  # noqa: F401

from runtime import get_runtime, sweb_canon_tag  # noqa: F401  (re-exported)


logger = get_logger(__name__)

DATASET_MAP = {
    "lite": "princeton-nlp/SWE-Bench_Lite",
    "verified": "princeton-nlp/SWE-Bench_Verified",
    "full": "princeton-nlp/SWE-Bench",
    "multimodal": "princeton-nlp/SWE-Bench_Multimodal",
}

# Default timeout per command issued by the agent (seconds). The SWE-bench
# harness itself times out at 3600s, so keep below that.
DEFAULT_CMD_TIMEOUT = 300.0

MODEL_NAME_OR_PATH_DEFAULT = "openhands"  # replaced by --model at runtime


# ──────────────────────────────────────────────────────────────────────────
# Sandbox helpers — delegate to runtime.get_runtime() so the same code paths
# work for local podman and remote (Daytona) sandboxes. Selected by env var
# SWE_RUNTIME ∈ {podman, daytona}.
# ──────────────────────────────────────────────────────────────────────────

_RUNTIME = get_runtime()


def start_instance_container(instance_id: str) -> str:
    """Start a sandbox for `instance_id`; returns an opaque handle."""
    return _RUNTIME.start(instance_id)


def stop_instance_container(handle: str) -> None:
    _RUNTIME.stop(handle)


def podman_exec(handle: str, command: str, cwd: str = "/testbed",
                timeout: float = DEFAULT_CMD_TIMEOUT) -> tuple[int, str]:
    """Run `command` inside the sandbox; returns (exit_code, merged_output).

    Name kept for backward compat — actual backend may be Daytona.
    """
    return _RUNTIME.exec(handle, command, cwd=cwd, timeout=timeout)


def _is_container_dead_error(exit_code: int, output: str) -> bool:
    return _RUNTIME.is_dead_error(exit_code, output)


def _request_conversation_pause(conversation, reason: str, iid_hint: str = "") -> None:
    # Best-effort: idempotent pause signal. pause() is thread-safe per SDK docs.
    if conversation is None:
        return
    try:
        conversation.pause()
        logger.warning(
            "%s%s → conversation.pause() requested",
            f"[{iid_hint}] " if iid_hint else "",
            reason,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("conversation.pause() failed: %s", exc)


# ──────────────────────────────────────────────────────────────────────────
# Custom TerminalTool executor routed through podman exec
# ──────────────────────────────────────────────────────────────────────────

class PodmanExecTerminalExecutor(ToolExecutor[TerminalAction, TerminalObservation]):
    """Stateless shell inside a podman container.

    Each TerminalAction becomes one `podman exec bash -lc` call. This does
    NOT preserve working directory between commands (each invocation is
    independent) — the agent must use absolute paths or chain with `&&`
    within a single command. The `is_input`, `reset`, and soft-timeout
    interrupt semantics of the upstream TerminalExecutor are NOT supported
    here; those map to stateful tmux sessions we don't run.
    """

    is_pooled = False  # consumed by TerminalTool.declared_resources

    def __init__(self, container: str, working_dir: str = "/testbed",
                 default_timeout: float = DEFAULT_CMD_TIMEOUT) -> None:
        self.container = container
        self.working_dir = working_dir
        self.default_timeout = default_timeout

    @property
    def _working_dir(self) -> str:  # some SDK code accesses this
        return self.working_dir

    def __call__(self, action: TerminalAction, conversation=None) -> TerminalObservation:
        cmd = (action.command or "").strip()
        if not cmd:
            return TerminalObservation.from_text(
                text="[empty command ignored; PodmanExecTerminalExecutor is stateless]",
                command="",
                exit_code=0,
            )
        if action.is_input:
            return TerminalObservation.from_text(
                text="[is_input=True is not supported: stateless podman-exec backend]",
                command=cmd,
                exit_code=1,
                is_error=True,
            )
        if action.reset:
            # Nothing to reset since every invocation is a fresh exec.
            return TerminalObservation.from_text(
                text="[terminal reset is a no-op with podman-exec backend]",
                command="[RESET]",
                exit_code=0,
            )
        timeout = action.timeout if action.timeout and action.timeout > 0 else self.default_timeout
        exit_code, out = podman_exec(self.container, cmd, cwd=self.working_dir, timeout=timeout)
        if _is_container_dead_error(exit_code, out):
            _request_conversation_pause(
                conversation,
                f"container {self.container} is gone (terminal)",
                iid_hint=self.container,
            )
        return TerminalObservation.from_text(
            text=out,
            command=cmd,
            exit_code=exit_code if exit_code is not None else -1,
            timeout=(exit_code == -1),
            is_error=(exit_code is not None and exit_code != 0),
        )

    def close(self) -> None:
        pass


class PodmanTerminalTool(TerminalTool):
    """Subclass that injects a PodmanExecTerminalExecutor via Tool(params=...)."""

    @classmethod
    def create(cls, conv_state, *, container: str,
               working_dir: str = "/testbed", **_kw) -> Sequence["PodmanTerminalTool"]:
        executor = PodmanExecTerminalExecutor(container=container, working_dir=working_dir)
        return [
            cls(
                action_type=TerminalAction,
                observation_type=TerminalObservation,
                description=(
                    "Execute a bash command inside the SWE-bench eval container "
                    f"(cwd defaults to {working_dir}). Each call is a fresh "
                    "`podman exec bash -lc`, so working-directory / env changes "
                    "from previous commands do NOT persist — use absolute paths "
                    "and chain with `&&` within a single command. The `is_input` "
                    "and `reset` flags are not supported."
                ),
                annotations=ToolAnnotations(
                    title="terminal",
                    readOnlyHint=False,
                    destructiveHint=True,
                    idempotentHint=False,
                    openWorldHint=True,
                ),
                executor=executor,
            )
        ]


register_tool("PodmanTerminalTool", PodmanTerminalTool)


# ──────────────────────────────────────────────────────────────────────────
# Custom FileEditorTool executor routed through podman exec
# ──────────────────────────────────────────────────────────────────────────

def _podman_read_file(container: str, path: str) -> tuple[int, str]:
    return _RUNTIME.read_file(container, path)


def _podman_write_file(container: str, path: str, text: str) -> tuple[int, str]:
    return _RUNTIME.write_file(container, path, text)


def _podman_stat(container: str, path: str) -> dict:
    return _RUNTIME.stat(container, path)


def _podman_listdir(container: str, path: str, maxdepth: int = 2) -> str:
    return _RUNTIME.listdir(container, path, maxdepth=maxdepth)


# Snippet context used when summarizing an edit in the observation (matches
# upstream FileEditor's default at utils/config.py::SNIPPET_CONTEXT_WINDOW).
_SNIPPET_CTX = 4
_MAX_VIEW_CHARS = 16000


def _fmt_numbered(text: str, start_line: int = 1) -> str:
    lines = text.splitlines()
    width = max(6, len(str(start_line + len(lines) - 1)))
    return "\n".join(f"{start_line + i:>{width}}\t{line}" for i, line in enumerate(lines))


def _truncate(text: str, limit: int = _MAX_VIEW_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n\n[...output truncated at {limit} chars]"


class PodmanFileEditorExecutor(ToolExecutor[FileEditorAction, FileEditorObservation]):
    """Container-native FileEditor.

    Mirrors the upstream openhands FileEditor's 5 commands (view / create /
    str_replace / insert / undo_edit) but routes every filesystem touch
    through `podman exec`. We keep per-path edit history in memory so
    `undo_edit` works within a single run.

    Trade-offs vs. the upstream implementation:
      * No encoding auto-detection — we decode everything as UTF-8 with
        replacement. Non-text files can still crash the agent; view
        refuses to open a path whose stat reports "binary file" is out
        of scope.
      * No Markdown / image support.
      * `view` of a directory does a `find -maxdepth 2`, no icons.
      * Snippet formatting around an edit matches upstream closely but
        isn't pixel-perfect (rich diff rendering is omitted).
    """

    def __init__(self, container: str, working_dir: str = "/testbed") -> None:
        self.container = container
        self.working_dir = working_dir
        # path → list[str] of prior file contents (most-recent last).
        self._history: dict[str, list[str]] = {}

    # ── validation ────────────────────────────────────────────────────────
    def _require_absolute(self, path: str) -> str | None:
        if not path.startswith("/"):
            return (f"The path `{path}` must be absolute (start with `/`). "
                    f"Relative paths are not supported.")
        return None

    def _err(self, command: str, msg: str, path: str | None = None) -> FileEditorObservation:
        return FileEditorObservation.from_text(
            text=msg, command=command, path=path, is_error=True,
        )

    # ── commands ──────────────────────────────────────────────────────────
    def _view(self, action: FileEditorAction) -> FileEditorObservation:
        stat = _podman_stat(self.container, action.path)
        if not stat["exists"]:
            return self._err("view", f"The path {action.path} does not exist.", action.path)
        if stat["is_dir"]:
            if action.view_range is not None:
                return self._err(
                    "view",
                    "`view_range` cannot be used with a directory path.", action.path,
                )
            listing = _podman_listdir(self.container, action.path)
            return FileEditorObservation.from_text(
                text=_truncate(listing),
                command="view",
                path=action.path,
                prev_exist=True,
            )
        rc, text = _podman_read_file(self.container, action.path)
        if rc != 0:
            return self._err("view", f"Failed to read {action.path}.", action.path)
        lines = text.splitlines(keepends=False)
        n = len(lines)
        start = 1
        end = n
        if action.view_range:
            if len(action.view_range) != 2:
                return self._err("view", "view_range must be [start, end].", action.path)
            start, end = action.view_range
            if start < 1 or start > n:
                return self._err(
                    "view",
                    f"Invalid view_range start={start}; file has {n} lines.",
                    action.path,
                )
            if end != -1 and (end < start or end > n):
                return self._err(
                    "view",
                    f"Invalid view_range end={end}; valid is [{start}, {n}].",
                    action.path,
                )
            if end == -1:
                end = n
            lines = lines[start - 1:end]
        snippet = _fmt_numbered("\n".join(lines), start_line=start)
        header = (f"Here's the result of running `cat -n` on {action.path}:"
                  if action.view_range is None
                  else f"Here's the result of running `cat -n {action.path}` from "
                       f"line {start} to {end}:")
        out = f"{header}\n{snippet}"
        return FileEditorObservation.from_text(
            text=_truncate(out),
            command="view",
            path=action.path,
            prev_exist=True,
        )

    def _create(self, action: FileEditorAction) -> FileEditorObservation:
        if action.file_text is None:
            return self._err("create", "`file_text` is required for `create`.", action.path)
        stat = _podman_stat(self.container, action.path)
        if stat["exists"]:
            return self._err(
                "create",
                f"File already exists at: {action.path}. Cannot overwrite via "
                "`create`; use `str_replace` or `insert` to modify it.",
                action.path,
            )
        rc, err = _podman_write_file(self.container, action.path, action.file_text)
        if rc != 0:
            return self._err("create", f"Failed to write {action.path}: {err}", action.path)
        self._history.setdefault(action.path, []).append(action.file_text)
        return FileEditorObservation.from_text(
            text=f"File created successfully at: {action.path}",
            command="create",
            path=action.path,
            new_content=action.file_text,
            prev_exist=False,
        )

    def _str_replace(self, action: FileEditorAction) -> FileEditorObservation:
        if action.old_str is None:
            return self._err("str_replace", "`old_str` is required.", action.path)
        if action.new_str is None:
            return self._err("str_replace", "`new_str` is required (use '' to delete).", action.path)
        if action.old_str == action.new_str:
            return self._err(
                "str_replace",
                "No replacement performed: `new_str` must differ from `old_str`.",
                action.path,
            )
        stat = _podman_stat(self.container, action.path)
        if not stat["exists"] or stat["is_dir"]:
            return self._err(
                "str_replace",
                f"{action.path} is not a regular file or does not exist.",
                action.path,
            )
        rc, content = _podman_read_file(self.container, action.path)
        if rc != 0:
            return self._err("str_replace", f"Failed to read {action.path}.", action.path)
        occurrences = content.count(action.old_str)
        if occurrences == 0:
            return self._err(
                "str_replace",
                f"`old_str` not found in {action.path}. Edit failed.",
                action.path,
            )
        if occurrences > 1:
            return self._err(
                "str_replace",
                (f"`old_str` is not unique in {action.path}: occurs "
                 f"{occurrences} times. Use a larger snippet with more "
                 "surrounding context so the match is unique."),
                action.path,
            )
        new_content = content.replace(action.old_str, action.new_str, 1)
        wrc, werr = _podman_write_file(self.container, action.path, new_content)
        if wrc != 0:
            return self._err("str_replace", f"Failed to write {action.path}: {werr}", action.path)
        self._history.setdefault(action.path, []).append(content)  # pre-edit
        # Build snippet around change for the observation.
        before_chunk = content.split(action.old_str, 1)[0]
        edit_start_line = before_chunk.count("\n") + 1
        new_lines = action.new_str.splitlines() or [""]
        edit_end_line = edit_start_line + len(new_lines) - 1
        new_content_lines = new_content.splitlines()
        ctx_start = max(1, edit_start_line - _SNIPPET_CTX)
        ctx_end = min(len(new_content_lines), edit_end_line + _SNIPPET_CTX)
        snippet = _fmt_numbered(
            "\n".join(new_content_lines[ctx_start - 1:ctx_end]),
            start_line=ctx_start,
        )
        out = (
            f"The file {action.path} has been edited. Here's a snippet of the "
            f"change with lines {ctx_start}-{ctx_end}:\n{snippet}"
        )
        return FileEditorObservation.from_text(
            text=_truncate(out),
            command="str_replace",
            path=action.path,
            old_content=content,
            new_content=new_content,
            prev_exist=True,
        )

    def _insert(self, action: FileEditorAction) -> FileEditorObservation:
        if action.insert_line is None:
            return self._err("insert", "`insert_line` is required.", action.path)
        if action.new_str is None:
            return self._err("insert", "`new_str` is required.", action.path)
        stat = _podman_stat(self.container, action.path)
        if not stat["exists"] or stat["is_dir"]:
            return self._err("insert",
                             f"{action.path} is not a regular file or does not exist.",
                             action.path)
        rc, content = _podman_read_file(self.container, action.path)
        if rc != 0:
            return self._err("insert", f"Failed to read {action.path}.", action.path)
        lines = content.splitlines(keepends=False)
        n = len(lines)
        if action.insert_line < 0 or action.insert_line > n:
            return self._err(
                "insert",
                f"insert_line={action.insert_line} is out of range [0, {n}].",
                action.path,
            )
        new_lines = action.new_str.split("\n")
        combined = lines[:action.insert_line] + new_lines + lines[action.insert_line:]
        new_content = "\n".join(combined)
        if content.endswith("\n"):
            new_content += "\n"
        wrc, werr = _podman_write_file(self.container, action.path, new_content)
        if wrc != 0:
            return self._err("insert", f"Failed to write {action.path}: {werr}", action.path)
        self._history.setdefault(action.path, []).append(content)
        edit_start_line = action.insert_line + 1
        edit_end_line = action.insert_line + len(new_lines)
        ctx_start = max(1, edit_start_line - _SNIPPET_CTX)
        new_content_lines = new_content.splitlines()
        ctx_end = min(len(new_content_lines), edit_end_line + _SNIPPET_CTX)
        snippet = _fmt_numbered(
            "\n".join(new_content_lines[ctx_start - 1:ctx_end]),
            start_line=ctx_start,
        )
        out = (
            f"The file {action.path} has been edited. {len(new_lines)} line(s) "
            f"inserted after line {action.insert_line}. Snippet "
            f"{ctx_start}-{ctx_end}:\n{snippet}"
        )
        return FileEditorObservation.from_text(
            text=_truncate(out),
            command="insert",
            path=action.path,
            old_content=content,
            new_content=new_content,
            prev_exist=True,
        )

    def _undo_edit(self, action: FileEditorAction) -> FileEditorObservation:
        hist = self._history.get(action.path, [])
        if not hist:
            return self._err("undo_edit", f"No prior edits recorded for {action.path}.", action.path)
        prev_content = hist.pop()
        wrc, werr = _podman_write_file(self.container, action.path, prev_content)
        if wrc != 0:
            return self._err("undo_edit", f"Failed to revert {action.path}: {werr}", action.path)
        return FileEditorObservation.from_text(
            text=f"Reverted last edit of {action.path}.",
            command="undo_edit",
            path=action.path,
            new_content=prev_content,
            prev_exist=True,
        )

    # ── entry point ───────────────────────────────────────────────────────
    def __call__(self, action: FileEditorAction, conversation=None) -> FileEditorObservation:
        err = self._require_absolute(action.path)
        if err:
            return self._err(action.command, err, action.path)
        # Cheap liveness probe: if the container is gone, short-circuit with a
        # clear error and pause the conversation so the agent loop exits.
        live_rc, live_out = podman_exec(
            self.container, "true", cwd="/", timeout=15,
        )
        if _is_container_dead_error(live_rc, live_out):
            _request_conversation_pause(
                conversation,
                f"container {self.container} is gone (file editor)",
                iid_hint=self.container,
            )
            return self._err(
                action.command,
                f"container is no longer running (rc={live_rc})",
                action.path,
            )
        try:
            if action.command == "view":
                return self._view(action)
            if action.command == "create":
                return self._create(action)
            if action.command == "str_replace":
                return self._str_replace(action)
            if action.command == "insert":
                return self._insert(action)
            if action.command == "undo_edit":
                return self._undo_edit(action)
            return self._err(action.command, f"Unknown command: {action.command}", action.path)
        except Exception as e:
            return self._err(
                action.command,
                f"PodmanFileEditorExecutor internal error: {e!r}",
                action.path,
            )


class PodmanFileEditorTool(FileEditorTool):
    """Injects a PodmanFileEditorExecutor; keeps upstream schema/description."""

    @classmethod
    def create(cls, conv_state, *, container: str,
               working_dir: str = "/testbed",
               **_kw) -> Sequence["PodmanFileEditorTool"]:
        executor = PodmanFileEditorExecutor(container=container, working_dir=working_dir)
        return [
            cls(
                action_type=FileEditorAction,
                observation_type=FileEditorObservation,
                description=(
                    "Edit files inside the SWE-bench eval container (at "
                    f"{working_dir} by default). Supports: view, create, "
                    "str_replace, insert, undo_edit. All operations are "
                    "executed via `podman exec` against the running "
                    "container; paths must be absolute (start with `/`). "
                    "No encoding auto-detection — UTF-8 is assumed."
                ),
                annotations=ToolAnnotations(
                    title="file_editor",
                    readOnlyHint=False,
                    destructiveHint=True,
                    idempotentHint=False,
                    openWorldHint=False,
                ),
                executor=executor,
            )
        ]


register_tool("PodmanFileEditorTool", PodmanFileEditorTool)


# ──────────────────────────────────────────────────────────────────────────
# Per-instance run
# ──────────────────────────────────────────────────────────────────────────

INSTRUCTION_TEMPLATE = """\
You are solving a GitHub issue in the repository at /testbed (a Python \
project checked out to the base commit {base_commit}). The environment is \
pre-configured: all dependencies are installed in the container's default \
Python environment.

<issue>
{problem_statement}
</issue>

Task: make the minimal changes under /testbed to satisfy the issue. You \
MUST NOT edit test files — the test suite has already been updated to \
express the correct behavior. You have two tools: a `file_editor` that \
supports view / create / str_replace / insert / undo_edit (absolute \
paths required, UTF-8 assumed), and a stateless bash `terminal` for \
exploration and running tests.

When you are done, call the `finish` tool. The evaluation harness will \
extract the git diff between HEAD and {base_commit} — there is no need \
to commit or push.
"""


def build_agent_for_instance(instance: dict, llm: LLM, container: str,
                             max_iter: int) -> Conversation:
    tools = [
        Tool(name="PodmanTerminalTool",
             params={"container": container, "working_dir": "/testbed"}),
        Tool(name="PodmanFileEditorTool",
             params={"container": container, "working_dir": "/testbed"}),
    ]
    agent = Agent(
        llm=llm,
        tools=tools,
        system_prompt_kwargs={"cli_mode": True},
    )
    # The Conversation workspace is only used for state bookkeeping now —
    # the terminal tool routes everything into podman, and we skip the
    # LocalWorkspace-touching tools (FileEditor/TaskTracker/Browser).
    workspace = LocalWorkspace(working_dir="/tmp")
    conv = Conversation(
        agent=agent,
        workspace=workspace,
        max_iteration_per_run=max_iter,
    )
    return conv


def _agent_finished(events: list) -> bool:
    for ev in reversed(events):
        if isinstance(ev, ActionEvent):
            return ev.action is not None and isinstance(ev.action, FinishAction)
        if isinstance(ev, MessageEvent) and ev.source == "agent":
            return False
    return False


def _agent_sent_message(events: list) -> bool:
    for ev in reversed(events):
        if isinstance(ev, MessageEvent) and ev.source == "agent":
            return True
        if isinstance(ev, ActionEvent):
            return False
    return False


def run_conversation(conv: Conversation, max_fake: int = 4) -> str:
    """Drive the conversation until finish / error / exhaustion. Returns status.

    `LocalConversation.run()` in openhands-sdk 1.17 takes no timeout
    argument — the per-step timeout is driven by LLM.timeout and the
    max_iteration cap is set at Conversation construction.
    """
    fake_n = 0
    while True:
        conv.run()
        if conv.state.execution_status != ConversationExecutionStatus.FINISHED:
            return conv.state.execution_status.value
        events = list(conv.state.events)
        if _agent_finished(events):
            return "finished_ok"
        if not _agent_sent_message(events):
            return "finished_no_msg"
        if fake_n >= max_fake:
            return "fake_limit"
        conv.send_message(
            "Please continue. When you believe the issue is resolved, call the "
            "`finish` tool. Do not ask for human help."
        )
        fake_n += 1


def extract_patch(container: str, base_commit: str) -> str:
    _rc, _ = podman_exec(container, "git add -A", timeout=60)
    _rc, out = podman_exec(
        container,
        f"git --no-pager diff --no-color {shlex.quote(base_commit)}",
        timeout=60,
    )
    return out


# Wall-clock deadline per instance. Enforced in two layers so a wedged agent
# loop (litellm retries, LLM hallucinating text after container is dead, etc.)
# can't hold a worker slot indefinitely:
#   1. Timer kills the container at T+deadline so podman_exec errors start
#   2. threading.Thread + join(timeout) abandons the worker thread itself,
#      letting the OUTER executor advance to the next instance.
# The orphaned worker thread (daemon) continues until it exits naturally or
# the process dies — we accept the thread leak in exchange for bounded wall
# time per instance.
# Overridable via env var (minutes).
INSTANCE_DEADLINE_MIN = float(os.environ.get("OPENHANDS_INSTANCE_DEADLINE_MIN", "60"))
INSTANCE_DEADLINE_SEC = INSTANCE_DEADLINE_MIN * 60


def run_one_instance(instance: dict, llm: LLM, max_iter: int,
                     out_dir: Path, model_name: str) -> dict:
    iid = instance["instance_id"]
    base_commit = instance["base_commit"]
    logger.info("── instance %s ──", iid)
    traj_path = out_dir / f"{iid}.traj.json"
    # Mutable boxes for cross-thread communication — `container` lets the
    # timer kill the container; `conv` lets the timer call pause() (OpenHands
    # SDK's thread-safe cancel API — breaks out of the run loop after the
    # current agent.step completes); `result` and `err` carry the outcome.
    container_box: list[str] = [""]
    conv_box: list[object] = [None]
    result: dict[str, object] = {}
    err: list[BaseException] = []

    def enforce_deadline() -> None:
        logger.warning(
            "instance %s: wall-clock deadline (%.0f min) exceeded — killing container + pausing conversation",
            iid, INSTANCE_DEADLINE_MIN,
        )
        cid = container_box[0]
        if cid:
            try:
                _RUNTIME.stop(cid)
            except Exception as exc:  # noqa: BLE001
                logger.warning("instance %s: deadline kill failed: %s", iid, exc)
        conv = conv_box[0]
        if conv is not None:
            try:
                conv.pause()  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001
                logger.warning("instance %s: conv.pause() failed: %s", iid, exc)

    # Background liveness probe: independent of the agent loop so it fires
    # even when the agent has stopped emitting tool calls (e.g., LLM-only
    # hallucination mode). Polls `podman container exists` every 15 s; on
    # first miss, calls conv.pause() — the conversation's `_state` lock
    # serializes naturally with the agent run loop, so no extra sync needed.
    # The pause still has to wait for the in-flight agent.step to return,
    # but a polling probe is much faster than waiting for the next tool call.
    liveness_stop = threading.Event()

    def liveness_check() -> None:
        while not liveness_stop.wait(15):
            cid = container_box[0]
            if not cid:
                continue
            try:
                alive = _RUNTIME.exists(cid)
            except Exception:
                continue
            if not alive:
                conv = conv_box[0]
                logger.warning(
                    "instance %s: container %s no longer exists (liveness probe) — pausing conversation",
                    iid, cid,
                )
                if conv is not None:
                    try:
                        conv.pause()  # type: ignore[attr-defined]
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("instance %s: conv.pause() failed: %s", iid, exc)
                return

    def worker() -> None:
        liveness_thread = threading.Thread(
            target=liveness_check, name=f"live-{iid}", daemon=True,
        )
        liveness_thread.start()
        try:
            container = start_instance_container(iid)
            container_box[0] = container
            conv = build_agent_for_instance(instance, llm, container, max_iter)
            conv_box[0] = conv
            instruction = INSTRUCTION_TEMPLATE.format(
                base_commit=base_commit,
                problem_statement=instance["problem_statement"],
            )
            conv.send_message(instruction)
            status = run_conversation(conv)
            patch = extract_patch(container, base_commit)
            stats = conv.conversation_stats.get_combined_metrics() \
                if hasattr(conv, "conversation_stats") else None
            traj = {
                "instance_id": iid,
                "model_name_or_path": model_name,
                "status": status,
                "patch_len": len(patch),
                "instruction": instruction,
                "history": [
                    ev.model_dump(mode="json") if hasattr(ev, "model_dump") else str(ev)
                    for ev in conv.state.events
                ],
                "metrics": stats.model_dump(mode="json") if stats and hasattr(stats, "model_dump") else None,
            }
            traj_path.write_text(json.dumps(traj, default=str))
            result["patch"] = patch
            result["status"] = status
        except BaseException as e:  # noqa: BLE001
            err.append(e)
        finally:
            # Signal liveness probe to exit on its next wait — the join is
            # bounded to 30 s so even if the probe is mid-`podman exec`, we
            # don't block the worker's exit.
            liveness_stop.set()
            try:
                liveness_thread.join(timeout=30)
            except Exception:
                pass
            cid = container_box[0]
            if cid:
                try:
                    stop_instance_container(cid)
                except Exception:
                    pass

    deadline_timer = threading.Timer(INSTANCE_DEADLINE_SEC, enforce_deadline)
    deadline_timer.daemon = True
    deadline_timer.start()
    t = threading.Thread(target=worker, name=f"inst-{iid}", daemon=True)
    t.start()
    # Give the worker a little slack past the container-kill deadline so the
    # kill has time to propagate into a raised exception before we abandon.
    t.join(INSTANCE_DEADLINE_SEC + 60)
    deadline_timer.cancel()

    if t.is_alive():
        logger.error(
            "instance %s: worker thread still alive at deadline+60s — abandoning (thread leaked)",
            iid,
        )
        # Ensure container is dead so the orphan thread's next exec fails.
        cid = container_box[0]
        if cid:
            try:
                _RUNTIME.stop(cid)
            except Exception:
                pass
        return {
            "instance_id": iid,
            "model_name_or_path": model_name,
            "model_patch": "",
            "status": f"error: deadline {INSTANCE_DEADLINE_MIN:.0f}m exceeded",
        }
    if err:
        e = err[0]
        logger.error("instance %s failed: %s\n%s", iid, e,
                     "".join(traceback.format_exception(type(e), e, e.__traceback__)))
        return {
            "instance_id": iid,
            "model_name_or_path": model_name,
            "model_patch": "",
            "status": f"error: {e!r}",
        }
    return {
        "instance_id": iid,
        "model_name_or_path": model_name,
        "model_patch": result.get("patch", ""),
        "status": result.get("status", ""),
    }


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def select_instances(df, *, instance: str | None, limit: int | None,
                     filter_re: str | None) -> list[dict]:
    rows = [row for _, row in df.iterrows()]
    if instance:
        rows = [r for r in rows if r["instance_id"] == instance]
    if filter_re:
        pat = re.compile(filter_re)
        rows = [r for r in rows if pat.search(r["instance_id"])]
    if limit:
        rows = rows[: int(limit)]
    return [dict(r) for r in rows]


def main() -> int:
    p = argparse.ArgumentParser(description="OpenHands SWE-bench driver (podman+vLLM)")
    p.add_argument("--subset", default="lite", choices=list(DATASET_MAP.keys()))
    p.add_argument("--split", default="test")
    p.add_argument("--instance")
    p.add_argument("--filter", help="regex over instance_id")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=1,
                   help="concurrent instances (each gets its own podman container)")
    p.add_argument("--max-iter", type=int, default=50,
                   help="max agent iterations per instance")
    p.add_argument("--model", required=True,
                   help="served model name, e.g. qwen3-30b-instruct")
    p.add_argument("--api-base", default=os.getenv("OPENAI_API_BASE",
                                                   "http://127.0.0.1:18020/v1"))
    p.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "dummy"))
    # Qwen3 official sampling recommendations: temp 0.6, top_p 0.8, top_k 20.
    # Without top_k the model samples from the full vocab and tends to drift
    # into long hallucinated outputs that don't terminate — observed in prior
    # 30B runs where agents never emitted a finish action.
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--top-k", type=float, default=20)
    p.add_argument("--max-input-tokens", type=int, default=131072)
    p.add_argument("--output-dir", required=True, help="directory for traj files + preds.json")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset %s/%s…", args.subset, args.split)
    ds_name = DATASET_MAP[args.subset]
    ds = load_dataset(ds_name, split=args.split)
    df = ds.to_pandas()
    instances = select_instances(df, instance=args.instance,
                                 limit=args.limit if args.limit > 0 else None,
                                 filter_re=args.filter)
    logger.info("Will evaluate %d instance(s)", len(instances))
    if not instances:
        logger.warning("No instances selected; nothing to do.")
        return 0

    # Build LLM (openhands-sdk.LLM is a pydantic model around litellm)
    # We go through litellm's openai provider by prefixing model with "openai/".
    # Use usage_id="agent" to register with OpenHands metrics.
    model = args.model if "/" in args.model else f"openai/{args.model}"
    # Shorter HTTP timeout so a stuck LLM call unblocks the state lock
    # (and pause()) within minutes instead of 25+. Empirically, 99.998% of
    # LLM calls in prior runs completed in <10 s — 90 s is 18× p99 headroom.
    # Keep num_retries at 5 (OpenHands default) so transient blips don't
    # fail an instance; with the shorter timeout, 5 retries cap worst case
    # at ~8 min (5 × 90 s + exponential backoff up to 16 s) instead of 30+.
    llm = LLM(
        usage_id="agent",
        model=model,
        base_url=args.api_base,
        api_key=args.api_key,  # pydantic SecretStr-compat: str is accepted
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_input_tokens=args.max_input_tokens,
        timeout=90,                  # per-request HTTP timeout (was 300)
        num_retries=5,               # OpenHands default
        retry_min_wait=4,            # was 8
        retry_max_wait=16,           # was 64
        native_tool_calling=True,
        drop_params=True,
        modify_params=True,
    )
    logger.info("LLM: %s @ %s", llm.model, llm.base_url)

    preds: dict[str, dict] = {}

    if args.workers <= 1 or len(instances) == 1:
        for inst in instances:
            r = run_one_instance(inst, llm, args.max_iter, out_dir, args.model)
            preds[r["instance_id"]] = {
                "instance_id": r["instance_id"],
                "model_name_or_path": args.model,
                "model_patch": r["model_patch"],
            }
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {
                ex.submit(run_one_instance, inst, llm, args.max_iter, out_dir, args.model): inst
                for inst in instances
            }
            for fut in as_completed(futs):
                r = fut.result()
                preds[r["instance_id"]] = {
                    "instance_id": r["instance_id"],
                    "model_name_or_path": args.model,
                    "model_patch": r["model_patch"],
                }

    preds_path = out_dir / "preds.json"
    preds_path.write_text(json.dumps(preds, indent=2))
    logger.info("Wrote %d predictions to %s", len(preds), preds_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
