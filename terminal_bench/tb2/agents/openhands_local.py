"""Harbor agent that runs openhands-sdk IN-PROCESS on the HPC host.

Unlike harbor's built-in `openhands` / `openhands-sdk` adapters which
install openhands-ai INSIDE the Daytona sandbox (LLM calls originate
from the sandbox and are blocked by Daytona Tier-2 SNI filter), this
adapter runs the openhands-sdk Agent/Conversation loop LOCALLY and
forwards the agent's terminal commands into the sandbox via harbor's
`environment.exec()` abstraction.

LLM call path:  HPC python → LLM() → your-vllm.example.com (works, same as terminus-2).
Tool call path: HPC python → environment.exec() → Daytona control plane
                → sandbox (works, bypasses SNI filter entirely).

Ported from ${CRANE_REPO_ROOT}/swe_bench/sh/openhands_swebench.py
(same pattern; podman exec → harbor env.exec).

Usage:
    harbor run ... \
        --agent openhands-local \
        --agent-import-path ${CRANE_REPO_ROOT}/terminal_bench/tb2/agents/openhands_local.py \
        --agent-kwarg api_base=https://your-vllm.example.com/v1 \
        --agent-kwarg model_name=qwen3-30b-instruct

Python env: ${CRANE_REPO_ROOT}/swe_bench/.venv-openhands (openhands-sdk 1.17.0).
"""
from __future__ import annotations

import asyncio
import contextvars
import shlex
import threading
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from openhands.sdk import LLM, Agent, Conversation, Tool, get_logger
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.event import ActionEvent, MessageEvent
from openhands.sdk.tool import ToolAnnotations, ToolExecutor, register_tool
from openhands.sdk.tool.builtins.finish import FinishAction
from openhands.sdk.workspace import LocalWorkspace
from openhands.tools.terminal import (
    TerminalAction,
    TerminalObservation,
    TerminalTool,
)


logger = get_logger(__name__)

DEFAULT_CMD_TIMEOUT = 300.0


# ── Patch harbor to schedule longest tasks first ────────────────────────────
def _patch_harbor_sort_trials_by_timeout() -> None:
    """Reorder Job._remaining_trial_configs so high-timeout_sec tasks run first.

    Why: in concurrent sweeps, a 60-min task (e.g. reshard-c4-data) submitted
    last blocks the entire sweep from completing. Submitting longest first
    lets short tasks fill the rest of the wall-clock and cuts overall time
    by ~20-40% in practice.
    """
    try:
        from harbor.job import Job
        from harbor.models.task.task import Task
    except Exception:
        return

    if getattr(Job, "_OH_LOCAL_TIMEOUT_SORT_PATCHED", False):
        return

    _orig = Job._init_remaining_trial_configs

    def _patched(self):
        _orig(self)

        def _key(tc):
            try:
                t = Task(tc.task.path)
                ts = t.config.agent.timeout_sec or 0
            except Exception:
                ts = 0
            return -float(ts)  # descending (negative = sort first)

        self._remaining_trial_configs.sort(key=_key)
        # Log the new order top 5
        top = self._remaining_trial_configs[:5]
        logger.info(
            "openhands-local patch: sorted trials by timeout_sec desc; top 5: %s",
            [pathlib.Path(tc.task.path).name for tc in top if tc.task.path],
        )

    Job._init_remaining_trial_configs = _patched
    Job._OH_LOCAL_TIMEOUT_SORT_PATCHED = True


_patch_harbor_sort_trials_by_timeout()
import pathlib  # used inside the patch above

# Per-trial registry indexed by trial_id (string). Tool params only carry
# scalars (the trial_id), so openhands Conversation.state can be JSON-
# serialized. The executor looks up the live (env, loop) pair from here
# at call time.
_TRIAL_REGISTRY: dict[str, tuple[BaseEnvironment, asyncio.AbstractEventLoop]] = {}
_registry_lock = threading.Lock()


# ── Terminal executor that routes through harbor env.exec ────────────────────

class _HarborEnvTerminalExecutor(ToolExecutor[TerminalAction, TerminalObservation]):
    """Stateless shell inside a harbor `BaseEnvironment` (Daytona, modal, …).

    Each TerminalAction becomes one `environment.exec()` call. WD is not
    preserved between invocations — the agent must use absolute paths or
    chain with `&&`. `is_input` / `reset` are no-ops for this backend.
    """

    is_pooled = False

    def __init__(
        self,
        trial_id: str,
        working_dir: str = "/app",
        default_timeout: float = DEFAULT_CMD_TIMEOUT,
    ) -> None:
        self._trial_id = trial_id
        self.working_dir = working_dir
        self.default_timeout = default_timeout

    @property
    def _working_dir(self) -> str:  # some SDK code accesses this
        return self.working_dir

    def __call__(self, action: TerminalAction, conversation=None) -> TerminalObservation:
        cmd = (action.command or "").strip()
        if not cmd:
            return TerminalObservation.from_text(
                text="[empty command ignored; stateless backend]",
                command="",
                exit_code=0,
            )
        if action.is_input:
            return TerminalObservation.from_text(
                text="[is_input=True not supported: stateless harbor-env backend]",
                command=cmd,
                exit_code=1,
                is_error=True,
            )
        if action.reset:
            return TerminalObservation.from_text(
                text="[terminal reset is a no-op with harbor-env backend]",
                command="[RESET]",
                exit_code=0,
            )
        timeout = (
            int(action.timeout) if action.timeout and action.timeout > 0
            else int(self.default_timeout)
        )

        # Wrap cmd so cwd is enforced per-invocation.
        full = f"cd {shlex.quote(self.working_dir)} && {cmd}"

        with _registry_lock:
            entry = _TRIAL_REGISTRY.get(self._trial_id)
        if entry is None:
            return TerminalObservation.from_text(
                text=f"[trial {self._trial_id} not in registry — agent bug]",
                command=cmd, exit_code=-1, is_error=True,
            )
        env, loop = entry
        # environment.exec is async; we're running inside openhands-sdk
        # Conversation.run() which is synchronous but invoked from a worker
        # thread. Dispatch the coroutine onto the main loop.
        fut = asyncio.run_coroutine_threadsafe(
            env.exec(command=full, timeout_sec=timeout),
            loop,
        )
        try:
            result = fut.result(timeout=timeout + 30)
        except Exception as e:  # noqa: BLE001
            return TerminalObservation.from_text(
                text=f"[exec dispatch failed: {type(e).__name__}: {e}]",
                command=cmd,
                exit_code=-1,
                timeout=True,
                is_error=True,
            )

        stdout = getattr(result, "stdout", "") or ""
        stderr = getattr(result, "stderr", "") or ""
        rc = getattr(result, "return_code", None)
        if rc is None:
            rc = getattr(result, "exit_code", -1)
        out = (stdout + stderr)
        if len(out) > 20000:
            out = out[:20000] + "\n\n[...output truncated at 20000 chars]"
        return TerminalObservation.from_text(
            text=out,
            command=cmd,
            exit_code=int(rc) if rc is not None else -1,
            is_error=(rc is not None and rc != 0),
        )

    def close(self) -> None:
        pass


class _HarborEnvTerminalTool(TerminalTool):
    # Force the LLM-visible tool name to "terminal" (parent TerminalTool's name).
    # `name` is ClassVar in ToolDefinition: __init_subclass__ skips auto-derivation
    # when the subclass's __dict__ already contains "name".
    name: ClassVar[str] = "terminal"

    @classmethod
    def create(
        cls,
        conv_state,
        *,
        trial_id: str,
        working_dir: str = "/app",
        **_kw,
    ) -> Sequence["_HarborEnvTerminalTool"]:
        executor = _HarborEnvTerminalExecutor(
            trial_id=trial_id, working_dir=working_dir,
        )
        return [
            cls(
                action_type=TerminalAction,
                observation_type=TerminalObservation,
                description=(
                    f"Execute a bash command inside the evaluation sandbox "
                    f"(cwd defaults to {working_dir}). Each call is a fresh "
                    f"invocation, so working-directory / env changes from "
                    f"previous commands do NOT persist — use absolute paths "
                    f"and chain with `&&` within a single command. The "
                    f"`is_input` and `reset` flags are not supported."
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


register_tool("HarborEnvTerminalTool", _HarborEnvTerminalTool)


# ── Harbor agent class ──────────────────────────────────────────────────────

SYSTEM_PROMPT_ADDENDUM = """\
You are solving a terminal-bench task. You have one tool: `terminal`, a stateless \
bash shell inside a sandbox. Each invocation is independent — use absolute paths \
and chain with `&&` within a single command when order matters.

When you believe the task is complete, call the `finish` tool. A separate verifier \
will run the tests automatically; you do not need to run them yourself unless you \
want to check your work.
"""


class OpenHandsLocal(BaseAgent):
    """Runs openhands-sdk in-process; routes tools via environment.exec()."""

    @staticmethod
    def name() -> str:
        return "openhands-local"

    def version(self) -> str:
        try:
            import openhands.sdk as _sdk
            v = getattr(_sdk, "__version__", None)
            if v:
                return f"sdk-{v}"
        except Exception:  # noqa: BLE001
            pass
        return "sdk-unknown"

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.6,
        top_p: float = 0.8,
        top_k: int = 20,
        max_input_tokens: int | None = None,
        max_iterations: int = 100,
        working_dir: str = "/app",
        **kwargs,
    ):
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        self._api_base = api_base
        self._api_key = api_key or "dummy-key-for-local-vllm"
        self._temperature = temperature
        self._top_p = top_p
        self._top_k = top_k
        self._max_input_tokens = max_input_tokens
        self._max_iterations = max_iterations
        self._working_dir = working_dir

    async def setup(self, environment: BaseEnvironment) -> None:
        return

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        # Normalize model to litellm's openai-compatible prefix.
        model = self.model_name or "qwen3-30b-instruct"
        if "/" not in model:
            model = f"openai/{model}"

        llm = LLM(
            usage_id="agent",
            model=model,
            base_url=self._api_base,
            api_key=self._api_key,
            temperature=self._temperature,
            top_p=self._top_p,
            top_k=self._top_k,
            max_input_tokens=self._max_input_tokens,
            timeout=90,
            num_retries=5,
            retry_min_wait=4,
            retry_max_wait=16,
            native_tool_calling=True,
            drop_params=True,
            modify_params=True,
        )

        loop = asyncio.get_running_loop()
        trial_id = uuid.uuid4().hex
        with _registry_lock:
            _TRIAL_REGISTRY[trial_id] = (environment, loop)
        tools = [
            Tool(
                name="HarborEnvTerminalTool",
                params={
                    "trial_id": trial_id,
                    "working_dir": self._working_dir,
                },
            ),
        ]
        agent = Agent(
            llm=llm,
            tools=tools,
            system_prompt_kwargs={"cli_mode": True},
            # Default = [FinishTool, ThinkTool]. Empirically:
            # - ThinkTool helps calibration; removing it causes premature
            #   finish_ok with reward=0 (agent skips the planning scaffold).
            # - FinishTool is essential: agent's clean exit signal.
            # Keep both.
        )
        workspace = LocalWorkspace(working_dir="/tmp")
        # persistence_dir = the trial's agent dir → events get written
        # incrementally, so even on timeout/cancellation we have a trace.
        persistence_dir = str(self.logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        conv = Conversation(
            agent=agent,
            workspace=workspace,
            max_iteration_per_run=self._max_iterations,
            stuck_detection=False,  # too aggressive on long agent traces
            persistence_dir=persistence_dir,
            conversation_id=uuid.UUID(trial_id),
        )
        conv.send_message(instruction + "\n\n" + SYSTEM_PROMPT_ADDENDUM)

        # Conversation.run() is synchronous; run in a thread so we don't block
        # the asyncio loop (which the TerminalTool executor dispatches back to).
        def _drive() -> str:
            fake_n = 0
            max_fake = 2
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
                    "Please continue. When you believe the task is complete, "
                    "call the `finish` tool."
                )
                fake_n += 1

        try:
            status = await asyncio.to_thread(_drive)
            logger.info("openhands-local finished with status=%s", status)
        finally:
            with _registry_lock:
                _TRIAL_REGISTRY.pop(trial_id, None)

        # Dump trajectory for inspection
        try:
            traj_path = self.logs_dir / "trajectory.json"
            traj_path.parent.mkdir(parents=True, exist_ok=True)
            events = [e.model_dump(mode="json") for e in conv.state.events]
            import json
            traj_path.write_text(json.dumps(events, indent=2, default=str))
        except Exception as e:  # noqa: BLE001
            logger.warning("failed to dump trajectory: %s", e)


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
