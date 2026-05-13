#!/usr/bin/env python3
"""
Roo Code Standalone Eval Runner
================================
Runs Roo Code evals against a local vLLM or any OpenAI-compatible model endpoint.
No Docker, PostgreSQL, or Redis required.

Usage:
    python3 roo-eval.py --config eval-config.yaml
    python3 roo-eval.py --config eval-config.yaml --language python --limit 3
    python3 roo-eval.py --config eval-config.yaml --exercises python/hello-world
    python3 roo-eval.py --config eval-config.yaml --dry-run

YAML config keys:
    provider        : openai (or anthropic / openrouter / etc.)
    model           : model name served by vLLM
    api_key         : API key (default: EMPTY for vLLM)
    base_url        : http://localhost:8003/v1
    context_window  : model's max context tokens (informational, default: 8192)
    languages       : list of languages to test
    evals_repo      : path to cloned Roo-Code-Evals repo
    concurrency     : parallel tasks (default: 1)
    iterations      : times to run each exercise (default: 1)
    timeout_seconds : per-exercise timeout in seconds (default: 300)
    temperature     : model temperature (default: 0, passed via ROO_MODEL_TEMPERATURE env var)
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from itertools import groupby

# Add pip-cache to path for pyyaml
try:
    import yaml
except ImportError:
    sys.exit("pyyaml is required: pip install pyyaml")


# ── Reference pricing (GPT-5.4 tiers, picked by model size) ──────────────────
# Used to compute a reference cost for local vLLM runs where totalCost is 0.
#   < 10B  params → nano   ($0.20 / $0.02 / $1.25 per 1M in/cached/out)
#   10–40B params → mini   ($0.75 / $0.075 / $4.50)
#   > 40B  params → gpt-5.4 ($2.50 / $0.25 / $15.00)
PRICING_TIERS = {
    "nano":    {"input": 0.20 / 1_000_000, "cached_input": 0.02 / 1_000_000, "output": 1.25  / 1_000_000},
    "mini":    {"input": 0.75 / 1_000_000, "cached_input": 0.075 / 1_000_000, "output": 4.50 / 1_000_000},
    "gpt-5.4": {"input": 2.50 / 1_000_000, "cached_input": 0.25 / 1_000_000, "output": 15.00 / 1_000_000},
}

_SIZE_B_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])")

def _parse_model_size_b(model: str) -> float | None:
    """Extract parameter count in billions from a model identifier.

    Matches patterns like '4b', '30B', '70b', '405B'. For MoE names like
    'Qwen3-30B-A3B' the max (total params) is used. Returns None if no
    size marker is found.
    """
    if not model:
        return None
    matches = _SIZE_B_RE.findall(model)
    if not matches:
        return None
    try:
        return max(float(m) for m in matches)
    except ValueError:
        return None


def _tier_for_model(model: str) -> str:
    size_b = _parse_model_size_b(model)
    if size_b is None:
        return "nano"  # fallback for names without an explicit parameter count
    if size_b < 10:
        return "nano"
    if size_b <= 40:
        return "mini"
    return "gpt-5.4"


def ref_cost(input_tokens: int, output_tokens: int, cached_input_tokens: int = 0,
             *, model: str = "") -> float:
    """Compute reference cost using the GPT-5.4 tier selected by model size."""
    p = PRICING_TIERS[_tier_for_model(model)]
    return (input_tokens * p["input"]
            + cached_input_tokens * p["cached_input"]
            + output_tokens * p["output"])


# ── Paths ──────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent.resolve()
ROO_CODE_DIR = SCRIPT_DIR / "Roo-Code"
DEFAULT_EVALS_REPO = SCRIPT_DIR / "evals"
RESULTS_DIR = SCRIPT_DIR / "eval-results"
CLI_DIST = ROO_CODE_DIR / "apps/cli/dist/index.js"

# ── Node.js detection ─────────────────────────────────────────────────────────
# Auto-detect node from: 1) NVM_DIR env, 2) common nvm paths, 3) system PATH

def _find_node_bin() -> Path:
    """Find node binary directory, preferring nvm if available."""
    nvm_dir = os.environ.get("NVM_DIR", "")
    search_dirs = []
    if nvm_dir:
        search_dirs.append(Path(nvm_dir))
    search_dirs += [Path.home() / ".nvm", Path("/data/.nvm")]
    for nvm in search_dirs:
        if nvm.is_dir():
            # Find highest version node binary
            versions = sorted(nvm.glob("versions/node/v*/bin/node"), reverse=True)
            if versions:
                return versions[0].parent
    # Fallback: use node from PATH
    node_path = shutil.which("node")
    if node_path:
        return Path(node_path).parent
    raise FileNotFoundError(
        "Node.js not found. Install via nvm (https://github.com/nvm-sh/nvm) "
        "or ensure 'node' is on PATH."
    )

NODE_BIN = _find_node_bin()
NODE = NODE_BIN / "node"


# ── Helpers ────────────────────────────────────────────────────────────────────

def env_with_node():
    env = os.environ.copy()
    # Build PATH: node bin + whatever the user already has
    extra_dirs = [str(NODE_BIN)]
    # Include active venv so python3/pytest resolve correctly
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        extra_dirs.insert(0, str(Path(venv) / "bin"))
    # Add common tool directories if they exist
    for p in [shutil.which("go"), shutil.which("cargo"), shutil.which("java")]:
        if p:
            extra_dirs.append(str(Path(p).parent))
    extra = ":".join(extra_dirs)
    env["PATH"] = f"{extra}:{env.get('PATH', '')}"
    return env


def log(msg, prefix=""):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {prefix}{msg}", flush=True)


def log_ok(msg):   log(msg, prefix="✓ ")
def log_err(msg):  log(msg, prefix="✗ ")
def log_info(msg): log(msg, prefix="  ")



# ── Token recovery from log ──────────────────────────────────────────────────

def _read_tokens_from_ui_messages(ui_path):
    """Read per-API-call token counts from a Roo CLI ui_messages.json file.

    Returns (input_tokens, cached_input_tokens, output_tokens) where:
    - If the API reported cacheReads > 0, use those directly.
    - Otherwise, estimate cached input via heuristic:
      call N's cached_input ≈ call (N-1)'s total input tokens
      (multi-turn prefix sharing).
    """
    try:
        msgs = json.load(open(ui_path))
    except (json.JSONDecodeError, OSError):
        return 0, 0, 0

    # Collect per-call data
    calls = []  # list of (tokensIn, tokensOut, cacheReads)
    for m in msgs:
        if m.get("type") == "say" and m.get("say") == "api_req_started" and m.get("text"):
            try:
                data = json.loads(m["text"])
                t_in = data.get("tokensIn", 0) or 0
                t_out = data.get("tokensOut", 0) or 0
                t_cache = data.get("cacheReads", 0) or 0
                if t_in > 0 or t_out > 0:
                    calls.append((t_in, t_out, t_cache))
            except (json.JSONDecodeError, KeyError, TypeError):
                pass

    if not calls:
        return 0, 0, 0

    total_out = sum(c[1] for c in calls)
    api_cache = sum(c[2] for c in calls)

    if api_cache > 0:
        # API reported real cache stats — use them
        total_in = sum(c[0] for c in calls)
        uncached_in = max(0, total_in - api_cache)
        return uncached_in, api_cache, total_out

    # Heuristic: call N's cached ≈ call (N-1)'s input tokens
    total_in = sum(c[0] for c in calls)
    est_cached = sum(calls[i - 1][0] for i in range(1, len(calls)))
    uncached_in = max(0, total_in - est_cached)
    return uncached_in, est_cached, total_out


def _estimate_cache_from_ephemeral(ephemeral_dir, token_result):
    """Estimate cached input tokens from per-API-call data in ui_messages.json.

    Always reads ephemeral dir to split input_tokens into uncached + cached
    using the heuristic (call N cached ≈ call N-1 input). Also serves as
    token recovery when the result event was missed.
    """
    import glob as glob_mod
    try:
        if not ephemeral_dir:
            return
        for ui_path in glob_mod.glob(os.path.join(ephemeral_dir, "global-storage/tasks/*/ui_messages.json")):
            uncached_in, cached_in, total_out = _read_tokens_from_ui_messages(ui_path)
            if uncached_in > 0 or cached_in > 0 or total_out > 0:
                token_result["input_tokens"] = uncached_in
                token_result["cached_input_tokens"] = cached_in
                token_result["output_tokens"] = total_out
                return
    except (OSError, ValueError):
        pass



# ── Process cleanup ───────────────────────────────────────────────────────────

def _kill_proc_tree(pid: int):
    """Kill a process and ALL its descendants recursively."""
    def _get_children(parent_pid):
        """Get all child PIDs recursively."""
        children = []
        try:
            with open(f"/proc/{parent_pid}/task/{parent_pid}/children") as f:
                direct = [int(p) for p in f.read().split()]
        except (OSError, ValueError):
            direct = []
        for child in direct:
            children.append(child)
            children.extend(_get_children(child))
        return children

    # Collect entire tree BEFORE killing (killing parent re-parents children to init)
    all_pids = _get_children(pid)
    all_pids.append(pid)

    # Kill process group first (catches most processes)
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, OSError, PermissionError):
        pass

    # Then kill each collected PID individually (catches escapees)
    for p in all_pids:
        try:
            os.kill(p, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass



# ── CLI runner ─────────────────────────────────────────────────────────────────

def run_roo_cli(
    *,
    prompt_file: Path,
    workspace: Path,
    provider: str,
    model: str,
    api_key: str,
    base_url: str | None,
    context_window: int | None,
    timeout: int,
    extension_path: str | None,
    extra_args: list,
    log_file: Path,
    label: str,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    no_think: bool = False,
) -> dict:
    """Run Roo Code CLI and return result with token usage.

    Returns dict with keys: success, input_tokens, cached_input_tokens, output_tokens, total_cost.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fail_result = {"success": False, "input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0, "total_cost": 0.0}

    # Optionally prepend /no_think to disable Qwen3 thinking mode
    if no_think:
        orig = prompt_file.read_text()
        tmp_prompt = log_file.parent / f"prompt-{log_file.stem}.md"
        tmp_prompt.write_text(f"/no_think\n{orig}")
        prompt_file = tmp_prompt

    cmd = [
        str(NODE), "--max-old-space-size=2048", str(CLI_DIST),
        "--prompt-file", str(prompt_file),
        "--workspace",   str(workspace),
        "--provider",    provider,
        "--model",       model,
        "--api-key",     api_key,
        "--oneshot",
        "--ephemeral",
        "--print",
        "--output-format", "stream-json",
    ]

    if base_url:
        cmd += ["--base-url", base_url]
    if context_window:
        cmd += ["--context-window", str(context_window)]
    if temperature is not None:
        cmd += ["--temperature", str(temperature)]
    if top_p is not None:
        cmd += ["--top-p", str(top_p)]
    if top_k is not None:
        cmd += ["--top-k", str(top_k)]
    if extension_path:
        cmd += ["--extension", extension_path]

    cmd += extra_args

    log_info(f"[{label}] Starting CLI → {provider}/{model}")
    start = time.time()

    token_result = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0, "total_cost": 0.0}

    import glob as glob_mod

    proc = None
    try:
        # Give each CLI process its own TMPDIR so the ephemeral dir
        # (created as $TMPDIR/roo-cli-{id}) is in a known, isolated location.
        # This avoids race conditions when detecting ephemeral dirs under concurrency.
        import uuid as _uuid
        isolate_dir = Path(f"/tmp/roo-eval-isolate-{_uuid.uuid4().hex[:12]}")
        isolate_dir.mkdir(parents=True, exist_ok=True)

        cli_env = env_with_node()
        cli_env["TMPDIR"] = str(isolate_dir)
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROO_CODE_DIR),
            env=cli_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        # Find the ephemeral dir — it's the only roo-cli-* dir inside our isolated TMPDIR
        ephemeral_dir = None
        for _ in range(20):  # poll up to 10s
            candidates = list(isolate_dir.glob("roo-cli-*"))
            if candidates:
                ephemeral_dir = str(candidates[0])
                break
            time.sleep(0.5)

        def _drain_stdout(proc, log_file, token_result):
            try:
                with open(log_file, "w") as lf:
                    for raw_line in proc.stdout:
                        line = raw_line.decode("utf-8", errors="replace")
                        lf.write(line)
                        stripped = line.strip()
                        if stripped.startswith("{"):
                            try:
                                evt = json.loads(stripped)
                                if evt.get("type") == "result" and isinstance(evt.get("cost"), dict):
                                    c = evt["cost"]
                                    total_in = c.get("inputTokens") or 0
                                    cache_reads = c.get("cacheReads") or 0
                                    token_result["input_tokens"] = max(0, total_in - cache_reads)
                                    token_result["cached_input_tokens"] = cache_reads
                                    token_result["output_tokens"] = c.get("outputTokens") or 0
                                    token_result["total_cost"] = c.get("totalCost") or 0.0
                            except (json.JSONDecodeError, KeyError, TypeError):
                                pass
            except (ValueError, OSError):
                pass

        reader = threading.Thread(target=_drain_stdout, args=(proc, log_file, token_result), daemon=True)
        reader.start()

        remaining = max(1, timeout - (time.time() - start))
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            _kill_proc_tree(proc.pid)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            reader.join(timeout=2)
            # Estimate cached tokens (and recover totals if missed) from ephemeral dir
            if ephemeral_dir:
                _estimate_cache_from_ephemeral(ephemeral_dir, token_result)
            shutil.rmtree(isolate_dir, ignore_errors=True)
            # Compute reference cost if CLI reported 0 (local vLLM)
            if token_result["total_cost"] == 0.0 and (token_result["input_tokens"] or token_result["output_tokens"]):
                token_result["total_cost"] = ref_cost(token_result["input_tokens"], token_result["output_tokens"], token_result["cached_input_tokens"], model=model)
            total_tok = token_result["input_tokens"] + token_result["cached_input_tokens"] + token_result["output_tokens"]
            log_err(f"[{label}] CLI timed out after {time.time()-start:.0f}s (killed, "
                    f"tokens: {token_result['input_tokens']} in / {token_result['cached_input_tokens']} cached / "
                    f"{token_result['output_tokens']} out (sum {total_tok}), "
                    f"ref_cost: ${token_result['total_cost']:.4f})")
            return {"success": False, **token_result}

        reader.join(timeout=5)
        _kill_proc_tree(proc.pid)
        elapsed = time.time() - start
        # Estimate cached tokens (and recover totals if missed) from ephemeral dir
        if ephemeral_dir:
            _estimate_cache_from_ephemeral(ephemeral_dir, token_result)
        shutil.rmtree(isolate_dir, ignore_errors=True)
        success = proc.returncode == 0
        # Compute reference cost if CLI reported 0 (local vLLM)
        if token_result["total_cost"] == 0.0 and (token_result["input_tokens"] or token_result["output_tokens"]):
            token_result["total_cost"] = ref_cost(token_result["input_tokens"], token_result["output_tokens"], token_result["cached_input_tokens"], model=model)
        total_tok = token_result["input_tokens"] + token_result["cached_input_tokens"] + token_result["output_tokens"]
        log_info(f"[{label}] CLI finished (rc={proc.returncode}, {elapsed:.0f}s, "
                 f"tokens: {token_result['input_tokens']} in / {token_result['cached_input_tokens']} cached / "
                 f"{token_result['output_tokens']} out (sum {total_tok}), "
                 f"ref_cost: ${token_result['total_cost']:.4f})")
        return {"success": success, **token_result}
    except Exception as e:
        if proc is not None:
            _kill_proc_tree(proc.pid)
            try:
                proc.wait(timeout=5)
            except:
                proc.kill()
                proc.wait()
        shutil.rmtree(isolate_dir, ignore_errors=True)
        log_err(f"[{label}] CLI error: {e}")
        return fail_result


# ── Unit test runner ───────────────────────────────────────────────────────────

TEST_COMMANDS: dict[str, list[str]] = {
    "go":         ["go test ./..."],
    "java":       ["./gradlew test"],
    "javascript": ["pnpm install", "pnpm test"],
    "python":     ["python3 -m pytest -o markers=task *_test.py"],
    "rust":       ["cargo test"],
}

def run_unit_tests(*, workspace: Path, language: str, label: str) -> bool:
    cwd = workspace
    commands = TEST_COMMANDS.get(language, [])
    if not commands:
        log_err(f"[{label}] No test commands for language: {language}")
        return False

    for cmd_str in commands:
        log_info(f"[{label}] Running: {cmd_str}")
        try:
            with subprocess.Popen(
                cmd_str,
                cwd=str(cwd),
                env=env_with_node(),
                shell=True,
                start_new_session=True,
            ) as proc:
                try:
                    proc.communicate(timeout=120)
                except subprocess.TimeoutExpired:
                    _kill_proc_tree(proc.pid)
                    proc.communicate()
                    log_err(f"[{label}] Test timed out: {cmd_str}")
                    return False
            if proc.returncode != 0:
                log_err(f"[{label}] Test failed: {cmd_str}")
                return False
        except subprocess.TimeoutExpired:
            log_err(f"[{label}] Test timed out: {cmd_str}")
            return False
        except Exception as e:
            log_err(f"[{label}] Test error: {e}")
            return False
    return True


# ── Exercise discovery ─────────────────────────────────────────────────────────

def discover_exercises(evals_repo: Path, languages: list) -> list:
    exercises = []
    for lang in languages:
        lang_dir = evals_repo / lang
        if not lang_dir.is_dir():
            log_err(f"Language directory not found: {lang_dir}")
            continue
        for entry in sorted(lang_dir.iterdir()):
            if entry.is_dir() and not entry.name.startswith("."):
                exercises.append((lang, entry.name))
    return exercises


# ── Single task (one exercise × one iteration) ────────────────────────────────

def run_task(
    *,
    evals_repo: Path,
    language: str,
    exercise: str,
    iteration: int,
    config: dict,
    results_dir: Path,
    lock: threading.Lock,
) -> dict:
    import tempfile
    iter_suffix = f" [iter {iteration}]" if config["iterations"] > 1 else ""
    label = f"{language}/{exercise}{iter_suffix}"

    result = {
        "language":   language,
        "exercise":   exercise,
        "iteration":  iteration,
        "passed":     False,
        "cli_success": False,
        "error":      None,
        "started_at": datetime.now().isoformat(),
        "finished_at": None,
        "duration_s": 0,
        "attempts":   0,
        "input_tokens":  0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "total_cost":    0.0,
    }

    start = time.time()
    per_attempt_timeout = config.get("timeout_seconds", 300)
    max_attempts = config.get("max_attempts", 11)
    total_timeout = config.get("total_timeout_seconds", 900)
    log(f"Starting: {label}")

    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        elapsed = time.time() - start
        if elapsed >= total_timeout:
            log_err(f"[{label}] Total timeout ({total_timeout}s) reached after {attempt-1} attempts")
            break

        # Use node-local scratch for workspaces — NFS doesn't support flock(),
        # which Go (go.mod) and Rust (cargo) need for file locking.
        ws_root = Path(os.environ.get("ROO_EVAL_WORKSPACES", os.environ.get("TMPDIR", "/tmp")) + "/roo-eval-workspaces")
        ws_root.mkdir(parents=True, exist_ok=True)
        tmp_dir = Path(tempfile.mkdtemp(prefix=f"roo-eval-{language}-{exercise}-iter{iteration}-", dir=ws_root))
        try:
            # 1. Copy exercise to isolated temp directory
            workspace = tmp_dir / exercise
            shutil.copytree(evals_repo / language / exercise, workspace)

            # 2. Run Roo Code CLI
            iter_tag = f"-iter{iteration}" if config["iterations"] > 1 else ""
            attempt_tag = f"-attempt{attempt}" if attempt > 1 else ""
            log_file = results_dir / "logs" / f"{language}-{exercise}{iter_tag}{attempt_tag}.log"

            cli_result = run_roo_cli(
                prompt_file=evals_repo / "prompts" / f"{language}.md",
                workspace=workspace,
                provider=config["provider"],
                model=config["model"],
                api_key=config.get("api_key", "EMPTY"),
                base_url=config.get("base_url"),
                context_window=config.get("context_window"),
                timeout=per_attempt_timeout,
                extension_path=config.get("extension_path"),
                extra_args=config.get("extra_cli_args", []),
                log_file=log_file,
                label=f"{label} attempt {attempt}",
                no_think=config.get("no_think", False),
                temperature=config.get("temperature"),
                top_p=config.get("top_p"),
                top_k=config.get("top_k"),
            )
            result["cli_success"] = cli_result["success"]
            result["input_tokens"]  += cli_result["input_tokens"]
            result["cached_input_tokens"] += cli_result["cached_input_tokens"]
            result["output_tokens"] += cli_result["output_tokens"]
            result["total_cost"]    += cli_result["total_cost"]

            # 3. Run unit tests in isolated workspace
            log_info(f"[{label}] Running unit tests (attempt {attempt})...")
            passed = run_unit_tests(
                workspace=workspace,
                language=language,
                label=label,
            )

            if passed:
                result["passed"] = True
                result["attempts"] = attempt
                log_ok(f"[{label}] PASSED on attempt {attempt}")
                break
            else:
                log_err(f"[{label}] FAILED attempt {attempt}/{max_attempts}")

        except Exception as e:
            log_err(f"[{label}] Unexpected error on attempt {attempt}: {e}")
            result["error"] = str(e)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    result["attempts"] = attempt
    result["finished_at"] = datetime.now().isoformat()
    result["duration_s"] = round(time.time() - start, 1)

    if not result["passed"]:
        log_err(f"[{label}] FAILED after {attempt} attempts ({result['duration_s']:.0f}s)")

    return result


# ── Config loader ──────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    config_file = Path(config_path).resolve()
    config_dir  = config_file.parent

    with open(config_file) as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("provider",        "openai")
    cfg.setdefault("model",           "default")
    cfg.setdefault("api_key",         "EMPTY")
    cfg.setdefault("languages",       ["python", "javascript", "go", "rust", "java"])
    cfg.setdefault("concurrency",     1)
    cfg.setdefault("iterations",      1)
    cfg.setdefault("timeout_seconds", 300)
    cfg.setdefault("context_window",  8192)
    cfg.setdefault("evals_repo",      str(DEFAULT_EVALS_REPO))

    # Resolve relative evals_repo relative to config file location
    repo = cfg["evals_repo"]
    if not Path(repo).is_absolute():
        cfg["evals_repo"] = str((config_dir / repo).resolve())

    return cfg


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Roo Code Standalone Eval Runner")
    parser.add_argument("--config",      required=True, help="Path to YAML config file")
    parser.add_argument("--exercises",   nargs="+",     help="Specific exercises (e.g. python/hello-world)")
    parser.add_argument("--language",                   help="Run only this language")
    parser.add_argument("--concurrency", type=int,      help="Override concurrency from config")
    parser.add_argument("--iterations",  type=int,      help="Override iterations from config")
    parser.add_argument("--limit",       type=int,      help="Max exercises per language")
    parser.add_argument("--dry-run",     action="store_true", help="List tasks without running")
    args = parser.parse_args()

    config = load_config(args.config)

    if not CLI_DIST.exists():
        log_err(f"Roo Code CLI not built: {CLI_DIST}")
        log_info("Run: bash setup-eval.sh")
        sys.exit(1)

    evals_repo = Path(config["evals_repo"]).resolve()
    if not evals_repo.exists():
        log_err(f"Evals repo not found: {evals_repo}")
        log_info(f"Clone it: git clone https://github.com/RooCodeInc/Roo-Code-Evals.git {evals_repo}")
        sys.exit(1)

    # Apply CLI overrides
    if args.concurrency: config["concurrency"] = args.concurrency
    if args.iterations:  config["iterations"]  = args.iterations

    concurrency = config["concurrency"]
    iterations  = config["iterations"]

    # Determine exercises
    if args.exercises:
        exercises = []
        for ex in args.exercises:
            parts = ex.split("/", 1)
            if len(parts) != 2:
                log_err(f"Invalid format (expected language/exercise): {ex}")
                sys.exit(1)
            exercises.append((parts[0], parts[1]))
    else:
        languages = [args.language] if args.language else config["languages"]
        exercises = discover_exercises(evals_repo, languages)

        if args.limit:
            limited = []
            for _, group in groupby(exercises, key=lambda x: x[0]):
                limited.extend(list(group)[:args.limit])
            exercises = limited

    if not exercises:
        log_err("No exercises found.")
        sys.exit(1)

    # Expand exercises × iterations into task list
    tasks = [(lang, ex, it) for lang, ex in exercises for it in range(1, iterations + 1)]
    total_tasks = len(tasks)

    # Print summary header
    log_info(f"Evals repo:     {evals_repo}")
    log_info(f"Provider:       {config['provider']}")
    log_info(f"Model:          {config['model']}  (pricing tier: {_tier_for_model(config['model'])})")
    log_info(f"Base URL:       {config.get('base_url', '(default)')}")
    log_info(f"Context window: {config['context_window']:,} tokens")
    log_info(f"Exercises:      {len(exercises)}")
    log_info(f"Iterations:     {iterations}")
    log_info(f"Total tasks:    {total_tasks}")
    log_info(f"Concurrency:    {concurrency}")
    log_info(f"Timeout:        {config['timeout_seconds']}s / task")
    print()

    if args.dry_run:
        for lang, ex, it in tasks:
            iter_tag = f" [iter {it}]" if iterations > 1 else ""
            print(f"  {lang}/{ex}{iter_tag}")
        return

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = RESULTS_DIR / run_ts
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "logs").mkdir(exist_ok=True)

    # Save config snapshot
    with open(results_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    log(f"Results: {results_dir}")
    print()

    lock = threading.Lock()
    all_results = []

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(
                run_task,
                evals_repo=evals_repo,
                language=lang,
                exercise=ex,
                iteration=it,
                config=config,
                results_dir=results_dir,
                lock=lock,
            ): (lang, ex, it)
            for lang, ex, it in tasks
        }

        for future in as_completed(futures):
            r = future.result()
            with lock:
                all_results.append(r)

    # ── Compute overall stats ─────────────────────────────────────────────────
    from collections import defaultdict
    ex_results: dict[tuple, list] = defaultdict(list)
    ex_iter1: dict[tuple, bool] = {}
    for r in all_results:
        key = (r["language"], r["exercise"])
        ex_results[key].append(r["passed"])
        if r["iteration"] == 1:
            ex_iter1[key] = r["passed"]

    total_ex = len(ex_results)
    total_tasks = len(all_results)
    any_pass  = sum(1 for v in ex_results.values() if any(v))
    all_pass  = sum(1 for v in ex_results.values() if all(v))
    pass_at_1 = sum(1 for v in ex_iter1.values() if v)
    passed_total = sum(1 for r in all_results if r["passed"])
    cli_success  = sum(1 for r in all_results if r["cli_success"])

    total_input  = sum(r.get("input_tokens", 0) for r in all_results)
    total_cached = sum(r.get("cached_input_tokens", 0) for r in all_results)
    total_output = sum(r.get("output_tokens", 0) for r in all_results)
    total_cost   = sum(r.get("total_cost", 0) for r in all_results)
    total_all    = total_input + total_cached + total_output
    tier         = _tier_for_model(config["model"])

    overall = {
        "model":        config["model"],
        "pricing_tier": tier,
        "exercises":    total_ex,
        "iterations":   iterations,
        "total_tasks":  total_tasks,
        "pass_at_1":    pass_at_1,
        "pass_at_k":    any_pass,
        "pass_all":     all_pass,
        "passed_total": passed_total,
        "cli_success":  cli_success,
        "pass_at_1_pct":    round(100 * pass_at_1 / total_ex, 1) if total_ex else 0,
        "pass_at_k_pct":    round(100 * any_pass / total_ex, 1) if total_ex else 0,
        "passed_total_pct": round(100 * passed_total / total_tasks, 1) if total_tasks else 0,
        "cli_success_pct":  round(100 * cli_success / total_tasks, 1) if total_tasks else 0,
        "total_input_tokens":  total_input,
        "total_cached_input_tokens": total_cached,
        "total_output_tokens": total_output,
        "total_tokens":        total_all,
        "total_cost":          round(total_cost, 4),
        "avg_input_per_task":  round(total_input / total_tasks) if total_tasks else 0,
        "avg_cached_input_per_task": round(total_cached / total_tasks) if total_tasks else 0,
        "avg_output_per_task": round(total_output / total_tasks) if total_tasks else 0,
        "avg_total_per_task":  round(total_all / total_tasks) if total_tasks else 0,
    }

    # Save results with overall summary
    with open(results_dir / "results.json", "w") as f:
        json.dump({"overall": overall, "tasks": all_results}, f, indent=2)

    print()
    print("=" * 64)
    if iterations == 1:
        print(f"  RESULTS: {passed_total}/{total_ex} passed  ({overall['passed_total_pct']}%)")
    else:
        print(f"  RESULTS  ({config['model']}, iterations={iterations})")
        print(f"  pass@1              : {pass_at_1}/{total_ex}  ({overall['pass_at_1_pct']}%)")
        print(f"  pass@{iterations}              : {any_pass}/{total_ex}  ({overall['pass_at_k_pct']}%)")
        print(f"  Passed (total)      : {passed_total}/{total_tasks}  ({overall['passed_total_pct']}%)")
        print(f"  CLI success         : {cli_success}/{total_tasks}  ({overall['cli_success_pct']}%)")
    # Recompute reference cost from tokens if total_cost is still 0
    if total_cost == 0.0 and (total_input > 0 or total_cached > 0 or total_output > 0):
        total_cost = ref_cost(total_input, total_output, total_cached, model=config["model"])
        overall["total_cost"] = round(total_cost, 4)
    if total_input > 0 or total_cached > 0 or total_output > 0:
        print(f"  Tokens              : {total_input:,} in / {total_cached:,} cached_in / {total_output:,} out  (sum {total_all:,})")
        print(f"  Avg per task        : {overall['avg_input_per_task']:,} in / {overall['avg_cached_input_per_task']:,} cached_in / {overall['avg_output_per_task']:,} out  (sum {overall['avg_total_per_task']:,})")
        print(f"  Ref cost ({tier:<7s}) : ${total_cost:.4f}")
    print("=" * 64)

    # Per-language breakdown
    by_lang: dict[str, dict] = defaultdict(lambda: {"ex": 0, "pass_all": 0, "pass_any": 0, "iter_pass": 0, "iter_total": 0, "input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0, "total_cost": 0.0})
    for r in all_results:
        lang = r["language"]
        by_lang[lang]["input_tokens"] += r.get("input_tokens", 0)
        by_lang[lang]["cached_input_tokens"] += r.get("cached_input_tokens", 0)
        by_lang[lang]["output_tokens"] += r.get("output_tokens", 0)
        by_lang[lang]["total_cost"] += r.get("total_cost", 0.0)

    for (lang, ex), passes in ex_results.items():
        by_lang[lang]["ex"] += 1
        by_lang[lang]["iter_total"] += len(passes)
        by_lang[lang]["iter_pass"]  += sum(passes)
        if all(passes): by_lang[lang]["pass_all"] += 1
        if any(passes): by_lang[lang]["pass_any"] += 1

    # Format per-language stats for JSON output
    by_lang_stats = {}
    for lang in sorted(by_lang):
        d = by_lang[lang]
        lang_total = d["input_tokens"] + d["cached_input_tokens"] + d["output_tokens"]
        by_lang_stats[lang] = {
            "exercises": d["ex"],
            "pass_all": d["pass_all"],
            "pass_at_k": d["pass_any"],
            "pass_at_1": sum(1 for (l, ex), passes in ex_iter1.items() if l == lang and passes) if iterations > 1 else d["pass_all"],
            "iterations": iterations,
            "iter_total": d["iter_total"],
            "iter_pass": d["iter_pass"],
            "iter_rate_pct": round(100 * d["iter_pass"] / d["iter_total"], 1) if d["iter_total"] else 0,
            "pass_all_pct": round(100 * d["pass_all"] / d["ex"], 1) if d["ex"] else 0,
            "pass_at_k_pct": round(100 * d["pass_any"] / d["ex"], 1) if d["ex"] else 0,
            "total_input_tokens": d["input_tokens"],
            "total_cached_input_tokens": d["cached_input_tokens"],
            "total_output_tokens": d["output_tokens"],
            "total_tokens": lang_total,
            "total_cost": round(d["total_cost"] if d["total_cost"] > 0 else ref_cost(d["input_tokens"], d["output_tokens"], d["cached_input_tokens"], model=config["model"]), 4),
        }

    # Save results with overall summary and per-language breakdown
    with open(results_dir / "results.json", "w") as f:
        json.dump({"overall": overall, "by_language": by_lang_stats, "tasks": all_results}, f, indent=2)

    for lang in sorted(by_lang):
        d = by_lang[lang]
        if iterations == 1:
            bar = "✓" * d["pass_all"] + "✗" * (d["ex"] - d["pass_all"])
            print(f"  {lang:12s} {d['pass_all']:3d}/{d['ex']:3d}  [{bar}]")
        else:
            lang_cost = d["total_cost"] if d["total_cost"] > 0 else ref_cost(d["input_tokens"], d["output_tokens"], d["cached_input_tokens"], model=config["model"])
            lang_total = d["input_tokens"] + d["cached_input_tokens"] + d["output_tokens"]
            print(f"  {lang:12s}  pass_all={d['pass_all']}/{d['ex']}  "
                  f"pass_any={d['pass_any']}/{d['ex']}  "
                  f"iter_rate={d['iter_pass']}/{d['iter_total']}  "
                  f"tokens={lang_total:,}  "
                  f"ref_cost=${lang_cost:.4f}")

    print()

    # List failed exercises
    failed_ex = [(lang, ex, passes) for (lang, ex), passes in sorted(ex_results.items()) if not all(passes)]
    if failed_ex:
        print("Failed / partial exercises:")
        for lang, ex, passes in failed_ex:
            rate = f"{sum(passes)}/{len(passes)}"
            print(f"  ✗ {lang}/{ex}  ({rate} iterations passed)")
        print()

    print(f"Full results : {results_dir}/results.json")
    print(f"Logs         : {results_dir}/logs/")
    sys.exit(0 if all_pass == total_ex else 1)


if __name__ == "__main__":
    main()
