#!/usr/bin/env python3
"""
Code-agent calibration sets for CRANE.

D_R (reasoning-transfer prompts) and D_T (tool-use preservation prompts)
are original rewrites in the style of public benchmarks rather than copies
of benchmark questions. Each item carries a `source_style` label so the
set can be audited or sliced by inspiration source.

Public benchmark styles used here:
  - SWE-bench-style: repository issue diagnosis and patch planning
  - LiveCodeBench-style: algorithmic / implementation reasoning for code tasks
  - LiveBench-coding-style: objective coding-agent reasoning and validation
"""

BENCHMARK_STYLE_REFERENCES = {
    "SWE-bench-style": {
        "benchmark": "SWE-bench",
        "url": "https://github.com/SWE-bench/SWE-bench",
        "notes": "real-repository issue resolution and patch-oriented debugging",
    },
    "LiveCodeBench-style": {
        "benchmark": "LiveCodeBench",
        "url": "https://arxiv.org/abs/2403.07974",
        "notes": "contamination-conscious coding and algorithmic reasoning",
    },
    "LiveBench-coding-style": {
        "benchmark": "LiveBench (coding)",
        "url": "https://github.com/LiveBench/LiveBench",
        "notes": "objective coding and agentic evaluation tasks",
    },
}


# Roo-Code system prompt used for D_T texts so the format-critical tool tokens
# (apply_diff, write_to_file, execute_command, etc.) are present in the
# preservation calibration distribution.
_ROO_SYSTEM = """\
You are Roo, a highly skilled software engineer with extensive knowledge in \
many programming languages, frameworks, design patterns, and best practices.

# Tools

You have access to the following tools to accomplish tasks:

## apply_diff
Apply precise, targeted modifications to an existing file using one or more \
search/replace blocks. This tool is for surgical edits only; the 'SEARCH' block \
must exactly match the existing content, including whitespace and indentation.

Parameters:
- path: The path of the file to modify, relative to the current workspace directory.
- diff: A string containing one or more search/replace blocks defining the changes.

Each block must follow this format:
<<<<<<< SEARCH
:start_line:[line_number]
-------
[exact content to find]
=======
[new content to replace with]
>>>>>>> REPLACE

## write_to_file
Write content to a file. Creates new files or overwrites existing ones. \
This tool will automatically create any directories needed.

Parameters:
- path: The path of the file to write to (relative to the current workspace directory)
- content: The content to write to the file. ALWAYS provide the COMPLETE intended \
content of the file, without any truncation or omissions.

## read_file
Read the contents of a file at the specified path.

Parameters:
- path: The path of the file to read (relative to the current workspace directory)

## execute_command
Execute a CLI command on the system.

Parameters:
- command: The CLI command to execute.
- cwd: (optional) The working directory to execute the command in.
- timeout: (optional) Timeout in seconds.

## list_files
List files and directories within the specified directory.

Parameters:
- path: The path of the directory to list contents for.
- recursive: (optional) Whether to list files recursively.

# Rules

- Your current working directory is the project root.
- You should prefer using apply_diff over write_to_file when making changes \
to existing files, since write_to_file is slower.
- Do not use the "ask_followup_question" tool.
- Run tests after making changes to verify correctness."""


# ── D_R: code-agent thinking set (36 prompts) ───────────────────────────────

D_R_ITEMS = [
    {"source_style": "SWE-bench-style", "prompt": "A Python helper is written as `def get_user(uid, cache={}): ...`. In production, results sometimes leak between requests and tests become order-dependent. Explain the root cause, why it is intermittent, and the smallest safe fix."},
    {"source_style": "SWE-bench-style", "prompt": "A React component fetches data inside `useEffect`, stores it in state, and includes that state in the dependency array. The page enters an infinite render loop. Explain exactly why this happens and how you would restructure the effect."},
    {"source_style": "SWE-bench-style", "prompt": "An API paginates with `ORDER BY created_at DESC LIMIT 50 OFFSET ?`. Users sometimes see duplicate rows or miss rows between page loads. Explain the failure mode and propose a more stable pagination strategy."},
    {"source_style": "SWE-bench-style", "prompt": "A payment worker retries failed requests after a timeout. Sometimes a customer is charged twice even though the provider claims requests are idempotent. Reason through the likely failure modes and what additional safeguards the client should add."},
    {"source_style": "LiveCodeBench-style", "prompt": "Consider a binary search implementation that updates `lo = mid` when the predicate fails and `hi = mid` when it succeeds. Explain when this can loop forever and how to write the invariant correctly."},
    {"source_style": "SWE-bench-style", "prompt": "A service compares `datetime.utcnow()` values with timezone-aware timestamps parsed from the database. The bug only appears around daylight-saving transitions and in some environments. Explain why the comparison is broken and how to normalize the code path."},
    {"source_style": "SWE-bench-style", "prompt": "A file-watcher cache keys on `(path, int(mtime))`. Some rapid file edits are ignored even though the file definitely changed. Explain the bug and propose a more reliable invalidation strategy."},
    {"source_style": "SWE-bench-style", "prompt": "Two threads each take two locks, but in opposite orders. The deadlock is rare and cannot be reproduced reliably under light load. Explain why the bug is load-sensitive and how you would remove the risk structurally rather than by adding sleeps or retries."},
    {"source_style": "LiveBench-coding-style", "prompt": "A feature-flag system reads defaults from code, overrides from YAML, and overrides again from environment variables. Different teams disagree on what should win. Explain how you would define a precedence rule that is debuggable and safe during rollouts."},
    {"source_style": "SWE-bench-style", "prompt": "An async function launches 20 coroutines with `asyncio.gather()`. One subtask fails quickly, but memory usage keeps climbing because the others continue doing expensive work. Explain what is happening and how cancellation should be handled."},
    {"source_style": "LiveCodeBench-style", "prompt": "A hand-written CSV parser splits on commas unless the parser is inside double quotes. It still breaks on inputs with escaped quotes like `\"\"`. Explain why this bug is harder than it first appears and what state the parser must track."},
    {"source_style": "SWE-bench-style", "prompt": "A service starts failing after a config refactor replaced `timeout_ms` with `request_timeout_ms`. Some callers still pass the old field, and the new code silently falls back to a default. Explain why silent fallback is dangerous here and how you would stage a safe migration."},
    {"source_style": "SWE-bench-style", "prompt": "You need to migrate a large system from referencing users by username to referencing them by immutable user_id, without downtime. Describe a staged migration plan that keeps reads and writes correct throughout the transition."},
    {"source_style": "SWE-bench-style", "prompt": "Webhook deliveries can be retried by the upstream provider, delivered out of order, and occasionally duplicated. Design a handler strategy that is idempotent and explain what state must be stored."},
    {"source_style": "LiveBench-coding-style", "prompt": "A product catalog is cached in memory on each app server. Writes are rare but must become visible quickly, and stale reads during deploys are causing incidents. Compare at least two invalidation strategies and argue for one."},
    {"source_style": "LiveBench-coding-style", "prompt": "A data sync pipeline currently polls a source database every minute for updated rows. The table is growing and the polling query is becoming expensive. Compare polling with change-data-capture and explain the operational tradeoffs."},
    {"source_style": "SWE-bench-style", "prompt": "A plugin system dynamically imports third-party Python modules. One plugin can crash process startup if its import side effects fail. Describe a safer loading model that keeps the host application resilient."},
    {"source_style": "LiveBench-coding-style", "prompt": "An inventory system currently uses optimistic locking but hot products see frequent write conflicts. Explain when pessimistic locking might help, what it costs, and what signals you would inspect before changing the design."},
    {"source_style": "LiveBench-coding-style", "prompt": "Background jobs are currently 'at least once'. The team wants 'exactly once' processing for a few critical workflows. Explain why exactly once is difficult in distributed systems and what practical approximation you would implement instead."},
    {"source_style": "SWE-bench-style", "prompt": "A monolith is being refactored into smaller modules, but teams keep creating circular dependencies through utility packages. Describe a boundary-setting strategy that preserves reuse without reintroducing a big ball of mud."},
    {"source_style": "SWE-bench-style", "prompt": "A configuration format has to evolve from a flat key-value file into nested sections, but old deployments must keep working for at least one release. Explain how you would support both formats while keeping error messages understandable."},
    {"source_style": "SWE-bench-style", "prompt": "A large table needs a backfill that computes a new derived column for hundreds of millions of rows. Describe how you would batch the work, avoid overwhelming the database, and verify correctness during the backfill."},
    {"source_style": "LiveBench-coding-style", "prompt": "A parser now supports comments, escaped delimiters, and multiline quoted strings. Propose a compact but high-value regression test suite that would catch most realistic bugs."},
    {"source_style": "LiveBench-coding-style", "prompt": "You are testing a token-bucket rate limiter. List the most important edge cases and invariants, especially around burst capacity, refill timing, and concurrent callers."},
    {"source_style": "LiveBench-coding-style", "prompt": "A CSV import pipeline validates rows, deduplicates by business key, and writes accepted rows in batches. Explain how you would test correctness without writing dozens of brittle end-to-end tests."},
    {"source_style": "SWE-bench-style", "prompt": "A flaky test fails only in CI with an error like 'address already in use'. Give a structured debugging plan for deciding whether the root cause is test ordering, leaked processes, parallelism, or environment reuse."},
    {"source_style": "SWE-bench-style", "prompt": "An auth middleware is placed before CORS handling, and some browser clients fail on preflight requests. Explain the ordering issue and how you would validate the fix."},
    {"source_style": "LiveBench-coding-style", "prompt": "A paginated API claims to be stable across retries. Describe the invariants your tests should enforce so the service cannot accidentally reintroduce duplicate or skipped results."},
    {"source_style": "LiveBench-coding-style", "prompt": "A validation layer was rewritten from hand-written checks to schema-based validation. Explain how property-based tests could help here and what kinds of generators you would build."},
    {"source_style": "SWE-bench-style", "prompt": "A concurrency bug is suspected, but the team only has one vague user report and no deterministic repro. Explain how you would build a minimal reproduction strategy before attempting a fix."},
    {"source_style": "SWE-bench-style", "prompt": "A web page shows a list of 100 orders and then loads each order's customer record separately. Explain the N+1 query pattern, how to detect it, and two ways to remove it."},
    {"source_style": "LiveCodeBench-style", "prompt": "You are building autocomplete for millions of strings with frequent reads and infrequent batch updates. Compare a trie, a sorted array with binary search, and a finite-state approach at a high level."},
    {"source_style": "LiveCodeBench-style", "prompt": "A pathfinding feature was first implemented with DFS on a weighted grid. Explain why the answers are wrong, and compare BFS, Dijkstra, and A* for this setting."},
    {"source_style": "LiveBench-coding-style", "prompt": "A database writer can either upsert records one at a time or batch them into chunks of 500. Explain the tradeoffs in throughput, latency, lock contention, and failure recovery."},
    {"source_style": "SWE-bench-style", "prompt": "A queue consumer keeps falling behind and memory usage grows steadily. The queue itself is durable, but the worker buffers messages internally before writing results. Explain what metrics and failure modes you would inspect first."},
    {"source_style": "LiveBench-coding-style", "prompt": "An event-sourced service replays too slowly on startup. Compare snapshotting, checkpointing, and log compaction as ways to reduce recovery time, and explain what each one gives up."},
]

D_R_PROMPTS = [item["prompt"] for item in D_R_ITEMS]


def get_d_r_items():
    """Return D_R items with source-style annotations."""
    return [dict(item) for item in D_R_ITEMS]


def get_d_r_texts(tokenizer, **kwargs):
    """Return D_R as chat-templated text strings."""
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": item["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for item in D_R_ITEMS
    ]


# ── D_T: code-agent tool-use set (16 prompts) ───────────────────────────────

D_T_ITEMS = [
    {"source_style": "SWE-bench-style", "prompt": "A targeted pytest job started failing after a config rename from `timeout_ms` to `request_timeout_ms`. Read the config loader, find where the old key is still used, patch the smallest compatible fix, and run only the relevant tests."},
    {"source_style": "SWE-bench-style", "prompt": "Our CLI advertises `--dry-run`, but users report that it still writes the output file. Read the command implementation, identify where the side effect happens, patch it so dry-run is truly side-effect free, and add or update a focused test."},
    {"source_style": "SWE-bench-style", "prompt": "A background worker makes HTTP calls without an explicit timeout and occasionally hangs forever in CI. Inspect the worker and its client wrapper, add a timeout plus conservative retry behavior, and run the smallest meaningful test scope."},
    {"source_style": "SWE-bench-style", "prompt": "The GitHub Actions workflow currently installs dependencies in three different ways across lint, test, and build jobs. Normalize the workflow to one install path, make caching consistent, and verify the YAML is still valid."},
    {"source_style": "SWE-bench-style", "prompt": "A constant was renamed from `DEFAULT_TIMEOUT` to `REQUEST_TIMEOUT`, but several imports, docs snippets, and tests still use the old name. Search the repo, patch all remaining references, and summarize anything you intentionally left untouched."},
    {"source_style": "SWE-bench-style", "prompt": "A React component crashes when an optional prop is `null` instead of `undefined`. Read the component and its tests, patch the null-safe behavior, and run the relevant frontend test file if available."},
    {"source_style": "SWE-bench-style", "prompt": "The app now prefers `NEW_API_URL`, but many environments still export `API_URL`. Update the config loader so the new variable wins while the old one remains as a deprecated fallback, and add a clear warning path."},
    {"source_style": "SWE-bench-style", "prompt": "A deployment shell script assumes unquoted paths and no spaces in environment variables. Inspect `scripts/deploy.sh`, add strict mode where appropriate, quote expansions conservatively, and run a syntax check."},
    {"source_style": "SWE-bench-style", "prompt": "The Makefile hardcodes `python` even though CI sets `PYTHON=python3.11`. Update the file so command targets respect the override everywhere, then run the affected target in dry or harmless form if possible."},
    {"source_style": "SWE-bench-style", "prompt": "Docker Compose brings up the app container before the database is truly ready, causing flaky integration failures. Read the compose file and startup script, add a safer readiness strategy, and explain the change."},
    {"source_style": "LiveBench-coding-style", "prompt": "The docs were updated after an API rename, but examples and code snippets still reference the old endpoint in a few places. Audit the docs, patch stale examples, and report any ambiguous cases instead of guessing."},
    {"source_style": "SWE-bench-style", "prompt": "Request logs include a request ID at the edge proxy, but the application logger drops it. Trace how context is passed through middleware, patch the logger wiring, and update any tests that assert log fields."},
    {"source_style": "SWE-bench-style", "prompt": "A flaky test suite reuses the same temp directory across tests and sometimes fails when cleanup races with background work. Read the relevant test helpers, isolate the shared-state bug, and make the least invasive fix."},
    {"source_style": "SWE-bench-style", "prompt": "The config system loads defaults, `.env`, environment variables, and a YAML file, but the precedence is undocumented and one code path applies them in a different order. Trace the loader behavior, make the precedence consistent, and document it briefly."},
    {"source_style": "SWE-bench-style", "prompt": "Several stale feature flags are marked deprecated but still referenced by docs and tests. Find every remaining reference, remove only the fully retired flags, and leave a short migration note listing the deleted names."},
    {"source_style": "LiveBench-coding-style", "prompt": "A repo-wide `readme_check` script fails because some relative markdown links point to files that moved. Use list_files and read_file to audit the broken references, patch only the incorrect paths, and rerun the checker if the script exists."},
]

D_T_PROMPTS = [item["prompt"] for item in D_T_ITEMS]


def get_d_t_items():
    """Return D_T items with source-style annotations."""
    return [dict(item) for item in D_T_ITEMS]


def get_d_t_texts(tokenizer, **kwargs):
    """Return D_T as Roo-style chat-templated text strings."""
    texts = []
    for item in D_T_ITEMS:
        msgs = [
            {"role": "system", "content": _ROO_SYSTEM},
            {"role": "user", "content": item["prompt"]},
        ]
        try:
            text = tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            merged = _ROO_SYSTEM + "\n\n" + item["prompt"]
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": merged}],
                tokenize=False,
                add_generation_prompt=True,
            )
        texts.append(text)
    return texts
