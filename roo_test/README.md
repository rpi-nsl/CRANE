# Roo Code Eval Runner

Standalone eval harness for benchmarking LLM coding ability via [Roo Code](https://github.com/RooCodeInc/Roo-Code) CLI. Runs exercises from [Roo-Code-Evals](https://github.com/RooCodeInc/Roo-Code-Evals) against any OpenAI-compatible endpoint (vLLM, Ollama, etc.) — no Docker, PostgreSQL, or Redis required.

## Quick Start

```bash
# 1. Clone Roo-Code and Roo-Code-Evals at the pinned commits (see UPSTREAM.md)

# 2. Edit config
cp yaml/eval-config.yaml my-model.yaml
vim my-model.yaml

# 3. Dry run
python3 roo-eval.py --config my-model.yaml --dry-run

# 4. Run evals
python3 roo-eval.py --config my-model.yaml --language python --limit 3
```

## Project Structure

```
.
├── roo-eval.py                          # Main eval runner (Python)
├── yaml/eval-config.yaml                # Config template
├── sh/env.sh                            # Sourced by all run scripts
├── sh/run_all_languages_3_b.sh          # 80B-A3B baseline merges, 5 languages
├── sh/run_all_languages_3_c.sh          # 80B-A3B CRANE merges, 5 languages
├── sh/run_80b_ablations.sh           # 80B canonical TGSP ablations
├── sh/run_rain.sh                       # RAIN baseline (Python)
├── sh/run_rain_planA_5lang.sh           # RAIN Plan A, 4 remaining languages
├── sh/run_rain_30b_proxy_5lang.sh       # RAIN 30B-proxy variant, 5 languages
├── Roo-Code/                            # Roo Code source (clone per UPSTREAM.md)
├── evals/                               # Roo-Code-Evals exercises + prompts
└── eval-results/                        # Output directory (timestamped runs)
```

## YAML Config Reference

```yaml
provider: openai                          # openai | anthropic | openrouter
model: "your-model-name"                  # Model name served by endpoint
api_key: "EMPTY"                          # API key ("EMPTY" for local vLLM)
base_url: "http://localhost:8003/v1"      # OpenAI-compatible endpoint

languages:                                # Languages to test
  - python
  - javascript
  - go
  - rust
  # - java

evals_repo: "./evals"                     # Path to Roo-Code-Evals clone

concurrency: 2                            # Parallel tasks (2 recommended for vLLM)
iterations: 3                             # Runs per exercise
timeout_seconds: 300                      # Per-attempt timeout
total_timeout_seconds: 900                # Total timeout per exercise (all attempts)
max_attempts: 11                          # Max retry attempts per exercise
context_window: 8192                      # Model context window (informational)
```

## CLI Usage

```bash
# Run all configured languages
python3 roo-eval.py --config eval-config.yaml

# Single language
python3 roo-eval.py --config eval-config.yaml --language python

# Specific exercises
python3 roo-eval.py --config eval-config.yaml --exercises python/hello-world python/leap

# Limit exercises per language
python3 roo-eval.py --config eval-config.yaml --limit 5

# Override concurrency / iterations
python3 roo-eval.py --config eval-config.yaml --concurrency 4 --iterations 3

# Dry run (list tasks without executing)
python3 roo-eval.py --config eval-config.yaml --dry-run
```

## Batch Scripts

Top-level run scripts under `sh/` invoke `roo-eval.py` directly with inline
YAML configs and run a list of models sequentially. See each script's header
for usage and the model registry.

```bash
sbatch sh/run_all_languages_3_b.sh                   # 80B baseline merges
sbatch sh/run_all_languages_3_c.sh                   # 80B CRANE merges
sbatch sh/run_80b_ablations.sh                    # 80B component-removal ablations
sbatch sh/run_rain.sh                                # RAIN baseline (python)
bash   sh/run_rain_planA_5lang.sh                    # RAIN Plan A (4 langs)
bash   sh/run_rain_30b_proxy_5lang.sh                # RAIN 30B-proxy (5 langs)

# Smoke test
bash sh/run_all_languages_3_b.sh --limit 3
```

## Results

Each run creates a timestamped directory under `eval-results/`:

```
eval-results/20260318_143000/
├── config.yaml       # Config snapshot for this run
├── results.json      # Full task-level results
└── logs/             # Per-task CLI output logs
    ├── python-hello-world.log
    ├── python-hello-world-attempt2.log
    └── ...
```

### Results JSON Format

Each entry in `results.json`:

| Field          | Description                                    |
|----------------|------------------------------------------------|
| `language`     | Exercise language (python, go, etc.)           |
| `exercise`     | Exercise name                                  |
| `iteration`    | Iteration number (1-based)                     |
| `passed`       | Whether unit tests passed                      |
| `cli_success`  | Whether CLI exited successfully                |
| `attempts`     | Number of attempts used                        |
| `duration_s`   | Total time for this task                       |
| `started_at`   | ISO timestamp                                  |
| `finished_at`  | ISO timestamp                                  |

## How It Works

1. **Exercise discovery** — scans `evals_repo/<language>/` for exercise directories
2. **Isolated workspace** — copies exercise to a temp directory via `shutil.copytree()`
3. **Roo Code CLI** — runs the agent loop headlessly (`node apps/cli/dist/index.js --oneshot --ephemeral`)
4. **Unit tests** — runs language-specific test commands (pytest, pnpm test, go test, cargo test)
5. **Retry** — if tests fail, retries up to `max_attempts` with fresh workspace each time
6. **Cleanup** — temp workspace removed after each attempt

## Prerequisites

- **Node.js 24** (via [nvm](https://github.com/nvm-sh/nvm))
- **Python 3** with pyyaml
- **Language runtimes**: python3, node/pnpm, go, cargo (as needed)
- **vLLM or OpenAI-compatible endpoint** serving your model
