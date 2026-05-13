#!/usr/bin/env python3
"""Format-token nullspace projector generation (long-context augmented).

Builds per-(layer, component) V_r and σ singular vectors that span the
instruct model's hidden-state subspace at format-token neighborhoods
(`<tool_call>`, `<|im_start|>`, etc.). Calibration mixes function-call,
chat, long-context (2k-20k token) tool-use, and completion conversations.
Outputs format_projectors.pt for use by Π_τ in the merge step.
"""

import argparse
import gc
import json
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# ── Configuration ────────────────────────────────────────────────────────────

MODEL_INSTRUCT = "Qwen/Qwen3-30B-A3B-Instruct-2507"
CACHE_DIR = os.environ.get("HF_HOME", "${HF_HOME}")
DEFAULT_OUT_DIR = os.environ.get(
    "CRANE_NULLSPACE_OUT_DIR",
    "${CRANE_DATA_DIR}/format_nullspace5_5_longctx",
)

MAX_SEQ_LEN = 4096
DEFAULT_SVD_RANK = 256
DEFAULT_THRESHOLD = 1e-5
DEFAULT_RADIUS = 3

FORMAT_PATTERNS = [
    "<tool_call>", "</tool_call>",
    "<tool_response>", "</tool_response>",
    "<|im_start|>", "<|im_end|>",
]

# Calibration datasets
HERMES_DATASET = "NousResearch/hermes-function-calling-v1"
HERMES_SUBSETS = [
    "func_calling_singleturn", "func_calling", "glaive_func_calling",
    "json_mode_agentic", "json_mode_singleturn",
]
ULTRACHAT_DATASET = "HuggingFaceH4/ultrachat_200k"

DEFAULT_N_TOOL = 300
DEFAULT_N_CHAT = 70   # reduced to make room for completion calibration
DEFAULT_N_LONG_CONTEXT = 30  # synthetic long-context tool-use conversations
DEFAULT_N_COMPLETION = 30    # completion-context conversations (attempt_completion after work)

HERMES_ROLE_MAP = {"system": "system", "human": "user", "gpt": "assistant", "tool": "tool"}


def locate_format_tokens(text: str, tokenizer, patterns: List[str] = None) -> List[int]:
    if patterns is None:
        patterns = FORMAT_PATTERNS
    bound_chars = []
    for pat in patterns:
        for m in re.finditer(re.escape(pat), text):
            bound_chars.append((m.start(), m.end(), pat))
    if not bound_chars:
        return []
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]

    def char_range_to_tokens(cs, ce):
        return [i for i, (s, e) in enumerate(offsets) if s < ce and e > cs]

    format_indices = set()
    for cs, ce, pat in bound_chars:
        format_indices.update(char_range_to_tokens(cs, ce))
    return sorted(format_indices)


def build_format_neighborhoods(format_indices: List[int], seq_len: int, radius: int) -> List[int]:
    neighborhood = set()
    for idx in format_indices:
        for t in range(max(0, idx - radius), min(seq_len, idx + radius + 1)):
            neighborhood.add(t)
    return sorted(neighborhood)


class FormatFeatureCollector:
    def __init__(self, model, num_layers: int, format_positions=None):
        self.num_layers = num_layers
        self._hooks = []
        self._features = defaultdict(list)
        self._current_positions = None

        for l in range(num_layers):
            layer = model.model.layers[l]
            key = f"layer_{l}_attn_input"
            hook = layer.input_layernorm.register_forward_hook(self._make_position_hook(key))
            self._hooks.append(hook)

            key_ffn = f"layer_{l}_ffn_input"
            hook_ffn = layer.post_attention_layernorm.register_forward_hook(self._make_position_hook(key_ffn))
            self._hooks.append(hook_ffn)

        print(f"  Registered {len(self._hooks)} format feature hooks")

    def _make_position_hook(self, key):
        def hook_fn(module, input, output):
            if self._current_positions is None or len(self._current_positions) == 0:
                return
            out = output[0] if isinstance(output, tuple) else output
            if out.dim() == 3:
                out = out[0]
            positions = [p for p in self._current_positions if p < out.shape[0]]
            if positions:
                pos_tensor = torch.tensor(positions, device=out.device)
                features_at_pos = out[pos_tensor].detach().float().cpu()
                self._features[key].append(features_at_pos)
        return hook_fn

    def set_positions(self, positions):
        self._current_positions = positions

    def get_features(self):
        return {k: torch.cat(ts, dim=0) for k, ts in self._features.items() if ts}

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ── Standard calibration ─────────────────────────────────────────────────────

def _hermes_to_messages(conversations):
    return [{"role": HERMES_ROLE_MAP.get(t["from"], t["from"]), "content": t["value"]}
            for t in conversations]


def _sample_hermes(tokenizer, n_total, rng):
    n_per = n_total // len(HERMES_SUBSETS)
    n_rem = n_total % len(HERMES_SUBSETS)
    texts = []
    for i, subset in enumerate(HERMES_SUBSETS):
        n = n_per + (1 if i < n_rem else 0)
        try:
            ds = load_dataset(HERMES_DATASET, subset, split="train", cache_dir=CACHE_DIR)
        except Exception as e:
            print(f"    [WARN] {subset}: {e}")
            continue
        indices = list(range(len(ds)))
        rng.shuffle(indices)
        sampled = 0
        for idx in indices:
            if sampled >= n:
                break
            row = ds[idx]
            conversations = row.get("conversations", [])
            if not conversations:
                continue
            messages = _hermes_to_messages(conversations)
            try:
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                if any(pat in text for pat in FORMAT_PATTERNS):
                    texts.append(text)
                    sampled += 1
            except Exception:
                continue
        print(f"    hermes/{subset}: {sampled}/{n}")
    return texts


def _sample_ultrachat(tokenizer, n_total, rng):
    texts = []
    try:
        ds = load_dataset(ULTRACHAT_DATASET, split="train_sft", cache_dir=CACHE_DIR)
    except Exception as e:
        print(f"    [WARN] ultrachat: {e}")
        return texts
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    sampled = 0
    for idx in indices:
        if sampled >= n_total:
            break
        messages = ds[idx].get("messages", [])
        if len(messages) < 4:
            continue
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            if any(pat in text for pat in FORMAT_PATTERNS):
                texts.append(text)
                sampled += 1
        except Exception:
            continue
    print(f"    ultrachat: {sampled}/{n_total}")
    return texts


# ── Long-context synthetic calibration (NEW) ─────────────────────────────────

ROO_SYSTEM_PROMPT = """\
You are Roo, a highly skilled software engineer with extensive knowledge in \
many programming languages, frameworks, design patterns, and best practices.

# Tools

## apply_diff
Apply precise, targeted modifications to an existing file.
Parameters:
- path: The path of the file to modify.
- diff: Search/replace blocks.

## read_file
Read the contents of a file.
Parameters:
- path: The path of the file to read

## execute_command
Execute a CLI command.
Parameters:
- command: The CLI command to execute.

## attempt_completion
Signal that the task is complete.
Parameters:
- result: A description of what was accomplished.
"""


def _generate_code_content(rng, min_lines=100, max_lines=500):
    """Generate diverse code content for file reads."""
    n_lines = rng.randint(min_lines, max_lines)
    templates = [
        # Python class
        lambda i: f"class Handler{i}:\n    def __init__(self, config):\n        self.config = config\n        self._state = {{}}\n\n    def process(self, data):\n        result = {{}}\n        for key, value in data.items():\n            if isinstance(value, str):\n                result[key] = value.strip().lower()\n            elif isinstance(value, (int, float)):\n                result[key] = value * self.config.get('scale', 1.0)\n            else:\n                result[key] = value\n        self._state[f'batch_{{len(self._state)}}'] = result\n        return result\n",
        # Python function
        lambda i: f"def validate_input_{i}(data, schema):\n    errors = []\n    for field, rules in schema.items():\n        if rules.get('required') and field not in data:\n            errors.append(f'Missing required field: {{field}}')\n        elif field in data:\n            value = data[field]\n            if 'type' in rules and not isinstance(value, rules['type']):\n                errors.append(f'Invalid type for {{field}}')\n            if 'min' in rules and value < rules['min']:\n                errors.append(f'{{field}} below minimum')\n            if 'max' in rules and value > rules['max']:\n                errors.append(f'{{field}} above maximum')\n    return errors\n",
        # Test code
        lambda i: f"class TestSuite{i}(unittest.TestCase):\n    def setUp(self):\n        self.handler = Handler{i}({{'scale': 2.0}})\n\n    def test_process_strings(self):\n        result = self.handler.process({{'name': '  Hello  ', 'value': 42}})\n        self.assertEqual(result['name'], 'hello')\n        self.assertEqual(result['value'], 84.0)\n\n    def test_process_empty(self):\n        result = self.handler.process({{}})\n        self.assertEqual(result, {{}})\n",
    ]
    lines = []
    for i in range(n_lines // 15):
        template = rng.choice(templates)
        lines.append(template(i))
    return "\n".join(lines)


def _build_long_context_conversation(tokenizer, rng, target_tokens=8000):
    """
    Build a synthetic multi-turn Roo-Code conversation that naturally
    accumulates long context with format tokens throughout.
    """
    messages = [{"role": "system", "content": ROO_SYSTEM_PROMPT}]

    # User task
    messages.append({"role": "user", "content": (
        "Fix the failing tests in this project. "
        "Use apply_diff for changes. Run tests after fixing."
    )})

    current_tokens = sum(len(tokenizer.encode(m["content"])) for m in messages)

    # Generate multi-turn conversation
    turn = 0
    file_names = ["app.py", "utils.py", "models.py", "tests.py", "config.py",
                  "handlers.py", "services.py", "middleware.py"]

    while current_tokens < target_tokens and turn < 20:
        fname = file_names[turn % len(file_names)]

        if turn % 4 == 0:
            # Read file turn
            messages.append({
                "role": "assistant",
                "content": (
                    f"Let me read {fname}.\n\n"
                    f"<tool_call>\n"
                    f'{{"name": "read_file", "arguments": {{"path": "{fname}"}}}}\n'
                    f"</tool_call>"
                ),
            })
            code = _generate_code_content(rng, min_lines=50, max_lines=200)
            messages.append({"role": "tool", "content": code})

        elif turn % 4 == 1:
            # Apply diff turn
            messages.append({
                "role": "assistant",
                "content": (
                    f"I see the issue. Let me fix it.\n\n"
                    f"<tool_call>\n"
                    f'{{"name": "apply_diff", "arguments": {{"path": "{fname}", '
                    f'"diff": "<<<<<<< SEARCH\\n:start_line:{rng.randint(10,50)}\\n-------\\n'
                    f'        result = {{}}\\n=======\\n        result = defaultdict(list)\\n>>>>>>> REPLACE"}}}}\n'
                    f"</tool_call>"
                ),
            })
            messages.append({"role": "tool", "content": "Applied diff successfully."})

        elif turn % 4 == 2:
            # Run tests turn
            messages.append({
                "role": "assistant",
                "content": (
                    f"Let me run the tests.\n\n"
                    f"<tool_call>\n"
                    f'{{"name": "execute_command", "arguments": {{"command": "pytest tests/ -v"}}}}\n'
                    f"</tool_call>"
                ),
            })
            n_pass = rng.randint(10, 30)
            n_fail = rng.randint(0, 3)
            if n_fail > 0:
                test_output = (
                    f"{'='*60} FAILURES {'='*60}\n"
                    + "\n".join(f"FAILED tests/test_{fname}::test_{rng.randint(1,20)} - AssertionError"
                               for _ in range(n_fail))
                    + f"\n{'='*20} {n_fail} failed, {n_pass} passed {'='*20}\n"
                )
            else:
                test_output = f"{'='*20} {n_pass} passed {'='*20}\n"
            messages.append({"role": "tool", "content": test_output})

        else:
            # Another read or apply
            messages.append({
                "role": "assistant",
                "content": (
                    f"<tool_call>\n"
                    f'{{"name": "read_file", "arguments": {{"path": "{fname}"}}}}\n'
                    f"</tool_call>"
                ),
            })
            code = _generate_code_content(rng, min_lines=30, max_lines=100)
            messages.append({"role": "tool", "content": code})

        current_tokens = sum(len(tokenizer.encode(m["content"])) for m in messages)
        turn += 1

    # 50% chance: end with attempt_completion after tests pass
    if rng.random() < 0.5:
        # Add a final test-pass + attempt_completion turn
        messages.append({
            "role": "assistant",
            "content": (
                "Let me run the tests to verify all fixes.\n\n"
                "<tool_call>\n"
                '{"name": "execute_command", "arguments": {"command": "pytest tests/ -v"}}\n'
                "</tool_call>"
            ),
        })
        n_pass = rng.randint(15, 40)
        messages.append({"role": "tool", "content": f"{'='*20} {n_pass} passed {'='*20}\n"})

        messages.append({
            "role": "assistant",
            "content": (
                "All tests pass. The fixes are complete.\n\n"
                "<tool_call>\n"
                '{"name": "attempt_completion", "arguments": {"result": '
                '"Fixed all failing tests. Applied changes to handle edge cases '
                'in data processing and validation logic."}}\n'
                "</tool_call>"
            ),
        })

    return messages


def build_long_context_calibration(tokenizer, n_conversations: int, rng,
                                   target_tokens_range=(4000, 16000)) -> List[str]:
    """
    Generate synthetic long-context multi-turn Roo-Code conversations.
    These cover format tokens at deep context positions (2k-16k+ tokens).
    """
    texts = []
    for i in range(n_conversations):
        target = rng.randint(*target_tokens_range)
        messages = _build_long_context_conversation(tokenizer, rng, target)
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            if any(pat in text for pat in FORMAT_PATTERNS):
                texts.append(text)
        except Exception:
            continue

        if (i + 1) % 10 == 0:
            n_tokens = len(tokenizer.encode(text))
            print(f"    long_context [{i+1}/{n_conversations}]: {n_tokens} tokens, "
                  f"{len(locate_format_tokens(text, tokenizer))} format tokens")

    print(f"    Long-context: generated {len(texts)}/{n_conversations}")
    return texts


def _build_completion_conversation(tokenizer, rng, target_tokens=4000):
    """
    Build a conversation focused on the completion context:
    user task → read_file → apply_diff → tests pass → attempt_completion.

    This captures format token activations specifically at the "task done,
    call attempt_completion" decision boundary.
    """
    messages = [{"role": "system", "content": ROO_SYSTEM_PROMPT}]

    # Diverse user tasks
    tasks = [
        "Fix the bug in the calculator — divide by zero should raise an exception.",
        "Fix the linked list — pop doesn't decrement length.",
        "Fix the hangman game — game doesn't end when guesses run out.",
        "Fix the matrix multiplication — wrong iteration range in inner loop.",
        "Fix the todo API — update endpoint ignores request body.",
        "Fix the authentication middleware — token expiry check is inverted.",
        "Fix the sorting algorithm — off-by-one error in partition.",
        "Fix the cache invalidation — stale entries not being removed.",
        "Fix the pagination — offset calculation is wrong.",
        "Fix the file parser — doesn't handle empty lines correctly.",
    ]
    messages.append({"role": "user", "content": rng.choice(tasks)})

    # Pad with file reads to reach target context
    current_tokens = sum(len(tokenizer.encode(m["content"])) for m in messages)
    file_names = ["app.py", "utils.py", "models.py", "handlers.py", "services.py"]
    turn = 0

    while current_tokens < target_tokens - 1500 and turn < 15:
        fname = file_names[turn % len(file_names)]

        if turn == 0:
            # First read
            messages.append({
                "role": "assistant",
                "content": (
                    f"Let me read the source file.\n\n"
                    f"<tool_call>\n"
                    f'{{"name": "read_file", "arguments": {{"path": "{fname}"}}}}\n'
                    f"</tool_call>"
                ),
            })
        else:
            messages.append({
                "role": "assistant",
                "content": (
                    f"<tool_call>\n"
                    f'{{"name": "read_file", "arguments": {{"path": "{fname}"}}}}\n'
                    f"</tool_call>"
                ),
            })

        code = _generate_code_content(rng, min_lines=40, max_lines=150)
        messages.append({"role": "tool", "content": code})
        current_tokens = sum(len(tokenizer.encode(m["content"])) for m in messages)
        turn += 1

    # Apply diff
    messages.append({
        "role": "assistant",
        "content": (
            "I found the bug. Let me fix it.\n\n"
            "<tool_call>\n"
            '{"name": "apply_diff", "arguments": {"path": "app.py", '
            '"diff": "<<<<<<< SEARCH\\n:start_line:15\\n-------\\n'
            '        return result\\n=======\\n'
            '        return validated_result\\n>>>>>>> REPLACE"}}\n'
            "</tool_call>"
        ),
    })
    messages.append({"role": "tool", "content": "Changes applied successfully."})

    # Run tests — they pass
    messages.append({
        "role": "assistant",
        "content": (
            "Let me run the tests to verify.\n\n"
            "<tool_call>\n"
            '{"name": "execute_command", "arguments": {"command": "pytest tests/ -v"}}\n'
            "</tool_call>"
        ),
    })
    n_pass = rng.randint(10, 30)
    messages.append({
        "role": "tool",
        "content": f"{'='*20} {n_pass} passed {'='*20}\n"
    })

    # THE KEY TURN: attempt_completion after tests pass
    completion_texts = [
        "All tests pass. The fix is complete.",
        "Tests are passing. I've successfully fixed the issue.",
        "The fix has been applied and verified with passing tests.",
        "All tests pass now. The bug has been resolved.",
        "I've fixed the bug and all tests pass.",
    ]
    messages.append({
        "role": "assistant",
        "content": (
            f"{rng.choice(completion_texts)}\n\n"
            "<tool_call>\n"
            '{"name": "attempt_completion", "arguments": {"result": '
            '"Fixed the bug by correcting the logic error. All tests pass."}}\n'
            "</tool_call>"
        ),
    })

    return messages


def build_completion_calibration(tokenizer, n_conversations: int, rng,
                                  target_tokens_range=(2000, 12000)) -> List[str]:
    """
    Generate completion-context calibration conversations.
    These capture format token features at the attempt_completion decision boundary.
    """
    texts = []
    for i in range(n_conversations):
        target = rng.randint(*target_tokens_range)
        messages = _build_completion_conversation(tokenizer, rng, target)
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            if any(pat in text for pat in FORMAT_PATTERNS):
                texts.append(text)
        except Exception:
            continue

        if (i + 1) % 10 == 0:
            n_tokens = len(tokenizer.encode(text))
            print(f"    completion [{i+1}/{n_conversations}]: {n_tokens} tokens, "
                  f"{len(locate_format_tokens(text, tokenizer))} format tokens")

    print(f"    Completion: generated {len(texts)}/{n_conversations}")
    return texts


# ── SVD with sigma output ────────────────────────────────────────────────────

def compute_projector_svd_with_sigma(
    features: torch.Tensor,
    max_rank: int = DEFAULT_SVD_RANK,
    threshold: float = DEFAULT_THRESHOLD,
    device: torch.device = torch.device("cuda:0"),
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Compute V_r and sigma from SVD of format token features.
    Returns V_r (d_in, r), sigma (r,), effective rank r.
    """
    n_tokens, d_in = features.shape
    k = min(max_rank, min(n_tokens, d_in))

    if k == 0 or n_tokens < 2:
        return torch.zeros(d_in, 0), torch.zeros(0), 0

    F = features.to(device=device, dtype=torch.float32)

    if min(n_tokens, d_in) > 2 * k:
        try:
            U, S, V = torch.svd_lowrank(F, q=k)
        except RuntimeError:
            U, S, Vh = torch.linalg.svd(F, full_matrices=False)
            S = S[:k]
            V = Vh[:k].T
    else:
        U, S, Vh = torch.linalg.svd(F, full_matrices=False)
        S = S[:k]
        V = Vh[:k].T

    s_max = S[0].item() if len(S) > 0 else 1.0
    mask = S > threshold * s_max
    r = mask.sum().item()

    V_r = V[:, :r].cpu()
    sigma = S[:r].cpu()

    del F, U, S, V
    torch.cuda.empty_cache()

    return V_r, sigma, r


# ── Main pipeline ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Format-token nullspace projectors (long-context augmented)"
    )
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--svd-rank", type=int, default=DEFAULT_SVD_RANK)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--radius", type=int, default=DEFAULT_RADIUS)
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--n-tool", type=int, default=DEFAULT_N_TOOL)
    parser.add_argument("--n-chat", type=int, default=DEFAULT_N_CHAT)
    parser.add_argument("--n-long-context", type=int, default=DEFAULT_N_LONG_CONTEXT)
    parser.add_argument("--n-completion", type=int, default=DEFAULT_N_COMPLETION)
    parser.add_argument("--long-context-min", type=int, default=4000)
    parser.add_argument("--long-context-max", type=int, default=16000)
    args = parser.parse_args()

    print("=" * 70)
    print("  Format-token nullspace projector generation (long-context)")
    print("=" * 70)
    print(f"  Model:          {MODEL_INSTRUCT}")
    print(f"  Output:         {args.out_dir}")
    print(f"  SVD rank:       {args.svd_rank}")
    print(f"  Max seq len:    {args.max_seq_len}")
    print(f"  Radius:         {args.radius}")
    print(f"  Calibration:    {args.n_tool} tool + {args.n_chat} chat + {args.n_long_context} long-context + {args.n_completion} completion")

    device = torch.device("cuda:0")
    print(f"\n  GPU: {torch.cuda.get_device_name(0)}")
    free, total = torch.cuda.mem_get_info(0)
    print(f"  Memory: {free/1024**3:.1f} GB free / {total/1024**3:.1f} GB total")

    # Load tokenizer
    print(f"\n  Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_INSTRUCT, cache_dir=CACHE_DIR, trust_remote_code=True)

    # Build calibration texts
    rng = random.Random(42)

    print(f"\n  === Building calibration data ===")
    print(f"  [1/4] Standard tool-calling ({args.n_tool}) ...")
    texts = _sample_hermes(tokenizer, args.n_tool, rng)

    print(f"  [2/4] Multi-turn chat ({args.n_chat}) ...")
    texts += _sample_ultrachat(tokenizer, args.n_chat, rng)

    print(f"  [3/4] Synthetic long-context ({args.n_long_context}) ...")
    long_texts = build_long_context_calibration(
        tokenizer, args.n_long_context, rng,
        target_tokens_range=(args.long_context_min, args.long_context_max),
    )
    texts += long_texts

    print(f"  [4/4] Completion-context ({args.n_completion}) ...")
    completion_texts = build_completion_calibration(
        tokenizer, args.n_completion, rng,
        target_tokens_range=(2000, 12000),
    )
    texts += completion_texts

    rng.shuffle(texts)
    print(f"\n  Total calibration: {len(texts)} texts")

    # Token length distribution
    lengths = [len(tokenizer.encode(t)) for t in texts]
    print(f"  Token lengths: min={min(lengths)}, median={sorted(lengths)[len(lengths)//2]}, max={max(lengths)}")

    # Load model
    print(f"\n  Loading model: {MODEL_INSTRUCT} ...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_INSTRUCT, torch_dtype=torch.bfloat16,
        device_map="auto", cache_dir=CACHE_DIR, trust_remote_code=True,
    )
    model.eval()
    print(f"  Loaded in {time.time()-t0:.0f}s")

    num_layers = model.config.num_hidden_layers
    os.makedirs(args.out_dir, exist_ok=True)

    # Step 1: Locate format tokens
    print(f"\n  === Step 1: Locating format tokens ===")
    all_positions = {}
    total_format = 0
    total_neighborhood = 0

    for idx, text in enumerate(texts):
        format_indices = locate_format_tokens(text, tokenizer)
        if not format_indices:
            continue
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        seq_len = enc["input_ids"].shape[1]
        neighborhood = build_format_neighborhoods(format_indices, seq_len, args.radius)
        all_positions[idx] = {
            "text": text,
            "input_ids": enc["input_ids"],
            "format_indices": format_indices,
            "neighborhood": neighborhood,
        }
        total_format += len(format_indices)
        total_neighborhood += len(neighborhood)

    print(f"  Texts with format tokens: {len(all_positions)}/{len(texts)}")
    print(f"  Total format token positions: {total_format}")
    print(f"  Total neighborhood positions: {total_neighborhood}")

    # Step 2: Collect features
    print(f"\n  === Step 2: Collecting features ===")
    collector = FormatFeatureCollector(model, num_layers)

    model.eval()
    with torch.no_grad():
        for idx, pos_data in all_positions.items():
            input_ids = pos_data["input_ids"]
            if input_ids.shape[1] > args.max_seq_len:
                input_ids = input_ids[:, :args.max_seq_len]
                positions = [p for p in pos_data["neighborhood"] if p < args.max_seq_len]
            else:
                positions = pos_data["neighborhood"]

            collector.set_positions(positions)
            input_ids = input_ids.to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                model(input_ids=input_ids)

            del input_ids
            if (idx + 1) % 50 == 0:
                print(f"    [{idx+1}/{len(all_positions)}] texts processed")

    print(f"  All {len(all_positions)} texts processed")
    features = collector.get_features()
    collector.remove_hooks()
    print(f"  Collected features for {len(features)} components")

    # Free model
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Step 3: Compute SVD projectors with sigma
    print(f"\n  === Step 3: Computing projectors (V_r + sigma) ===")
    projectors = {}
    stats = {"total_components": 0, "per_component": {}}

    for key in sorted(features.keys()):
        feat = features[key]
        t0 = time.time()
        V_r, sigma, r = compute_projector_svd_with_sigma(
            feat, args.svd_rank, args.threshold, device
        )
        elapsed = time.time() - t0

        if r > 0:
            projectors[key] = {"V_r": V_r, "sigma": sigma}
            stats["total_components"] += 1
            stats["per_component"][key] = {
                "n_tokens": feat.shape[0],
                "d_in": feat.shape[1],
                "effective_rank": r,
                "sigma_1": sigma[0].item(),
                "sigma_r": sigma[-1].item(),
                "ratio": (sigma[-1] / sigma[0]).item(),
                "svd_time": elapsed,
            }

            m = re.search(r"layer_(\d+)", key)
            if m:
                layer_idx = int(m.group(1))
                if layer_idx % 12 == 0 or layer_idx == num_layers - 1:
                    print(f"    {key}: features ({feat.shape[0]}, {feat.shape[1]}) → "
                          f"rank {r}, σ1={sigma[0].item():.4f}, σr={sigma[-1].item():.6f}, "
                          f"ratio={sigma[-1].item()/sigma[0].item():.6f} [{elapsed:.1f}s]")

    # Save
    proj_path = os.path.join(args.out_dir, "format_projectors.pt")
    torch.save(projectors, proj_path)
    fsize = os.path.getsize(proj_path) / 1024**2
    print(f"\n  Saved projectors → {proj_path} ({fsize:.1f} MB)")
    print(f"  Total components: {stats['total_components']}")

    # Save stats
    stats_path = os.path.join(args.out_dir, "format_stats.json")
    stats_json = {
        "model": MODEL_INSTRUCT,
        "svd_rank": args.svd_rank,
        "threshold": args.threshold,
        "radius": args.radius,
        "max_seq_len": args.max_seq_len,
        "n_texts": len(texts),
        "n_tool": args.n_tool,
        "n_chat": args.n_chat,
        "n_long_context": args.n_long_context,
        "n_completion": args.n_completion,
        "long_context_range": [args.long_context_min, args.long_context_max],
        "format_patterns": FORMAT_PATTERNS,
        "total_format_tokens": total_format,
        "total_neighborhood_tokens": total_neighborhood,
        "total_components": stats["total_components"],
    }
    with open(stats_path, "w") as f:
        json.dump(stats_json, f, indent=2)

    print(f"\n  === Projector generation complete ===")
    print(f"  Projectors: {proj_path}")
    print(f"  Stats: {stats_path}")


if __name__ == "__main__":
    main()
