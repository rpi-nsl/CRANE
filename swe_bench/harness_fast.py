#!/usr/bin/env python3
"""Thin wrapper around `python -m swebench.harness.run_evaluation`.

`docker_utils.list_images()` calls `client.images.list(all=True)` with no
filter. On this node's podman (1200+ images from the prebuild cache), that
one call hangs past docker-py's 600s timeout. We monkey-patch it to a
no-op (returns an empty set). Downstream effect: the final image-cleanup
step sees "no existing images" and cleans nothing, which is the safe
default and also what we want — we rely on the prebuild cache and don't
want the harness tearing images down between runs.

We also override the default docker-py timeout to 600s so container /
exec calls made during evaluation (not the images list) have more
breathing room too.

Usage is identical to the upstream CLI — we just forward argv.
"""
from __future__ import annotations

import os
import sys

# Make the default docker-py client more tolerant of a busy podman.
os.environ.setdefault("DOCKER_CLIENT_TIMEOUT", "600")

import docker as _docker  # noqa: E402

_orig_from_env = _docker.from_env


def _patched_from_env(*args, **kwargs):
    kwargs.setdefault("timeout", int(os.environ.get("DOCKER_CLIENT_TIMEOUT", "600")))
    return _orig_from_env(*args, **kwargs)


_docker.from_env = _patched_from_env

# Skip the expensive full-image listing; harness uses it only for cache
# cleanup at the end of a run, which we don't need.
from swebench.harness import docker_utils  # noqa: E402


def _fast_list_images(_client):
    return set()


docker_utils.list_images = _fast_list_images

# Forward to the real CLI.
from swebench.harness.run_evaluation import main as _main  # noqa: E402
import argparse  # noqa: E402

if __name__ == "__main__":
    # Reuse the upstream argparser by re-importing and calling main via
    # parser.parse_args pattern — simplest is to just reach into the
    # module's __main__ block path.
    import runpy

    runpy.run_module("swebench.harness.run_evaluation", run_name="__main__")
