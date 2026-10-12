#!/usr/bin/env -S uv run --no-project --python 3.12 -- python
"""Submit the serving job with an allocation derived from its scenario."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from scenario import resolve_scenario, select_nodes

ROOT = Path(__file__).resolve().parents[1]
SBATCH_SCRIPT = Path("sbatch/serve.sbatch")


# Build the Slurm command without duplicating scenario allocation policy.
def build_command(args: argparse.Namespace, nodes: int) -> list[str]:
    command = [
        "sbatch",
        f"--nodes={nodes}",
        "--export=ALL",
        "--uenv-passthrough=ignore",
    ]
    for option in ("account", "partition", "time"):
        value = getattr(args, option)
        if value is not None:
            command.append(f"--{option.replace('_', '-')}={value}")
    command.append(str(SBATCH_SCRIPT))
    return command


# Build the exported job environment separately from sbatch's comma-delimited syntax.
def build_environment(args: argparse.Namespace, nodes: int) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "SCENARIO": args.scenario,
            "PLATFORM": args.platform,
            "EXPECTED_NODES": str(nodes),
        }
    )
    if args.model is not None:
        environment["MODEL"] = args.model
    if args.benchmark_suite is not None:
        environment["BENCHMARK_SUITE"] = args.benchmark_suite
    if args.jit_cache_mode is not None:
        environment["JIT_CACHE_MODE"] = args.jit_cache_mode
    if args.smoke_only:
        environment["SMOKE_ONLY"] = "1"
    return environment


# Parse serving and common Slurm submission options.
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit a GLM-5.3 serving job with scenario-aware resources."
    )
    parser.add_argument("--scenario", default=os.environ.get("SCENARIO", "baseline"))
    parser.add_argument(
        "--platform",
        default=os.environ.get("PLATFORM", "cuda"),
        choices=("cuda", "rocm"),
    )
    parser.add_argument("--nodes", type=int)
    parser.add_argument("--account")
    parser.add_argument("--partition")
    parser.add_argument("--time")
    parser.add_argument("--model")
    parser.add_argument("--benchmark-suite")
    parser.add_argument(
        "--jit-cache-mode",
        choices=("off", "read", "readwrite"),
        help="persistent JIT cache behavior (default: readwrite)",
    )
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="exit after readiness, smoke validation, and cache publication",
    )
    return parser.parse_args(argv)


# Resolve the allocation before asking Slurm to create the job.
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        resolved = resolve_scenario(args.scenario, args.platform)
        nodes = select_nodes(resolved, args.nodes)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    command = build_command(args, nodes)
    environment = build_environment(args, nodes)
    print(f"scenario={resolved['id']} platform={args.platform} nodes={nodes}")
    try:
        return subprocess.run(command, cwd=ROOT, env=environment).returncode
    except OSError as error:
        print(f"error: unable to run sbatch: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
