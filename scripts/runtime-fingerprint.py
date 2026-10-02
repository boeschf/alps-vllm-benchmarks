#!/usr/bin/env python3
"""Capture the software, accelerator, environment, and exact server command per node."""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
from pathlib import Path
from typing import Any

from script_utils import atomic_json, distribution_versions

RUNTIME_PACKAGES = (
    "vllm",
    "torch",
    "triton",
    "flashinfer-python",
    "deep-gemm",
    "uccl",
    "mooncake-transfer-engine",
    "nixl",
    "runai-model-streamer",
)


# Describe every accelerator visible to the serving process.
def accelerator_identity() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"available": False}

    devices = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": properties.name,
                "total_memory": properties.total_memory,
                "multi_processor_count": properties.multi_processor_count,
                "major": properties.major,
                "minor": properties.minor,
            }
        )
    return {
        "available": torch.cuda.is_available(),
        "cuda": torch.version.cuda,
        "hip": torch.version.hip,
        "devices": devices,
    }


# Capture runtime identity and the exact server command for one node.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--command-output", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("server command is empty")

    scenario_names = filter(None, os.getenv("SCENARIO_ENV_NAMES", "").split(","))
    fixed_names = {
        "ALL2ALL",
        "IMAGE_REF",
        "LOAD_FORMAT",
        "PLATFORM",
        "SCENARIO_ID",
        "JIT_CACHE_MODE",
        "JIT_CACHE_ROOT",
        "TP_SIZE",
    }
    environment = {
        name: os.environ[name]
        for name in sorted(fixed_names.union(scenario_names))
        if name in os.environ
    }
    jit_cache = None
    status_file = os.getenv("JIT_CACHE_STATUS_FILE")
    if status_file:
        jit_cache = json.loads(Path(status_file).read_text())
    identity = {
        "schema_version": 1,
        "hostname": socket.gethostname(),
        "node_rank": int(os.getenv("SLURM_NODEID", "0")),
        "deployment_role": os.getenv("DEPLOYMENT_ROLE", "single"),
        "python": sys.version,
        "platform": platform.platform(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "packages": distribution_versions(RUNTIME_PACKAGES),
        "accelerator": accelerator_identity(),
        "environment": environment,
        "jit_cache": jit_cache,
    }
    atomic_json(args.output, identity)
    atomic_json(args.command_output, {"argv": command})


if __name__ == "__main__":
    main()
