#!/usr/bin/env python3
"""Run a declarative vLLM serving suite and emit fingerprinted archival results."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from provenance import (
    artifact_inventory,
    load_runtime_records,
    model_identity,
    repository_identity,
    stable_runtime_identity,
)
from script_utils import atomic_json, canonical_sha256, file_sha256

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_TEMPERATURE = 0.0
SUMMARY_FIELDS = (
    "name",
    "scenario",
    "input_len",
    "effective_input_len",
    "output_len",
    "prompts",
    "concurrency",
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
    "mean_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "p99_tpot_ms",
)
BENCHER_MEASURES = {
    "request_throughput": "request-throughput-req-per-sec",
    "output_throughput": "output-throughput-tok-per-sec",
    "total_token_throughput": "total-throughput-tok-per-sec",
    "mean_ttft_ms": "mean-ttft-ms",
    "p99_ttft_ms": "p99-ttft-ms",
    "mean_tpot_ms": "mean-tpot-ms",
    "p99_tpot_ms": "p99-tpot-ms",
}


# Load and validate every point in a tab-separated benchmark suite.
def load_suite(path: Path) -> list[dict[str, Any]]:
    required = {"name", "scenario", "input_len", "output_len", "prompts", "concurrency"}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"suite is missing columns: {', '.join(sorted(missing))}")
        points: list[dict[str, Any]] = []
        names: set[str] = set()
        for line, row in enumerate(reader, start=2):
            name = row["name"].strip()
            if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", name):
                raise ValueError(f"line {line}: invalid benchmark name {name!r}")
            if name in names:
                raise ValueError(f"line {line}: duplicate benchmark name {name!r}")
            names.add(name)
            point: dict[str, Any] = {"name": name, "scenario": row["scenario"].strip()}
            for field in ("input_len", "output_len", "prompts", "concurrency"):
                point[field] = int(row[field])
                if point[field] <= 0:
                    raise ValueError(f"line {line}: {field} must be positive")
            points.append(point)
    if not points:
        raise ValueError("suite contains no benchmark points")
    return points


# Convert an arbitrary label into a stable identifier component.
def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


# Verify both raw-token and chat-template tokenization through the serving endpoint.
def verify_token_alignment(base_url: str, model: str) -> dict[str, Any]:
    endpoint = f"{base_url.rstrip('/')}/tokenize"
    probe = "benchmark token alignment probe"

    def token_count(payload: dict[str, Any]) -> int:
        request = Request(
            endpoint,
            data=json.dumps({"model": model, **payload}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=30) as response:
                result = json.load(response)
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"mandatory token alignment failed at {endpoint}: {error}"
            ) from error
        tokens = result.get("tokens") if isinstance(result, dict) else None
        if not isinstance(tokens, list) or not all(
            isinstance(token, int) for token in tokens
        ):
            raise RuntimeError(
                f"mandatory token alignment returned an invalid response from {endpoint}"
            )
        return len(tokens)

    raw_tokens = token_count({"prompt": probe, "add_special_tokens": False})
    post_template_tokens = token_count(
        {
            "messages": [{"role": "user", "content": probe}],
            "add_generation_prompt": True,
            "add_special_tokens": False,
        }
    )
    overhead = post_template_tokens - raw_tokens
    if overhead < 0:
        raise RuntimeError(
            "chat template removed tokens from the alignment probe: "
            f"raw={raw_tokens}, post_template={post_template_tokens}"
        )
    return {
        "endpoint": endpoint,
        "model": model,
        "raw_probe_tokens": raw_tokens,
        "post_template_probe_tokens": post_template_tokens,
        "chat_template_token_overhead": overhead,
    }


# Parse benchmark execution and provenance options.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model-name", default="glm-5.3")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results")
    parser.add_argument("--run-id")
    parser.add_argument("--platform", required=True, choices=("cuda", "rocm"))
    parser.add_argument("--image-ref", required=True)
    parser.add_argument("--scenario-config", type=Path, required=True)
    parser.add_argument("--edf", type=Path, required=True)
    parser.add_argument("--logdir", type=Path, required=True)
    parser.add_argument("--bencher-project", required=True)
    parser.add_argument("--bencher-testbed", required=True)
    parser.add_argument("--warmups", type=int, default=1)
    return parser.parse_args()


# Execute the benchmark suite and persist its complete run record.
def main() -> int:
    args = parse_args()
    suite_path = args.suite.resolve()
    scenario_path = args.scenario_config.resolve()
    edf_path = args.edf.resolve()
    logdir = args.logdir.resolve()
    points = load_suite(suite_path)
    scenario = json.loads(scenario_path.read_text())
    scenario_id = str(scenario["id"])
    timestamp = datetime.now(timezone.utc)
    run_scenario = slug(scenario_id)
    default_run_id = (
        f"{timestamp:%Y%m%dT%H%M%SZ}-{os.getenv('SLURM_JOB_ID', 'local')}-"
        f"{args.platform}-{run_scenario}"
    )
    run_id = args.run_id or default_run_id
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        raise ValueError(f"invalid run id: {run_id!r}")

    output_dir = args.output_root.resolve() / run_id
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=False)

    nodes = int(os.getenv("SLURM_JOB_NUM_NODES", "1"))
    gpus_per_node = int(os.getenv("GPUS_PER_NODE", "4"))
    tp_size = int(scenario["topology"]["tensor_parallel_size"])
    dp_size = nodes * gpus_per_node // tp_size
    topology = {
        "nodes": nodes,
        "node_list": list(filter(None, os.getenv("SLURM_NODELIST_EXPANDED", "").split(","))),
        "gpus_per_node": gpus_per_node,
        "tensor_parallel_size": tp_size,
        "data_parallel_size": dp_size,
        "expert_parallel_size": dp_size * tp_size,
        "rank_distribution": "block:cyclic",
        "cpu_bind": "none",
    }
    repository = repository_identity(ROOT)
    model = model_identity(Path(args.model), os.getenv("MODEL_REPO", "zai-org/GLM-5.3"))
    runtime_records, server_commands = load_runtime_records(logdir)
    cache_publication_path = logdir / "jit-cache-publication.json"
    cache_publication = (
        json.loads(cache_publication_path.read_text())
        if cache_publication_path.is_file()
        else {"status": "not-published", "mode": os.getenv("JIT_CACHE_MODE", "off")}
    )
    token_alignment = verify_token_alignment(args.base_url, args.served_model_name)
    suite_identity = {
        "path": str(
            suite_path.relative_to(ROOT) if suite_path.is_relative_to(ROOT) else suite_path
        ),
        "sha256": file_sha256(suite_path),
        "points": points,
        "warmups_per_point": args.warmups,
        "seed": 0,
        "dataset": "random",
        "ignore_eos": True,
        "temperature": BENCHMARK_TEMPERATURE,
        "token_alignment": token_alignment,
    }
    model_for_fingerprint = {key: value for key, value in model.items() if key != "path"}
    experiment_identity = {
        "schema_version": 1,
        "source": {
            "revision": repository["revision"],
            "source_tree_sha256": repository["source_tree_sha256"],
        },
        "model": model_for_fingerprint,
        "image_ref": args.image_ref,
        "platform": args.platform,
        "edf_sha256": file_sha256(edf_path),
        "scenario": scenario,
        "topology": {key: value for key, value in topology.items() if key != "node_list"},
        "runtime": stable_runtime_identity(runtime_records),
        "suite": suite_identity,
    }
    experiment_fingerprint = canonical_sha256(experiment_identity)
    spec = slug(
        f"glm53-{args.platform}-{nodes}n-{gpus_per_node}g-tp{tp_size}-"
        f"dp{dp_size}-ep{dp_size * tp_size}-{scenario_id}"
    )
    publishable = bool(
        repository["clean"]
        and re.fullmatch(r"[0-9a-f]{40}", repository["revision"] or "")
    )
    publication = {
        "project": args.bencher_project,
        "branch": repository["branch"],
        "testbed": args.bencher_testbed,
        "spec": spec,
        "hash": repository["revision"] if publishable else None,
        "publishable": publishable,
        "reason": None if publishable else "Bencher publication requires a clean Git worktree",
    }
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "status": "running",
        "run_id": run_id,
        "created_at": timestamp.isoformat(),
        "experiment_fingerprint": experiment_fingerprint,
        "experiment_identity": experiment_identity,
        "repository": repository,
        "hostname": socket.gethostname(),
        "model": model,
        "served_model_name": args.served_model_name,
        "platform": args.platform,
        "image_ref": args.image_ref,
        "image_source": "https://github.com/eth-cscs/alps-extended-images/pull/63",
        "edf": {"path": str(edf_path), "sha256": file_sha256(edf_path)},
        "scenario": scenario,
        "suite": suite_identity,
        "token_alignment": token_alignment,
        "slurm": {
            "job_id": os.getenv("SLURM_JOB_ID"),
            "account": os.getenv("SLURM_JOB_ACCOUNT"),
            "partition": os.getenv("SLURM_JOB_PARTITION"),
        },
        "topology": topology,
        "runtime": runtime_records,
        "server_commands": server_commands,
        "publication": publication,
        "jit_cache_publication": cache_publication,
        "points": [],
    }
    atomic_json(output_dir / "manifest.json", manifest)

    summary: list[dict[str, Any]] = []
    bencher: dict[str, dict[str, dict[str, float]]] = {}
    try:
        for index, point in enumerate(points, start=1):
            effective_input_len = (
                point["input_len"] + token_alignment["chat_template_token_overhead"]
            )
            print(
                f"[{index}/{len(points)}] {point['name']}: "
                f"in={point['input_len']} effective_in={effective_input_len} "
                f"out={point['output_len']} prompts={point['prompts']} "
                f"concurrency={point['concurrency']}",
                flush=True,
            )
            raw_file = raw_dir / f"{point['name']}.json"
            command = [
                "vllm",
                "bench",
                "serve",
                "--backend",
                "openai-chat",
                "--endpoint",
                "/v1/chat/completions",
                "--base-url",
                args.base_url,
                "--model",
                args.served_model_name,
                "--served-model-name",
                args.served_model_name,
                "--tokenizer",
                args.model,
                "--ignore-eos",
                "--temperature",
                str(BENCHMARK_TEMPERATURE),
                "--dataset-name",
                "random",
                "--input-len",
                str(point["input_len"]),
                "--output-len",
                str(point["output_len"]),
                "--num-prompts",
                str(point["prompts"]),
                "--max-concurrency",
                str(point["concurrency"]),
                "--num-warmups",
                str(args.warmups),
                "--seed",
                "0",
                "--disable-tqdm",
                "--percentile-metrics",
                "ttft,tpot",
                "--metric-percentiles",
                "99",
                "--save-result",
                "--result-dir",
                str(raw_dir),
                "--result-filename",
                raw_file.name,
                "--label",
                point["name"],
            ]
            subprocess.run(command, check=True)
            raw_file.with_suffix(".pytorch.json").unlink(missing_ok=True)
            result = json.loads(raw_file.read_text())
            completed = int(result.get("completed", 0))
            failed = int(result.get("failed", 0))
            if completed != point["prompts"] or failed:
                raise RuntimeError(
                    f"{point['name']}: completed={completed}/{point['prompts']}, failed={failed}"
                )
            total_input_tokens = result.get("total_input_tokens")
            if not isinstance(total_input_tokens, int):
                raise RuntimeError(
                    f"{point['name']}: vLLM result is missing integer total_input_tokens"
                )
            expected_input_tokens = effective_input_len * completed
            if total_input_tokens != expected_input_tokens:
                raise RuntimeError(
                    f"{point['name']}: post-template input length mismatch: "
                    f"observed={total_input_tokens}, expected={expected_input_tokens} "
                    f"({effective_input_len} per request)"
                )

            row = {**point, "effective_input_len": effective_input_len}
            for field in BENCHER_MEASURES:
                if field not in result:
                    raise RuntimeError(f"{point['name']}: vLLM result is missing {field}")
                row[field] = float(result[field])
            summary.append(row)
            bencher[point["name"]] = {
                measure: {"value": row[field]} for field, measure in BENCHER_MEASURES.items()
            }
            manifest["points"].append(
                {
                    **point,
                    "effective_input_len": effective_input_len,
                    "total_input_tokens": total_input_tokens,
                    "raw_result": str(raw_file.relative_to(output_dir)),
                    "raw_sha256": file_sha256(raw_file),
                    "command": command,
                }
            )
            atomic_json(output_dir / "manifest.json", manifest)
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = repr(error)
        manifest["artifacts"] = artifact_inventory(output_dir)
        atomic_json(output_dir / "manifest.json", manifest)
        raise

    with (output_dir / "summary.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(summary)
    atomic_json(output_dir / "bencher.json", bencher)
    manifest["artifacts"] = artifact_inventory(output_dir)
    manifest["record_fingerprint"] = canonical_sha256(
        {
            "experiment_fingerprint": experiment_fingerprint,
            "artifacts": manifest["artifacts"],
        }
    )
    manifest["status"] = "complete"
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    atomic_json(output_dir / "manifest.json", manifest)
    print(f"Results: {output_dir}")
    print(f"Experiment fingerprint: {experiment_fingerprint}")
    print(f"Record fingerprint: {manifest['record_fingerprint']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
