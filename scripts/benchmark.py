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
import statistics
import sys
import time
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
MIN_PERCENTILE_SAMPLES = 100
STATUS_INTERVAL_SECONDS = 60
SUMMARY_FIELDS = (
    "name",
    "scenario",
    "input_len",
    "effective_input_len",
    "output_len",
    "prompts",
    "warmup_prompts",
    "concurrency",
    "repetitions",
    "total_prompts",
    "request_throughput",
    "request_throughput_sample_stddev",
    "output_throughput",
    "output_throughput_sample_stddev",
    "total_token_throughput",
    "total_token_throughput_sample_stddev",
    "mean_ttft_ms",
    "mean_ttft_ms_sample_stddev",
    "p99_ttft_ms",
    "p99_ttft_ms_sample_stddev",
    "mean_tpot_ms",
    "mean_tpot_ms_sample_stddev",
    "p99_tpot_ms",
    "p99_tpot_ms_sample_stddev",
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
    required = {
        "name",
        "scenario",
        "input_len",
        "output_len",
        "prompts",
        "warmup_prompts",
        "concurrency",
    }
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
            for field in (
                "input_len",
                "output_len",
                "prompts",
                "warmup_prompts",
                "concurrency",
            ):
                point[field] = int(row[field])
                if point[field] <= 0:
                    raise ValueError(f"line {line}: {field} must be positive")
            if point["prompts"] < MIN_PERCENTILE_SAMPLES:
                raise ValueError(
                    f"line {line}: prompts must be at least {MIN_PERCENTILE_SAMPLES} "
                    "for p99 metrics"
                )
            if point["warmup_prompts"] > point["prompts"]:
                raise ValueError(
                    f"line {line}: warmup_prompts cannot exceed prompts"
                )
            if point["warmup_prompts"] < point["concurrency"]:
                raise ValueError(
                    f"line {line}: warmup_prompts must reach concurrency"
                )
            points.append(point)
    if not points:
        raise ValueError("suite contains no benchmark points")
    return points


# Replace the statistical measurement volume with the point's explicit warmup volume.
def make_warmup_point(point: dict[str, Any]) -> dict[str, Any]:
    return {
        **point,
        "measurement_prompts": point["prompts"],
        "prompts": point["warmup_prompts"],
    }


# Convert an arbitrary label into a stable identifier component.
def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


# Emit one UTC timestamped line that remains readable in redirected Slurm logs.
def log_status(message: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[status {timestamp}] {message}", flush=True)


# Format monotonic elapsed time without wall-clock assumptions.
def format_elapsed(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    return f"{minutes}m{seconds:02d}s"


# Run a long command with periodic liveness messages while preserving its output.
def run_with_heartbeat(
    command: list[str],
    label: str,
    *,
    interval_seconds: float = STATUS_INTERVAL_SECONDS,
) -> None:
    if interval_seconds <= 0:
        raise ValueError("status interval must be positive")
    started = time.monotonic()
    process = subprocess.Popen(command)
    try:
        while True:
            try:
                return_code = process.wait(timeout=interval_seconds)
                break
            except subprocess.TimeoutExpired:
                log_status(
                    f"{label}: still running; "
                    f"elapsed={format_elapsed(time.monotonic() - started)}"
                )
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    log_status(
        f"{label}: complete; elapsed={format_elapsed(time.monotonic() - started)}"
    )


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


# Build one reproducible vLLM benchmark invocation.
def build_benchmark_command(
    args: argparse.Namespace,
    point: dict[str, Any],
    result_file: Path,
    *,
    label: str,
    warmups: int,
) -> list[str]:
    return [
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
        str(warmups),
        "--seed",
        "0",
        "--disable-tqdm",
        "--percentile-metrics",
        "ttft,tpot",
        "--metric-percentiles",
        "99",
        "--save-result",
        "--result-dir",
        str(result_file.parent),
        "--result-filename",
        result_file.name,
        "--label",
        label,
    ]


# Load one completed invocation and enforce request and post-template token counts.
def load_benchmark_result(
    result_file: Path,
    point: dict[str, Any],
    effective_input_len: int,
) -> tuple[dict[str, Any], int]:
    result_file.with_suffix(".pytorch.json").unlink(missing_ok=True)
    result = json.loads(result_file.read_text())
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
    return result, total_input_tokens


# Aggregate independent suite passes and expose their run-to-run variation.
def aggregate_measurements(
    name: str,
    results: list[dict[str, Any]],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    if len(results) < 2:
        raise ValueError(f"{name}: at least two measurement repetitions are required")
    summary: dict[str, float] = {}
    bencher: dict[str, dict[str, float]] = {}
    for field, measure in BENCHER_MEASURES.items():
        try:
            values = [float(result[field]) for result in results]
        except KeyError as error:
            raise RuntimeError(f"{name}: vLLM result is missing {field}") from error
        mean = statistics.fmean(values)
        sample_stddev = statistics.stdev(values)
        summary[field] = mean
        summary[f"{field}_sample_stddev"] = sample_stddev
        bencher[measure] = {
            "value": mean,
            "lower_value": mean - sample_stddev,
            "upper_value": mean + sample_stddev,
        }
    return summary, bencher


# Publish all node-local JIT artifacts after the full warmup and before measurement.
def publish_jit_cache_after_warmup(
    logdir: Path,
) -> tuple[dict[str, Any], list[str] | None]:
    mode = os.getenv("JIT_CACHE_MODE", "off")
    phase = "after-full-suite-warmup"
    if mode != "readwrite":
        return {"status": "not-published", "mode": mode, "phase": phase}, None

    required = ("JIT_CACHE_TRIGGER", "JIT_CACHE_STAGING")
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(
            "JIT cache publication is missing environment variables: "
            + ", ".join(missing)
        )
    trigger = Path(os.environ["JIT_CACHE_TRIGGER"])
    output = logdir / "jit-cache-publication.json"
    command = [
        sys.executable,
        str(ROOT / "scripts/jit-cache.py"),
        "publish",
        "--staging",
        os.environ["JIT_CACHE_STAGING"],
        "--output",
        str(output),
        "--expected-bundles",
        os.getenv("SLURM_JOB_NUM_NODES", "1"),
        "--strict",
    ]
    trigger.touch()
    run_with_heartbeat(command, "JIT cache publication")
    publication = json.loads(output.read_text())
    if publication.get("status") not in {"published", "unchanged", "conflict"}:
        raise RuntimeError(f"JIT cache publication did not complete: {publication}")
    return {**publication, "mode": mode, "phase": phase}, command


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
    parser.add_argument("--batch-script", type=Path, required=True)
    parser.add_argument("--bencher-project", required=True)
    parser.add_argument("--bencher-testbed", required=True)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.repetitions < 2:
        parser.error("--repetitions must be at least 2")
    return args


# Execute the benchmark suite and persist its complete run record.
def main() -> int:
    args = parse_args()
    suite_path = args.suite.resolve()
    scenario_path = args.scenario_config.resolve()
    edf_path = args.edf.resolve()
    logdir = args.logdir.resolve()
    batch_script_path = args.batch_script.resolve()
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
    warmup_dir = output_dir / "warmup"
    warmup_dir.mkdir()
    archived_batch_script = output_dir / "submitted-batch-script.sbatch"
    archived_batch_script.write_bytes(batch_script_path.read_bytes())
    batch_script_identity = {
        "path": archived_batch_script.relative_to(output_dir).as_posix(),
        "sha256": file_sha256(archived_batch_script),
    }

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
    cache_mode = os.getenv("JIT_CACHE_MODE", "off")
    cache_publication = {
        "status": "pending" if cache_mode == "readwrite" else "not-published",
        "mode": cache_mode,
        "phase": "after-full-suite-warmup",
    }
    token_alignment = verify_token_alignment(args.base_url, args.served_model_name)
    suite_identity = {
        "path": str(
            suite_path.relative_to(ROOT) if suite_path.is_relative_to(ROOT) else suite_path
        ),
        "sha256": file_sha256(suite_path),
        "points": points,
        "measurement_repetitions": args.repetitions,
        "measurement_warmups_per_repetition": args.warmups,
        "pre_measurement_warmup": {
            "full_suite_passes": 1,
            "prompts": "warmup_prompts",
            "point_order": [point["name"] for point in points],
        },
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
            "slurm_batch_script_sha256": batch_script_identity["sha256"],
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
            "batch_script": batch_script_identity,
        },
        "topology": topology,
        "runtime": runtime_records,
        "server_commands": server_commands,
        "publication": publication,
        "jit_cache_publication": cache_publication,
        "jit_cache_publication_command": None,
        "points": [],
        "warmup": {
            "status": "pending",
            "full_suite_passes": 1,
            "points": [],
        },
    }
    atomic_json(output_dir / "manifest.json", manifest)
    log_status(
        f"benchmark run {run_id}: starting; points={len(points)} "
        f"warmup_passes=1 measurement_passes={args.repetitions}"
    )

    summary: list[dict[str, Any]] = []
    bencher: dict[str, dict[str, dict[str, float]]] = {}
    try:
        manifest["warmup"]["status"] = "running"
        atomic_json(output_dir / "manifest.json", manifest)
        log_status(f"full-suite warmup: starting; points={len(points)}")
        for index, point in enumerate(points, start=1):
            warmup = make_warmup_point(point)
            effective_input_len = (
                point["input_len"] + token_alignment["chat_template_token_overhead"]
            )
            point_label = f"warmup {index}/{len(points)} {point['name']}"
            log_status(
                f"{point_label}: starting; "
                f"in={point['input_len']} effective_in={effective_input_len} "
                f"out={point['output_len']} prompts={warmup['prompts']} "
                f"measurement_prompts={point['prompts']} "
                f"concurrency={point['concurrency']}"
            )
            warmup_file = warmup_dir / f"{point['name']}.json"
            command = build_benchmark_command(
                args,
                warmup,
                warmup_file,
                label=f"warmup-{point['name']}",
                warmups=0,
            )
            run_with_heartbeat(command, point_label)
            _, total_input_tokens = load_benchmark_result(
                warmup_file, warmup, effective_input_len
            )
            manifest["warmup"]["points"].append(
                {
                    **warmup,
                    "effective_input_len": effective_input_len,
                    "total_input_tokens": total_input_tokens,
                    "raw_result": str(warmup_file.relative_to(output_dir)),
                    "raw_sha256": file_sha256(warmup_file),
                    "command": command,
                }
            )
            atomic_json(output_dir / "manifest.json", manifest)
        manifest["warmup"]["status"] = "complete"
        manifest["warmup"]["completed_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(output_dir / "manifest.json", manifest)
        log_status("full-suite warmup: complete")
        log_status(f"JIT cache publication phase: starting; mode={cache_mode}")
        cache_publication, cache_command = publish_jit_cache_after_warmup(logdir)
        manifest["jit_cache_publication"] = cache_publication
        manifest["jit_cache_publication_command"] = cache_command
        atomic_json(output_dir / "manifest.json", manifest)
        log_status(
            "JIT cache publication phase: "
            f"complete; status={cache_publication['status']}"
        )

        measurements: dict[str, list[dict[str, Any]]] = {}
        point_records: dict[str, dict[str, Any]] = {}
        for point in points:
            effective_input_len = (
                point["input_len"] + token_alignment["chat_template_token_overhead"]
            )
            point_raw_dir = raw_dir / point["name"]
            point_raw_dir.mkdir()
            measurements[point["name"]] = []
            point_record = {
                **point,
                "effective_input_len": effective_input_len,
                "repetitions": [],
                "aggregate": None,
            }
            point_records[point["name"]] = point_record
            manifest["points"].append(point_record)
        atomic_json(output_dir / "manifest.json", manifest)
        log_status(
            f"measurements: starting; passes={args.repetitions} "
            f"invocations={args.repetitions * len(points)}"
        )

        for repetition in range(1, args.repetitions + 1):
            for index, point in enumerate(points, start=1):
                effective_input_len = point_records[point["name"]][
                    "effective_input_len"
                ]
                point_label = (
                    f"measure pass {repetition}/{args.repetitions} "
                    f"point {index}/{len(points)} {point['name']}"
                )
                log_status(
                    f"{point_label}: starting; "
                    f"in={point['input_len']} effective_in={effective_input_len} "
                    f"out={point['output_len']} prompts={point['prompts']} "
                    f"concurrency={point['concurrency']}"
                )
                raw_file = (
                    raw_dir / point["name"] / f"repetition-{repetition}.json"
                )
                command = build_benchmark_command(
                    args,
                    point,
                    raw_file,
                    label=f"{point['name']}-repetition-{repetition}",
                    warmups=args.warmups,
                )
                run_with_heartbeat(command, point_label)
                result, total_input_tokens = load_benchmark_result(
                    raw_file, point, effective_input_len
                )
                missing_metrics = BENCHER_MEASURES.keys() - result.keys()
                if missing_metrics:
                    raise RuntimeError(
                        f"{point['name']}: vLLM result is missing "
                        + ", ".join(sorted(missing_metrics))
                    )
                measurements[point["name"]].append(result)
                point_records[point["name"]]["repetitions"].append(
                    {
                        "repetition": repetition,
                        "total_input_tokens": total_input_tokens,
                        "raw_result": str(raw_file.relative_to(output_dir)),
                        "raw_sha256": file_sha256(raw_file),
                        "command": command,
                    }
                )
                atomic_json(output_dir / "manifest.json", manifest)

        log_status("measurements: complete; aggregating metrics")
        for point in points:
            aggregate, bencher_measures = aggregate_measurements(
                point["name"], measurements[point["name"]]
            )
            summary.append(
                {
                    **point,
                    "effective_input_len": point_records[point["name"]][
                        "effective_input_len"
                    ],
                    "repetitions": args.repetitions,
                    "total_prompts": point["prompts"] * args.repetitions,
                    **aggregate,
                }
            )
            bencher[point["name"]] = bencher_measures
            point_records[point["name"]]["aggregate"] = {
                field: {
                    "mean": aggregate[field],
                    "sample_stddev": aggregate[f"{field}_sample_stddev"],
                }
                for field in BENCHER_MEASURES
            }
        atomic_json(output_dir / "manifest.json", manifest)
        log_status("metric aggregation: complete")
    except BaseException as error:
        log_status(f"benchmark run {run_id}: failed; error={error!r}")
        if manifest["warmup"]["status"] == "running":
            manifest["warmup"]["status"] = "failed"
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
    log_status(f"benchmark run {run_id}: complete; results={output_dir}")
    log_status(f"experiment fingerprint: {experiment_fingerprint}")
    log_status(f"record fingerprint: {manifest['record_fingerprint']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
