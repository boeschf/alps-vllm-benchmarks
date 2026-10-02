"""Canonical benchmark provenance and artifact fingerprint helpers."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from script_utils import canonical_sha256, file_sha256

EXCLUDED_SOURCE_PARTS = {".git", "__pycache__", "logs", "models", "results"}
EXCLUDED_SOURCE_SUFFIXES = {".pyc", ".pyo", ".swp", ".swo", ".swn"}
EXCLUDED_ROOT_SOURCE_GLOBS = ("core", "core.[0-9]*", "core_nid[0-9]*_[0-9]*")
MODEL_METADATA_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
)


# Run a read-only Git query against the benchmark repository.
def run_git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )


# Decide whether a repository path contributes to source identity.
def source_path_is_relevant(path: Path) -> bool:
    root_artifact = len(path.parts) == 1 and any(
        path.match(pattern) for pattern in EXCLUDED_ROOT_SOURCE_GLOBS
    )
    return (
        not root_artifact
        and not EXCLUDED_SOURCE_PARTS.intersection(path.parts)
        and path.suffix not in EXCLUDED_SOURCE_SUFFIXES
    )


# Fingerprint the repository revision, changes, and relevant source files.
def repository_identity(root: Path) -> dict[str, Any]:
    revision_result = run_git(root, "rev-parse", "HEAD")
    revision = revision_result.stdout.strip() if revision_result.returncode == 0 else None
    branch_result = run_git(root, "symbolic-ref", "--quiet", "--short", "HEAD")
    branch = branch_result.stdout.strip() if branch_result.returncode == 0 else "detached"
    files_result = run_git(root, "ls-files", "-co", "--exclude-standard", "-z")
    if files_result.returncode != 0:
        raise RuntimeError(f"cannot enumerate repository files under {root}")

    entries = []
    for raw_path in files_result.stdout.split("\0"):
        if not raw_path:
            continue
        relative = Path(raw_path)
        path = root / relative
        if not source_path_is_relevant(relative) or not path.is_file():
            continue
        entries.append(
            {
                "path": relative.as_posix(),
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    entries.sort(key=lambda item: item["path"])

    diff_result = run_git(root, "diff", "--binary", "HEAD", "--")
    diff = diff_result.stdout if diff_result.returncode == 0 else ""
    tracked_result = run_git(root, "ls-files", "-m", "-d", "-o", "--exclude-standard", "-z")
    changed_paths = sorted(
        path
        for path in tracked_result.stdout.split("\0")
        if path and source_path_is_relevant(Path(path))
    )
    return {
        "revision": revision,
        "branch": branch,
        "clean": not changed_paths,
        "changed_paths": changed_paths,
        "tracked_diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
        "source_tree_sha256": canonical_sha256(entries),
        "source_file_count": len(entries),
    }


# Resolve the checkpoint revision from configuration or Hugging Face metadata.
def model_revision(model: Path) -> str | None:
    configured = os.getenv("MODEL_REVISION")
    if configured:
        return configured
    trees = model / ".cache" / "huggingface" / "trees"
    revisions = sorted(path.stem for path in trees.glob("*.json")) if trees.is_dir() else []
    return revisions[0] if len(revisions) == 1 else None


# Fingerprint checkpoint metadata and its shard inventory.
def model_identity(model: Path, repository: str) -> dict[str, Any]:
    metadata = {}
    for name in MODEL_METADATA_FILES:
        path = model / name
        if path.is_file():
            metadata[name] = {"size": path.stat().st_size, "sha256": file_sha256(path)}
    shards = sorted(model.glob("*.safetensors"))
    shard_inventory = [{"name": path.name, "size": path.stat().st_size} for path in shards]
    return {
        "path": str(model.resolve()),
        "repository": repository,
        "revision": model_revision(model),
        "metadata": metadata,
        "metadata_sha256": canonical_sha256(metadata),
        "shard_count": len(shard_inventory),
        "shard_bytes": sum(item["size"] for item in shard_inventory),
        "shard_inventory_sha256": canonical_sha256(shard_inventory),
        "content_sha256": os.getenv("MODEL_CONTENT_SHA256") or None,
    }


# Load per-node runtime identities and server commands.
def load_runtime_records(logdir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    runtime = [json.loads(path.read_text()) for path in sorted(logdir.glob("runtime-*.json"))]
    commands = [
        json.loads(path.read_text())
        for path in sorted(logdir.glob("server-command-*.json"))
    ]
    if not runtime or not commands:
        raise RuntimeError(f"runtime identity or server command records are missing under {logdir}")
    return runtime, commands


# Deduplicate runtime records after removing node-specific fields.
def stable_runtime_identity(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stable = []
    for record in records:
        item = {
            key: record[key]
            for key in (
                "deployment_role",
                "python",
                "platform",
                "kernel",
                "machine",
                "packages",
                "accelerator",
                "environment",
            )
        }
        cache = record.get("jit_cache")
        if cache:
            item["jit_cache"] = {
                key: cache.get(key)
                for key in (
                    "mode",
                    "compatibility_key",
                    "hit",
                    "generation",
                    "archive_sha256",
                )
            }
        stable.append(item)
    unique = {canonical_sha256(item): item for item in stable}
    return [unique[key] for key in sorted(unique)]


# Inventory completed run artifacts for the final manifest.
def artifact_inventory(output_dir: Path) -> list[dict[str, Any]]:
    artifacts = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name == "manifest.json" or path.suffix == ".tmp":
            continue
        artifacts.append(
            {
                "path": path.relative_to(output_dir).as_posix(),
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return artifacts
