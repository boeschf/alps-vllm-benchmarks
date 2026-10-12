#!/usr/bin/env python3
"""Hydrate, capture, and atomically publish node-local JIT cache snapshots.

Design invariants:
- Compiler hot paths stay in container-local /tmp; shared storage holds snapshots only.
- Compatibility keys isolate image, Python, package, platform, and accelerator ABI changes.
- Published generations are immutable; readers switch generations through atomic CURRENT updates.
- Every node captures its cache because distributed ranks may compile different artifacts.
- Ready and COMPLETE markers commit bundles and generations only after their data is durable.
- Publication serializes with deadline-driven nonblocking locks suitable for the project NFS.
- Three-way base/current/job merges preserve unrelated concurrent additions and reject conflicts.
- Invalid or incomplete snapshots clear local roots and become cold starts, never serving failures.
- Explicitly allowlisted image seeds must match every non-image compatibility field.
- Rank-local mutable autotune state is excluded because partial hits can deadlock collectives.

Persistent namespaces contain CURRENT, publish.lock, and immutable generations; per-job staging
contains one archive, metadata record, and ready marker for each serving node.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import platform
import re
import shutil
import socket
import sys
import tarfile
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

from ninja_cache import restore_ninja_mtimes
from script_utils import (
    atomic_json,
    atomic_text,
    canonical_sha256,
    distribution_versions,
    file_sha256,
    stream_sha256,
)

SCHEMA_VERSION = 2
# Logical names become top-level archive directories. Environment variables
# point to the node-local directories used by each compiler.
CACHE_ROOTS = {
    "flashinfer": ("FLASHINFER_WORKSPACE_BASE", "/tmp/flashinfer-cache"),
    "triton": ("TRITON_CACHE_DIR", "/tmp/triton-cache"),
    "inductor": ("TORCHINDUCTOR_CACHE_DIR", "/tmp/inductor-cache"),
    "torch-extensions": ("TORCH_EXTENSIONS_DIR", "/tmp/torch-extensions"),
    "vllm": ("VLLM_CACHE_ROOT", "/tmp/vllm-cache"),
    "tensorrt-llm": ("TRTLLM_DG_CACHE_DIR", "/tmp/tensorrt-llm-cache"),
}
# Rank-local autotune choices cannot be merged safely across nodes.
EXCLUDED_ARCHIVE_PREFIXES = (PurePosixPath("vllm/flashinfer_autotune_cache"),)
# The versions of these packages are part of the cache compatibility key.
PACKAGE_NAMES = (
    "vllm",
    "torch",
    "triton",
    "flashinfer-python",
    "deep-gemm",
)
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


# Any identity change selects a different persistent namespace.
def compatibility_identity() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        raise RuntimeError("cache identity requires a visible CUDA or ROCm accelerator")
    properties = torch.cuda.get_device_properties(0)
    hip = torch.version.hip
    architecture = (
        getattr(properties, "gcnArchName", properties.name)
        if hip
        else f"sm{properties.major}{properties.minor}"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "platform": os.environ["PLATFORM"],
        "image_ref": os.environ["IMAGE_REF"],
        "image_digest": os.environ["IMAGE_DIGEST"],
        "python_abi": sys.implementation.cache_tag,
        "python_version": platform.python_version(),
        "packages": distribution_versions(PACKAGE_NAMES),
        "accelerator": {
            "architecture": architecture,
            "name": properties.name,
            "major": properties.major,
            "minor": properties.minor,
        },
        "cuda": torch.version.cuda,
        "hip": hip,
    }


# Read and validate the configured cache mode.
def cache_mode() -> str:
    mode = os.getenv("JIT_CACHE_MODE", "readwrite")
    if mode not in {"off", "read", "readwrite"}:
        raise ValueError(f"invalid JIT_CACHE_MODE={mode!r}; expected off, read, or readwrite")
    return mode


# Resolve the shared persistent cache root.
def persistent_root() -> Path:
    configured = os.getenv("JIT_CACHE_ROOT")
    if not configured:
        raise ValueError("JIT_CACHE_ROOT is required")
    return Path(configured).resolve()


# Resolve logical cache names to their live node-local directories.
def live_roots() -> dict[str, Path]:
    return {
        logical: Path(os.getenv(variable, default))
        for logical, (variable, default) in CACHE_ROOTS.items()
    }


# Build the persistent namespace for one compatibility key.
def namespace_path(
    identity_key: str,
    *,
    root: Path | None = None,
    platform_name: str | None = None,
) -> Path:
    base = root or persistent_root()
    platform_value = platform_name or os.environ["PLATFORM"]
    return base / f"schema-v{SCHEMA_VERSION}" / platform_value / identity_key


# Remove a file, symlink, or directory if it exists.
def remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


# Start each process with empty local caches before optional hydration.
def reset_live_roots() -> None:
    for path in live_roots().values():
        remove_path(path)
        path.mkdir(parents=True)


# Check whether an archive path belongs to excluded rank-local state.
def is_excluded_archive_path(path: PurePosixPath) -> bool:
    return any(path == prefix or prefix in path.parents for prefix in EXCLUDED_ARCHIVE_PREFIXES)


# Shared archives may contain only regular entries under known cache roots.
def safe_member_path(name: str) -> PurePosixPath:
    member = PurePosixPath(name)
    if member.is_absolute() or not member.parts or ".." in member.parts:
        raise ValueError(f"unsafe archive member path: {name!r}")
    if member.parts[0] not in CACHE_ROOTS:
        raise ValueError(f"unknown archive cache root: {name!r}")
    if is_excluded_archive_path(member):
        raise ValueError(f"excluded cache entry in archive: {name!r}")
    return member


# Per-file hashes support three-way merges between concurrent jobs.
def archive_file_manifest(archive: Path) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    with tarfile.open(archive, "r") as bundle:
        for member in bundle:
            path = safe_member_path(member.name)
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError(f"unsupported archive member type: {member.name!r}")
            source = bundle.extractfile(member)
            if source is None:
                raise ValueError(f"cannot read archive member: {member.name!r}")
            files[path.as_posix()] = {
                "sha256": stream_sha256(source),
                "size": member.size,
                "mode": member.mode & 0o777,
            }
    return files


# Describe an archive for its manifest.
def archive_descriptor(archive: Path) -> dict[str, Any]:
    return {
        "name": archive.name,
        "size": archive.stat().st_size,
        "sha256": file_sha256(archive),
    }


# Verify an archive checksum and optional per-file manifest.
def validate_archive(
    archive: Path,
    descriptor: dict[str, Any],
    expected_files: dict[str, dict[str, Any]] | None = None,
) -> None:
    if file_sha256(archive) != descriptor["sha256"]:
        raise ValueError(f"archive checksum mismatch: {archive}")
    if expected_files is not None and archive_file_manifest(archive) != expected_files:
        raise ValueError(f"archive file manifest mismatch: {archive}")


# Extract one archive member through an atomic file replacement.
def extract_file(bundle: tarfile.TarFile, member: tarfile.TarInfo, target: Path) -> None:
    source = bundle.extractfile(member)
    if source is None:
        raise ValueError(f"cannot read archive member: {member.name!r}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as output:
        shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
    temporary.chmod(member.mode & 0o777)
    os.replace(temporary, target)


# Archive all live cache roots without following symlinks.
def create_archive_from_roots(archive: Path, roots: dict[str, Path]) -> None:
    # Drop rank-local paths while tar walks each cache root.
    def archive_filter(member: tarfile.TarInfo) -> tarfile.TarInfo | None:
        return None if is_excluded_archive_path(PurePosixPath(member.name)) else member

    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w", format=tarfile.PAX_FORMAT, dereference=False) as bundle:
        for logical, root in sorted(roots.items()):
            root.mkdir(parents=True, exist_ok=True)
            bundle.add(root, arcname=logical, recursive=True, filter=archive_filter)


# Archive the logical cache roots stored under an extracted tree.
def create_archive_from_tree(archive: Path, tree: Path) -> None:
    create_archive_from_roots(archive, {name: tree / name for name in CACHE_ROOTS})


# Extract a trusted archive and restore build-system timestamps.
def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r") as bundle:
        members = [(member, safe_member_path(member.name)) for member in bundle.getmembers()]
        snapshot_mtime = max((member.mtime for member, _ in members), default=time.time())
        extracted_files: list[Path] = []
        for member, relative in members:
            target = destination.joinpath(*relative.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                target.chmod(member.mode & 0o777)
                continue
            if not member.isfile():
                raise ValueError(f"unsupported archive member type: {member.name!r}")
            extract_file(bundle, member, target)
            extracted_files.append(target)
        # Ninja rebuild decisions depend on recorded mtimes, not extraction order.
        restore_ninja_mtimes(
            destination,
            extracted_files,
            snapshot_mtime,
            CACHE_ROOTS,
        )


# Extraction happens in a temporary tree; each live root is replaced by rename.
def install_extracted_tree(tree: Path) -> None:
    for logical, destination in live_roots().items():
        source = tree / logical
        source.mkdir(exist_ok=True)
        remove_path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, destination)


# Load and validate a complete generation manifest.
def generation_manifest(namespace: Path, generation: str) -> dict[str, Any]:
    if not SAFE_NAME.fullmatch(generation):
        raise ValueError(f"invalid cache generation name: {generation!r}")
    directory = namespace / "generations" / generation
    if not (directory / "COMPLETE").is_file():
        raise ValueError(f"cache generation is incomplete: {generation}")
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported cache manifest schema: {manifest.get('schema_version')!r}")
    if manifest.get("generation") != generation:
        raise ValueError("cache generation manifest identity mismatch")
    return manifest


# Read the generation named by the namespace's CURRENT pointer.
def current_generation(namespace: Path) -> str | None:
    pointer = namespace / "CURRENT"
    return pointer.read_text().strip() if pointer.is_file() else None


# A seed may differ only by an explicitly allowed image reference and its digest.
def image_seed_compatible(identity: dict[str, Any], candidate: dict[str, Any]) -> bool:
    allowed_images = {
        value.strip()
        for value in os.getenv("JIT_CACHE_SEED_IMAGE_REFS", "").split(",")
        if value.strip()
    }
    candidate_image = candidate.get("image_ref")
    candidate_digest = candidate.get("image_digest")
    return (
        candidate_image in allowed_images
        and candidate
        == {
            **identity,
            "image_ref": candidate_image,
            "image_digest": candidate_digest,
        }
    )


# Find the newest allowlisted generation compatible with the current image.
def compatible_seed_generation(
    identity: dict[str, Any], target_namespace: Path
) -> tuple[Path, str] | None:
    if not os.getenv("JIT_CACHE_SEED_IMAGE_REFS", "").strip():
        return None
    # Prefer the newest compatible generation from another image namespace.
    candidates: list[tuple[str, Path, str]] = []
    for namespace in target_namespace.parent.iterdir():
        if namespace == target_namespace or not namespace.is_dir():
            continue
        try:
            generation = current_generation(namespace)
            if generation is None:
                continue
            manifest = generation_manifest(namespace, generation)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
        if image_seed_compatible(identity, manifest.get("compatibility", {})):
            candidates.append((manifest.get("created_at", ""), namespace, generation))
    if not candidates:
        return None
    _, namespace, generation = max(candidates, key=lambda item: item[0])
    return namespace, generation


# Hydrate node-local caches from the current compatible generation.
def hydrate(status_path: Path) -> dict[str, Any]:
    started = time.monotonic()
    mode = cache_mode()
    identity = compatibility_identity()
    identity_key = canonical_sha256(identity)
    namespace = namespace_path(identity_key)
    status: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "compatibility": identity,
        "compatibility_key": identity_key,
        "namespace": str(namespace),
        "hit": False,
        "generation": None,
        "base_generation": None,
        "archive_sha256": None,
        "seed_compatibility_key": None,
        "reason": None,
    }
    # This also clears stale local files when the cache is off or misses.
    reset_live_roots()
    if mode == "off":
        status["reason"] = "cache disabled"
        status["hydrate_seconds"] = time.monotonic() - started
        atomic_json(status_path, status)
        return status

    # A broken cache is a cold start, not a serving failure.
    try:
        source_namespace = namespace
        generation = current_generation(namespace)
        seeded = False
        if generation is None:
            seed = compatible_seed_generation(identity, namespace)
            if seed is not None:
                source_namespace, generation = seed
                seeded = True
        if generation is None:
            status["reason"] = "no published generation"
        else:
            manifest = generation_manifest(source_namespace, generation)
            if not seeded and manifest.get("compatibility_key") != identity_key:
                raise ValueError("cache generation compatibility key mismatch")
            archive_info = manifest["archive"]
            source = source_namespace / "generations" / generation / archive_info["name"]
            with tempfile.TemporaryDirectory(
                prefix="glm53-cache-hydrate-", dir="/tmp"
            ) as temporary:
                # Verify the local copy before replacing live cache roots.
                local_archive = Path(temporary) / "cache.tar"
                shutil.copyfile(source, local_archive)
                validate_archive(local_archive, archive_info)
                extracted = Path(temporary) / "extracted"
                safe_extract(local_archive, extracted)
                install_extracted_tree(extracted)
            status.update(
                {
                    "hit": True,
                    "generation": generation,
                    "base_generation": None if seeded else generation,
                    "archive_sha256": archive_info["sha256"],
                    "seed_compatibility_key": (
                        manifest.get("compatibility_key") if seeded else None
                    ),
                    "reason": "hydrated compatible image seed" if seeded else "hydrated",
                }
            )
    except Exception as error:
        reset_live_roots()
        status["reason"] = f"hydrate error: {error}"
    status["hydrate_seconds"] = time.monotonic() - started
    atomic_json(status_path, status)
    return status


# Capture one node's live caches as a publication bundle.
def capture_bundle(status_path: Path, staging: Path, role: str) -> dict[str, Any]:
    status = json.loads(status_path.read_text())
    if status.get("mode") != "readwrite":
        return {"status": "skipped", "reason": f"cache mode is {status.get('mode')}"}
    hostname = socket.gethostname()
    prefix = f"{role}-{hostname}"
    staging.mkdir(parents=True, exist_ok=True)
    # Build locally, then publish the archive before its ready marker.
    with tempfile.TemporaryDirectory(prefix="glm53-cache-bundle-", dir="/tmp") as temporary:
        local_archive = Path(temporary) / "cache.tar"
        create_archive_from_roots(local_archive, live_roots())
        files = archive_file_manifest(local_archive)
        destination = staging / f"{prefix}.tar"
        shutil.copyfile(local_archive, destination)
        archive_info = archive_descriptor(destination)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "kind": "bundle",
        "job_id": os.getenv("SLURM_JOB_ID"),
        "scenario_id": os.getenv("SCENARIO_ID"),
        "role": role,
        "hostname": hostname,
        "base_generation": status["base_generation"],
        "compatibility": status["compatibility"],
        "compatibility_key": status["compatibility_key"],
        "archive": archive_info,
        "files": files,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    # Metadata must be durable before the ready marker commits the bundle.
    atomic_json(staging / f"{prefix}.json", metadata)
    atomic_text(staging / f"{prefix}.ready", "ready\n")
    return metadata


# Wait for the serving trigger, then capture this node's bundle.
def wait_and_capture(status_path: Path, trigger: Path, staging: Path, role: str) -> None:
    # Benchmark jobs trigger after the full-suite warmup; jobs without a suite
    # trigger after their smoke request succeeds.
    while not trigger.exists():
        time.sleep(2)
    try:
        metadata = capture_bundle(status_path, staging, role)
        print(
            "JIT cache bundle ready: "
            f"role={role} archive={metadata['archive']['name']} files={len(metadata['files'])}",
            flush=True,
        )
    except Exception as error:
        atomic_json(
            staging / f"{role}-{socket.gethostname()}.error.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": "error",
                "role": role,
                "hostname": socket.gethostname(),
                "error": repr(error),
            },
        )
        print(f"JIT cache bundle failed: {error}", file=sys.stderr, flush=True)


# Validate bundle contents before using their hashes in the merge.
def validate_bundle(staging: Path, metadata_path: Path) -> tuple[dict[str, Any], Path]:
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("schema_version") != SCHEMA_VERSION or metadata.get("kind") != "bundle":
        raise ValueError(f"invalid bundle metadata: {metadata_path}")
    archive = staging / metadata["archive"]["name"]
    validate_archive(archive, metadata["archive"], metadata["files"])
    return metadata, archive


# Three-way merge one bundle against its hydrated base and the locked current generation.
def merge_bundle_files(
    base: dict[str, dict[str, Any]],
    merged: dict[str, dict[str, Any]],
    job: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[str], list[str]]:
    result = dict(merged)
    accepted: list[str] = []
    conflicts: list[str] = []
    for path, job_value in sorted(job.items()):
        base_value = base.get(path)
        current_value = result.get(path)
        # No job change, or another contributor already supplied identical content.
        if job_value == base_value or job_value == current_value:
            continue
        # Accept only when current still matches base; otherwise preserve current and conflict.
        if current_value == base_value:
            result[path] = job_value
            accepted.append(path)
        else:
            conflicts.append(path)
    return result, accepted, conflicts


# Copy only files accepted by the merge from each node bundle.
def extract_selected(archive: Path, destination: Path, selected: set[str]) -> None:
    if not selected:
        return
    with tarfile.open(archive, "r") as bundle:
        members = {
            safe_member_path(member.name).as_posix(): member
            for member in bundle
            if member.isfile()
        }
        missing = selected.difference(members)
        if missing:
            raise ValueError(f"bundle is missing selected files: {sorted(missing)[:5]}")
        for path in sorted(selected):
            member = members[path]
            extract_file(
                bundle,
                member,
                destination.joinpath(*PurePosixPath(path).parts),
            )


# Serialize namespace updates with deadline-driven nonblocking locks for NFS.
@contextmanager
def publication_lock(lock_path: Path, timeout_seconds: float) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as stream:
        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"timed out acquiring cache publication lock: {lock_path}"
                    )
                time.sleep(1)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


# Load the file manifest for a base generation.
def load_generation_files(namespace: Path, generation: str | None) -> dict[str, dict[str, Any]]:
    if generation is None:
        return {}
    return generation_manifest(namespace, generation).get("files", {})


# Remove old complete generations while retaining CURRENT.
def prune_generations(namespace: Path, current: str) -> None:
    keep = max(1, int(os.getenv("JIT_CACHE_KEEP_GENERATIONS", "3")))
    generations_root = namespace / "generations"
    complete = sorted(
        (
            path
            for path in generations_root.iterdir()
            if path.is_dir() and not path.name.startswith(".") and (path / "COMPLETE").is_file()
        ),
        key=lambda path: path.name,
        reverse=True,
    )
    # Always retain CURRENT, even if timestamp ordering puts it outside the limit.
    retained = {path.name for path in complete[:keep]}
    retained.add(current)
    for path in complete:
        if path.name not in retained:
            shutil.rmtree(path)


# Merge ready node bundles and publish one immutable generation.
def publish_bundles(staging: Path, expected: int) -> dict[str, Any]:
    started = time.monotonic()
    deadline = started + float(os.getenv("JIT_CACHE_BUNDLE_SECONDS", "1800"))
    # Every node writes its ready marker after its archive and metadata.
    ready: list[Path] = []
    while time.monotonic() < deadline:
        ready = sorted(staging.glob("*.ready"))
        errors = sorted(staging.glob("*.error.json"))
        if errors:
            details = [json.loads(path.read_text()) for path in errors]
            raise RuntimeError(f"cache bundle agent failed: {details}")
        if len(ready) >= expected:
            break
        time.sleep(2)
    if len(ready) < expected:
        raise TimeoutError(f"received {len(ready)}/{expected} cache bundles under {staging}")

    bundles = [
        validate_bundle(staging, marker.with_suffix(".json"))
        for marker in ready
    ]
    compatibility_keys = {
        metadata["compatibility_key"] for metadata, _ in bundles
    }
    if len(compatibility_keys) != 1:
        raise ValueError(f"cache bundles have different compatibility keys: {compatibility_keys}")
    identity_key = compatibility_keys.pop()
    identity = bundles[0][0]["compatibility"]
    job_ids = {metadata.get("job_id") for metadata, _ in bundles}
    base_generations = {
        metadata.get("base_generation") for metadata, _ in bundles
    }
    namespace = namespace_path(identity_key, platform_name=identity["platform"])
    lock_seconds = float(os.getenv("JIT_CACHE_LOCK_SECONDS", "300"))

    with publication_lock(namespace / "publish.lock", lock_seconds):
        current = current_generation(namespace)
        current_manifest = generation_manifest(namespace, current) if current else None
        merged_files = dict(current_manifest.get("files", {}) if current_manifest else {})
        accepted_by_archive: list[tuple[Path, set[str]]] = []
        conflicts: list[str] = []
        node_divergences: list[str] = []
        # Nodes in one job may compile different valid files. Keep the first
        # version instead of treating this as a concurrent-job conflict.
        coalesce_node_divergences = len(job_ids) == 1 and base_generations == {current}
        contributors: list[dict[str, Any]] = []
        for metadata, archive in bundles:
            base_generation = metadata.get("base_generation")
            base_files = load_generation_files(namespace, base_generation)
            merged_files, accepted, bundle_conflicts = merge_bundle_files(
                base_files, merged_files, metadata["files"]
            )
            accepted_by_archive.append((archive, set(accepted)))
            (node_divergences if coalesce_node_divergences else conflicts).extend(
                bundle_conflicts
            )
            contributors.append(
                {
                    "job_id": metadata.get("job_id"),
                    "scenario_id": metadata.get("scenario_id"),
                    "role": metadata.get("role"),
                    "hostname": metadata.get("hostname"),
                    "base_generation": base_generation,
                }
            )

        if conflicts:
            return {
                "schema_version": SCHEMA_VERSION,
                "status": "conflict",
                "compatibility_key": identity_key,
                "generation": current,
                "conflicts": sorted(set(conflicts)),
                "publish_seconds": time.monotonic() - started,
            }
        if not any(selected for _, selected in accepted_by_archive):
            return {
                "schema_version": SCHEMA_VERSION,
                "status": "unchanged",
                "compatibility_key": identity_key,
                "generation": current,
                "publish_seconds": time.monotonic() - started,
            }

        # Rebuild a complete generation from the current cache and accepted files.
        with tempfile.TemporaryDirectory(prefix="glm53-cache-publish-", dir="/tmp") as temporary:
            temporary_path = Path(temporary)
            merged_tree = temporary_path / "merged"
            merged_tree.mkdir()
            for logical in CACHE_ROOTS:
                (merged_tree / logical).mkdir()
            if current_manifest:
                current_archive = (
                    namespace
                    / "generations"
                    / current
                    / current_manifest["archive"]["name"]
                )
                validate_archive(current_archive, current_manifest["archive"])
                safe_extract(current_archive, merged_tree)
            for archive, selected in accepted_by_archive:
                extract_selected(archive, merged_tree, selected)

            local_archive = temporary_path / "cache.tar"
            create_archive_from_tree(local_archive, merged_tree)
            final_files = archive_file_manifest(local_archive)
            archive_info = archive_descriptor(local_archive)
            timestamp = datetime.now(timezone.utc)
            generation = (
                f"{timestamp:%Y%m%dT%H%M%SZ}-"
                f"{os.getenv('SLURM_JOB_ID', 'local')}-"
                f"{archive_info['sha256'][:12]}"
            )
            generations = namespace / "generations"
            generations.mkdir(parents=True, exist_ok=True)
            destination = generations / generation
            # Keep the new directory hidden until all generation files exist.
            generation_staging = generations / f".{generation}.{os.getpid()}.tmp"
            generation_staging.mkdir()
            shutil.copyfile(local_archive, generation_staging / archive_info["name"])
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "generation": generation,
                "created_at": timestamp.isoformat(),
                "compatibility": identity,
                "compatibility_key": identity_key,
                "previous_generation": current,
                "contributors": contributors,
                "coalesced_node_divergences": len(set(node_divergences)),
                "archive": archive_info,
                "files": final_files,
            }
            atomic_json(generation_staging / "manifest.json", manifest)
            atomic_text(generation_staging / "COMPLETE", "complete\n")
            # Publish the generation first, then switch readers with CURRENT.
            os.replace(generation_staging, destination)
            atomic_text(namespace / "CURRENT", generation + "\n")
            prune_generations(namespace, generation)

        return {
            "schema_version": SCHEMA_VERSION,
            "status": "published",
            "compatibility_key": identity_key,
            "generation": generation,
            "previous_generation": current,
            "archive_sha256": archive_info["sha256"],
            "archive_size": archive_info["size"],
            "coalesced_node_divergences": len(set(node_divergences)),
            "publish_seconds": time.monotonic() - started,
        }


# Publication is best effort unless the caller requests strict failure.
def publish(staging: Path, output: Path, expected: int, strict: bool) -> int:
    try:
        result = publish_bundles(staging, expected)
    except Exception as error:
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "error",
            "error": repr(error),
        }
        atomic_json(output, result)
        print(f"JIT cache publication failed: {error}", file=sys.stderr, flush=True)
        return 1 if strict else 0
    atomic_json(output, result)
    print(f"JIT cache publication: {result}", flush=True)
    if result["status"] in {"published", "unchanged", "conflict"}:
        shutil.rmtree(staging, ignore_errors=True)
    return 0


# Define the hydrate, agent, and publish command-line interfaces.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hydrate, capture, and publish persistent JIT cache snapshots."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    hydrate_parser = subparsers.add_parser("hydrate")
    hydrate_parser.add_argument("--status", type=Path, required=True)

    agent_parser = subparsers.add_parser("agent")
    agent_parser.add_argument("--status", type=Path, required=True)
    agent_parser.add_argument("--trigger", type=Path, required=True)
    agent_parser.add_argument("--staging", type=Path, required=True)
    agent_parser.add_argument("--role", required=True)

    publish_parser = subparsers.add_parser("publish")
    publish_parser.add_argument("--staging", type=Path, required=True)
    publish_parser.add_argument("--output", type=Path, required=True)
    publish_parser.add_argument("--expected-bundles", type=int, required=True)
    publish_parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


# Dispatch the selected cache command.
def main() -> int:
    args = parse_args()
    if args.command == "hydrate":
        status = hydrate(args.status)
        print(
            f"JIT cache: mode={status['mode']} hit={status['hit']} "
            f"generation={status['generation']} reason={status['reason']}",
            flush=True,
        )
        return 0
    if args.command == "agent":
        wait_and_capture(args.status, args.trigger, args.staging, args.role)
        return 0
    if args.command == "publish":
        if args.expected_bundles <= 0:
            raise ValueError("--expected-bundles must be positive")
        return publish(args.staging, args.output, args.expected_bundles, args.strict)
    raise AssertionError(args.command)


if __name__ == "__main__":
    sys.exit(main())
