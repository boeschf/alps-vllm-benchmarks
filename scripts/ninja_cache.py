"""Restore Ninja output timestamps after cache archive extraction."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from pathlib import Path

CacheRoots = Mapping[str, tuple[str, str]]


# Resolve one .ninja_log output path inside the extracted tree.
def _archived_output_path(
    tree: Path,
    output: str,
    log_parent: Path,
    cache_roots: CacheRoots,
) -> Path | None:
    path = Path(output)
    if not path.is_absolute():
        candidate = log_parent / path
        return candidate if candidate.is_file() else None

    # Absolute paths still point to the original live /tmp cache.
    for logical, (environment, default) in cache_roots.items():
        live_root = Path(os.environ.get(environment, default))
        try:
            relative = path.relative_to(live_root)
        except ValueError:
            continue
        candidate = tree / logical / relative
        return candidate if candidate.is_file() else None
    return None


# Collect valid output mtimes from all extracted .ninja_log files.
def _recorded_output_mtimes(tree: Path, cache_roots: CacheRoots) -> dict[Path, int]:
    mtimes: dict[Path, int] = {}
    for log in tree.rglob(".ninja_log"):
        for line in log.read_text(errors="replace").splitlines():
            fields = line.split("\t")
            if len(fields) < 5 or line.startswith("#"):
                continue
            try:
                recorded = int(fields[2])
            except ValueError:
                continue
            target = _archived_output_path(tree, fields[3], log.parent, cache_roots)
            if target is None:
                continue
            # Old Ninja logs use milliseconds; current logs use nanoseconds.
            mtimes[target] = recorded * 1_000_000 if recorded < 10**16 else recorded
    return mtimes


# Normalize extracted mtimes, then restore times recorded by Ninja.
def restore_ninja_mtimes(
    tree: Path,
    extracted_files: Iterable[Path],
    snapshot_mtime: float,
    cache_roots: CacheRoots,
) -> None:
    output_mtimes = _recorded_output_mtimes(tree, cache_roots)
    snapshot_mtime_ns = int(snapshot_mtime * 1_000_000_000)
    baseline_mtime_ns = max(
        0,
        min(output_mtimes.values(), default=snapshot_mtime_ns) - 1_000_000_000,
    )
    for target in extracted_files:
        os.utime(
            target,
            ns=(baseline_mtime_ns, baseline_mtime_ns),
            follow_symlinks=False,
        )
    for target, output_mtime_ns in output_mtimes.items():
        os.utime(
            target,
            ns=(output_mtime_ns, output_mtime_ns),
            follow_symlinks=False,
        )
