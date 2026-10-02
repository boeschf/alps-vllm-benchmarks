"""Shared file, hashing, and package helpers for repository scripts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any, BinaryIO


# Serialize a value into the stable JSON representation used for fingerprints.
def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


# Hash a value using its stable JSON representation.
def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


# Hash a binary stream without loading it into memory.
def stream_sha256(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


# Hash a file without loading it into memory.
def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return stream_sha256(stream)


# Atomically replace a text file after syncing its complete contents.
def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


# Serialize JSON through the shared atomic text writer.
def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


# Read installed distribution versions while retaining missing packages.
def distribution_versions(names: Iterable[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions
