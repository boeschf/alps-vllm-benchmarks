#!/usr/bin/env python3
"""Resolve one installed uenv reference to its immutable SHA-256 digest."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections.abc import Callable
from typing import Any

DIGEST = re.compile(r"^[0-9a-f]{64}$")


def resolve_uenv_digest(
    reference: str,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> str:
    if not reference or reference.startswith("-"):
        raise ValueError(f"invalid uenv reference: {reference!r}")
    result = runner(
        [
            "uenv",
            "image",
            "ls",
            "--no-partials",
            "--json",
            reference,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload: Any = json.loads(result.stdout)
    records = payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(records, list) or len(records) != 1:
        count = len(records) if isinstance(records, list) else 0
        raise ValueError(
            f"uenv reference must resolve to exactly one installed image; got {count}: {reference}"
        )
    digest = records[0].get("digest") if isinstance(records[0], dict) else None
    if not isinstance(digest, str) or not DIGEST.fullmatch(digest):
        raise ValueError(f"uenv returned an invalid digest for {reference}: {digest!r}")
    return digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    args = parser.parse_args()
    print(resolve_uenv_digest(args.reference))


if __name__ == "__main__":
    main()
