#!/usr/bin/env python3
"""Export an EDF's environment table as a sourceable shell file."""

from __future__ import annotations

import argparse
import re
import shlex
import tomllib
from pathlib import Path

from script_utils import atomic_text

ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def load_edf_environment(path: Path) -> dict[str, str]:
    with path.open("rb") as stream:
        document = tomllib.load(stream)
    environment = document.get("env", {})
    if not isinstance(environment, dict):
        raise ValueError("EDF env must be a table")
    result: dict[str, str] = {}
    for name, value in environment.items():
        if not isinstance(name, str) or not ENVIRONMENT_NAME.fullmatch(name):
            raise ValueError(f"invalid EDF environment name: {name!r}")
        if not isinstance(value, str):
            raise ValueError(f"EDF environment value for {name} must be a string")
        result[name] = value
    return result


def shell_exports(environment: dict[str, str]) -> str:
    names = ",".join(sorted(environment))
    lines = [f"export EDF_ENV_NAMES={shlex.quote(names)}"]
    lines.extend(
        f"export {name}={shlex.quote(value)}"
        for name, value in sorted(environment.items())
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    atomic_text(args.output, shell_exports(load_edf_environment(args.edf)))


if __name__ == "__main__":
    main()
