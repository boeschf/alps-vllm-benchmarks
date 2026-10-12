from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "submit_serve",
    ROOT / "scripts" / "submit-serve.py",
)
assert SPEC is not None and SPEC.loader is not None
submit_serve = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(submit_serve)


class SubmissionCommandTests(unittest.TestCase):
    def test_submission_ignores_active_uenv(self) -> None:
        args = submit_serve.parse_args(
            [
                "--account",
                "test-account",
                "--partition",
                "test-partition",
                "--time",
                "01:00:00",
            ]
        )

        command = submit_serve.build_command(args, 4)

        self.assertEqual(
            command,
            [
                "sbatch",
                "--nodes=4",
                "--export=ALL",
                "--uenv-passthrough=ignore",
                "--account=test-account",
                "--partition=test-partition",
                "--time=01:00:00",
                "sbatch/serve.sbatch",
            ],
        )


if __name__ == "__main__":
    unittest.main()
