from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLISHER = ROOT / "scripts" / "publish-bencher.sh"


class BencherPublisherTests(unittest.TestCase):
    def test_local_results_do_not_enable_remote_runner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            run_dir.mkdir()
            (run_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "publication": {
                            "project": "project",
                            "testbed": "alps-gh200",
                            "spec": "glm53-cuda-4n-baseline",
                            "branch": "main",
                            "publishable": True,
                            "hash": "a" * 40,
                        },
                    }
                )
            )
            metric = {"value": 1.0}
            (run_dir / "bencher.json").write_text(
                json.dumps({"conc-c1": {"throughput": metric}})
            )

            fake_bencher = root / "bencher"
            fake_bencher.write_text(
                f"#!{sys.executable}\n"
                "import json, sys\n"
                "args = sys.argv[1:]\n"
                "path = args[args.index('--file') + 1]\n"
                "print(json.dumps({'args': args, 'payload': json.load(open(path))}))\n"
            )
            fake_bencher.chmod(0o755)

            environment = os.environ.copy()
            environment.update(
                {
                    "BENCHER_API_KEY": "bencher_run_test",
                    "BENCHER_BIN": str(fake_bencher),
                }
            )
            completed = subprocess.run(
                [str(PUBLISHER), str(run_dir)],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )
            invocation = json.loads(completed.stdout.splitlines()[-1])

            self.assertNotIn("--image", invocation["args"])
            self.assertNotIn("--spec", invocation["args"])
            self.assertEqual(
                invocation["payload"],
                {"glm53-cuda-4n-baseline/conc-c1": {"throughput": metric}},
            )


if __name__ == "__main__":
    unittest.main()
