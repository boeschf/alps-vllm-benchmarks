from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark import (
    BENCHER_MEASURES,
    MIN_PERCENTILE_SAMPLES,
    aggregate_measurements,
    load_suite,
    publish_jit_cache_after_warmup,
)


class BenchmarkStatisticsTests(unittest.TestCase):
    def test_standard_suite_has_enough_requests_for_p99(self) -> None:
        points = load_suite(ROOT / "benchmarks" / "standard.tsv")

        self.assertTrue(points)
        self.assertTrue(
            all(point["prompts"] >= MIN_PERCENTILE_SAMPLES for point in points)
        )

    def test_repetitions_emit_mean_and_sample_standard_deviation(self) -> None:
        results = [
            {field: 10.0 for field in BENCHER_MEASURES},
            {field: 14.0 for field in BENCHER_MEASURES},
        ]

        summary, bencher = aggregate_measurements("point", results)

        for field, measure in BENCHER_MEASURES.items():
            self.assertEqual(summary[field], 12.0)
            self.assertAlmostEqual(
                summary[f"{field}_sample_stddev"],
                8.0**0.5,
            )
            self.assertEqual(bencher[measure]["value"], 12.0)
            self.assertAlmostEqual(
                bencher[measure]["lower_value"],
                12.0 - 8.0**0.5,
            )
            self.assertAlmostEqual(
                bencher[measure]["upper_value"],
                12.0 + 8.0**0.5,
            )


class JitCachePublicationTests(unittest.TestCase):
    def test_readwrite_publication_triggers_agents_and_is_strict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            logdir = directory / "logs"
            staging = directory / "staging"
            trigger = directory / "publish.trigger"
            logdir.mkdir()
            staging.mkdir()

            def publish(command: list[str], *, check: bool) -> None:
                self.assertTrue(check)
                self.assertTrue(trigger.is_file())
                output = Path(command[command.index("--output") + 1])
                output.write_text(
                    json.dumps({"status": "published", "generation": "test"})
                )

            environment = {
                "JIT_CACHE_MODE": "readwrite",
                "JIT_CACHE_TRIGGER": str(trigger),
                "JIT_CACHE_STAGING": str(staging),
                "SLURM_JOB_NUM_NODES": "4",
            }
            with (
                mock.patch.dict(os.environ, environment, clear=False),
                mock.patch("benchmark.subprocess.run", side_effect=publish) as run,
            ):
                publication, command = publish_jit_cache_after_warmup(logdir)

            run.assert_called_once()
            self.assertEqual(publication["status"], "published")
            self.assertEqual(publication["phase"], "after-full-suite-warmup")
            self.assertIsNotNone(command)
            assert command is not None
            self.assertIn("--strict", command)
            self.assertEqual(
                command[command.index("--expected-bundles") + 1],
                "4",
            )

    def test_read_mode_does_not_trigger_or_publish(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            logdir = Path(temporary)
            with (
                mock.patch.dict(os.environ, {"JIT_CACHE_MODE": "read"}, clear=False),
                mock.patch("benchmark.subprocess.run") as run,
            ):
                publication, command = publish_jit_cache_after_warmup(logdir)

            run.assert_not_called()
            self.assertIsNone(command)
            self.assertEqual(
                publication,
                {
                    "status": "not-published",
                    "mode": "read",
                    "phase": "after-full-suite-warmup",
                },
            )


if __name__ == "__main__":
    unittest.main()
