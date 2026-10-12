from __future__ import annotations

import json
import os
import subprocess
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
    archive_reproduction_bundle,
    load_suite,
    make_warmup_point,
    publish_jit_cache_after_warmup,
    run_with_heartbeat,
)
from provenance import artifact_inventory
from script_utils import canonical_sha256, file_sha256


class BenchmarkStatisticsTests(unittest.TestCase):
    def test_standard_suite_has_enough_requests_for_p99(self) -> None:
        points = load_suite(ROOT / "benchmarks" / "standard.tsv")

        self.assertTrue(points)
        self.assertTrue(
            all(point["prompts"] >= MIN_PERCENTILE_SAMPLES for point in points)
        )

    def test_standard_suite_uses_two_wave_warmup_counts(self) -> None:
        points = load_suite(ROOT / "benchmarks" / "standard.tsv")

        for point in points:
            expected = min(
                point["prompts"],
                max(16, 2 * point["concurrency"]),
            )
            self.assertEqual(point["warmup_prompts"], expected)
            warmup = make_warmup_point(point)
            self.assertEqual(warmup["prompts"], expected)
            self.assertEqual(warmup["measurement_prompts"], point["prompts"])

    def test_warmup_must_reach_requested_concurrency(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            suite = Path(temporary) / "suite.tsv"
            suite.write_text(
                "name\tscenario\tinput_len\toutput_len\tprompts\t"
                "warmup_prompts\tconcurrency\n"
                "invalid\ttest\t32\t8\t100\t3\t4\n"
            )

            with self.assertRaisesRegex(
                ValueError,
                "warmup_prompts must reach concurrency",
            ):
                load_suite(suite)

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



class ReproductionBundleTests(unittest.TestCase):
    def test_archives_exact_configs_and_allowlisted_environment(self) -> None:

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scenario_dir = root / "scenario"
            scenario_dir.mkdir()
            output_dir = root / "result"
            output_dir.mkdir()
            config_path = scenario_dir / "vllm-single-node0.yaml"
            config_path.write_text(
                "model: /models/GLM-5.3\n"
                "data-parallel-address: 10.0.0.1\n"
                "host: 0.0.0.0\n"
            )
            identity_sha256 = canonical_sha256(
                {
                    "model": "/models/GLM-5.3",
                    "data-parallel-address": "<runtime-address>",
                    "host": "0.0.0.0",
                }
            )
            scenario_path = scenario_dir / "scenario.json"
            scenario_path.write_text(
                json.dumps(
                    {
                        "id": "baseline",
                        "mode": "single",
                        "vllm_configs": {
                            "single": {
                                "0": {
                                    "path": config_path.name,
                                    "sha256": file_sha256(config_path),
                                    "identity_sha256": identity_sha256,
                                }
                            }
                        },
                    }
                )
            )
            runtime_records = [
                {
                    "deployment_role": "single",
                    "node_rank": 0,
                    "environment": {
                        "IMAGE_REF": "example/image@sha256:test",
                        "JIT_CACHE_ROOT": "/machine-specific/cache",
                    },
                }
            ]

            reproduction = archive_reproduction_bundle(
                scenario_path,
                output_dir,
                runtime_records,
            )

            archived = output_dir / "reproduction" / "vllm" / config_path.name
            self.assertEqual(archived.read_bytes(), config_path.read_bytes())
            self.assertEqual(reproduction["vllm_config_count"], 1)
            self.assertEqual(
                reproduction["sha256"],
                file_sha256(output_dir / reproduction["path"]),
            )
            environment = json.loads(
                (output_dir / "reproduction" / "environment.json").read_text()
            )
            self.assertEqual(environment["records"], runtime_records)
            bundle = json.loads(
                (output_dir / "reproduction" / "bundle.json").read_text()
            )
            self.assertEqual(bundle["identity_sha256"], reproduction["identity_sha256"])
            self.assertEqual(
                bundle["vllm_configs"][0]["identity_sha256"],
                identity_sha256,
            )


            artifact_paths = {
                artifact["path"] for artifact in artifact_inventory(output_dir)
            }
            self.assertEqual(
                artifact_paths,
                {
                    "reproduction/bundle.json",
                    "reproduction/environment.json",
                    "reproduction/vllm/vllm-single-node0.yaml",
                },
            )

            second_output = root / "second-result"
            second_output.mkdir()
            second_runtime = json.loads(json.dumps(runtime_records))
            second_runtime[0]["environment"]["JIT_CACHE_ROOT"] = "/another/cache"
            second_reproduction = archive_reproduction_bundle(
                scenario_path,
                second_output,
                second_runtime,
            )
            self.assertEqual(
                second_reproduction["identity_sha256"],
                reproduction["identity_sha256"],
            )
            self.assertNotEqual(second_reproduction["sha256"], reproduction["sha256"])

    def test_rejects_changed_source_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scenario_dir = root / "scenario"
            scenario_dir.mkdir()
            output_dir = root / "result"
            output_dir.mkdir()
            config_path = scenario_dir / "vllm-single-node0.yaml"
            config_path.write_text("model: /models/GLM-5.3\n")
            scenario_path = scenario_dir / "scenario.json"
            scenario_path.write_text(
                json.dumps(
                    {
                        "id": "baseline",
                        "vllm_configs": {
                            "single": {
                                "0": {
                                    "path": config_path.name,
                                    "sha256": "0" * 64,
                                    "identity_sha256": "1" * 64,
                                }
                            }
                        },
                    }
                )
            )

            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                archive_reproduction_bundle(scenario_path, output_dir, [])


class BenchmarkHeartbeatTests(unittest.TestCase):
    def test_long_command_reports_periodic_and_completion_status(self) -> None:
        process = mock.Mock()
        process.wait.side_effect = [
            subprocess.TimeoutExpired(cmd=["fake"], timeout=60),
            0,
        ]

        with (
            mock.patch("benchmark.subprocess.Popen", return_value=process) as popen,
            mock.patch("benchmark.log_status") as status,
            mock.patch("benchmark.time.monotonic", side_effect=[0.0, 61.0, 62.0]),
        ):
            run_with_heartbeat(["fake"], "benchmark point")

        popen.assert_called_once_with(["fake"])
        self.assertEqual(
            status.call_args_list,
            [
                mock.call("benchmark point: still running; elapsed=1m01s"),
                mock.call("benchmark point: complete; elapsed=1m02s"),
            ],
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

            def publish(command: list[str], label: str) -> None:
                self.assertEqual(label, "JIT cache publication")
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
                mock.patch("benchmark.run_with_heartbeat", side_effect=publish) as run,
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

    def test_concurrent_cache_conflict_does_not_abort_benchmark(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            logdir = directory / "logs"
            staging = directory / "staging"
            trigger = directory / "publish.trigger"
            logdir.mkdir()
            staging.mkdir()

            def publish(command: list[str], label: str) -> None:
                output = Path(command[command.index("--output") + 1])
                output.write_text(
                    json.dumps(
                        {
                            "status": "conflict",
                            "generation": "concurrent",
                            "conflicts": ["vllm/kernel.cubin"],
                        }
                    )
                )

            environment = {
                "JIT_CACHE_MODE": "readwrite",
                "JIT_CACHE_TRIGGER": str(trigger),
                "JIT_CACHE_STAGING": str(staging),
                "SLURM_JOB_NUM_NODES": "4",
            }
            with (
                mock.patch.dict(os.environ, environment, clear=False),
                mock.patch("benchmark.run_with_heartbeat", side_effect=publish),
            ):
                publication, command = publish_jit_cache_after_warmup(logdir)

            self.assertEqual(publication["status"], "conflict")
            self.assertEqual(publication["generation"], "concurrent")
            self.assertIsNotNone(command)

    def test_read_mode_does_not_trigger_or_publish(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            logdir = Path(temporary)
            with (
                mock.patch.dict(os.environ, {"JIT_CACHE_MODE": "read"}, clear=False),
                mock.patch("benchmark.run_with_heartbeat") as run,
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
