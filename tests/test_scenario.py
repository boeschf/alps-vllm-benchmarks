from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from scenario import resolve_scenario, select_nodes, validate_server


class ScenarioAllocationTests(unittest.TestCase):
    def test_baseline_defaults_to_minimum_nodes(self) -> None:
        resolved = resolve_scenario("baseline", "cuda")

        self.assertEqual(select_nodes(resolved), 4)
        self.assertEqual(
            resolved["server"]["chat-template-content-format"],
            "openai",
        )
        self.assertEqual(resolved["server"]["kv-cache-dtype"], "bfloat16")
        self.assertEqual(resolved["server"]["gpu-memory-utilization"], 0.88)

    def test_rocm_uses_explicit_mi300a_numa_binding(self) -> None:
        resolved = resolve_scenario("baseline", "rocm")
        server = resolved["server"]

        self.assertTrue(server["numa-bind"])
        self.assertEqual(server["numa-bind-nodes"], [0, 1, 2, 3])
        self.assertEqual(
            server["numa-bind-cpus"],
            [
                "0-23,96-119",
                "24-47,120-143",
                "48-71,144-167",
                "72-95,168-191",
            ],
        )
        self.assertEqual(resolved["environment"]["VLLM_ROCM_USE_AITER"], "1")
        self.assertEqual(server["gpu-memory-utilization"], 0.95)

    def test_cuda_uses_explicit_gh200_numa_binding(self) -> None:
        resolved = resolve_scenario("baseline", "cuda")
        server = resolved["server"]

        self.assertTrue(server["numa-bind"])
        self.assertEqual(server["numa-bind-nodes"], [0, 1, 2, 3])
        self.assertEqual(
            server["numa-bind-cpus"],
            ["0-71", "72-143", "144-215", "216-287"],
        )

    def test_platform_defaults_apply_to_every_supported_scenario(self) -> None:
        expected_numa = {
            "cuda": ["0-71", "72-143", "144-215", "216-287"],
            "rocm": [
                "0-23,96-119",
                "24-47,120-143",
                "48-71,144-167",
                "72-95,168-191",
            ],
        }
        scenario_names = sorted(
            path.stem for path in (ROOT / "config" / "scenarios").glob("*.json")
        )

        for scenario_name in scenario_names:
            for platform, cpu_bindings in expected_numa.items():
                try:
                    resolved = resolve_scenario(scenario_name, platform)
                except ValueError as error:
                    if "does not support platform" in str(error):
                        continue
                    raise
                server = resolved["server"]
                with self.subTest(scenario=scenario_name, platform=platform):
                    self.assertTrue(server["numa-bind"])
                    self.assertEqual(server["numa-bind-nodes"], [0, 1, 2, 3])
                    self.assertEqual(server["numa-bind-cpus"], cpu_bindings)
                    if platform == "rocm":
                        self.assertEqual(
                            resolved["environment"]["VLLM_ROCM_USE_AITER"],
                            "1",
                        )
                    self.assertIn(
                        f"config/platforms/{platform}.json",
                        {
                            source["path"]
                            for source in resolved["source_files"]
                        },
                    )

    def test_intentional_scenario_memory_overrides_platform_default(self) -> None:
        self.assertEqual(
            resolve_scenario("production", "cuda")["server"][
                "gpu-memory-utilization"
            ],
            0.87,
        )
        self.assertEqual(
            resolve_scenario("uccl-ll", "cuda")["server"][
                "gpu-memory-utilization"
            ],
            0.9,
        )
        self.assertEqual(
            resolve_scenario("uccl-ll", "rocm")["server"][
                "gpu-memory-utilization"
            ],
            0.7,
        )

    def test_uccl_ll_bounds_low_latency_rdma_capacity(self) -> None:
        baseline = resolve_scenario("baseline", "cuda")
        resolved = resolve_scenario("uccl-ll", "cuda")
        composed = resolve_scenario("uccl-ll+moe-deep-gemm", "cuda")

        self.assertEqual(baseline["server"]["max-num-batched-tokens"], 2048)
        self.assertEqual(resolved["server"]["max-num-batched-tokens"], 1024)
        self.assertEqual(composed["server"]["max-num-batched-tokens"], 1024)
        self.assertEqual(resolved["server"]["gpu-memory-utilization"], 0.9)
        self.assertEqual(composed["server"]["gpu-memory-utilization"], 0.9)

    def test_uccl_ht_uses_cuda_deep_gemm_workaround(self) -> None:
        baseline = resolve_scenario("baseline", "cuda")
        resolved = resolve_scenario("uccl-ht", "cuda")

        validate_server(resolved, resolved["environment"])
        self.assertEqual(
            resolved["server"]["all2all-backend"],
            "deepep_high_throughput",
        )
        self.assertEqual(resolved["server"]["moe-backend"], "deep_gemm")
        self.assertEqual(
            resolved["server"]["gpu-memory-utilization"],
            baseline["server"]["gpu-memory-utilization"],
        )

    def test_uccl_ht_keeps_rocm_backend_selection(self) -> None:
        resolved = resolve_scenario("uccl-ht", "rocm")

        validate_server(resolved, resolved["environment"])
        self.assertEqual(
            resolved["server"]["all2all-backend"],
            "deepep_high_throughput",
        )
        self.assertNotIn("moe-backend", resolved["server"])
        self.assertEqual(resolved["environment"]["VLLM_ROCM_USE_AITER"], "1")

    def test_deep_gemm_component_composes_with_baseline_and_low_latency(self) -> None:
        scenarios = {
            "moe-deep-gemm": ("allgather_reducescatter", 0.88),
            "uccl-ll+moe-deep-gemm": ("deepep_low_latency", 0.9),
        }

        for scenario, (all2all_backend, gpu_memory_utilization) in scenarios.items():
            with self.subTest(scenario=scenario):
                resolved = resolve_scenario(scenario, "cuda")
                validate_server(resolved, resolved["environment"])
                self.assertEqual(
                    resolved["server"]["all2all-backend"],
                    all2all_backend,
                )
                self.assertEqual(resolved["server"]["moe-backend"], "deep_gemm")
                self.assertEqual(
                    resolved["server"]["gpu-memory-utilization"],
                    gpu_memory_utilization,
                )

    def test_deep_gemm_component_is_cuda_only(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not support platform rocm"):
            resolve_scenario("moe-deep-gemm", "rocm")

    def test_scalable_scenario_accepts_larger_allocation(self) -> None:
        resolved = resolve_scenario("uccl-ll+eplb", "cuda")

        self.assertEqual(select_nodes(resolved, 8), 8)

    def test_scalable_scenario_rejects_allocation_below_minimum(self) -> None:
        resolved = resolve_scenario("baseline", "cuda")

        with self.assertRaisesRegex(ValueError, "requires at least 4 nodes"):
            select_nodes(resolved, 3)

    def test_production_requires_exact_allocation(self) -> None:
        resolved = resolve_scenario("production", "cuda")

        self.assertEqual(select_nodes(resolved), 8)
        with self.assertRaisesRegex(ValueError, "requires exactly 8 nodes"):
            select_nodes(resolved, 4)

    def test_disaggregated_allocation_comes_from_deployment_roles(self) -> None:
        resolved = resolve_scenario("disaggregated-pd", "cuda")

        self.assertNotIn("exact_nodes", resolved["constraints"])
        self.assertEqual(select_nodes(resolved), 8)
        with self.assertRaisesRegex(ValueError, "requires exactly 8 nodes"):
            select_nodes(resolved, 12)
        prefill_server = resolved["deployments"]["prefill"]["server"]
        self.assertNotIn("moe-backend", prefill_server)
        self.assertNotIn("gpu-memory-utilization", prefill_server)
        decode_server = resolved["deployments"]["decode"]["server"]
        self.assertEqual(decode_server["max-num-batched-tokens"], 1024)
        self.assertEqual(decode_server["gpu-memory-utilization"], 0.9)

    def test_platform_constraint_is_checked_before_submission(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not support platform rocm"):
            resolve_scenario("production", "rocm")


if __name__ == "__main__":
    unittest.main()
