from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from scenario import resolve_scenario, select_nodes


class ScenarioAllocationTests(unittest.TestCase):
    def test_baseline_defaults_to_minimum_nodes(self) -> None:
        resolved = resolve_scenario("baseline", "cuda")

        self.assertEqual(select_nodes(resolved), 4)
        self.assertEqual(
            resolved["server"]["chat-template-content-format"],
            "openai",
        )

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

    def test_cuda_uses_explicit_gh200_numa_binding(self) -> None:
        resolved = resolve_scenario("baseline", "cuda")
        server = resolved["server"]

        self.assertTrue(server["numa-bind"])
        self.assertEqual(server["numa-bind-nodes"], [0, 1, 2, 3])
        self.assertEqual(
            server["numa-bind-cpus"],
            ["0-71", "72-143", "144-215", "216-287"],
        )


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

    def test_platform_constraint_is_checked_before_submission(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not support platform rocm"):
            resolve_scenario("production", "rocm")


if __name__ == "__main__":
    unittest.main()
