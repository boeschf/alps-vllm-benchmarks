from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("jit_cache", ROOT / "scripts" / "jit-cache.py")
assert SPEC and SPEC.loader
jit_cache = importlib.util.module_from_spec(SPEC)
PROVENANCE_SPEC = importlib.util.spec_from_file_location(
    "provenance", ROOT / "scripts" / "provenance.py"
)
assert PROVENANCE_SPEC and PROVENANCE_SPEC.loader
provenance = importlib.util.module_from_spec(PROVENANCE_SPEC)
PROVENANCE_SPEC.loader.exec_module(provenance)

SPEC.loader.exec_module(jit_cache)

class RuntimeIdentityTests(unittest.TestCase):
    def test_cache_timing_does_not_change_stable_identity(self) -> None:
        record = {
            "deployment_role": "single",
            "python": "test",
            "platform": "test",
            "kernel": "test",
            "machine": "test",
            "packages": {},
            "accelerator": {},
            "environment": {},
            "jit_cache": {
                "mode": "readwrite",
                "compatibility_key": "key",
                "hit": True,
                "generation": "generation",
                "archive_sha256": "archive",
                "hydrate_seconds": 1.0,
            },
        }
        other = json.loads(json.dumps(record))
        other["jit_cache"]["hydrate_seconds"] = 99.0
        self.assertEqual(
            provenance.stable_runtime_identity([record]),
            provenance.stable_runtime_identity([other]),
        )



class CacheIdentityTests(unittest.TestCase):
    def test_cross_image_seed_requires_allowlist_and_identical_runtime(self) -> None:
        current = {
            "schema_version": 2,
            "platform": "cuda",
            "image_ref": "new",
            "python_abi": "cpython-312",
            "packages": {"torch": "test"},
            "accelerator": {"architecture": "sm90"},
            "cuda": "13.3",
            "hip": None,
        }
        candidate = {**current, "image_ref": "old"}
        with patch.dict(os.environ, {"JIT_CACHE_SEED_IMAGE_REFS": "old"}):
            self.assertTrue(jit_cache.image_seed_compatible(current, candidate))
            incompatible = {
                **candidate,
                "packages": {"torch": "different"},
            }
            self.assertFalse(jit_cache.image_seed_compatible(current, incompatible))
        with patch.dict(os.environ, {"JIT_CACHE_SEED_IMAGE_REFS": "other"}):
            self.assertFalse(jit_cache.image_seed_compatible(current, candidate))


class MergeTests(unittest.TestCase):
    def test_disjoint_and_identical_files_merge(self) -> None:
        base = {"vllm/base": {"sha256": "base", "size": 1, "mode": 420}}
        current = {
            **base,
            "triton/current": {"sha256": "current", "size": 1, "mode": 420},
        }
        job = {
            **base,
            "flashinfer/job": {"sha256": "job", "size": 1, "mode": 420},
            "triton/current": {"sha256": "current", "size": 1, "mode": 420},
        }
        merged, accepted, conflicts = jit_cache.merge_bundle_files(base, current, job)
        self.assertEqual(conflicts, [])
        self.assertEqual(accepted, ["flashinfer/job"])
        self.assertEqual(merged["flashinfer/job"], job["flashinfer/job"])

    def test_divergent_change_conflicts(self) -> None:
        base = {"vllm/config": {"sha256": "base", "size": 1, "mode": 420}}
        current = {"vllm/config": {"sha256": "current", "size": 1, "mode": 420}}
        job = {"vllm/config": {"sha256": "job", "size": 1, "mode": 420}}
        merged, accepted, conflicts = jit_cache.merge_bundle_files(base, current, job)
        self.assertEqual(merged, current)
        self.assertEqual(accepted, [])
        self.assertEqual(conflicts, ["vllm/config"])


class ArchiveTests(unittest.TestCase):
    def test_traversal_member_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "bad.tar"
            with tarfile.open(archive, "w") as bundle:
                member = tarfile.TarInfo("../escape")
                payload = b"bad"
                member.size = len(payload)
                bundle.addfile(member, io.BytesIO(payload))
            with self.assertRaisesRegex(ValueError, "unsafe archive member"):
                jit_cache.safe_extract(archive, root / "out")
            self.assertFalse((root / "escape").exists())

    def test_files_receive_one_snapshot_mtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "cache.tar"
            with tarfile.open(archive, "w") as bundle:
                for name, mtime in (
                    ("triton/kernel.cu", 1_600_000_000),
                    ("triton/kernel.o", 1_600_000_100),
                ):
                    member = tarfile.TarInfo(name)
                    payload = name.encode()
                    member.size = len(payload)
                    member.mtime = mtime
                    bundle.addfile(member, io.BytesIO(payload))
            destination = root / "out"
            jit_cache.safe_extract(archive, destination)
            self.assertEqual(
                (destination / "triton" / "kernel.cu").stat().st_mtime_ns,
                (destination / "triton" / "kernel.o").stat().st_mtime_ns,
            )

    def test_ninja_output_mtime_is_restored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "cache.tar"
            recorded_ns = 1_600_000_100_123_456_789
            files = {
                "triton/op/.ninja_log": (
                    "# ninja log v6\n"
                    f"0\t1\t{recorded_ns}\t/tmp/triton-cache/op/kernel.o\thash\n"
                ).encode(),
                "triton/op/kernel.cu": b"source",
                "triton/op/kernel.o": b"object",
            }
            with tarfile.open(archive, "w") as bundle:
                for name, payload in files.items():
                    member = tarfile.TarInfo(name)
                    member.size = len(payload)
                    member.mtime = 1_700_000_000
                    bundle.addfile(member, io.BytesIO(payload))
            destination = root / "out"
            jit_cache.safe_extract(archive, destination)
            source = destination / "triton" / "op" / "kernel.cu"
            output = destination / "triton" / "op" / "kernel.o"
            self.assertEqual(output.stat().st_mtime_ns, recorded_ns)
            self.assertLess(source.stat().st_mtime_ns, output.stat().st_mtime_ns)

    def test_rank_local_autotune_configs_are_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            roots = {
                logical: root / logical
                for logical in jit_cache.CACHE_ROOTS
            }
            for path in roots.values():
                path.mkdir(parents=True)
            excluded = roots["vllm"] / "flashinfer_autotune_cache" / "configs.json"
            excluded.parent.mkdir(parents=True)
            excluded.write_text("{}")
            included = roots["triton"] / "kernel.bin"
            included.write_bytes(b"kernel")
            archive = root / "cache.tar"
            jit_cache.create_archive_from_roots(archive, roots)
            manifest = jit_cache.archive_file_manifest(archive)
            self.assertNotIn("vllm/flashinfer_autotune_cache/configs.json", manifest)
            self.assertIn("triton/kernel.bin", manifest)


class LifecycleTests(unittest.TestCase):
    def test_cold_capture_publish_and_warm_hydrate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            environment = {
                "JIT_CACHE_ROOT": str(root / "persistent"),
                "JIT_CACHE_MODE": "readwrite",
                "JIT_CACHE_KEEP_GENERATIONS": "3",
                "PLATFORM": "cuda",
                "IMAGE_REF": "example/image@sha256:test",
            }
            environment.update(
                {
                    variable: str(root / "live" / logical)
                    for logical, (variable, _) in jit_cache.CACHE_ROOTS.items()
                }
            )
            identity = {
                "schema_version": jit_cache.SCHEMA_VERSION,
                "platform": "cuda",
                "image_ref": environment["IMAGE_REF"],
                "python_abi": "cpython-test",
                "python_version": "test",
                "packages": {},
                "accelerator": {
                    "architecture": "sm90",
                    "name": "test",
                    "major": 9,
                    "minor": 0,
                },
                "cuda": "test",
                "hip": None,
            }
            status_path = root / "cold.json"
            with (
                patch.dict(os.environ, environment),
                patch.object(jit_cache, "compatibility_identity", return_value=identity),
            ):
                cold = jit_cache.hydrate(status_path)
                self.assertFalse(cold["hit"])
                kernel = Path(environment["TRITON_CACHE_DIR"]) / "kernel.bin"
                kernel.write_bytes(b"compiled")

                staging = root / "staging"
                jit_cache.capture_bundle(status_path, staging, "test")
                published = jit_cache.publish_bundles(staging, 1)
                self.assertEqual(published["status"], "published")

                warm = jit_cache.hydrate(root / "warm.json")
                self.assertTrue(warm["hit"])
                self.assertEqual(kernel.read_bytes(), b"compiled")


class PublicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.previous = os.environ.get("JIT_CACHE_ROOT")
        os.environ["JIT_CACHE_ROOT"] = str(self.root / "persistent")
        os.environ["JIT_CACHE_KEEP_GENERATIONS"] = "3"

    def tearDown(self) -> None:
        if self.previous is None:
            os.environ.pop("JIT_CACHE_ROOT", None)
        else:
            os.environ["JIT_CACHE_ROOT"] = self.previous
        self.temporary.cleanup()

    def make_bundle(
        self,
        staging: Path,
        name: str,
        files: dict[str, bytes],
        *,
        base_generation: str | None = None,
    ) -> None:
        tree = self.root / f"tree-{name}"
        for logical in jit_cache.CACHE_ROOTS:
            (tree / logical).mkdir(parents=True, exist_ok=True)
        for relative, content in files.items():
            path = tree / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        archive = staging / f"{name}.tar"
        jit_cache.create_archive_from_tree(archive, tree)
        identity = {
            "schema_version": jit_cache.SCHEMA_VERSION,
            "platform": "cuda",
            "image_ref": "example/image@sha256:test",
            "python_abi": "cpython-test",
            "python_version": "test",
            "packages": {},
            "accelerator": {"architecture": "sm90", "name": "test", "major": 9, "minor": 0},
            "cuda": "test",
            "hip": None,
        }
        metadata = {
            "schema_version": jit_cache.SCHEMA_VERSION,
            "kind": "bundle",
            "job_id": "test",
            "scenario_id": name,
            "role": name,
            "hostname": name,
            "base_generation": base_generation,
            "compatibility": identity,
            "compatibility_key": jit_cache.canonical_sha256(identity),
            "archive": {
                "name": archive.name,
                "size": archive.stat().st_size,
                "sha256": jit_cache.file_sha256(archive),
            },
            "files": jit_cache.archive_file_manifest(archive),
        }
        (staging / f"{name}.json").write_text(json.dumps(metadata))
        (staging / f"{name}.ready").write_text("ready\n")

    def test_disjoint_bundles_publish_one_atomic_generation(self) -> None:
        staging = self.root / "staging"
        staging.mkdir()
        self.make_bundle(staging, "prefill", {"triton/a.bin": b"a"})
        self.make_bundle(staging, "decode", {"flashinfer/b.bin": b"b"})
        result = jit_cache.publish_bundles(staging, 2)
        self.assertEqual(result["status"], "published")
        namespace = jit_cache.namespace_path(result["compatibility_key"], platform_name="cuda")
        generation = (namespace / "CURRENT").read_text().strip()
        manifest = jit_cache.generation_manifest(namespace, generation)
        self.assertIn("triton/a.bin", manifest["files"])
        self.assertIn("flashinfer/b.bin", manifest["files"])

    def test_same_job_node_divergence_uses_canonical_bundle(self) -> None:
        staging = self.root / "staging"
        staging.mkdir()
        self.make_bundle(staging, "a", {"triton/kernel.cubin": b"canonical"})
        self.make_bundle(staging, "b", {"triton/kernel.cubin": b"other-node"})
        result = jit_cache.publish_bundles(staging, 2)
        self.assertEqual(result["status"], "published")
        self.assertEqual(result["coalesced_node_divergences"], 1)
        namespace = jit_cache.namespace_path(result["compatibility_key"], platform_name="cuda")
        manifest = jit_cache.generation_manifest(namespace, result["generation"])
        extracted = self.root / "extracted"
        jit_cache.safe_extract(
            namespace / "generations" / result["generation"] / manifest["archive"]["name"],
            extracted,
        )
        self.assertEqual((extracted / "triton/kernel.cubin").read_bytes(), b"canonical")

    def test_concurrent_divergent_update_leaves_current_unchanged(self) -> None:
        first = self.root / "first"
        first.mkdir()
        self.make_bundle(first, "base", {"vllm/config.json": b"base"})
        initial = jit_cache.publish_bundles(first, 1)
        base_generation = initial["generation"]

        advance = self.root / "advance"
        advance.mkdir()
        self.make_bundle(
            advance,
            "advance",
            {"vllm/config.json": b"current"},
            base_generation=base_generation,
        )
        advanced = jit_cache.publish_bundles(advance, 1)

        stale = self.root / "stale"
        stale.mkdir()
        self.make_bundle(
            stale,
            "stale",
            {"vllm/config.json": b"stale"},
            base_generation=base_generation,
        )
        result = jit_cache.publish_bundles(stale, 1)
        self.assertEqual(result["status"], "conflict")
        namespace = jit_cache.namespace_path(result["compatibility_key"], platform_name="cuda")
        self.assertEqual((namespace / "CURRENT").read_text().strip(), advanced["generation"])


if __name__ == "__main__":
    unittest.main()
