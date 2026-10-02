from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.ninja_cache import restore_ninja_mtimes

CACHE_ROOTS = {"triton": ("TRITON_CACHE_DIR", "/tmp/triton-cache")}


class NinjaTimestampTests(unittest.TestCase):
    def test_files_without_ninja_outputs_share_one_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tree = Path(temporary)
            source = tree / "triton" / "kernel.cu"
            output = tree / "triton" / "kernel.o"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"source")
            output.write_bytes(b"output")

            restore_ninja_mtimes(tree, [source, output], 1_700_000_000, CACHE_ROOTS)

            self.assertEqual(source.stat().st_mtime_ns, output.stat().st_mtime_ns)

    def test_relative_output_is_resolved_beside_ninja_log(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tree = Path(temporary)
            directory = tree / "triton" / "op"
            directory.mkdir(parents=True)
            output = directory / "kernel.o"
            output.write_bytes(b"output")
            recorded_ns = 1_600_000_100_123_456_789
            (directory / ".ninja_log").write_text(
                f"# ninja log v6\n0\t1\t{recorded_ns}\tkernel.o\thash\n"
            )

            restore_ninja_mtimes(tree, [output], 1_700_000_000, CACHE_ROOTS)

            self.assertEqual(output.stat().st_mtime_ns, recorded_ns)

    def test_absolute_output_is_remapped_from_live_cache_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tree = Path(temporary)
            directory = tree / "triton" / "op"
            directory.mkdir(parents=True)
            output = directory / "kernel.o"
            output.write_bytes(b"output")
            recorded_ns = 1_600_000_100_123_456_789
            (directory / ".ninja_log").write_text(
                f"# ninja log v6\n0\t1\t{recorded_ns}\t/live/triton/op/kernel.o\thash\n"
            )

            with patch.dict(os.environ, {"TRITON_CACHE_DIR": "/live/triton"}):
                restore_ninja_mtimes(tree, [output], 1_700_000_000, CACHE_ROOTS)

            self.assertEqual(output.stat().st_mtime_ns, recorded_ns)


if __name__ == "__main__":
    unittest.main()
