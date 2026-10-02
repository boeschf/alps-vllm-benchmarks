from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from script_utils import atomic_text


class AtomicTextTests(unittest.TestCase):
    def test_failed_replace_preserves_destination_and_removes_temporary_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            destination = directory / "state.json"
            destination.write_text("old\n")

            with mock.patch("script_utils.os.replace", side_effect=OSError("replace failed")):
                with self.assertRaisesRegex(OSError, "replace failed"):
                    atomic_text(destination, "new\n")

            self.assertEqual(destination.read_text(), "old\n")
            self.assertEqual(list(directory.iterdir()), [destination])


if __name__ == "__main__":
    unittest.main()
