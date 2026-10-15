from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "smoke_request",
    ROOT / "scripts" / "smoke-request.py",
)
assert SPEC is not None and SPEC.loader is not None
SMOKE_REQUEST = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SMOKE_REQUEST)


class LongContextPromptTests(unittest.TestCase):
    def test_repeated_prompt_reaches_exact_effective_token_target(self) -> None:
        def token_count(prompt: str) -> int:
            return 12 + len(prompt.split())

        prompt, observed = SMOKE_REQUEST.build_repeated_prompt(1_000_000, token_count)

        self.assertEqual(observed, 1_000_000)
        self.assertEqual(token_count(prompt), 1_000_000)

    def test_repeated_prompt_never_exceeds_target(self) -> None:
        def token_count(prompt: str) -> int:
            return 11 + 2 * len(prompt.split())

        prompt, observed = SMOKE_REQUEST.build_repeated_prompt(100, token_count)

        self.assertEqual(observed, 99)
        self.assertEqual(token_count(prompt), 99)


if __name__ == "__main__":
    unittest.main()
