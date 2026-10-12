from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

RUNTIME_SPEC = importlib.util.spec_from_file_location(
    "runtime_fingerprint", ROOT / "scripts" / "runtime-fingerprint.py"
)
assert RUNTIME_SPEC and RUNTIME_SPEC.loader
runtime_fingerprint = importlib.util.module_from_spec(RUNTIME_SPEC)
RUNTIME_SPEC.loader.exec_module(runtime_fingerprint)

RESOLVER_SPEC = importlib.util.spec_from_file_location(
    "resolve_image_digest", ROOT / "scripts" / "resolve-image-digest.py"
)
assert RESOLVER_SPEC and RESOLVER_SPEC.loader
resolve_image_digest = importlib.util.module_from_spec(RESOLVER_SPEC)
RESOLVER_SPEC.loader.exec_module(resolve_image_digest)


class RuntimeEnvironmentTests(unittest.TestCase):
    def test_edf_environment_is_allowlisted_without_capturing_unrelated_values(self) -> None:
        edf_names: set[str] = set()
        for name in ("vllm-cuda.toml", "vllm-rocm.toml"):
            with (ROOT / "edf" / name).open("rb") as stream:
                edf_names.update(tomllib.load(stream)["env"])
        self.assertLessEqual(edf_names, runtime_fingerprint.RUNTIME_ENVIRONMENT_NAMES)

        with patch.dict(
            os.environ,
            {
                "IMAGE_DIGEST": "sha256:" + "1" * 64,
                "TRITON_CACHE_DIR": "/tmp/triton-cache",
                "SCENARIO_ENV_NAMES": "CUSTOM_SCENARIO_VALUE",
                "CUSTOM_SCENARIO_VALUE": "enabled",
                "UNRELATED_SECRET": "not-for-provenance",
            },
            clear=True,
        ):
            self.assertEqual(
                runtime_fingerprint.captured_environment(),
                {
                    "CUSTOM_SCENARIO_VALUE": "enabled",
                    "IMAGE_DIGEST": "sha256:" + "1" * 64,
                    "TRITON_CACHE_DIR": "/tmp/triton-cache",
                },
            )


class Response:
    def __init__(self, body: bytes = b"", headers: dict[str, str] | None = None):
        self.body = body
        self.headers = headers or {}

    def __enter__(self) -> Response:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


class ImageDigestTests(unittest.TestCase):
    def test_anonymous_bearer_challenge_resolves_jfrog_manifest_digest(self) -> None:
        image = (
            "jfrog.svc.cscs.ch/docker-group-csstaff/alps-images/vllm-cuda:"
            "alps7-dev-test"
        )
        digest = "sha256:" + "a" * 64
        manifest_url = (
            "https://jfrog.svc.cscs.ch/artifactory/api/docker/"
            "docker-group-csstaff/v2/alps-images/vllm-cuda/manifests/alps7-dev-test"
        )
        requests = []

        def opener(request):
            requests.append(request)
            if request.full_url.startswith("https://jfrog.svc.cscs.ch/v2/token?"):
                return Response(json.dumps({"token": "anonymous-token"}).encode())
            if request.full_url != manifest_url:
                raise AssertionError(request.full_url)
            if request.get_header("Authorization") is None:
                raise HTTPError(
                    request.full_url,
                    401,
                    "Unauthorized",
                    {
                        "WWW-Authenticate": (
                            'Bearer realm="https://jfrog.svc.cscs.ch/v2/token",'
                            'service="jfrog.svc.cscs.ch",'
                            'scope="repository:alps-images/vllm-cuda:pull"'
                        )
                    },
                    io.BytesIO(b""),
                )
            self.assertEqual(
                request.get_header("Authorization"), "Bearer anonymous-token"
            )
            return Response(headers={"Docker-Content-Digest": digest})

        self.assertEqual(resolve_image_digest.resolve_image_digest(image, opener), digest)
        self.assertEqual([request.get_method() for request in requests], ["HEAD", "GET", "HEAD"])
        self.assertIn(
            "scope=repository%3Aalps-images%2Fvllm-cuda%3Apull",
            requests[1].full_url,
        )

    def test_reference_requires_explicit_tag_or_digest(self) -> None:
        with self.assertRaisesRegex(ValueError, "explicit tag or digest"):
            resolve_image_digest.split_reference("registry.example/repository/image")


if __name__ == "__main__":
    unittest.main()
