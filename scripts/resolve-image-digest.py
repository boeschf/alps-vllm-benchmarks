#!/usr/bin/env -S uv run --no-project --python 3.12 -- python
"""Resolve the immutable manifest digest for an OCI image reference."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable
from typing import Any
from urllib.error import HTTPError
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit
from urllib.request import Request, parse_http_list, parse_keqv_list, urlopen

MANIFEST_ACCEPT = ", ".join(
    (
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    )
)
DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")


def split_reference(reference: str) -> tuple[str, str, str]:
    """Split an explicit registry reference into registry, repository, and tag/digest."""
    if "/" not in reference:
        raise ValueError(f"image reference has no registry: {reference!r}")
    registry, repository_reference = reference.split("/", 1)
    if "@" in repository_reference:
        repository, manifest_reference = repository_reference.rsplit("@", 1)
    else:
        repository, separator, manifest_reference = repository_reference.rpartition(":")
        if not separator or "/" in manifest_reference:
            raise ValueError(f"image reference has no explicit tag or digest: {reference!r}")
    if not registry or not repository or not manifest_reference:
        raise ValueError(f"invalid image reference: {reference!r}")
    return registry, repository, manifest_reference


def manifest_url(registry: str, repository: str, manifest_reference: str) -> str:
    """Build the registry manifest endpoint, including JFrog virtual repositories."""
    encoded_reference = quote(manifest_reference, safe=":")
    if registry == "jfrog.svc.cscs.ch":
        virtual_repository, separator, image_repository = repository.partition("/")
        if not separator:
            raise ValueError("JFrog image reference has no virtual repository")
        return (
            f"https://{registry}/artifactory/api/docker/"
            f"{quote(virtual_repository, safe='')}/v2/"
            f"{quote(image_repository, safe='/')}/manifests/{encoded_reference}"
        )
    return (
        f"https://{registry}/v2/{quote(repository, safe='/')}/manifests/"
        f"{encoded_reference}"
    )


def bearer_parameters(challenge: str) -> dict[str, str]:
    """Parse a Docker Registry bearer challenge."""
    scheme, separator, parameters = challenge.partition(" ")
    if not separator or scheme.lower() != "bearer":
        raise RuntimeError(f"unsupported registry authentication challenge: {challenge!r}")
    return parse_keqv_list(parse_http_list(parameters))


def token_url(parameters: dict[str, str]) -> str:
    """Build the anonymous bearer-token request from challenge parameters."""
    realm = parameters.get("realm")
    if not realm:
        raise RuntimeError("registry authentication challenge has no realm")
    parts = urlsplit(realm)
    query = parse_qsl(parts.query, keep_blank_values=True)
    query.extend(
        (name, parameters[name])
        for name in ("service", "scope")
        if parameters.get(name)
    )
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def resolve_image_digest(
    reference: str,
    opener: Callable[[Request], Any] = urlopen,
) -> str:
    """Resolve a tag or digest reference and return its registry manifest digest."""
    registry, repository, manifest_reference = split_reference(reference)
    url = manifest_url(registry, repository, manifest_reference)
    headers = {"Accept": MANIFEST_ACCEPT, "User-Agent": "glm-5.3-inference/1"}
    request = Request(url, headers=headers, method="HEAD")
    try:
        response = opener(request)
    except HTTPError as error:
        if error.code != 401:
            raise RuntimeError(
                f"registry manifest request failed for {reference}: HTTP {error.code}"
            ) from error
        challenge = error.headers.get("WWW-Authenticate", "")
        parameters = bearer_parameters(challenge)
        token_request = Request(
            token_url(parameters),
            headers={"User-Agent": headers["User-Agent"]},
        )
        with opener(token_request) as token_response:
            token_document = json.loads(token_response.read())
        token = token_document.get("token") or token_document.get("access_token")
        if not token:
            raise RuntimeError("registry token response has no token")
        request.add_header("Authorization", f"Bearer {token}")
        try:
            response = opener(request)
        except HTTPError as authenticated_error:
            raise RuntimeError(
                f"authenticated registry manifest request failed for {reference}: "
                f"HTTP {authenticated_error.code}"
            ) from authenticated_error

    with response:
        digest = response.headers.get("Docker-Content-Digest", "")
    if not DIGEST_PATTERN.fullmatch(digest):
        raise RuntimeError(f"registry returned an invalid manifest digest: {digest!r}")
    return digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    args = parser.parse_args()
    print(resolve_image_digest(args.reference))


if __name__ == "__main__":
    main()
