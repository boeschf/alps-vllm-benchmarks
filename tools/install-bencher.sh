#!/usr/bin/env bash
# Install the Bencher CLI into an architecture-specific directory under tools/bin
# (ignored by Git): tools/bin/<arch>/bencher, where arch is either aarch64 or
# x86_64.
#
# Downloads the official pinned release directly from bencher.dev and verifies it
# against the checksum recorded in this script, so the binary used for benchmark
# publication is pinned the same way as everything else.
#
# Usage:
#   tools/install-bencher.sh               # install/upgrade to the pinned version
#   tools/bin/$(uname -m)/bencher --version
#
# Environment:
#   BENCHER_VERSION  override the pinned version; the expected checksum must then
#                    also be updated from https://bencher.dev/download/<version>/
#                    <artifact>.sha256 when pinning a new version
#
# Notes:
# - bencher.dev publishes bare binaries for Linux, not archives.
# - The install is local to the repository, so $HOME is never modified and the
#   PATH is never touched by an installer.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BENCHER_VERSION="${BENCHER_VERSION:-0.6.13}"

case "$(uname -s)/$(uname -m)" in
  Linux/x86_64)
    ARCH_DIR="x86_64"
    ARTIFACT="bencher-v${BENCHER_VERSION}-linux-x86-64"
    EXPECTED_SHA256="41b172864a1cd347f75a68688100688453b02f28e1d1b831b326931b895b4b3d"
    ;;
  Linux/aarch64)
    ARCH_DIR="aarch64"
    ARTIFACT="bencher-v${BENCHER_VERSION}-linux-arm-64"
    EXPECTED_SHA256="eefe56680908ad1772ec19a7f7e35cc64039f27be05cc5891e5cb5eca24b824c"
    ;;
  *)
    echo "unsupported platform: $(uname -s)/$(uname -m)" >&2
    exit 2
    ;;
esac

INSTALL_DIR="${ROOT}/tools/bin/${ARCH_DIR}"

command -v curl >/dev/null || { echo "curl is required" >&2; exit 2; }
command -v sha256sum >/dev/null || { echo "sha256sum is required" >&2; exit 2; }

URL="https://bencher.dev/download/${BENCHER_VERSION}/${ARTIFACT}"
TEMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TEMP_DIR}"' EXIT
TARGET="${TEMP_DIR}/${ARTIFACT}"

echo "downloading ${URL}"
curl --fail --silent --show-error --location --max-time 300 --output "${TARGET}" "${URL}"

ACTUAL_SHA256="$(sha256sum "${TARGET}" | awk '{print $1}')"
if [[ "${ACTUAL_SHA256}" != "${EXPECTED_SHA256}" ]]; then
  {
    echo "checksum mismatch for ${ARTIFACT}"
    echo "  expected ${EXPECTED_SHA256}"
    echo "  actual   ${ACTUAL_SHA256}"
    if [[ -z "${BENCHER_VERSION:-}" ]]; then
      echo "if bencher ${BENCHER_VERSION} was re-released, verify the new checksum at"
      echo "  ${URL}.sha256"
      echo "and update tools/install-bencher.sh before installing it"
    fi
  } >&2
  exit 1
fi

mkdir -p "${INSTALL_DIR}"
install -m 0755 "${TARGET}" "${INSTALL_DIR}/bencher"
"${INSTALL_DIR}/bencher" --version
echo "installed: ${INSTALL_DIR}/bencher"
echo "usage: BENCHER_API_KEY=... ${INSTALL_DIR}/bencher run --help"
