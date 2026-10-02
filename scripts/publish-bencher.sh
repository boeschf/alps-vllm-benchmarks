#!/usr/bin/env bash
set -euo pipefail

# Upload an existing result to Bencher. The run context (project, testbed, spec, hash,
# branch) comes from the run's own manifest.json, so the upload is tied to what was
# actually measured instead of operator-typed values. BENCHER_API_KEY (and optionally
# BENCHER_HOST) must be provided by the operator; secrets never enter the repository.
#
#   BENCHER_API_KEY=... scripts/publish-bencher.sh results/<run>/
#
# BENCHER_PROJECT, BENCHER_TESTBED, BENCHER_BRANCH, or BENCHER_SPEC override the
# manifest values deliberately, for example to redirect an upload during a Bencher
# migration. The manifest is not modified.

# Print the required invocation and terminate with a usage error.
usage() {
  echo "usage: BENCHER_API_KEY=... $0 results/<run-id>/" >&2
  exit 2
}

(( $# == 1 )) || usage
RUN_DIR="$1"
[[ -d "${RUN_DIR}" ]] || { echo "missing run directory: ${RUN_DIR}" >&2; exit 2; }
MANIFEST="${RUN_DIR}/manifest.json"
BENCHER_JSON="${RUN_DIR}/bencher.json"
[[ -f "${MANIFEST}" && -f "${BENCHER_JSON}" ]] || {
  echo "run directory lacks manifest.json or bencher.json: ${RUN_DIR}" >&2
  exit 2
}
BENCHER_BIN="${BENCHER_BIN:-}"
if [[ -z "${BENCHER_BIN}" ]]; then
  if command -v bencher >/dev/null; then
    BENCHER_BIN="$(command -v bencher)"
  else
    LOCAL_BENCHER="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/tools/bin/$(uname -m)/bencher"
    if [[ -x "${LOCAL_BENCHER}" ]]; then
      BENCHER_BIN="${LOCAL_BENCHER}"
      echo "using repository-local ${BENCHER_BIN}"
    fi
  fi
fi
[[ -n "${BENCHER_BIN}" ]] || {
  echo "bencher is missing; install it with tools/install-bencher.sh" >&2
  echo "(BENCHER_BIN can point at an explicit binary)" >&2
  exit 1
}
command -v jq >/dev/null || { echo "jq is missing; it is required to read the manifest" >&2; exit 2; }
: "${BENCHER_API_KEY:?set BENCHER_API_KEY (the CLI also honors BENCHER_HOST)}"

# Read one value from the completed run manifest.
read_manifest() {
  jq -r "$1" "${MANIFEST}"
}

STATUS="$(read_manifest '.status')"
[[ "${STATUS}" == "complete" ]] || {
  echo "refusing to publish a run with status '${STATUS}'" >&2
  exit 1
}

PROJECT="${BENCHER_PROJECT:-$(read_manifest '.publication.project')}"
TESTBED="${BENCHER_TESTBED:-$(read_manifest '.publication.testbed')}"
SPEC="${BENCHER_SPEC:-$(read_manifest '.publication.spec')}"
BRANCH="${BENCHER_BRANCH:-$(read_manifest '.publication.branch')}"
PUBLISHABLE="$(read_manifest '.publication.publishable')"

[[ -n "${PROJECT}" && "${PROJECT}" != "null" ]] || { echo "manifest has no Bencher project" >&2; exit 1; }
[[ -n "${TESTBED}" && "${TESTBED}" != "null" ]] || { echo "manifest has no Bencher testbed" >&2; exit 1; }
[[ -n "${SPEC}" && "${SPEC}" != "null" ]] || { echo "manifest has no Bencher spec" >&2; exit 1; }
[[ -n "${BRANCH}" && "${BRANCH}" != "null" ]] || { echo "manifest has no Bencher branch" >&2; exit 1; }

ARGS=(
  run
  --project "${PROJECT}"
  --branch "${BRANCH}"
  --testbed "${TESTBED}"
  --spec "${SPEC}"
  --adapter json
  --file "${BENCHER_JSON}"
)

if [[ "${PUBLISHABLE}" == "true" ]]; then
  HASH="${BENCHER_HASH:-$(read_manifest '.publication.hash')}"
  [[ -n "${HASH}" && "${HASH}" != "null" ]] || { echo "manifest has no Git hash to publish under" >&2; exit 1; }
  ARGS+=(--hash "${HASH}")
else
  [[ -n "${BENCHER_HASH:-}" ]] || {
    echo "the manifest records a dirty worktree; set BENCHER_HASH to publish anyway" >&2
    read_manifest '.publication.reason' >&2 || true
    exit 1
  }
  ARGS+=(--hash "${BENCHER_HASH}")
fi

echo "publishing ${RUN_DIR}"
echo "  project=${PROJECT} testbed=${TESTBED} spec=${SPEC} branch=${BRANCH}"
exec "${BENCHER_BIN}" "${ARGS[@]}"
