#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
command -v hf >/dev/null || {
  echo "hf is missing; install it with: uv tool install 'huggingface_hub[cli]'" >&2
  exit 1
}

REPO="${MODEL_REPO:-zai-org/GLM-5.3}"
REVISION="${MODEL_REVISION:-aca966e4e02791568aa6a4ced368624b3d897f42}"
MODEL_DIR="${ROOT}/models/GLM-5.3"
WORKERS="${HF_MAX_WORKERS:-16}"
[[ ! -L "${MODEL_DIR}" ]] || {
  echo "refusing to download through model symlink: ${MODEL_DIR}" >&2
  exit 2
}

export HF_XET_HIGH_PERFORMANCE=1
mkdir -p "${MODEL_DIR}"

echo "Downloading ${REPO}@${REVISION} to ${MODEL_DIR}"
hf download "${REPO}" \
  --revision "${REVISION}" \
  --local-dir "${MODEL_DIR}" \
  --max-workers "${WORKERS}" \
  --exclude '.eval_results/*'

echo "Model ready: ${MODEL_DIR}"
