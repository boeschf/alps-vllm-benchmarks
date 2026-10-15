#!/usr/bin/env bash
#SBATCH --job-name=glm53-serve
#SBATCH --account=csstaff
#SBATCH --partition=normal
#SBATCH --ntasks-per-node=1
##SBATCH --cpus-per-task=288
##SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --output=logs/serve-%j.out
#SBATCH --error=logs/serve-%j.err

# Slurm copies this script into its spool. Never derive paths from $0 or BASH_SOURCE:
# SLURM_SUBMIT_DIR is the directory from which sbatch was invoked. Submit through
# scripts/submit-serve.py so the scenario determines the allocation before this
# script starts.
#
# SCENARIO selects config/scenarios/<name>.json; components compose with '+'.
set -euo pipefail

# Emit one UTC timestamped orchestration line in the Slurm output.
status() {
  printf '[status %s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

EXPECTED_NODES="${EXPECTED_NODES:?submit with scripts/submit-serve.py}"
case "${EXPECTED_NODES}" in
  0|*[!0-9]*) echo "invalid EXPECTED_NODES=${EXPECTED_NODES}" >&2; exit 2 ;;
esac
[[ "${SLURM_JOB_NUM_NODES:?missing Slurm allocation size}" == "${EXPECTED_NODES}" ]] || {
  echo "allocation has ${SLURM_JOB_NUM_NODES} nodes; scenario submission requested ${EXPECTED_NODES}" >&2
  exit 2
}

ROOT="$(realpath "${SLURM_SUBMIT_DIR:?submit this job from the repository root}")"
[[ -f "${ROOT}/config/platforms/cuda.env" && -f "${ROOT}/scripts/serve-worker.sh" ]] || {
  echo "SLURM_SUBMIT_DIR is not the repository root: ${ROOT}" >&2
  exit 2
}
cd "${ROOT}"

# All Python on the host (outside the container) runs through uv: the system
# python3 is 3.6 and cannot run these scripts. uv run --no-project resolves a
# managed interpreter (>= 3.12) without creating or requiring a venv.
command -v uv >/dev/null || {
  echo "uv is missing; install it with: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 2
}
PY=(uv run --no-project --python 3.12 -- python)
"${PY[@]}" --version >/dev/null

PLATFORM="${PLATFORM:-cuda}"
PROFILE="${ROOT}/config/platforms/${PLATFORM}.env"
[[ -f "${PROFILE}" ]] || { echo "unsupported PLATFORM=${PLATFORM}" >&2; exit 2; }
# shellcheck source=/dev/null
source "${PROFILE}"
IMAGE_DIGEST="$("${PY[@]}" "${ROOT}/scripts/resolve-image-digest.py" "${IMAGE_REF}")"

MODEL_SOURCE="${MODEL:-${ROOT}/models/GLM-5.3}"
[[ -d "${MODEL_SOURCE}" ]] || { echo "model directory is missing: ${MODEL_SOURCE}" >&2; exit 2; }
MODEL_SOURCE="$(realpath "${MODEL_SOURCE}")"
MODEL_CONTAINER="${MODEL_CONTAINER:-/models/GLM-5.3}"
[[ "${MODEL_CONTAINER}" = /* && "${MODEL_CONTAINER}" != "/" ]] || {
  echo "MODEL_CONTAINER must be an absolute non-root path: ${MODEL_CONTAINER}" >&2
  exit 2
}
case "${MODEL_SOURCE}" in
  *[,:]*) echo "MODEL cannot contain ':' or ',': ${MODEL_SOURCE}" >&2; exit 2 ;;
esac
case "${MODEL_CONTAINER}" in
  *[,:]*) echo "MODEL_CONTAINER cannot contain ':' or ',': ${MODEL_CONTAINER}" >&2; exit 2 ;;
esac
MODEL="${MODEL_CONTAINER}"

EDF="${EDF:-${ROOT}/${EDF_REL}}"
[[ -f "${EDF}" ]] || { echo "missing EDF: ${EDF}" >&2; exit 2; }

SCENARIO="${SCENARIO:-baseline}"
BENCHER_PROJECT="${BENCHER_PROJECT:-glm-5-3-inference}"
case "${PLATFORM}" in
  cuda) BENCHER_TESTBED="${BENCHER_TESTBED:-alps-gh200}" ;;
  rocm) BENCHER_TESTBED="${BENCHER_TESTBED:-alps-mi300a}" ;;
esac

LOGROOT="${LOGROOT:-${ROOT}/logs}"
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT}/results}"
LOGDIR="${LOGROOT}/run-${SLURM_JOB_ID}"
mkdir -p "${LOGDIR}" "${RESULTS_ROOT}"
SLURM_BATCH_SCRIPT="${LOGDIR}/submitted-batch-script.sbatch"
# BASH_SOURCE points at Slurm's immutable spool copy while this job is running.
# Archive those executed bytes; never use the spool path to locate the repository.
cp -- "${BASH_SOURCE[0]}" "${SLURM_BATCH_SCRIPT}"
status "job ${SLURM_JOB_ID} starting; submit_dir=${ROOT} logdir=${LOGDIR}"
SCENARIO_DIR="${LOGDIR}/scenario"
mkdir -p "${SCENARIO_DIR}"
JIT_CACHE_MODE="${JIT_CACHE_MODE:-readwrite}"
case "${JIT_CACHE_MODE}" in
  off|read|readwrite) ;;
  *) echo "invalid JIT_CACHE_MODE=${JIT_CACHE_MODE}; expected off, read, or readwrite" >&2; exit 2 ;;
esac
JIT_CACHE_ROOT="${JIT_CACHE_ROOT:-${ROOT}/.cache/jit}"
[[ "${JIT_CACHE_ROOT}" = /* ]] || JIT_CACHE_ROOT="${ROOT}/${JIT_CACHE_ROOT}"
JIT_CACHE_STAGING="${JIT_CACHE_ROOT}/staging/${SLURM_JOB_ID}"
JIT_CACHE_TRIGGER="${LOGDIR}/jit-cache.publish.trigger"
JIT_CACHE_LOCK_SECONDS="${JIT_CACHE_LOCK_SECONDS:-300}"
JIT_CACHE_BUNDLE_SECONDS="${JIT_CACHE_BUNDLE_SECONDS:-1800}"
JIT_CACHE_KEEP_GENERATIONS="${JIT_CACHE_KEEP_GENERATIONS:-3}"
mkdir -p "${JIT_CACHE_STAGING}"
rm -f "${JIT_CACHE_TRIGGER}"


mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}" | sort)
HEAD_NODE="${NODES[0]}"
HEAD_IP="$(getent hosts "${HEAD_NODE}" | awk 'NR == 1 {print $1}')"
status "materializing scenario=${SCENARIO} platform=${PLATFORM}"
HEAD_IP="${HEAD_IP:-${HEAD_NODE}}"

GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
API_PORT="${API_PORT:-8000}"
DP_RPC_PORT="${DP_RPC_PORT:-13345}"

"${PY[@]}" "${ROOT}/scripts/scenario.py" \
  --scenario "${SCENARIO}" \
  --platform "${PLATFORM}" \
  --nodes "${SLURM_JOB_NUM_NODES}" \
  --gpus-per-node "${GPUS_PER_NODE}" \
  --model "${MODEL}" \
  --load-format "${LOAD_FORMAT}" \
  --head-ip "${HEAD_IP}" \
  --api-port "${API_PORT}" \
  --dp-rpc-port "${DP_RPC_PORT}" \
  --output-dir "${SCENARIO_DIR}"
status "scenario materialized; config=${SCENARIO_DIR}/scenario.json"

# shellcheck source=/dev/null
source "${SCENARIO_DIR}/scenario.env"

if [[ "${SERVING_MODE}" == "single" ]]; then
  BASE_URL="http://${HEAD_IP}:${API_PORT}"
else
  BASE_URL="http://${HEAD_IP}:${PROXY_PORT}"
fi

export ROOT MODEL MODEL_SOURCE MODEL_CONTAINER EDF PLATFORM IMAGE_REF IMAGE_DIGEST LOGDIR HEAD_IP API_PORT DP_RPC_PORT SCENARIO_ID
export SCENARIO_ENV="${SCENARIO_DIR}/scenario.env"
export SCENARIO_RESOLVED="${SCENARIO_DIR}/scenario.json"
export SCENARIO_VLLM_CONFIG_DIR SCENARIO_ENV_NAMES
export MODEL_REPO="${MODEL_REPO:-zai-org/GLM-5.3}"
export MODEL_REVISION="${MODEL_REVISION:-}"
export GPUS_PER_NODE
export HF_HUB_OFFLINE=1
export JIT_CACHE_MODE JIT_CACHE_ROOT JIT_CACHE_STAGING JIT_CACHE_TRIGGER
export JIT_CACHE_LOCK_SECONDS JIT_CACHE_BUNDLE_SECONDS JIT_CACHE_KEEP_GENERATIONS
export JIT_CACHE_SEED_IMAGE_REFS="${JIT_CACHE_SEED_IMAGE_REFS:-}"

ENVVARS="ROOT,MODEL,MODEL_REPO,MODEL_REVISION,PLATFORM,IMAGE_REF,IMAGE_DIGEST,LOAD_FORMAT,LOGDIR,HEAD_IP,API_PORT,DP_RPC_PORT,SCENARIO_ID,SCENARIO_ENV,SCENARIO_RESOLVED,SCENARIO_VLLM_CONFIG_DIR,SCENARIO_ENV_NAMES,GPUS_PER_NODE,HF_HUB_OFFLINE,JIT_CACHE_MODE,JIT_CACHE_ROOT,JIT_CACHE_STAGING,JIT_CACHE_TRIGGER,JIT_CACHE_LOCK_SECONDS,JIT_CACHE_BUNDLE_SECONDS,JIT_CACHE_KEEP_GENERATIONS,JIT_CACHE_SEED_IMAGE_REFS,${SCENARIO_ENV_NAMES}"

echo "platform=${PLATFORM} image=${IMAGE_REF} digest=${IMAGE_DIGEST}"
echo "nodes=${NODES[*]} head=${HEAD_NODE} (${HEAD_IP}) model=${MODEL_SOURCE}:${MODEL_CONTAINER}:ro"
echo "scenario=${SCENARIO_ID} mode=${SERVING_MODE} TP=${TP_SIZE} all2all=${ALL2ALL}"

WORKER_ARGS=(
  --distribution=block:cyclic
  --cpu-bind=none
  --export="ALL,${ENVVARS}"
  --container-env="${ENVVARS}"
  --network=disable_rdzv_get
  --mpi=pmix
  --environment="${EDF}"
)

SERVE_PIDS=()

# Launch one serving role across its assigned nodes.
run_workers() {
  # role node-list; single mode passes the full allocation (no --nodelist), because
  # allocated node IDs are not contiguous and must never be turned into an A-B range.
  local role="$1" node_list="$2"
  local -a nodelist_args=()
  if [[ -n "${node_list}" ]]; then
    nodelist_args=(--nodelist="${node_list}")
  fi
  srun --overlap \
    "${WORKER_ARGS[@]}" \
    "${nodelist_args[@]}" \
    -u env DEPLOYMENT_ROLE="${role}" bash "${ROOT}/scripts/serve-worker.sh" &
  SERVE_PIDS+=("$!")
  status "serving role launched; role=${role} nodes=${node_list:-all} step_pid=$!"
}

if [[ "${SERVING_MODE}" == "single" ]]; then
  run_workers single ""
else
  PREFILL_COUNT="${PREFILL_NODES:?}"
  run_workers prefill "$(IFS=,; echo "${NODES[*]:0:${PREFILL_COUNT}}")"
  run_workers decode "$(IFS=,; echo "${NODES[*]:${PREFILL_COUNT}}")"
fi

(( ${#SERVE_PIDS[@]} > 0 )) || { echo "no serving steps were launched" >&2; exit 2; }

PROXY_PID=""
if [[ "${SERVING_MODE}" == "disaggregated" ]]; then
  srun --overlap --nodes=1 --ntasks=1 --nodelist="${HEAD_NODE}" --cpus-per-task=8 \
    "${WORKER_ARGS[@]}" \
    -u bash "${ROOT}/scripts/proxy-worker.sh" &
  PROXY_PID=$!
  status "proxy launched; node=${HEAD_NODE} step_pid=${PROXY_PID}"
fi

# Terminate every serving step and the optional proxy.
cleanup() {
  if [[ -n "${PROXY_PID}" ]]; then
    kill "${PROXY_PID}" 2>/dev/null || true
    wait "${PROXY_PID}" 2>/dev/null || true
  fi
  kill "${SERVE_PIDS[@]}" 2>/dev/null || true
  wait "${SERVE_PIDS[@]}" 2>/dev/null || true
}
trap cleanup EXIT TERM INT

# Check that all serving and proxy processes are still running.
serving_processes_alive() {
  local pid
  for pid in "${SERVE_PIDS[@]}"; do
    kill -0 "${pid}" 2>/dev/null || return 1
  done
  [[ -z "${PROXY_PID}" ]] || kill -0 "${PROXY_PID}" 2>/dev/null
}

# Wait for one serving endpoint to become healthy.
wait_for_health() {
  local url="$1" label="$2" ready=0
  status "readiness wait starting; label=${label} url=${url}"
  for attempt in $(seq 1 "${READY_ATTEMPTS:-480}"); do
    if ! serving_processes_alive; then
      echo "a serving or proxy step exited before ${label} readiness" >&2
      return 1
    fi
    if curl --silent --fail --max-time 5 "${url}/health" >/dev/null; then
      ready=1
      break
    fi
    sleep "${READY_INTERVAL_SECONDS:-15}"
    (( attempt % 4 != 0 )) || status \
      "readiness pending; label=${label} url=${url} attempts=${attempt}"
  done
  (( ready == 1 )) || { echo "${label} did not become ready; inspect ${LOGDIR}" >&2; return 1; }
}

if [[ "${SERVING_MODE}" == "disaggregated" ]]; then
  wait_for_health "http://${HEAD_IP}:${PREFILL_API_PORT}" "prefill instance" || exit 1
  wait_for_health "http://${HEAD_IP}:${DECODE_API_PORT}" "decode instance" || exit 1
  wait_for_health "${BASE_URL}" "proxy" || exit 1
else
  wait_for_health "${BASE_URL}" "server" || exit 1
fi

status "server ready; url=${BASE_URL}"
SMOKE_ARGS=(
  --base-url "${BASE_URL}"
  --model "${SERVED_MODEL_NAME:?scenario did not define served-model-name}"
  --output "${LOGDIR}/completion.json"
)
if [[ -n "${SMOKE_INPUT_TOKENS:-}" ]]; then
  SMOKE_ARGS+=(--input-tokens "${SMOKE_INPUT_TOKENS}")
  SMOKE_ARGS+=(--max-tokens "${SMOKE_MAX_TOKENS:-16}")
fi
status "smoke request starting"
"${PY[@]}" "${ROOT}/scripts/smoke-request.py" "${SMOKE_ARGS[@]}"
status "smoke request complete; output=${LOGDIR}/completion.json"
if [[ "${JIT_CACHE_MODE}" == "readwrite" && -z "${BENCHMARK_SUITE:-}" ]]; then
  # Benchmark jobs publish after their full-suite warmup. Jobs without a suite
  # have no later warmup boundary, so capture the smoke-initialized cache now.
  status "JIT cache publication starting after smoke"
  touch "${JIT_CACHE_TRIGGER}"
  JIT_CACHE_EXPECTED_BUNDLES="${SLURM_NNODES}"
  "${PY[@]}" "${ROOT}/scripts/jit-cache.py" publish \
    --staging "${JIT_CACHE_STAGING}" \
    --output "${LOGDIR}/jit-cache-publication.json" \
    --expected-bundles "${JIT_CACHE_EXPECTED_BUNDLES}" \
    --strict
  status "JIT cache publication complete; output=${LOGDIR}/jit-cache-publication.json"
fi

if [[ -n "${BENCHMARK_SUITE:-}" ]]; then
  SUITE="${BENCHMARK_SUITE}"
  [[ "${SUITE}" = /* ]] || SUITE="${ROOT}/${SUITE}"
  [[ -f "${SUITE}" ]] || { echo "benchmark suite not found: ${SUITE}" >&2; exit 2; }
  status "benchmark suite starting; suite=${SUITE}"
  srun --overlap --nodes=1 --ntasks=1 --nodelist="${HEAD_NODE}" --cpus-per-task=72 \
    "${WORKER_ARGS[@]}" \
    -u python3 "${ROOT}/scripts/benchmark.py" \
      --suite "${SUITE}" \
      --base-url "${BASE_URL}" \
      --model "${MODEL}" \
      --served-model-name "${SERVED_MODEL_NAME:?scenario did not define served-model-name}" \
      --output-root "${RESULTS_ROOT}" \
      --platform "${PLATFORM}" \
      --image-ref "${IMAGE_REF}" \
      --image-digest "${IMAGE_DIGEST}" \
      --scenario-config "${SCENARIO_RESOLVED}" \
      --edf "${EDF}" \
      --logdir "${LOGDIR}" \
      --batch-script "${SLURM_BATCH_SCRIPT}" \
      --bencher-project "${BENCHER_PROJECT}" \
      --bencher-testbed "${BENCHER_TESTBED}"
  status "benchmark suite complete"
  echo "benchmark complete"
elif [[ "${SMOKE_ONLY:-0}" == "1" ]]; then
  echo "smoke-only run complete"
else
  echo "server remains active until the job is cancelled"
  wait "${SERVE_PIDS[@]}"
fi
