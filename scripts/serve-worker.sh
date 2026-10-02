#!/usr/bin/env bash
# Per-node worker inside the container. The scenario resolver decides what this node
# runs: single-mode serving (SCENARIO_VLLM_CONFIG), disaggregated prefill or decode
# (PREFILL_SCENARIO_VLLM_CONFIG / DECODE_SCENARIO_VLLM_CONFIG with DEPLOYMENT_ROLE),
# or a mooncake master (MOONCAKE_STORE_ENABLED=1 on the head node). This script only
# launches what the resolver produced.
set -euo pipefail

: "${MODEL:?MODEL must be an absolute checkpoint path}"
: "${HEAD_IP:?HEAD_IP is required}"
: "${LOGDIR:?LOGDIR is required}"
: "${SCENARIO_ENV:?SCENARIO_ENV must be the generated scenario.env}"
: "${ROOT:?ROOT must be the repository root}"
: "${SERVING_MODE:?SERVING_MODE must be single or disaggregated}"

# shellcheck source=/dev/null
source "${SCENARIO_ENV}"

# The EDF bind-mounts the selected host checkpoint read-only at MODEL. A node
# whose mount is missing (observed once as a transient NFS/pyxis failure) must
# fail loudly here instead of sending vLLM into an offline HF hub lookup that
# produces a confusing LocalEntryNotFoundError. Retry briefly because the NFS
# source can take a moment to appear.
for _ in {1..30}; do
  [[ -f "${MODEL}/config.json" ]] && break
  sleep 2
done
[[ -f "${MODEL}/config.json" ]] || {
  echo "model directory is not visible in this container: ${MODEL}" >&2
  echo "the bind mount from the launcher is missing on node $(hostname)" >&2
  exit 2
}
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
TP_SIZE="${TP_SIZE:-2}"
(( GPUS_PER_NODE % TP_SIZE == 0 )) || {
  echo "GPUS_PER_NODE=${GPUS_PER_NODE} is not divisible by TP_SIZE=${TP_SIZE}" >&2
  exit 2
}

# One Slurm task runs per serving node. PROCID is step-local, so disaggregated
# prefill and decode steps each get contiguous data-parallel ranks from zero.
NODE_RANK="${SLURM_PROCID:?SLURM_PROCID is required}"
DP_LOCAL=$((GPUS_PER_NODE / TP_SIZE))
API_PORT="${API_PORT:-8000}"
DP_RPC_PORT="${DP_RPC_PORT:-13345}"

role="${DEPLOYMENT_ROLE:-single}"
if [[ "${SERVING_MODE}" == "single" ]]; then
  [[ "${role}" == "single" ]] || { echo "SERVING_MODE=single but DEPLOYMENT_ROLE=${role}" >&2; exit 2; }
  : "${SCENARIO_VLLM_CONFIG:?SCENARIO_VLLM_CONFIG is required for single-mode serving}"
  CONFIG="${SCENARIO_VLLM_CONFIG}"
  DP_TOTAL=$((SLURM_JOB_NUM_NODES * DP_LOCAL))
else
  if [[ "${role}" != "prefill" && "${role}" != "decode" ]]; then
    echo "SERVING_MODE=disaggregated requires DEPLOYMENT_ROLE=prefill|decode, got '${role}'" >&2
    exit 2
  fi
  if [[ "${role}" == "prefill" ]]; then
    : "${PREFILL_SCENARIO_VLLM_CONFIG:?PREFILL_SCENARIO_VLLM_CONFIG is required}"
    CONFIG="${PREFILL_SCENARIO_VLLM_CONFIG}"
    API_PORT="${PREFILL_API_PORT:?PREFILL_API_PORT is required}"
    DP_RPC_PORT="${PREFILL_RPC_PORT:?PREFILL_RPC_PORT is required}"
    DP_TOTAL=$((PREFILL_NODES * DP_LOCAL))
  else
    : "${DECODE_SCENARIO_VLLM_CONFIG:?DECODE_SCENARIO_VLLM_CONFIG is required}"
    CONFIG="${DECODE_SCENARIO_VLLM_CONFIG}"
    API_PORT="${DECODE_API_PORT:?DECODE_API_PORT is required}"
    DP_RPC_PORT="${DECODE_RPC_PORT:?DECODE_RPC_PORT is required}"
    DP_TOTAL=$((DECODE_NODES * DP_LOCAL))
  fi
fi
START_RANK=$((NODE_RANK * DP_LOCAL))
CACHE_STATUS="${LOGDIR}/jit-cache-${role}-node${NODE_RANK}.json"
python3 "${ROOT}/scripts/jit-cache.py" hydrate --status "${CACHE_STATUS}"
export JIT_CACHE_STATUS_FILE="${CACHE_STATUS}"

# Every node captures its container-local /tmp cache after health and smoke
# checks. Publishing merges all node bundles into the next shared generation.
if [[ "${JIT_CACHE_MODE:-readwrite}" == "readwrite" ]]; then
  python3 "${ROOT}/scripts/jit-cache.py" agent \
    --status "${CACHE_STATUS}" \
    --trigger "${JIT_CACHE_TRIGGER:?JIT_CACHE_TRIGGER is required}" \
    --staging "${JIT_CACHE_STAGING:?JIT_CACHE_STAGING is required}" \
    --role "${role}" \
    > "${LOGDIR}/jit-cache-agent-${role}-node${NODE_RANK}.log" 2>&1 &
  echo "[node ${NODE_RANK}] JIT cache agent started for role=${role} (pid $!)"
fi

printf '[node %s] host=%s platform=%s role=%s scenario=%s TP=%s DP=%s local-DP=%s EP=%s\n' \
  "${NODE_RANK}" "$(hostname)" "${PLATFORM:-unknown}" "${role}" "${SCENARIO_ID:-unknown}" \
  "${TP_SIZE}" "${DP_TOTAL}" "${DP_LOCAL}" "$((DP_TOTAL * TP_SIZE))"
CONFIG_ARGS=(--config "${CONFIG}")
python -c 'import torch, vllm; print(f"torch={torch.__version__} cuda={torch.version.cuda} hip={torch.version.hip} vllm={vllm.__version__}")'

# The runtime-fingerprint capture must match the interpreter that vLLM runs on
python3 "${ROOT}/scripts/runtime-fingerprint.py" \
  --output "${LOGDIR}/runtime-${role}-node${NODE_RANK}.json" \
  --command-output "${LOGDIR}/server-command-${role}-node${NODE_RANK}.json" \
  -- vllm serve "${CONFIG_ARGS[@]}"

COMMON=(
  "${CONFIG_ARGS[@]}"
  --tensor-parallel-size "${TP_SIZE}"
  --data-parallel-size "${DP_TOTAL}"
  --data-parallel-size-local "${DP_LOCAL}"
  --data-parallel-address "${HEAD_IP}"
  --data-parallel-rpc-port "${DP_RPC_PORT}"
)

if [[ "${MOONCAKE_STORE_ENABLED:-0}" == "1" && "${NODE_RANK}" == "0" ]]; then
  command -v mooncake_master >/dev/null || {
    echo "MOONCAKE_STORE_ENABLED=1 but mooncake_master is not in the image" >&2
    exit 2
  }
  MOONCAKE_MASTER_PORT="${MOONCAKE_MASTER_PORT:?MOONCAKE_MASTER_PORT is required}"
  mooncake_master --port "${MOONCAKE_MASTER_PORT}" \
    > "${LOGDIR}/mooncake-master.log" 2>&1 &
  echo "[node 0] mooncake_master started on ${MOONCAKE_MASTER_PORT} (pid $!)"
fi

if (( NODE_RANK == 0 )); then
  exec vllm serve "${COMMON[@]}" \
    --host 0.0.0.0 --port "${API_PORT}" \
    2>&1 | tee "${LOGDIR}/serve-${role}-node0.log"
else
  exec vllm serve "${COMMON[@]}" --headless \
    --data-parallel-start-rank "${START_RANK}" \
    2>&1 | tee "${LOGDIR}/serve-${role}-node${NODE_RANK}.log"
fi
