#!/usr/bin/env bash
# Runs on the head node inside the container for SERVING_MODE=disaggregated.
# Forwards the OpenAI endpoints through the official vLLM Mooncake disaggregation
# proxy so that prefill and decode instances cooperate per request.
set -euo pipefail

: "${SCENARIO_ENV:?SCENARIO_ENV must be the generated scenario.env}"
: "${LOGDIR:?LOGDIR is required}"
: "${HEAD_IP:?HEAD_IP is required}"

# shellcheck source=/dev/null
source "${SCENARIO_ENV}"

: "${PREFILL_API_PORT:?PREFILL_API_PORT is required}"
: "${DECODE_API_PORT:?DECODE_API_PORT is required}"
: "${PREFILL_BOOTSTRAP_PORT:?PREFILL_BOOTSTRAP_PORT is required}"
: "${PROXY_PORT:?PROXY_PORT is required}"

exec python3 /opt/vllm-proxy/mooncake_connector_proxy.py \
  --prefill "http://${HEAD_IP}:${PREFILL_API_PORT}" "${PREFILL_BOOTSTRAP_PORT}" \
  --decode "http://${HEAD_IP}:${DECODE_API_PORT}" \
  --host 0.0.0.0 --port "${PROXY_PORT}" \
  2>&1 | tee "${LOGDIR}/proxy.log"
