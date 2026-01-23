#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_NAME="${SCRIPT_NAME:-benchmark_crypten_multi_host_cpu.py}"

# defaults
WORLD_SIZE="${WORLD_SIZE:-3}"
IFACE_DEFAULT="${IFACE_DEFAULT:-}"   # e.g., ens5 / eth0 (optional)
RUN_ID_DEFAULT="${RUN_ID_DEFAULT:-0}"
LOG_DIR_DEFAULT="${LOG_DIR_DEFAULT:-./logs}"

usage() {
  cat <<EOF
Usage: $0 --rank <rank> --master-addr <addr> --master-port <port> [options] [-- <extra python args>]

Required:
  --rank         Party rank (0..WORLD_SIZE-1)
  --master-addr  MASTER_ADDR
  --master-port  MASTER_PORT

Options:
  --world-size   WORLD_SIZE (default: ${WORLD_SIZE})
  --iface        GLOO_SOCKET_IFNAME (e.g., ens5, eth0) (optional)
  --run-id       Run index for env logging (default: ${RUN_ID_DEFAULT})
  --log-dir      Directory for latency logs (default: ${LOG_DIR_DEFAULT})
  --quiet        Reduce debug prints (recommended)

Examples:
  bash run_node.sh --rank 0 --master-addr 10.0.5.35 --master-port 29500 --run-id 0 --log-dir ./logs --quiet -- --model resnet18 --batch-size 1 --mpc-iters 5 --ensemble-k 2
  bash run_node.sh --rank 1 --master-addr 10.0.5.35 --master-port 29500 --run-id 0 --log-dir ./logs --quiet -- --model resnet18 --batch-size 1 --mpc-iters 5 --ensemble-k 2
  bash run_node.sh --rank 2 --master-addr 10.0.5.35 --master-port 29500 --run-id 0 --log-dir ./logs --quiet -- --model resnet18 --batch-size 1 --mpc-iters 5 --ensemble-k 2
EOF
  exit 1
}

# ========= arguments ==========
RANK=""
MASTER_ADDR=""
MASTER_PORT=""
IFACE="${IFACE_DEFAULT}"
RUN_ID="${RUN_ID_DEFAULT}"
LOG_DIR="${LOG_DIR_DEFAULT}"
QUIET="0"
PY_ARGS=""

# allow "--" to separate script args and python args
SEEN_DASHDASH="0"

while [[ $# -gt 0 ]]; do
  if [[ "${SEEN_DASHDASH}" == "1" ]]; then
    PY_ARGS+="$1 "
    shift
    continue
  fi

  case "$1" in
    --rank)        RANK="$2"; shift 2;;
    --master-addr) MASTER_ADDR="$2"; shift 2;;
    --master-port) MASTER_PORT="$2"; shift 2;;
    --world-size)  WORLD_SIZE="$2"; shift 2;;
    --iface)       IFACE="$2"; shift 2;;
    --run-id)      RUN_ID="$2"; shift 2;;
    --log-dir)     LOG_DIR="$2"; shift 2;;
    --quiet)       QUIET="1"; shift 1;;
    --)            SEEN_DASHDASH="1"; shift 1;;
    *)             # passthrough (old style)
                   PY_ARGS+="$1 "
                   shift;;
  esac
done

# ========= sanity check ==========
if [[ -z "${RANK}" || -z "${MASTER_ADDR}" || -z "${MASTER_PORT}" ]]; then
  usage
fi

echo "=== CrypTen node (env launcher) ==="
echo "SCRIPT        : ${SCRIPT_NAME}"
echo "PYTHON        : ${PYTHON_BIN}"
echo "RANK          : ${RANK}"
echo "WORLD_SIZE    : ${WORLD_SIZE}"
echo "MASTER_ADDR   : ${MASTER_ADDR}"
echo "MASTER_PORT   : ${MASTER_PORT}"
echo "RUN_ID        : ${RUN_ID}"
echo "LOG_DIR       : ${LOG_DIR}"
echo "IFACE         : ${IFACE:-<unset>}"
echo "QUIET         : ${QUIET}"
echo "PYTHON ARGS   : ${PY_ARGS}"
echo

COMMON_ARGS="--world-size ${WORLD_SIZE} --launcher env --run-id ${RUN_ID} --log-dir ${LOG_DIR}"

if [[ "${QUIET}" == "1" ]]; then
  COMMON_ARGS="${COMMON_ARGS} --quiet"
fi

# ===== env vars for CrypTen distributed init =====
export WORLD_SIZE="${WORLD_SIZE}"
export RANK="${RANK}"
export MASTER_ADDR="${MASTER_ADDR}"
export MASTER_PORT="${MASTER_PORT}"
export RENDEZVOUS="env://"
export DISTRIBUTED_BACKEND="gloo"

# optional: network interface binding for gloo
if [[ -n "${IFACE}" ]]; then
  export GLOO_SOCKET_IFNAME="${IFACE}"
fi

# run
${PYTHON_BIN} "${SCRIPT_NAME}" ${COMMON_ARGS} ${PY_ARGS}
