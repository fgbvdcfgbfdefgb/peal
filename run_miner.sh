#!/usr/bin/env bash
#
# run_miner.sh - start the Pearl solo miner on the offline Linux machine.
#
# The miner reads jobs from jobs.txt (you git-pull that file) and appends
# shares to share.txt (you git-push that file).  It never touches the
# network itself.
#
# Usage:
#   ./run_miner.sh                 # mine with the default configuration
#   ./run_miner.sh --batch 128     # smaller batches if RAM is tight
#   ./run_miner.sh --help          # all options
#   ./run_miner.sh --install       # create a venv, install numpy/blake3
#   ./run_miner.sh --install-gpu   # also install CuPy for the GPU engine
#   ./run_miner.sh --gpu-selftest  # validate the GPU engine, then exit
#   ./run_miner.sh --kernel-logic   # kernel algorithms, no GPU needed
#   PEARL_BACKEND=numpy ./run_miner.sh   # force the CPU engine
#
# First time on a new machine:
#   ./run_miner.sh --install
#   ./run_miner.sh --install-gpu   # optional, for A10G / sm86 GPUs

set -euo pipefail
cd "$(dirname "$0")"

MINER_DIR="miner"
VENV_DIR=".venv"
PYTHON="${PYTHON:-python3}"

if [[ "${1:-}" == "--install" ]]; then
    echo "[run_miner] creating virtual environment in ${VENV_DIR}"
    "${PYTHON}" -m venv "${VENV_DIR}"
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
    pip install --upgrade pip
    pip install -r requirements.txt
    echo "[run_miner] installed.  Run ./run_miner.sh to start mining."
    exit 0
fi

if [[ "${1:-}" == "--install-gpu" ]]; then
    [[ -f "${VENV_DIR}/bin/activate" ]] && source "${VENV_DIR}/bin/activate"
    echo "[run_miner] installing CuPy (wheel must match your CUDA version)"
    echo "[run_miner]   CUDA 12.x -> cupy-cuda12x    CUDA 11.x -> cupy-cuda11x"
    if ! pip install "cupy-cuda12x>=13.0"; then
        echo "[run_miner] cupy-cuda12x failed, trying cupy-cuda11x"
        pip install "cupy-cuda11x>=13.0" || {
            echo "[run_miner] CuPy install failed - the miner will use numpy."
            echo "[run_miner] You can install a local wheel instead:"
            echo "[run_miner]   pip install ./cupy-cuda12x-*.whl"
            exit 1
        }
    fi
    echo "[run_miner] CuPy installed.  Run ./run_miner.sh --gpu-selftest."
    exit 0
fi

if [[ "${1:-}" == "--kernel-logic" ]]; then
    [[ -f "${VENV_DIR}/bin/activate" ]] && source "${VENV_DIR}/bin/activate"
    exec "${PYTHON}" "${MINER_DIR}/kernel_logic_test.py" "${@:2}"
fi

if [[ "${1:-}" == "--gpu-selftest" ]]; then
    [[ -f "${VENV_DIR}/bin/activate" ]] && source "${VENV_DIR}/bin/activate"
    exec "${PYTHON}" "${MINER_DIR}/gpu_selftest.py" "${@:2}"
fi

if [[ -d "${VENV_DIR}" ]]; then
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
else
    echo "[run_miner] no virtual environment found; using system python."
    echo "[run_miner] (run ./run_miner.sh --install for an isolated setup)"
fi

# Sanity: numpy must be importable.
if ! "${PYTHON}" -c "import numpy" >/dev/null 2>&1; then
    echo "[run_miner] numpy is missing.  Run: ./run_miner.sh --install"
    exit 1
fi

# Optional self-test (validates against the official Rust module when present).
if [[ "${PEARL_SELFTEST:-0}" == "1" ]]; then
    "${PYTHON}" "${MINER_DIR}/selftest.py"
fi

if [[ "${PEARL_BACKEND:-}" == "numpy" ]]; then
    exec "${PYTHON}" -m pearl_miner --backend numpy "$@"
fi

cd "${MINER_DIR}"
exec "${PYTHON}" -m pearl_miner "$@"
