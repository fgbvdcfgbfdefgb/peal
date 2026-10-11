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
#   ./run_miner.sh --install-gpu   # install CuPy for the GPU engine
#   ./run_miner.sh --gpu-info      # show CuPy/GPU status, then exit
#   ./run_miner.sh --gpu-selftest  # validate the GPU engine, then exit
#   ./run_miner.sh --kernel-logic  # kernel algorithms, no GPU needed
#   PEARL_BACKEND=numpy ./run_miner.sh   # force the CPU engine
#
# If the file lost its execute bit (some ZIP downloads do that):
#   chmod +x run_miner.sh
#   ... or simply run it as:  bash run_miner.sh <same arguments>
#
# First time on a new machine:
#   ./run_miner.sh --install
#   ./run_miner.sh --install-gpu
#   ./run_miner.sh --gpu-info      # confirm the 4 A10Gs are visible
#   ./run_miner.sh --gpu-selftest  # GPU vs CPU, once per machine
#   ./run_miner.sh --backend cuda --devices 0,1,2,3

set -euo pipefail
cd "$(dirname "$0")"

MINER_DIR="miner"
VENV_DIR=".venv"
PYTHON="${PYTHON:-python3}"

# Enter the virtual environment when it exists.  Written as an if-block on
# purpose: a bare "[[ -f ... ]] && source ..." would abort the script under
# `set -e` when the venv is missing, which silently skipped the CuPy install.
enter_venv() {
    if [[ -d "${VENV_DIR}" ]]; then
        # shellcheck disable=SC1091
        source "${VENV_DIR}/bin/activate"
        return 0
    fi
    return 1
}

if [[ "${1:-}" == "--install" ]]; then
    echo "[run_miner] creating virtual environment in ${VENV_DIR}"
    "${PYTHON}" -m venv "${VENV_DIR}"
    enter_venv
    pip install --upgrade pip
    if ! pip install -r requirements.txt; then
        echo "[run_miner] WARNING: could not install every requirement"
        echo "[run_miner] (see the pip errors above).  Installing numpy only -"
        echo "[run_miner] the miner falls back to its own BLAKE3 port, so it"
        echo "[run_miner] still works, just a little slower."
        pip install "numpy>=1.24"
    fi
    echo "[run_miner] installed.  Run ./run_miner.sh to start mining."
    echo "[run_miner] for the GPU engine: ./run_miner.sh --install-gpu"
    exit 0
fi

if [[ "${1:-}" == "--install-gpu" ]]; then
    if ! enter_venv; then
        echo "[run_miner] no virtual environment found - creating one first"
        "${PYTHON}" -m venv "${VENV_DIR}"
        enter_venv
    fi
    echo "[run_miner] installing CuPy (the wheel must match your CUDA version)"
    echo "[run_miner]   CUDA 12.x -> cupy-cuda12x    CUDA 11.x -> cupy-cuda11x"
    if pip install "cupy-cuda12x>=13.0"; then
        echo "[run_miner] CuPy (CUDA 12) installed."
    elif pip install "cupy-cuda11x>=13.0"; then
        echo "[run_miner] CuPy (CUDA 11) installed."
    else
        echo "[run_miner] CuPy install FAILED - the miner will use numpy."
        echo "[run_miner] Check the error above.  Common causes:"
        echo "[run_miner]   * no NVIDIA driver           -> nvidia-smi"
        echo "[run_miner]   * CUDA older than 11         -> upgrade the driver"
        echo "[run_miner]   * offline machine             -> download the wheel"
        echo "[run_miner]     elsewhere and:  pip install ./cupy-cuda12x-*.whl"
        exit 1
    fi
    echo "[run_miner] now run:  ./run_miner.sh --gpu-info"
    exit 0
fi

if [[ "${1:-}" == "--gpu-info" ]]; then
    enter_venv || true
    exec "${PYTHON}" - "${MINER_DIR}" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from pearl_miner.cuda_kernels import gpu_status
print(gpu_status())
PY
fi

if [[ "${1:-}" == "--kernel-logic" ]]; then
    enter_venv || true
    exec "${PYTHON}" "${MINER_DIR}/kernel_logic_test.py" "${@:2}"
fi

if [[ "${1:-}" == "--gpu-selftest" ]]; then
    enter_venv || true
    exec "${PYTHON}" "${MINER_DIR}/gpu_selftest.py" "${@:2}"
fi

if ! enter_venv; then
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
