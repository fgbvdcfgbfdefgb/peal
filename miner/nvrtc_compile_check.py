"""
Compile-check the miner's CUDA kernel source with NVRTC (no GPU required).

This mirrors what cupy.RawModule(code=..., backend="nvrtc") does at runtime,
so a syntax or type error in cuda_kernels.py is caught on a machine that has
no GPU at all.

Usage:  python nvrtc_compile_check.py [--arch compute_86]
"""

import ctypes
import sys

NVRTC_CANDIDATES = (
    "/tmp/nvrtcwheel/lib/nvidia/cuda_nvrtc/lib/libnvrtc.so.12",
    "libnvrtc.so.12",
    "libnvrtc.so",
)

NVRTC_SUCCESS = 0
NVRTC_PROGRAM_CREATION_FAILURE = 1
NVRTC_COMPILE_ERROR = 2
NVRTC_OUT_OF_MEMORY = 5
NVRTC_INVALID_INPUT = 6


def main() -> int:
    arch = "compute_86"
    if "--arch" in sys.argv:
        arch = sys.argv[sys.argv.index("--arch") + 1]

    import os
    here = os.path.dirname(os.path.abspath(__file__))
    # import only the source string, not the module (CuPy is absent here)
    src = open(os.path.join(here, "pearl_miner", "cuda_kernels.py"),
               encoding="utf-8").read()
    start = src.index('_CUDA_SOURCE = r"""') + len('_CUDA_SOURCE = r"""')
    end = src.index('"""', start)
    code = src[start:end]
    print(f"kernel source: {len(code)} characters")

    last_error = None
    for candidate in NVRTC_CANDIDATES:
        try:
            lib = ctypes.CDLL(candidate)
            break
        except OSError as exc:
            last_error = exc
    else:
        print(f"cannot load NVRTC ({last_error}); install it with "
              f"pip install nvidia-cuda-nvrtc-cu12")
        return 2
    lib.nvrtcGetErrorString.restype = ctypes.c_char_p

    prog = ctypes.c_void_p()
    rc = lib.nvrtcCreateProgram(ctypes.byref(prog), code.encode(),
                                b"pearl_miner.cu", 0, None, None)
    if rc != NVRTC_SUCCESS:
        print(f"nvrtcCreateProgram failed: "
              f"{lib.nvrtcGetErrorString(rc).decode()}")
        return 1

    options = [b"-std=c++14", f"--gpu-architecture={arch}".encode()]
    arr = (ctypes.c_char_p * len(options))(*options)
    rc = lib.nvrtcCompileProgram(prog, len(options), arr)

    log_size = ctypes.c_size_t()
    lib.nvrtcGetProgramLogSize(prog, ctypes.byref(log_size))
    log = ctypes.create_string_buffer(log_size.value)
    lib.nvrtcGetProgramLog(prog, log)
    if log_size.value > 1:
        print("--- compile log ---")
        print(log.value.decode())

    if rc != NVRTC_SUCCESS:
        print(f"nvrtcCompileProgram FAILED ({arch}): "
              f"{lib.nvrtcGetErrorString(rc).decode()}")
        return 1

    ptx_size = ctypes.c_size_t()
    lib.nvrtcGetPTXSize(prog, ctypes.byref(ptx_size))
    ptx = ctypes.create_string_buffer(ptx_size.value)
    lib.nvrtcGetPTX(prog, ptx)
    print(f"nvrtcCompileProgram OK for {arch}: {ptx_size.value} bytes of PTX")

    # list the kernels the module exports
    kernels = [line.strip().split()[-1].rstrip(";")
               for line in code.splitlines()
               if line.startswith('extern "C" __global__')]
    print("kernels:", ", ".join(kernels))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
