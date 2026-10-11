"""
GPU mining engine (CuPy + NVRTC kernels, no compiler toolchain needed).

Mirrors :class:`pearl_miner.engine_np.NumpyEngine`:

    engine = GpuEngine(m, n, k, rank, rows_pattern, cols_pattern,
                       batch_size=256, device_id=0)
    engine.set_job(header, cert_version)
    hits = engine.run_batch()

Each batch runs the whole pipeline on the GPU:

  1. random int8 matrices A (m, k) and B^T (n, k) in [-64, 64]
  2. keyed BLAKE3 Merkle roots of the padded matrices (commitments)
  3. noise-seed derivation (Legacy or Salted, matching the cert version)
  4. noise matrices (uniform rows + permutation pairs)
  5. noised GEMM in rank-sized k-chunks with the jackpot fold
  6. jackpot hashing and the difficulty check; hits are copied back

Supported configurations (checked in :meth:`supports`): dense patterns
(``[0, 1, ..., size-1]``), m and n multiples of 32, k a multiple of rank,
rank a multiple of 32, and <= 1536 chunks per matrix.  Anything else falls
back to the numpy engine.

Because the kernels cannot be exercised on a machine without a GPU, always
run ``gpu_selftest.py`` once on the mining machine: it compares the GPU
engine against the numpy reference trial by trial.
"""

from __future__ import annotations

import numpy as np

from .core import (
    MiningConfiguration, IncompleteBlockHeader, compute_job_key,
    extract_difficulty_bound, derivation_for_cert_version,
    SEED_LABEL_A, SEED_LABEL_B, SEED_SALT_A, SEED_SALT_B,
)
from .patterns import PeriodicPattern, threads_partition

TILE_M = 32
TILE_N = 32
NTHREADS = 256
MAX_TREE_CHUNKS = 1536          # shared-memory limit of blake3_tree_root
MAX_GRID_DIM = 65535            # CUDA gridDim.y / gridDim.z limit
SMEM_STATIC_LIMIT = 48 * 1024   # bytes; above this an opt-in is required

# CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
_CU_FUNC_ATTR_MAX_DYN_SMEM = 8

# flags for a single 64-byte block that is the whole (root) chunk
_FLAGS_UNKEYED_ROOT = 1 | 2 | 8      # CHUNK_START | CHUNK_END | ROOT
_FLAGS_KEYED_ROOT = 1 | 2 | 8 | 16   # + KEYED_HASH

_IV_HEX = ("67e6096a85ae67bb72ef6e3c3af54fa5"
           "7f520e518c68059babd9831f19cde05b")


def _u32_words(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype="<u4").copy()


class GpuEngine:
    """Vectorised Pearl PoW search on one GPU."""

    def __init__(self, m: int, n: int, k: int, rank: int,
                 rows_pattern, cols_pattern, batch_size: int = 256,
                 device_id: int = 0, seed: int | None = None):
        self.m, self.n, self.k, self.rank = m, n, k, rank
        self.rows_pattern = rows_pattern
        self.cols_pattern = cols_pattern
        self.config = MiningConfiguration(k, rank, rows_pattern, cols_pattern)
        self.config.sanity_check(m, n)
        self.device_id = int(device_id)
        self.batch_size = int(batch_size)

        ok, why = self.supports(m, n, k, rank, rows_pattern, cols_pattern)
        if not ok:
            raise ValueError(f"GPU engine cannot mine this configuration: "
                             f"{why}")

        import cupy as cp
        self.cp = cp
        with cp.cuda.Device(self.device_id):
            from .cuda_kernels import kernels_for_device
            self.kernels = kernels_for_device(self.device_id)
            self._alloc_buffers()

        self.rng = np.random.default_rng(seed)
        self._base_seed = int(self.rng.integers(0, 2 ** 62))
        self._batch_index = 0
        self._job_key = None
        self._bound = None
        self._derivation = None

    # -- static capability check -------------------------------------------

    @staticmethod
    def supports(m, n, k, rank, rows_pattern, cols_pattern):
        """``(ok, reason)`` for mining this configuration on the GPU."""
        if rows_pattern.to_list() != list(range(rows_pattern.size())):
            return False, "rows pattern must be dense [0..size-1]"
        if cols_pattern.to_list() != list(range(cols_pattern.size())):
            return False, "cols pattern must be dense [0..size-1]"
        if m % TILE_M or n % TILE_N:
            return False, f"m and n must be multiples of {TILE_M}"
        if TILE_M % rows_pattern.size() or TILE_N % cols_pattern.size():
            return False, "tile dims must divide 32"
        if k % rank:
            return False, "k must be a multiple of rank"
        if rank % 32:
            return False, "rank must be a multiple of 32"
        if (m * k) % 1024 or (n * k) % 1024:
            return False, "matrix bytes must be a multiple of 1024"
        if (m * k) // 1024 > MAX_TREE_CHUNKS:
            return False, f"A has more than {MAX_TREE_CHUNKS} chunks"
        if (n * k) // 1024 > MAX_TREE_CHUNKS:
            return False, f"B^T has more than {MAX_TREE_CHUNKS} chunks"
        return True, ""

    # -- buffers ------------------------------------------------------------

    def _alloc_buffers(self):
        cp = self.cp
        m, n, k, B = self.m, self.n, self.k, self.batch_size
        self.d_a = cp.zeros((B, m, k), dtype=cp.int8)
        self.d_bt = cp.zeros((B, n, k), dtype=cp.int8)
        self.d_na = cp.zeros((B, m, k), dtype=cp.int8)
        self.d_nb = cp.zeros((B, n, k), dtype=cp.int8)
        self.d_roots_a = cp.zeros((B, 8), dtype=cp.uint32)
        self.d_roots_b = cp.zeros((B, 8), dtype=cp.uint32)
        self.d_a_seed = cp.zeros((B, 8), dtype=cp.uint32)
        self.d_b_seed = cp.zeros((B, 8), dtype=cp.uint32)
        self.d_e_al = cp.zeros((B, m, self.rank), dtype=cp.int8)
        self.d_e_br = cp.zeros((B, n, self.rank), dtype=cp.int8)
        self.d_e_ar = cp.zeros((B, k, 2), dtype=cp.int32)
        self.d_e_bl = cp.zeros((B, k, 2), dtype=cp.int32)
        self.max_hits = 4096
        self.d_hit_count = cp.zeros(1, dtype=cp.int32)
        self.d_hit_trial = cp.zeros(self.max_hits, dtype=cp.int32)
        self.d_hit_part = cp.zeros(self.max_hits, dtype=cp.int32)
        self.d_hit_jack = cp.zeros((self.max_hits, 16), dtype=cp.uint32)
        self.d_hit_hash = cp.zeros((self.max_hits, 8), dtype=cp.uint32)

        self.h_iv = _u32_words(bytes.fromhex(_IV_HEX))
        self.d_iv = cp.asarray(self.h_iv)
        self.d_salt_a = cp.asarray(_u32_words(SEED_SALT_A))
        self.d_salt_b = cp.asarray(_u32_words(SEED_SALT_B))
        self.d_label_a = cp.asarray(_u32_words(SEED_LABEL_A))
        self.d_label_b = cp.asarray(_u32_words(SEED_LABEL_B))
        self.d_job_key = cp.zeros((B, 8), dtype=cp.uint32)

        if self.batch_size > MAX_GRID_DIM:
            raise ValueError(
                f"batch size {self.batch_size} exceeds the CUDA grid limit "
                f"of {MAX_GRID_DIM} (the trial index is a grid dimension)")

        self.partitions_a = threads_partition(self.rows_pattern, m)
        self.partitions_b = threads_partition(self.cols_pattern, n)
        self.P = len(self.partitions_a) * len(self.partitions_b)
        self.pr = self.rows_pattern.size()
        self.pc = self.cols_pattern.size()
        self.tiles = (m // TILE_M) * (n // TILE_N)
        self._pow_shared = ((TILE_M + TILE_N) * (self.rank // 4) * 4
                            + TILE_M * TILE_N * 4
                            + (TILE_M // self.pr) * (TILE_N // self.pc)
                            * 16 * 4)
        self._tree_shared = ((m * k) // 1024) * 32

        # Kernels using more than 48 KB of dynamic shared memory need an
        # explicit opt-in (sm_70+); CuPy does not do it for us.
        for name, nbytes in (("pearl_pow", self._pow_shared),
                             ("blake3_tree_root", self._tree_shared)):
            self._opt_in_shared(self.kernels.kernel(name), nbytes)

    # -- shared-memory opt-in ------------------------------------------------

    def _opt_in_shared(self, kernel, nbytes: int):
        """Allow a kernel to use ``nbytes`` of dynamic shared memory."""
        if nbytes <= SMEM_STATIC_LIMIT:
            return
        driver = self.cp.cuda.driver
        limit = driver.funcGetAttribute(_CU_FUNC_ATTR_MAX_DYN_SMEM, kernel.ptr)
        if nbytes > limit:
            raise ValueError(
                f"kernel needs {nbytes} bytes of shared memory but the "
                f"device only allows {limit}; lower --rank/--m/--k or use "
                f"--backend numpy")
        driver.funcSetAttribute(kernel.ptr, _CU_FUNC_ATTR_MAX_DYN_SMEM,
                                int(nbytes))

    # -- job ----------------------------------------------------------------

    def set_job(self, header: IncompleteBlockHeader, cert_version: int):
        self._job_key = compute_job_key(header, self.config)
        self._bound = extract_difficulty_bound(header.nbits, self.config)
        self._derivation = derivation_for_cert_version(cert_version)
        cp = self.cp
        with cp.cuda.Device(self.device_id):
            self.d_job_key = cp.asarray(
                np.tile(_u32_words(self._job_key), (self.batch_size, 1)))
            bound = int(self._bound)
            words = np.array([(bound >> (32 * i)) & 0xFFFFFFFF
                              for i in range(8)], dtype=np.uint32)
            self.d_bound = cp.asarray(words)

    @property
    def job_key(self) -> bytes:
        if self._job_key is None:
            raise RuntimeError("call set_job() first")
        return self._job_key

    # -- small helpers ------------------------------------------------------

    def _single_block(self, messages: np.ndarray, key: np.ndarray,
                      flags: int):
        """Batched 64-byte single-block BLAKE3 -> (N, 8) uint32 device array."""
        cp = self.cp
        N = messages.shape[0]
        if key.ndim == 1:
            key = np.tile(key, (N, 1))
        d_msg = cp.asarray(np.ascontiguousarray(messages, dtype=np.uint32))
        d_keys = cp.asarray(np.ascontiguousarray(key, dtype=np.uint32))
        d_out = cp.zeros((N, 8), dtype=cp.uint32)
        threads = 256
        blocks = (N + threads - 1) // threads
        self.kernels.kernel("blake3_single64")(
            (blocks,), (threads,),
            (d_msg, d_keys, cp.uint32(flags), cp.int32(N), d_out))
        return d_out

    def _launch_tree(self, data, keys, out):
        cp = self.cp
        B = data.shape[0]
        nbytes = data.shape[1] * data.shape[2]
        num_chunks = nbytes // 1024
        self.kernels.kernel("blake3_tree_root")(
            (B,), (NTHREADS,),
            (data, cp.uint64(nbytes), keys, cp.uint32(1), out),
            shared_mem=num_chunks * 32)

    # -- one batch ----------------------------------------------------------

    def run_batch(self, max_hits: int | None = None, matrices=None):
        """
        Mine one batch.  ``matrices`` (A, Bt) replaces the random matrices
        when given - used by ``gpu_selftest.py`` to compare the GPU pipeline
        against the CPU reference on identical inputs.
        """
        cp = self.cp
        with cp.cuda.Device(self.device_id):
            if matrices is not None:
                A, Bt = matrices
                self.d_a[:] = cp.asarray(np.ascontiguousarray(A,
                                                              dtype=np.int8))
                self.d_bt[:] = cp.asarray(np.ascontiguousarray(Bt,
                                                               dtype=np.int8))
            return self._run_batch(skip_generation=matrices is not None)

    def _run_batch(self, skip_generation: bool = False):
        cp = self.cp
        m, n, k, B = self.m, self.n, self.k, self.batch_size
        batch_index = self._batch_index
        self._batch_index += 1

        # 1. random matrices (counter-based streams, distinct per batch)
        threads = 256
        if not skip_generation:
            for seed_off, total, buf in ((0, B * m * k, self.d_a),
                                         (1, B * n * k, self.d_bt)):
                seed = (self._base_seed + 0x9E3779B97F4A7C15 * (
                    batch_index * 2 + seed_off + 1))
                blocks = (total + threads - 1) // threads
                self.kernels.kernel("gen_matrices")(
                    (blocks,), (threads,),
                    (cp.uint64(seed), cp.int64(total), buf))

        # 2. commitment roots (keyed BLAKE3 Merkle roots of padded matrices)
        self._launch_tree(self.d_a, self.d_job_key, self.d_roots_a)
        self._launch_tree(self.d_bt, self.d_job_key, self.d_roots_b)

        # 3. noise-seed derivation (single-block hashes)
        roots_a = self.d_roots_a.get()
        roots_b = self.d_roots_b.get()
        if self._derivation == "salted":
            bind_a = self._bind_roots(roots_a, m, self.d_salt_a)
            bind_b = self._bind_roots(roots_b, n, self.d_salt_b)
        else:
            bind_a, bind_b = roots_a, roots_b

        job_words = _u32_words(self._job_key)
        msg_b = np.concatenate([np.tile(job_words, (B, 1)), bind_b],
                               axis=1).astype(np.uint32)
        d_b_seed = self._single_block(msg_b, self.h_iv, _FLAGS_UNKEYED_ROOT)
        msg_a = np.concatenate([d_b_seed.get(), bind_a],
                               axis=1).astype(np.uint32)
        d_a_seed = self._single_block(msg_a, self.h_iv, _FLAGS_UNKEYED_ROOT)
        self.d_a_seed = d_a_seed
        self.d_b_seed = d_b_seed

        # 4. noise bases
        self.kernels.kernel("noise_rows")(
            (B,), (NTHREADS,),
            (d_a_seed, self.d_label_a, cp.int32(m), cp.int32(self.rank),
             cp.uint32(1), self.d_e_al))
        self.kernels.kernel("noise_rows")(
            (B,), (NTHREADS,),
            (d_b_seed, self.d_label_b, cp.int32(n), cp.int32(self.rank),
             cp.uint32(1), self.d_e_br))
        self.kernels.kernel("noise_pairs")(
            (B,), (NTHREADS,),
            (d_a_seed, self.d_label_a, cp.int32(k), cp.int32(self.rank),
             cp.uint32(1), self.d_e_ar))
        self.kernels.kernel("noise_pairs")(
            (B,), (NTHREADS,),
            (d_b_seed, self.d_label_b, cp.int32(k), cp.int32(self.rank),
             cp.uint32(1), self.d_e_bl))

        # 5. noised matrices
        for matrix, e_rows, e_pairs, rows, out in (
                (self.d_a, self.d_e_al, self.d_e_ar, m, self.d_na),
                (self.d_bt, self.d_e_br, self.d_e_bl, n, self.d_nb)):
            per = rows * k
            blocks = (per + 255) // 256
            self.kernels.kernel("noise_apply")(
                (blocks, 1, B), (256,),
                (matrix, e_rows, e_pairs, cp.int32(rows),
                 cp.int32(self.rank), cp.int32(k), out))

        # 6. the PoW kernel
        self.d_hit_count.fill(0)
        self.kernels.kernel("pearl_pow")(
            (self.tiles, B), (NTHREADS,),
            (self.d_na, self.d_nb, d_a_seed, cp.int32(m), cp.int32(n),
             cp.int32(k), cp.int32(self.rank), cp.int32(self.pr),
             cp.int32(self.pc), self.d_bound, self.d_hit_count,
             self.d_hit_trial, self.d_hit_part, self.d_hit_jack,
             self.d_hit_hash, cp.int32(self.max_hits)),
            shared_mem=self._pow_shared)

        count = int(self.d_hit_count.get()[0])
        if count <= 0:
            return []
        overflow = count > self.max_hits
        n_hits = min(count, self.max_hits)
        trials = self.d_hit_trial.get()[:n_hits]
        parts = self.d_hit_part.get()[:n_hits]
        jacks = self.d_hit_jack.get()[:n_hits]
        hashes = self.d_hit_hash.get()[:n_hits]
        if overflow:
            print(f"[gpu] WARNING: {count} hits but only {self.max_hits} "
                  f"slots - some hits were dropped", flush=True)

        hits = []
        for i in range(n_hits):
            t, p = int(trials[i]), int(parts[i])
            ai, bi = divmod(p, len(self.partitions_b))
            hits.append({
                "trial": t,
                "A": self.d_a[t].get(),
                "Bt": self.d_bt[t].get(),
                "a_rows": list(self.partitions_a[ai]),
                "b_cols": list(self.partitions_b[bi]),
                "job_key": self.job_key,
                "jackpot": [int(w) for w in jacks[i]],
                "jackpot_hash": hashes[i].astype("<u4").tobytes(),
            })
            if max_hits is not None and len(hits) >= max_hits:
                break
        return hits

    def _bind_roots(self, roots: np.ndarray, dim: int, salt):
        """V3 salting: blake3(root || u32le(dim) || 28 zeros, key=salt)."""
        B = roots.shape[0]
        msg = np.zeros((B, 16), dtype=np.uint32)
        msg[:, :8] = roots
        msg[:, 8] = np.uint32(dim)
        return self._single_block(msg, salt.get(), _FLAGS_KEYED_ROOT).get()
