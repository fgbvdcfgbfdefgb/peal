"""
CUDA kernels for the Pearl miner (CuPy + NVRTC, no nvcc required).

These kernels implement on the GPU exactly what the numpy engine does on the
CPU (and what ``zk-pow`` does in Rust):

  * ``blake3_single64``  - batched single-block BLAKE3 of 64-byte messages
                           (noise-seed derivation)
  * ``blake3_tree_root`` - keyed BLAKE3 Merkle root of chunk-padded matrices
                           (the commitment hashes); one thread block per
                           matrix
  * ``gen_matrices``     - counter-based random int8 matrices in [-64, 64]
  * ``noise_rows``       - E_AL / E_BR uniform rows from the noise seeds
  * ``noise_pairs``      - E_AR / E_BL permutation pairs from the noise seeds
  * ``noise_apply``      - A + noise and B^T + noise (int8)
  * ``pearl_pow``        - the main kernel: per (trial, tile) GEMM in
                           rank-sized k-chunks with the jackpot fold, the
                           jackpot hash and the difficulty check

All integer arithmetic is exact: noised values fit in int8, ``__dp4a``
accumulates into int32, and every partial sum stays below 2^24.

Requires ``pip install cupy-cuda12x`` (the wheel matching the installed
driver) and a GPU with compute capability >= 7.5 (``dp4a``) - for example
the A10G (sm86).  Kernels are JIT-compiled by NVRTC on first use.

Run ``python gpu_selftest.py`` once on the mining machine to validate the
GPU engine against the CPU reference before trusting it with real jobs.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# CUDA C source
# ---------------------------------------------------------------------------

_CUDA_SOURCE = r"""
typedef unsigned int u32;
typedef unsigned long long u64;
typedef unsigned char u8;

#define CHUNK_START 1u
#define CHUNK_END   2u
#define PARENT      4u
#define ROOT        8u
#define KEYED_HASH  16u

__device__ __constant__ u32 IV[8] = {
    0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
    0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u
};

__device__ __constant__ u32 MSG_PERM[16] = {
    2u, 6u, 3u, 10u, 7u, 0u, 4u, 13u,
    1u, 11u, 12u, 5u, 9u, 14u, 15u, 8u
};

__device__ __forceinline__ u32 rotr32(u32 x, u32 r) {
    return __funnelshift_r(x, x, r);
}

__device__ __forceinline__ u32 rotl32(u32 x, u32 r) {
    return __funnelshift_r(x, x, 32u - r);
}

__device__ __forceinline__ void g_fn(u32* s, const u32* m,
                                     int a, int b, int c, int d,
                                     int mx, int my) {
    s[a] = s[a] + s[b] + m[mx];
    s[d] = rotr32(s[d] ^ s[a], 16u);
    s[c] = s[c] + s[d];
    s[b] = rotr32(s[b] ^ s[c], 12u);
    s[a] = s[a] + s[b] + m[my];
    s[d] = rotr32(s[d] ^ s[a], 8u);
    s[c] = s[c] + s[d];
    s[b] = rotr32(s[b] ^ s[c], 7u);
}

__device__ __forceinline__ void compress(u32* s, u32* m) {
    #pragma unroll
    for (int r = 0; r < 7; r++) {
        g_fn(s, m, 0, 4, 8, 12, 0, 1);
        g_fn(s, m, 1, 5, 9, 13, 2, 3);
        g_fn(s, m, 2, 6, 10, 14, 4, 5);
        g_fn(s, m, 3, 7, 11, 15, 6, 7);
        g_fn(s, m, 0, 5, 10, 15, 8, 9);
        g_fn(s, m, 1, 6, 11, 12, 10, 11);
        g_fn(s, m, 2, 7, 8, 13, 12, 13);
        g_fn(s, m, 3, 4, 9, 14, 14, 15);
        u32 p[16];
        #pragma unroll
        for (int i = 0; i < 16; i++) p[i] = m[MSG_PERM[i]];
        #pragma unroll
        for (int i = 0; i < 16; i++) m[i] = p[i];
    }
}

// One compression of a single 64-byte block.
__device__ __forceinline__ void compress_block(const u32* key, const u32* m,
                                               u32 counter_lo,
                                               u32 block_len, u32 flags,
                                               u32* out_state) {
    #pragma unroll
    for (int j = 0; j < 8; j++) out_state[j] = key[j];
    out_state[8]  = IV[0];
    out_state[9]  = IV[1];
    out_state[10] = IV[2];
    out_state[11] = IV[3];
    out_state[12] = counter_lo;
    out_state[13] = 0u;
    out_state[14] = block_len;
    out_state[15] = flags;
    u32 mm[16];
    #pragma unroll
    for (int j = 0; j < 16; j++) mm[j] = m[j];
    compress(out_state, mm);
}

__device__ __forceinline__ void fold_cv(const u32* s, u32* out8) {
    #pragma unroll
    for (int j = 0; j < 8; j++) out8[j] = s[j] ^ s[j + 8];
}

// ---------------------------------------------------------------------------
// gen_matrices: counter-based PRNG producing int8 values in [-64, 64].
// ---------------------------------------------------------------------------

__device__ __forceinline__ u64 splitmix64(u64 x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

extern "C" __global__ void gen_matrices(
    u64 seed, long long total, signed char* __restrict__ out)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    u64 h = splitmix64(seed ^ (u64)i);
    out[i] = (signed char)((int)(h % 129ull) - 64);
}

// ---------------------------------------------------------------------------
// blake3_single64: batched single-block BLAKE3 of 64-byte messages.
// ---------------------------------------------------------------------------

extern "C" __global__ void blake3_single64(
    const u32* __restrict__ messages, const u32* __restrict__ keys,
    u32 flags, int N, u32* __restrict__ out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    u32 key[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) key[j] = keys[i * 8 + j];
    u32 m[16];
    #pragma unroll
    for (int j = 0; j < 16; j++) m[j] = messages[i * 16 + j];
    u32 s[16];
    compress_block(key, m, 0u, 64u, flags, s);
    fold_cv(s, out + i * 8);
}

// ---------------------------------------------------------------------------
// blake3_tree_root: keyed BLAKE3 Merkle root of chunk-padded matrices.
//   data: (B, nbytes) u8 with nbytes % 1024 == 0
//   keys: (B, 8) u32 ; out: (B, 8) u32
// One block per matrix.  Shared memory holds num_chunks * 8 u32 (chunk CVs).
// The per-level reduction is race free: every thread first loads its inputs
// into registers, a barrier separates all loads from all writes.
// ---------------------------------------------------------------------------

extern "C" __global__ void blake3_tree_root(
    const u8* __restrict__ data, u64 nbytes,
    const u32* __restrict__ keys, u32 keyed, u32* __restrict__ out)
{
    extern __shared__ u32 cv[];                 // num_chunks * 8 words
    int b = blockIdx.x;
    const u8* base = data + (u64)b * nbytes;
    int num_chunks = (int)(nbytes >> 10);
    int nt = blockDim.x;
    u32 base_flags = keyed ? KEYED_HASH : 0u;
    // A single-chunk tree has no parent level: the ROOT flag belongs to the
    // chunk's own last block (matches blake3k._tree_layers).
    u32 chunk_root = (num_chunks == 1) ? ROOT : 0u;

    u32 key[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) key[j] = keys[b * 8 + j];

    // --- chunk chaining values: 16 chained 64-byte blocks per chunk -------
    for (int c = threadIdx.x; c < num_chunks; c += nt) {
        const u8* chunk = base + (u64)c * 1024;
        u32 cvw[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) cvw[j] = key[j];
        #pragma unroll
        for (int blk = 0; blk < 16; blk++) {
            u32 m[16];
            const u32* src = (const u32*)(chunk + blk * 64);
            #pragma unroll
            for (int j = 0; j < 16; j++) m[j] = src[j];
            u32 flags = base_flags;
            if (blk == 0) flags |= CHUNK_START;
            if (blk == 15) flags |= CHUNK_END | chunk_root;
            u32 s[16];
            compress_block(cvw, m, (u32)c, 64u, flags, s);
            fold_cv(s, cvw);
        }
        #pragma unroll
        for (int j = 0; j < 8; j++) cv[c * 8 + j] = cvw[j];
    }
    __syncthreads();

    // --- pairwise parent levels until <= 2 nodes remain -------------------
    int len = num_chunks;
    while (len > 2) {
        int npairs = len / 2;
        int odd = len & 1;
        int iters = (npairs + nt - 1) / nt;
        for (int it = 0; it < iters; it++) {
            int p = it * nt + threadIdx.x;
            bool active = p < npairs;
            u32 m[16];
            if (active) {
                #pragma unroll
                for (int j = 0; j < 8; j++) m[j] = cv[(2 * p) * 8 + j];
                #pragma unroll
                for (int j = 0; j < 8; j++) m[8 + j] = cv[(2 * p + 1) * 8 + j];
            }
            __syncthreads();                 // all loads done
            if (active) {
                u32 s[16];
                compress_block(key, m, 0u, 64u, base_flags | PARENT, s);
                u32 out8[8];
                fold_cv(s, out8);
                #pragma unroll
                for (int j = 0; j < 8; j++) cv[p * 8 + j] = out8[j];
            }
            __syncthreads();                 // all writes done
        }
        if (odd) {
            if (threadIdx.x == 0) {
                #pragma unroll
                for (int j = 0; j < 8; j++)
                    cv[npairs * 8 + j] = cv[(len - 1) * 8 + j];
            }
            __syncthreads();
        }
        len = npairs + odd;
    }

    // --- final root -------------------------------------------------------
    if (threadIdx.x == 0) {
        if (len == 2) {
            u32 m[16];
            #pragma unroll
            for (int j = 0; j < 8; j++) m[j] = cv[j];
            #pragma unroll
            for (int j = 0; j < 8; j++) m[8 + j] = cv[8 + j];
            u32 s[16];
            compress_block(key, m, 0u, 64u, base_flags | PARENT | ROOT, s);
            fold_cv(s, out + b * 8);
        } else {
            #pragma unroll
            for (int j = 0; j < 8; j++) out[b * 8 + j] = cv[j];
        }
    }
}

// ---------------------------------------------------------------------------
// noise_rows: E_AL / E_BR uniform rows - values (byte & 63) - 32.
//   seeds:  (B, 8) u32   noise-seed words
//   labels: (8,) u32     tensor label words ("A_tensor" / "B_tensor")
//   e_rows: (B, rows, rank) int8
// One block per batch element.
// ---------------------------------------------------------------------------

extern "C" __global__ void noise_rows(
    const u32* __restrict__ seeds, const u32* __restrict__ labels,
    int rows, int rank, u32 keyed,
    signed char* __restrict__ e_rows)
{
    int b = blockIdx.x;
    int nt = blockDim.x;
    u32 key[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) key[j] = seeds[b * 8 + j];
    u32 label[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) label[j] = labels[j];
    u32 flags = (keyed ? KEYED_HASH : 0u) | CHUNK_START | CHUNK_END | ROOT;

    int blocks_per_row = rank / 32;              // 4 for rank 128, up to 32
    for (int r = threadIdx.x; r < rows; r += nt) {
        for (int blk = 0; blk < blocks_per_row; blk++) {
            int block_index = r * blocks_per_row + blk;
            u32 m[16];
            #pragma unroll
            for (int j = 0; j < 16; j++) m[j] = 0u;
            m[0] = (u32)(block_index + 1);
            #pragma unroll
            for (int j = 0; j < 8; j++) m[8 + j] = label[j];
            u32 s[16];
            compress_block(key, m, 0u, 64u, flags, s);
            u32 digest[8];
            fold_cv(s, digest);
            const u8* bytes = (const u8*)digest;
            #pragma unroll
            for (int j = 0; j < 32; j++)
                e_rows[((long long)b * rows + r) * rank + blk * 32 + j] =
                    (signed char)((bytes[j] & 63) - 32);
        }
    }
}

// ---------------------------------------------------------------------------
// noise_pairs: E_AR / E_BL permutation pairs (8 lines per hash).
//   e_pairs: (B, k, 2) int32
// ---------------------------------------------------------------------------

extern "C" __global__ void noise_pairs(
    const u32* __restrict__ seeds, const u32* __restrict__ labels,
    int k, int rank, u32 keyed,
    int* __restrict__ e_pairs)
{
    int b = blockIdx.x;
    int nt = blockDim.x;
    u32 key[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) key[j] = seeds[b * 8 + j];
    u32 label[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) label[j] = labels[j];
    u32 flags = (keyed ? KEYED_HASH : 0u) | CHUNK_START | CHUNK_END | ROOT;

    int lines_per_hash = 8;
    int num_chunks = (k + lines_per_hash - 1) / lines_per_hash;
    for (int c = threadIdx.x; c < num_chunks; c += nt) {
        u32 m[16];
        #pragma unroll
        for (int j = 0; j < 16; j++) m[j] = 0u;
        m[1] = (u32)(c + 1);
        #pragma unroll
        for (int j = 0; j < 8; j++) m[8 + j] = label[j];
        u32 s[16];
        compress_block(key, m, 0u, 64u, flags, s);
        u32 digest[8];
        fold_cv(s, digest);
        u32 rank_mask = (u32)(rank - 1);
        #pragma unroll
        for (int j = 0; j < lines_per_hash; j++) {
            int col = c * lines_per_hash + j;
            if (col >= k) break;
            u32 u = digest[j];
            u32 first = u & rank_mask;
            u32 mulhi = (u32)(((u64)(rank - 1) * (u64)u) >> 32);
            u32 second = first ^ (1u + mulhi);
            e_pairs[((long long)b * k + col) * 2 + 0] = (int)first;
            e_pairs[((long long)b * k + col) * 2 + 1] = (int)second;
        }
    }
}

// ---------------------------------------------------------------------------
// noise_apply: noised = matrix + noise (int8; matrix in [-64,64] and noise
// in [-63,63], so no saturation is possible).
//   grid = (ceil(rows*k / 256), 1, B), block = (256)
// ---------------------------------------------------------------------------

extern "C" __global__ void noise_apply(
    const signed char* __restrict__ matrix,
    const signed char* __restrict__ e_rows,
    const int* __restrict__ e_pairs,
    int rows, int rank, int k,
    signed char* __restrict__ noised)
{
    long long per_matrix = (long long)rows * k;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= per_matrix) return;
    long long n = blockIdx.z;
    int r = (int)(idx / k);
    int j = (int)(idx % k);
    int first = e_pairs[((long long)n * k + j) * 2 + 0];
    int second = e_pairs[((long long)n * k + j) * 2 + 1];
    const signed char* er = e_rows + ((long long)n * rows + r) * rank;
    int noise = (int)er[first] - (int)er[second];
    long long base = n * per_matrix;
    noised[base + idx] = (signed char)((int)matrix[base + idx] + noise);
}

// ---------------------------------------------------------------------------
// pearl_pow: the mining kernel.
//
// Grid:  x = tile index within a trial ((m/32) * (n/32)), y = trial.
// Block: 256 threads; each block owns a 32 x 32 output tile and each thread
//        a 2 x 2 sub-tile (accumulated in int32 registers via dp4a).
//
// For every rank-sized k-chunk the block accumulates its tile, publishes it
// to shared memory and folds it into the jackpot words of every partition
// the block covers.  After all chunks each partition's jackpot is hashed
// (keyed BLAKE3, single block) and compared against the difficulty bound;
// hits are appended to a global buffer with an atomic counter.
//
// Requirements: m % 32 == 0, n % 32 == 0, 32 % pr == 0, 32 % pc == 0,
//               k % rank == 0, rank % 32 == 0.
// ---------------------------------------------------------------------------

#define TILE_M 32
#define TILE_N 32
#define NTHREADS 256

extern "C" __global__ void pearl_pow(
    const signed char* __restrict__ noised_a,   // (B, m, k)
    const signed char* __restrict__ noised_bt,  // (B, n, k)
    const u32* __restrict__ a_seeds,            // (B, 8)
    int m, int n, int k, int rank, int pr, int pc,
    const u32* __restrict__ bound_words,        // (8,) little-endian u256
    int* __restrict__ hit_count,
    int* __restrict__ hit_trial,
    int* __restrict__ hit_partition,
    u32* __restrict__ hit_jackpot,              // (max_hits, 16)
    u32* __restrict__ hit_hash,                 // (max_hits, 8)
    int max_hits)
{
    extern __shared__ u32 smem[];
    u32* As = smem;                                   // TILE_M * (rank/4)
    u32* Bs = As + TILE_M * (rank / 4);               // TILE_N * (rank/4)
    int* tile = (int*)(Bs + TILE_N * (rank / 4));     // TILE_M * TILE_N
    u32* jackpot = (u32*)(tile + TILE_M * TILE_N);    // nparts * 16

    int tile_idx = blockIdx.x;
    int trial = blockIdx.y;

    int tiles_n = n / TILE_N;
    int row0 = (tile_idx / tiles_n) * TILE_M;
    int col0 = (tile_idx % tiles_n) * TILE_N;

    int parts_m = TILE_M / pr;
    int parts_n = TILE_N / pc;
    int nparts = parts_m * parts_n;
    int parts_total_n = n / pc;

    int tid = threadIdx.x;
    int ty = tid >> 4;
    int tx = tid & 15;
    int r0 = ty * 2, c0 = tx * 2;

    int acc[2][2];
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 2; j++) acc[i][j] = 0;

    const signed char* A = noised_a + (long long)trial * m * k;
    const signed char* B = noised_bt + (long long)trial * n * k;
    int rank4 = rank / 4;

    for (int p = tid; p < nparts * 16; p += NTHREADS) jackpot[p] = 0u;
    __syncthreads();

    int steps = k / rank;
    for (int step = 0; step < steps; step++) {
        int koff = step * rank;
        for (int i = tid; i < TILE_M * rank4; i += NTHREADS) {
            int r = i / rank4;
            int w = i % rank4;
            As[i] = *(const u32*)(A + (long long)(row0 + r) * k + koff + w * 4);
        }
        for (int i = tid; i < TILE_N * rank4; i += NTHREADS) {
            int c = i / rank4;
            int w = i % rank4;
            Bs[i] = *(const u32*)(B + (long long)(col0 + c) * k + koff + w * 4);
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < 2; i++) {
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                int v = acc[i][j];
                const u32* arow = As + (r0 + i) * rank4;
                const u32* brow = Bs + (c0 + j) * rank4;
                for (int w = 0; w < rank4; w++)
                    v = __dp4a((int)arow[w], (int)brow[w], v);
                acc[i][j] = v;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < 2; i++)
            #pragma unroll
            for (int j = 0; j < 2; j++)
                tile[(r0 + i) * TILE_N + (c0 + j)] = acc[i][j];
        __syncthreads();

        for (int p = tid; p < nparts; p += NTHREADS) {
            int rg = p / parts_n;
            int cg = p % parts_n;
            u32 x = 0u;
            for (int i = 0; i < pr; i++)
                for (int j = 0; j < pc; j++)
                    x ^= (u32)tile[(rg * pr + i) * TILE_N + (cg * pc + j)];
            int tidslot = step % 16;
            jackpot[p * 16 + tidslot] = rotl32(jackpot[p * 16 + tidslot], 13u) ^ x;
        }
        __syncthreads();
    }

    // ---- hash the jackpots and check the difficulty ----------------------
    u32 key[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) key[j] = a_seeds[(long long)trial * 8 + j];

    for (int p = tid; p < nparts; p += NTHREADS) {
        u32 m[16];
        #pragma unroll
        for (int j = 0; j < 16; j++) m[j] = jackpot[p * 16 + j];
        u32 s[16];
        compress_block(key, m, 0u, 64u,
                       KEYED_HASH | CHUNK_START | CHUNK_END | ROOT, s);
        u32 digest[8];
        fold_cv(s, digest);

        bool le = true;
        #pragma unroll
        for (int w = 7; w >= 0; w--) {
            if (digest[w] > bound_words[w]) { le = false; break; }
            if (digest[w] < bound_words[w]) break;
        }
        if (!le) continue;

        int slot = atomicAdd(hit_count, 1);
        if (slot < max_hits) {
            hit_trial[slot] = trial;
            int rg = p / parts_n;
            int cg = p % parts_n;
            int global_part = (row0 / pr + rg) * parts_total_n
                              + (col0 / pc + cg);
            hit_partition[slot] = global_part;
            #pragma unroll
            for (int j = 0; j < 16; j++)
                hit_jackpot[(long long)slot * 16 + j] = m[j];
            #pragma unroll
            for (int j = 0; j < 8; j++)
                hit_hash[(long long)slot * 8 + j] = digest[j];
        }
    }
}
"""

# ---------------------------------------------------------------------------
# Kernel registry / lazy compilation
# ---------------------------------------------------------------------------

_KERNEL_NAMES = (
    "gen_matrices",
    "blake3_single64",
    "blake3_tree_root",
    "noise_rows",
    "noise_pairs",
    "noise_apply",
    "pearl_pow",
)


class CudaKernels:
    """Kernels JIT-compiled by NVRTC for one device."""

    def __init__(self):
        import cupy as cp

        self.cp = cp
        self._module = cp.RawModule(code=_CUDA_SOURCE,
                                    options=("-std=c++14",),
                                    backend="nvrtc")
        self._kernels = {name: self._module.get_function(name)
                         for name in _KERNEL_NAMES}

    def kernel(self, name: str):
        return self._kernels[name]


_compiled: dict = {}


def kernels_for_device(device_id: int = 0) -> CudaKernels:
    """Kernels for ``device_id`` (compiled once per process)."""
    if device_id not in _compiled:
        import cupy as cp
        with cp.cuda.Device(device_id):
            _compiled[device_id] = CudaKernels()
    return _compiled[device_id]


def gpu_devices() -> list:
    """Ids of devices that can run these kernels (compute capability >= 7.5)."""
    try:
        import cupy as cp
    except Exception:
        return []
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except Exception:
        return []
    devices = []
    for dev in range(count):
        try:
            props = cp.cuda.runtime.getDeviceProperties(dev)
            major, minor = props["computeCapability"]
            if (major, minor) >= (7, 5):
                devices.append(dev)
        except Exception:
            continue
    return devices


def cuda_available() -> bool:
    return bool(gpu_devices())


def gpu_status() -> str:
    """
    Human-readable CuPy / GPU status, for ``run_miner.sh --gpu-info``.

    Says exactly why the GPU engine is or is not usable: CuPy missing, CuPy
    present but no visible device, or a device whose compute capability is
    too old for ``__dp4a``.
    """
    try:
        import cupy as cp
    except Exception as exc:  # pragma: no cover - environment dependent
        return (f"CuPy is NOT importable ({type(exc).__name__}: {exc}).\n"
                f"  Install it with:  ./run_miner.sh --install-gpu")

    try:
        count = cp.cuda.runtime.getDeviceCount()
    except Exception as exc:  # pragma: no cover - driver problems
        return (f"CuPy {cp.__version__} is installed but no CUDA device is "
                f"visible ({type(exc).__name__}: {exc}).\n"
                f"  Check the driver with nvidia-smi, and that the CuPy "
                f"wheel matches the CUDA version (cupy-cuda12x needs CUDA "
                f">= 12, cupy-cuda11x needs CUDA 11.x).")

    if count == 0:
        return "CuPy is installed but reports 0 CUDA devices."

    lines = [f"CuPy {cp.__version__}: {count} CUDA device(s)"]
    usable = gpu_devices()
    for dev in range(count):
        try:
            props = cp.cuda.runtime.getDeviceProperties(dev)
        except Exception as exc:  # pragma: no cover
            lines.append(f"  [{dev}] <unreadable: {exc}>")
            continue
        name = props.get("name", b"?")
        if isinstance(name, bytes):
            name = name.decode(errors="replace")
        major, minor = props.get("computeCapability", (0, 0))
        total = props.get("totalGlobalMem", 0) / 1024 ** 3
        sm_count = props.get("multiProcessorCount", 0)
        optin = props.get("sharedMemPerBlockOptin", 0) / 1024
        ok = dev in usable
        lines.append(
            f"  [{dev}] {name}: sm_{major}{minor}, {total:.1f} GB, "
            f"{sm_count} SMs, {optin:.0f} KB shared memory -> "
            + ("USABLE" if ok else "TOO OLD (needs compute capability >= 7.5)"))
    return "\n".join(lines)
