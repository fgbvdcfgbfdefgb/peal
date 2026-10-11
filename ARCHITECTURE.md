# Architecture & protocol notes

Everything below was reverse-engineered from the Pearl monorepo
(`pearl-research-labs/pearl`: `zk-pow`, `pearl-blake3`, `miner/pearl-gateway`,
`node`) and is pinned by `miner/selftest.py`, which cross-checks this
implementation against the official Rust module `pearl_mining` when it is
installed.

## 1. The mining job

The gateway serves `getMiningInfo` over line-delimited JSON-RPC (TCP
`127.0.0.1:8337` by default, or a UNIX socket):

```json
{"jsonrpc": "2.0", "method": "getMiningInfo", "id": 1}
→ {"jsonrpc": "2.0", "result": {
     "incomplete_header_bytes": "<base64, 76 bytes>",
     "target": 123456789,
     "cert_version": 3}, "id": 1}
```

`incomplete_header_bytes` is the 76-byte header **without** the proof
commitment:

```
version        u32 LE
prev_block     32 bytes, stored reversed (internal byte order)
merkle_root    32 bytes, stored reversed
timestamp      u32 LE
nbits          u32 LE   (Bitcoin compact difficulty)
```

Shares go back as `submitPlainProof` with the base64 `plain_proof` and the
same `mining_job` dict. The gateway rebuilds the full block (proof
commitment + coinbase), builds the Plonky2 ZK proof and submits it via
`submitblock`. Shares whose header no longer matches the current template
are skipped by the gateway — the relay pre-checks this to avoid wasting a
proving run.

## 2. What the miner computes

For each trial (a fresh random pair of matrices):

```
A  (m, k) int8, entries in [-64, 64]      (row-major)
Bᵀ (n, k) int8, entries in [-64, 64]      (B transposed)
```

1. **Job key** — `job_key = blake3(header ‖ config)` where `config` is the
   52-byte mining configuration:

   ```
   common_dim (k)  u32 LE
   rank            u16 LE
   mma_type        u16 LE   (0 = Int7xInt7ToInt32)
   rows_pattern    6 bytes  (3 dims × (factor-1, length-1))
   cols_pattern    6 bytes
   trailer         32 bytes (zeros ⇒ dense / non-MoE)
   ```

2. **Commitments** — `hash_a = keyed_blake3(pad1024(A), job_key)`,
   `hash_b = keyed_blake3(pad1024(Bᵀ), job_key)`; the matrices are
   zero-padded to a multiple of the 1024-byte BLAKE3 chunk.

3. **Noise seeds** — Legacy (cert V1/V2): `hash_a`, `hash_b` as-is. Salted
   (cert V3): `bind_root_a(hash_a, m) = blake3(hash_a ‖ u32le(m) ‖ 28×0,
   key=SEED_SALT_A)`, likewise for B. Then
   `b_noise_seed = blake3(job_key ‖ hash_b)`,
   `a_noise_seed = blake3(b_noise_seed ‖ hash_a)`.

4. **Noise** — for each row/column index `i`:

   ```
   e_al[i][:] = (bytes of keyed_blake3(64B: u32le(1+block) ‖ "A_tensor", a_noise_seed) at row·rank …) & 63) - 32
   e_ar[j]    = permutation pair from keyed_blake3(64B: slot1=u32le(1+chunk) ‖ "A_tensor", a_noise_seed)
   noise_a[i][j] = e_al[i][first_j] - e_al[i][second_j]
   ```
   (`first = u32 & (rank-1)`, `second = first ^ (1 + mulhi(rank-1, u32))`;
   the same construction with the `B_tensor` label and `b_noise_seed` for
   `noise_b`.) Noise values live in [-63, 63].

5. **Jackpot** — for every (rows tile, cols tile) partition, accumulate the
   noised GEMM in rank-sized k-chunks; after each chunk fold the tile:

   ```
   jackpot[tid] = rotl32(jackpot[tid], 13) ^ xor(tile[i][j] as u32)
   tid = (chunk_index) % 16
   ```

6. **Hit test** — `hit ⇔ u256le(blake3_keyed(16×u32le(jackpot), a_noise_seed)) ≤ bound`
   with `bound = target · tile_size · (k − k % rank)` (the *unpenalised,
   saturating* bound — exactly what `mine.rs` and `verify.rs` use; the rank
   penalty only bites for rank ≠ 128 through the separate consensus check).

On a hit the miner builds `MatrixMerkleProof`s for the tile's rows of A and
columns of Bᵀ and emits a `PlainProof`.

## 3. PlainProof wire format (bincode fixint, base64)

```
m, n, k, noise_rank                     u64 LE each
a:  MatrixMerkleProof
bt: MatrixMerkleProof
moe: Option tag                         0x00 = None (dense proofs only)

MatrixMerkleProof:
    proof: MerkleProof
    row_indices: Vec<u64>

MerkleProof:
    leaf_data:    u64 count, then per leaf (u64 len = 1024 ‖ 1024 bytes)
    leaf_indices: Vec<u64>          (sorted, unique)
    total_leaves: u64
    root:         [u8; 32]
    siblings:     u64 count ‖ 32 bytes each
```

`row_indices` must equal the pattern list plus the partition offset, and the
leaf indices must be exactly the chunks spanned by those rows
(`compute_leaf_indices_from_rows`). The sibling order is the level-walk over
the sorted leaf set (odd index → left sibling; even index → right sibling
when it exists and is absent from the set).

## 4. Local share verification

Before a share is written to `share.txt`, `proof.verify_plain_proof`
re-implements `zk-pow/src/api/verify.rs` without the ZK circuit:

1. tree sizes match `(m·k + 1023) / 1024` and `(n·k + 1023) / 1024`;
2. the pattern/offset is recovered from `row_indices` and re-checked;
3. the Merkle roots reconstruct from `leaf_data` + `siblings`;
4. the strips are extracted from the leaf data (range-checked to [-64, 64]);
5. noise, the jackpot and the difficulty bound are recomputed from the
   proof's own roots.

The selftest additionally round-trips a mined share through the official
Rust `PlainProof::from_base64` + `verify_plain_proof_v2` and checks that a
corrupted share is rejected.

## 5. Pinned constants (selftest)

* `SEED_SALT_A = 8249406c a0ed1516 9616f692 fcf076f8 92dbdb2a 7023b852 f0d47719 c390017b`
  = `blake3("pearl/cert-v3/noise-seed/A")`
* `SEED_SALT_B = 11300632 ec6301ca 2be2af71 8b3f4d4f 1ae9c639 88e8cc04 4844301d 71b89aa9`
  = `blake3("pearl/cert-v3/noise-seed/B")`
* Legacy seeds for `job_key=0x11…`, `hash_a=0xAA…`, `hash_b=0xBB…`,
  `new_for_tests(192, 320, 256)`:
  * b = `add6f7ea5feebf89c8a77e2ebfa0d82442e7dbb0046dbd48971861d12fcb0177`
  * a = `483b07b6f73105030b9482255f37723f3fed69ae916724ee8291848b8c28794b`
* Salted seeds for the same inputs:
  * b = `60ed9b73c5a9599b200b6cd563e7f0d5d9a67d2402d85fd4ef966c580080d0e5`
  * a = `301784168005ec833ab0aa60006f7fe7faaa95307d8c1fc6819b2ffdd717eccf`

## 6. GPU engine (A10G / sm86)

The monorepo's `pearl-gemm` miner is H100-only (`arch=compute_90a`, needs
nvcc + CUTLASS + torch 2.11), so this repo ships its own GPU engine:
**CuPy + NVRTC** — `pip install cupy-cuda12x` is the whole toolchain, the
kernels are compiled by the driver at runtime, and no CUDA compiler, CUTLASS
or PyTorch is involved. Requires compute capability ≥ 7.5 (`dp4a`).

`pearl_miner/cuda_kernels.py` holds the kernels; `pearl_miner/engine_cuda.py`
drives them. Per batch:

| Stage | Kernel | Notes |
|-------|--------|-------|
| random A / Bᵀ | `gen_matrices` | counter-based splitmix64, int8 in [-64, 64] |
| commitment roots | `blake3_tree_root` | one block per matrix, chunk CVs in shared memory, race-free per-level reduction |
| noise-seed derivation | `blake3_single64` | 64-byte single-block hashes, Legacy and Salted |
| noise bases | `noise_rows` / `noise_pairs` | slot 0 uniform rows, slot 1 permutation pairs |
| noised matrices | `noise_apply` | `matrix + e_row[first] - e_row[second]` |
| the PoW itself | `pearl_pow` | 32×32 tile per block, 256 threads × 2×2 sub-tile, `__dp4a` into int32, per-rank-chunk fold, jackpot hash + bound check, atomic hit append |

Everything is exact: noised values fit in int8, `__dp4a` accumulates in int32,
and the accumulated tile stays below 2³¹ (≤ k · 64 · 127 ≈ 3.3·10⁷ for the
default config). fp32/fp16 accumulation would not be exact.

Supported configurations (checked in `GpuEngine.supports`, otherwise the
numpy engine is used): dense patterns, `m`/`n` multiples of 32, tile dims
dividing 32, `k` a multiple of `rank`, `rank` a multiple of 32, matrices a
multiple of 1024 bytes with ≤ 1536 chunks each, `--gpu-batch` ≤ 65535.
Kernels that need more than 48 KB of dynamic shared memory opt in through
`cuFuncSetAttribute` (CuPy does not do this automatically).

Validation without a GPU:

* `miner/kernel_logic_test.py` — transcribes every kernel's index arithmetic
  into numpy and compares it against the CPU reference (tree roots for 1…64
  chunks, noise rows/pairs/apply, the per-tile jackpot fold and partition
  mapping, the noise-seed derivation). Runs anywhere, no GPU needed.
* `miner/nvrtc_compile_check.py` — compiles the kernel source through NVRTC
  (the same path CuPy uses) so syntax/type errors are caught without a GPU.
  Needs `pip install nvidia-cuda-nvrtc-cu12`.
* `miner/gpu_selftest.py` — the real thing: GPU vs CPU on identical matrices,
  including a saturating bound where every partition must hit and a realistic
  bound where the hit sets must match exactly. Run it once per machine.

## 7. Performance notes

* The engine batches `--batch` trials at once; every step is vectorised
  (numpy). GEMMs run in fp32 chunks of `rank` columns — exact, because each
  partial sum stays below 2²⁴ — and accumulate into an int32 tile.
* The commitment hashes use the pip `blake3` package when available (the
  miner falls back to its own vectorised BLAKE3 port otherwise).
* Merkle trees are only built on hits; the tree build and the verification
  path batch all chunk/parent compressions into single vectorised calls.
* Rule of thumb: `--batch` trials cost roughly
  `batch · (m·k + n·k)` bytes for the matrices plus `batch · m · n · 4`
  bytes for the tile — 256 trials of the default config ≈ 400 MB.

## 8. Known limits

* Dense (non-MoE) proofs only — the miner refuses to emit MoE shares.
* The relay submits shares but cannot see whether the node accepted the
  block (the gateway returns `submitted` before proving); watch the
  `pearl-gateway` log for `Block submission result`.
* Plonky2 proving happens on the Windows machine inside the gateway; it
  needs a few GB of RAM and takes on the order of a minute per share.
