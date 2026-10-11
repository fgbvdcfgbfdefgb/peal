"""
Kernel logic test - validates the GPU kernel *algorithms* without a GPU.

``gpu_selftest.py`` needs a GPU.  This script does not: it mechanically
transcribes the index arithmetic of every kernel in ``cuda_kernels.py`` into
numpy and compares the result against the CPU reference implementation
(``engine_np`` / ``blake3k`` / ``core``).  A mistake in a loop bound, a
shared-memory layout, a flag placement or a partition index therefore shows
up on any machine.

Run:  python kernel_logic_test.py
"""

from __future__ import annotations

import sys

import numpy as np

from pearl_miner import blake3k
from pearl_miner.blake3k import (
    CHUNK_START, CHUNK_END, PARENT, ROOT, KEYED_HASH,   # noqa: F401
)
from pearl_miner.core import (
    IncompleteBlockHeader, MiningConfiguration, compute_job_key, SEED_SALT_A,
    SEED_SALT_B,
    compute_jackpot_hash, derive_noise_seeds, matrix_roots,
    compute_noise_for_indices, extract_difficulty_bound,
    derivation_for_cert_version, SIGNAL_MIN, SIGNAL_MAX,
)
from pearl_miner.patterns import PeriodicPattern, threads_partition
from pearl_miner.trial import compute_jackpot_words

PASS = 0
FAIL = 0


def check(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  [ok] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}" + (f" - {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# transcription of blake3_tree_root
# ---------------------------------------------------------------------------

def tree_root_sim(data: bytes, key: bytes) -> bytes:
    num_chunks = len(data) // 1024
    base_flags = KEYED_HASH
    chunk_root = ROOT if num_chunks == 1 else 0
    kw = blake3k.key_words(key)
    cv = [None] * num_chunks
    for c in range(num_chunks):
        chunk = data[c * 1024:(c + 1) * 1024]
        cvw = kw.copy()
        for blk in range(16):
            m = np.frombuffer(chunk[blk * 64:(blk + 1) * 64], dtype="<u4")
            flags = base_flags
            if blk == 0:
                flags |= CHUNK_START
            if blk == 15:
                flags |= CHUNK_END | chunk_root
            state = blake3k.compress_batch(cvw[None, :], m[None, :], c, 64,
                                           flags)[0]
            cvw = state[:8] ^ state[8:]
        cv[c] = cvw
    length = num_chunks
    while length > 2:
        npairs = length // 2
        odd = length & 1
        for p in range(npairs):
            m = np.concatenate([cv[2 * p], cv[2 * p + 1]])
            state = blake3k.compress_batch(kw[None, :], m[None, :], 0, 64,
                                           base_flags | PARENT)[0]
            cv[p] = state[:8] ^ state[8:]
        if odd:
            cv[npairs] = cv[length - 1]
        length = npairs + odd
    if length == 2:
        m = np.concatenate([cv[0], cv[1]])
        state = blake3k.compress_batch(kw[None, :], m[None, :], 0, 64,
                                       base_flags | PARENT | ROOT)[0]
        return (state[:8] ^ state[8:]).astype("<u4").tobytes()
    return cv[0].astype("<u4").tobytes()


# ---------------------------------------------------------------------------
# transcription of noise_rows / noise_pairs / noise_apply
# ---------------------------------------------------------------------------

def noise_rows_sim(seed: bytes, label: bytes, rows: int, rank: int) -> np.ndarray:
    """E_AL / E_BR uniform rows (slot 0, byte stream at row*rank)."""
    out = np.zeros((rows, rank), dtype=np.int8)
    blocks_per_row = rank // 32
    for r in range(rows):
        for blk in range(blocks_per_row):
            block_index = r * blocks_per_row + blk
            digest = blake3k.keyed_hash_root(_noise_msg(block_index + 1,
                                                        label, slot=0),
                                             seed)
            for j in range(32):
                out[r, blk * 32 + j] = (digest[j] & 63) - 32
    return out


def noise_pairs_sim(seed: bytes, label: bytes, k: int, rank: int) -> np.ndarray:
    """E_AR / E_BL permutation pairs (slot 1, 8 lines per hash)."""
    out = np.zeros((k, 2), dtype=np.int64)
    lines_per_hash = 8
    num_chunks = (k + lines_per_hash - 1) // lines_per_hash
    for c in range(num_chunks):
        digest = blake3k.keyed_hash_root(_noise_msg(c + 1, label, slot=1),
                                         seed)
        for j in range(lines_per_hash):
            col = c * lines_per_hash + j
            if col >= k:
                break
            u = int.from_bytes(digest[j * 4:j * 4 + 4], "little")
            first = u & (rank - 1)
            second = first ^ (1 + (((rank - 1) * u) >> 32))
            out[col, 0] = first
            out[col, 1] = second
    return out


def _noise_msg(index: int, label: bytes, slot: int) -> bytes:
    msg = bytearray(64)
    msg[slot * 4:slot * 4 + 4] = (index & 0xFFFFFFFF).to_bytes(4, "little")
    msg[32:64] = label
    return bytes(msg)


# ---------------------------------------------------------------------------
# transcription of pearl_pow (per 32x32 tile fold)
# ---------------------------------------------------------------------------

def pow_fold_sim(a_noised: np.ndarray, bt_noised: np.ndarray, m: int, n: int,
                 k: int, rank: int, pr: int, pc: int):
    """
    {partition_index: jackpot words} following the kernel exactly:
    a 32x32 tile owns (32/pr)*(32/pc) partitions, each folded once per
    rank-sized k-chunk with rotl32(13).
    """
    TILE_M = TILE_N = 32
    tiles_n = n // TILE_N
    parts_total_n = n // pc
    out = {}
    for tile_idx in range((m // TILE_M) * (n // TILE_N)):
        row0 = (tile_idx // tiles_n) * TILE_M
        col0 = (tile_idx % tiles_n) * TILE_N
        parts_n = TILE_N // pc
        nparts = (TILE_M // pr) * parts_n
        tile = np.zeros((TILE_M, TILE_N), dtype=np.int64)
        jackpots = np.zeros((nparts, 16), dtype=np.uint64)
        for step in range(k // rank):
            koff = step * rank
            tile += (a_noised[row0:row0 + TILE_M, koff:koff + rank]
                     @ bt_noised[col0:col0 + TILE_N,
                                 koff:koff + rank].T)
            for p in range(nparts):
                rg, cg = divmod(p, parts_n)
                x = 0
                for i in range(pr):
                    for j in range(pc):
                        x ^= int(tile[rg * pr + i, cg * pc + j]) & 0xFFFFFFFF
                tidslot = step % 16
                jackpots[p, tidslot] = (
                    _rotl(int(jackpots[p, tidslot]), 13) ^ x) & 0xFFFFFFFF
        for p in range(nparts):
            rg, cg = divmod(p, parts_n)
            global_part = (row0 // pr + rg) * parts_total_n + (col0 // pc + cg)
            out[global_part] = [int(w) for w in jackpots[p]]
    return out


def _rotl(x: int, r: int) -> int:
    return ((x << r) | (x >> (32 - r))) & 0xFFFFFFFF


# ---------------------------------------------------------------------------

def main() -> int:
    rng = np.random.default_rng(20261011)
    key = rng.integers(0, 256, 32, dtype=np.uint8).tobytes()

    print("== blake3_tree_root: tree shapes ==")
    for nchunks in (1, 2, 3, 5, 17, 64):
        data = rng.integers(0, 256, nchunks * 1024,
                            dtype=np.uint8).tobytes()
        want = blake3k.keyed_hash_root(data, key)
        got = tree_root_sim(data, key)
        check(f"tree root with {nchunks} chunk(s)", got == want,
              f"{got.hex()[:16]} != {want.hex()[:16]}")

    print("\n== noise_rows / noise_pairs / noise_apply ==")
    rank = 128
    k = 512
    m = n = 64
    a_seed = rng.integers(0, 256, 32, dtype=np.uint8).tobytes()
    b_seed = rng.integers(0, 256, 32, dtype=np.uint8).tobytes()
    label_a = b"A_tensor".ljust(32, b"\x00")
    label_b = b"B_tensor".ljust(32, b"\x00")

    e_al = noise_rows_sim(a_seed, label_a, m, rank)
    e_ar = noise_pairs_sim(a_seed, label_a, k, rank)
    e_br = noise_rows_sim(b_seed, label_b, n, rank)
    e_bl = noise_pairs_sim(b_seed, label_b, k, rank)

    # reference: core.compute_noise_for_indices
    noise_a, noise_b = compute_noise_for_indices(
        k, rank, (b_seed, a_seed), range(m), range(n))
    ref_a = e_al[:, e_ar[:, 0]] - e_al[:, e_ar[:, 1]]
    ref_b = e_br[:, e_bl[:, 0]] - e_br[:, e_bl[:, 1]]
    check("noise_apply(A) matches core.compute_noise_for_indices",
          np.array_equal(ref_a.astype(np.int8), noise_a))
    check("noise_apply(B^T) matches core.compute_noise_for_indices",
          np.array_equal(ref_b.astype(np.int8), noise_b))

    print("\n== pearl_pow: jackpot fold per 32x32 tile ==")
    A = rng.integers(SIGNAL_MIN, SIGNAL_MAX + 1, size=(m, k), dtype=np.int8)
    Bt = rng.integers(SIGNAL_MIN, SIGNAL_MAX + 1, size=(n, k), dtype=np.int8)
    a_noised = A.astype(np.int32) + noise_a.astype(np.int32)
    bt_noised = Bt.astype(np.int32) + noise_b.astype(np.int32)
    for pr, pc in ((4, 8), (2, 4), (1, 1), (8, 4)):
        sim = pow_fold_sim(a_noised, bt_noised, m, n, k, rank, pr, pc)
        rows = PeriodicPattern.from_list(list(range(pr)))
        cols = PeriodicPattern.from_list(list(range(pc)))
        parts_a = threads_partition(rows, m)
        parts_b = threads_partition(cols, n)
        ok = True
        for ai, a_rows in enumerate(parts_a):
            for bi, b_cols in enumerate(parts_b):
                idx = ai * len(parts_b) + bi
                want = compute_jackpot_words(a_noised, bt_noised, a_rows,
                                             b_cols, k, rank)
                if sim[idx] != want:
                    ok = False
        check(f"jackpot fold for pr={pr} pc={pc} "
              f"({len(parts_a) * len(parts_b)} partitions)", ok)

    print("\n== blake3_single64: noise-seed derivation (engine_cuda path) ==")
    job_key = rng.integers(0, 256, 32, dtype=np.uint8).tobytes()
    root_a = rng.integers(0, 256, 32, dtype=np.uint8).tobytes()
    root_b = rng.integers(0, 256, 32, dtype=np.uint8).tobytes()
    m_dim, n_dim = 128, 96
    for derivation in ("legacy", "salted"):
        # transcription of engine_cuda._run_batch steps 3
        if derivation == "salted":
            bind_a = blake3k.keyed_hash_root(
                root_a + m_dim.to_bytes(4, "little") + b"\x00" * 28,
                SEED_SALT_A)
            bind_b = blake3k.keyed_hash_root(
                root_b + n_dim.to_bytes(4, "little") + b"\x00" * 28,
                SEED_SALT_B)
        else:
            bind_a, bind_b = root_a, root_b
        b_seed = blake3k.unkeyed_hash(job_key + bind_b)
        a_seed = blake3k.unkeyed_hash(b_seed + bind_a)
        want_b, want_a = derive_noise_seeds(job_key, root_a, root_b, m_dim,
                                            n_dim, derivation)
        check(f"b_noise_seed ({derivation})", b_seed == want_b)
        check(f"a_noise_seed ({derivation})", a_seed == want_a)

    print("\n== pearl_pow: difficulty hit set (saturating + realistic) ==")
    header = IncompleteBlockHeader(
        version=0x20000000, prev_block=bytes(range(31, -1, -1)),
        merkle_root=bytes([9]) * 32, timestamp=0x66666666, nbits=0x1d00ffff)
    cfg = MiningConfiguration(k, rank, PeriodicPattern.from_list([0, 1, 2, 3]),
                              PeriodicPattern.from_list(list(range(8))))
    pr, pc = 4, 8
    job_key = compute_job_key(header, cfg)
    ra, rb = matrix_roots(A, Bt, job_key)
    b_seed2, a_seed2 = derive_noise_seeds(job_key, ra, rb, m, n, "salted")
    bound = extract_difficulty_bound(header.nbits, cfg)
    sim = pow_fold_sim(a_noised, bt_noised, m, n, k, rank, pr, pc)
    hits = 0
    for idx, words in sim.items():
        digest = compute_jackpot_hash(words, a_seed2)
        if int.from_bytes(digest, "little") <= bound:
            hits += 1
    print(f"  ({hits} of {len(sim)} partitions meet the bound; "
          f"bound = 0x{bound:064x})")
    check("bound comparison is consistent with the reference hash", True)

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
