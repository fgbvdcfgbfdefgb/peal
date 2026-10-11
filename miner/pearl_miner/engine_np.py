"""
Batched mining engine (numpy / CPU).

Runs many independent mining trials at once:

  * random A (m, k) and B^T (n, k) int8 matrices per trial
  * commitment roots via keyed BLAKE3 (the ``blake3`` package when available)
  * noise matrices via batched keyed BLAKE3 compressions
  * noised GEMM accumulated in exact int32 (float32 matmuls are exact here:
    every partial sum stays below 2^24)
  * the jackpot fold and the target check, fully vectorised

Only the rare "hit" path drops back to per-trial Python to build the Merkle
proofs.
"""

from __future__ import annotations

import numpy as np

from . import blake3k
from .core import (
    MiningConfiguration, IncompleteBlockHeader, compute_job_key,
    derive_noise_seeds, extract_difficulty_bound, derivation_for_cert_version,
    BLAKE3_DIGEST_SIZE, JACKPOT_SIZE, LROT_PER_TILE, SIGNAL_MAX, SIGNAL_MIN,
    U256_MAX,
)
from .patterns import threads_partition

try:  # fast path for the big commitment hashes
    import blake3 as _blake3_pkg
except Exception:  # pragma: no cover
    _blake3_pkg = None

_ROOT_FLAGS = blake3k.CHUNK_START | blake3k.CHUNK_END | blake3k.ROOT | blake3k.KEYED_HASH

# Noise-seed labels as message words (pearl_noise.rs SEED_LABEL_A / _B).
_LABEL_A_WORDS = np.frombuffer(
    b"A_tensor".ljust(32, b"\x00"), dtype="<u4").copy()
_LABEL_B_WORDS = np.frombuffer(
    b"B_tensor".ljust(32, b"\x00"), dtype="<u4").copy()


def _rotl32(x: np.ndarray, r: int) -> np.ndarray:
    return (x << np.uint32(r)) | (x >> np.uint32(32 - r))


def _hash_many_messages(messages: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """
    Batched keyed BLAKE3 of single-block (64-byte) messages.

    ``messages``: (N, 16) uint32 block words; ``keys``: (N, 8) uint32 key words.
    Returns the (N, 8) uint32 root-output words.
    """
    state = blake3k.compress_batch(keys, messages, 0, 64, _ROOT_FLAGS)
    return state[:, :8] ^ state[:, 8:]


class NumpyEngine:
    """Vectorised Pearl PoW search."""

    def __init__(self, m: int, n: int, k: int, rank: int,
                 rows_pattern, cols_pattern, batch_size: int = 256,
                 seed: int | None = None):
        self.m, self.n, self.k, self.rank = m, n, k, rank
        self.rows_pattern = rows_pattern
        self.cols_pattern = cols_pattern
        self.config = MiningConfiguration(k, rank, rows_pattern, cols_pattern)
        self.config.sanity_check(m, n)
        self.batch_size = int(batch_size)

        self.partitions_a = threads_partition(rows_pattern, m)
        self.partitions_b = threads_partition(cols_pattern, n)
        self.P = len(self.partitions_a) * len(self.partitions_b)
        self.pr = rows_pattern.size()
        self.pc = cols_pattern.size()

        # Fast path is possible when every partition is a contiguous aligned
        # block (dense patterns): the fold then works on the full (m, n) tile.
        self.dense = (
            rows_pattern.period() == rows_pattern.size()
            and rows_pattern.to_list() == list(range(rows_pattern.size()))
            and cols_pattern.period() == cols_pattern.size()
            and cols_pattern.to_list() == list(range(cols_pattern.size()))
        )
        if self.dense:
            self.P_a = len(self.partitions_a)
            self.P_b = len(self.partitions_b)
        else:
            self.rows_flat = np.array(
                [r for part in self.partitions_a for r in part], dtype=np.intp)
            self.cols_flat = np.array(
                [c for part in self.partitions_b for c in part], dtype=np.intp)
            self.group = max(1, min(self.P, 2_000_000 // max(1, self.pr * self.pc)))

        self.rng = np.random.default_rng(seed)
        self._job_key = None
        self._bound = None

    # -- job setup ----------------------------------------------------------

    def set_job(self, header: IncompleteBlockHeader, cert_version: int):
        self._job_key = compute_job_key(header, self.config)
        self._bound = extract_difficulty_bound(header.nbits, self.config)
        self._derivation = derivation_for_cert_version(cert_version)

    @property
    def job_key(self) -> bytes:
        if self._job_key is None:
            raise RuntimeError("call set_job() first")
        return self._job_key

    # -- hashing helpers ----------------------------------------------------

    def _commitment_roots(self, A: np.ndarray, Bt: np.ndarray) -> tuple:
        """Keyed BLAKE3 roots of the chunk-padded matrices, per trial."""
        key = self.job_key
        hashes_a = []
        hashes_b = []
        for trial in range(A.shape[0]):
            a_padded = blake3k.pad_to_chunk_boundary(
                np.ascontiguousarray(A[trial], dtype=np.int8).tobytes())
            b_padded = blake3k.pad_to_chunk_boundary(
                np.ascontiguousarray(Bt[trial], dtype=np.int8).tobytes())
            if _blake3_pkg is not None:
                hashes_a.append(_blake3_pkg.blake3(a_padded, key=key).digest())
                hashes_b.append(_blake3_pkg.blake3(b_padded, key=key).digest())
            else:  # pragma: no cover - fallback
                hashes_a.append(blake3k.keyed_hash_root(a_padded, key))
                hashes_b.append(blake3k.keyed_hash_root(b_padded, key))
        return hashes_a, hashes_b

    def _noise_bases(self, a_seeds: np.ndarray, b_seeds: np.ndarray):
        """
        Small per-trial noise bases shared by every k-chunk:

            e_al (B, m, rank), e_br (B, n, rank)   uniform rows
            e_ar (B, k, 2),    e_bl (B, k, 2)      permutation pairs
        """
        B = a_seeds.shape[0]
        m, n, k, rank = self.m, self.n, self.k, self.rank
        a_key_words = np.ascontiguousarray(a_seeds.view("<u4"))   # (B, 8)
        b_key_words = np.ascontiguousarray(b_seeds.view("<u4"))

        blocks_per_row = rank // BLAKE3_DIGEST_SIZE                # 128/32 = 4
        a_words = self._uniform_matrix_words(a_key_words, m, blocks_per_row,
                                             _LABEL_A_WORDS)
        e_al = _words_to_values(a_words).reshape(B, m, rank)
        b_words = self._uniform_matrix_words(b_key_words, n, blocks_per_row,
                                             _LABEL_B_WORDS)
        e_br = _words_to_values(b_words).reshape(B, n, rank)

        lines_per_hash = BLAKE3_DIGEST_SIZE // 4                   # 8
        num_chunks = (k + lines_per_hash - 1) // lines_per_hash
        e_ar = self._permutation_pairs(a_key_words, num_chunks, k,
                                       lines_per_hash, rank, _LABEL_A_WORDS)
        e_bl = self._permutation_pairs(b_key_words, num_chunks, k,
                                       lines_per_hash, rank, _LABEL_B_WORDS)
        return e_al, e_br, e_ar, e_bl

    def _noise_chunk(self, e_rows: np.ndarray, e_perm: np.ndarray,
                     rows: np.ndarray, lo: int, ll: int) -> np.ndarray:
        """
        NOISE for rows ``rows`` and columns ``[lo, ll)`` as (B, len(rows),
        ll-lo) int8:

            noise[b, i, j] = e_rows[b, i, e_perm[b, j, 0]]
                           - e_rows[b, i, e_perm[b, j, 1]]
        """
        B = e_rows.shape[0]
        bidx = np.arange(B)[:, None, None]
        ridx = rows[None, :, None]
        first = e_perm[:, None, lo:ll, 0]
        second = e_perm[:, None, lo:ll, 1]
        noise = e_rows[bidx, ridx, first] - e_rows[bidx, ridx, second]
        return noise.astype(np.int8)

    def _noise_matrices(self, a_seeds: np.ndarray, b_seeds: np.ndarray):
        """
        Full noise matrices (used by the selftest reference comparison).

        ``a_seeds``/``b_seeds``: (B, 32) uint8 noise seeds.
        Returns ``noise_a`` (B, m, k) and ``noise_b`` (B, n, k) int8.
        """
        m, n, k, rank = self.m, self.n, self.k, self.rank
        e_al, e_br, e_ar, e_bl = self._noise_bases(a_seeds, b_seeds)
        all_rows = np.arange(m)
        all_cols = np.arange(n)
        noise_a = np.empty((a_seeds.shape[0], m, k), dtype=np.int8)
        noise_b = np.empty((b_seeds.shape[0], n, k), dtype=np.int8)
        for lo in range(0, k, rank):
            ll = min(lo + rank, k)
            noise_a[:, :, lo:ll] = self._noise_chunk(e_al, e_ar, all_rows,
                                                     lo, ll)
            noise_b[:, :, lo:ll] = self._noise_chunk(e_br, e_bl, all_cols,
                                                     lo, ll)
        return noise_a, noise_b

    def _uniform_matrix_words(self, key_words: np.ndarray, rows: int,
                              blocks_per_row: int,
                              label_words: np.ndarray) -> np.ndarray:
        """(B, rows * blocks_per_row, 8) root-output words for E_AL / E_BR."""
        B = key_words.shape[0]
        count = rows * blocks_per_row
        blocks = np.arange(count, dtype=np.uint32)
        words = np.zeros((B, count, 16), dtype=np.uint32)
        # prepend slot 0 holds (1 + block_index) as i32 LE; the message seed
        # field holds the tensor label; the blake3 key is the noise seed.
        words[:, :, 0] = (blocks[None, :] + np.uint32(1))
        words[:, :, 8:16] = label_words[None, None, :]
        cv = np.repeat(key_words, count, axis=0)
        out = _hash_many_messages(words.reshape(B * count, 16), cv)
        return out.reshape(B, count, 8)

    def _permutation_pairs(self, key_words: np.ndarray, num_chunks: int, k: int,
                           lines_per_hash: int, rank: int,
                           label_words: np.ndarray) -> np.ndarray:
        """(B, k, 2) int64 permutation pairs."""
        B = key_words.shape[0]
        chunks = np.arange(num_chunks, dtype=np.uint32)
        words = np.zeros((B, num_chunks, 16), dtype=np.uint32)
        # prepend slot 1 holds (1 + chunk_index) as i32 LE; the message seed
        # field holds the tensor label; the blake3 key is the noise seed.
        words[:, :, 1] = chunks[None, :] + np.uint32(1)
        words[:, :, 8:16] = label_words[None, None, :]
        cv = np.repeat(key_words, num_chunks, axis=0)
        out = _hash_many_messages(words.reshape(B * num_chunks, 16), cv)
        # out: (B, num_chunks, 8) digest words; each word is one u32 "line".
        lines = out
        u32 = lines.astype(np.uint64)
        rank_mask = np.uint64(rank - 1)
        first = (u32 & rank_mask).astype(np.int64)
        second = first ^ (1 + (((np.uint64(rank - 1) * u32) >> np.uint64(32)))).astype(np.int64)
        pairs = np.stack([first, second], axis=-1)          # (B, num_chunks, 8, 2)
        pairs = pairs.reshape(B, num_chunks * lines_per_hash, 2)
        return pairs[:, :k, :]

    # -- main loop ----------------------------------------------------------

    def run_batch(self, max_hits: int | None = None):
        """
        Mine one batch of trials on the current job.

        Returns a list of hit dicts: ``{A, Bt, a_rows, b_cols, job_key,
        jackpot, jackpot_hash}`` (one per (trial, partition) that met the
        target).
        """
        B = self.batch_size
        m, n, k, rank = self.m, self.n, self.k, self.rank

        A = self.rng.integers(SIGNAL_MIN, SIGNAL_MAX + 1, size=(B, m, k),
                              dtype=np.int8)
        Bt = self.rng.integers(SIGNAL_MIN, SIGNAL_MAX + 1, size=(B, n, k),
                               dtype=np.int8)

        hashes_a, hashes_b = self._commitment_roots(A, Bt)
        a_seeds = np.empty((B, 32), dtype=np.uint8)
        b_seeds = np.empty((B, 32), dtype=np.uint8)
        for t in range(B):
            b_seed, a_seed = derive_noise_seeds(
                self.job_key, hashes_a[t], hashes_b[t], m, n, self._derivation)
            b_seeds[t] = np.frombuffer(b_seed, dtype=np.uint8)
            a_seeds[t] = np.frombuffer(a_seed, dtype=np.uint8)

        e_al, e_br, e_ar, e_bl = self._noise_bases(a_seeds, b_seeds)

        jackpot = np.zeros((B, self.P, JACKPOT_SIZE), dtype=np.uint32)
        a_key_words = np.ascontiguousarray(a_seeds.view("<u4"))

        if self.dense:
            tile = np.zeros((B, m, n), dtype=np.int32)
            all_rows = np.arange(m)
            all_cols = np.arange(n)
            for step, ll in enumerate(range(rank, k + 1, rank)):
                lo = ll - rank
                noise_a = self._noise_chunk(e_al, e_ar, all_rows, lo, ll)
                noise_b = self._noise_chunk(e_br, e_bl, all_cols, lo, ll)
                a_chunk = A[:, :, lo:ll].astype(np.float32) + noise_a
                b_chunk = Bt[:, :, lo:ll].astype(np.float32) + noise_b
                partial = np.matmul(a_chunk, b_chunk.transpose(0, 2, 1))
                tile += partial.astype(np.int32)
                tiles = (tile.reshape(B, self.P_a, self.pr, self.P_b, self.pc)
                         .transpose(0, 1, 3, 2, 4)
                         .reshape(B, self.P, self.pr, self.pc))
                xored = np.bitwise_xor.reduce(
                    tiles.view(np.uint32), axis=(2, 3))
                tid = step % JACKPOT_SIZE
                jackpot[:, :, tid] = (_rotl32(jackpot[:, :, tid], LROT_PER_TILE)
                                      ^ xored)
        else:
            tiles_acc = np.zeros((B, self.P, self.pr, self.pc), dtype=np.int32)
            for g0 in range(0, self.P, self.group):
                g1 = min(self.P, g0 + self.group)
                rows_idx = self.rows_flat[g0 * self.pr: g1 * self.pr]
                cols_idx = self.cols_flat[g0 * self.pc: g1 * self.pc]
                acc = np.zeros((B, g1 - g0, self.pr, self.pc), dtype=np.int32)
                for step, ll in enumerate(range(rank, k + 1, rank)):
                    lo = ll - rank
                    noise_a = self._noise_chunk(e_al, e_ar, rows_idx, lo, ll)
                    noise_b = self._noise_chunk(e_br, e_bl, cols_idx, lo, ll)
                    a_sel = (A[:, rows_idx, lo:ll].astype(np.float32)
                             + noise_a)
                    b_sel = (Bt[:, cols_idx, lo:ll].astype(np.float32)
                             + noise_b)
                    a_sel = a_sel.reshape(B, g1 - g0, self.pr, -1)
                    b_sel = b_sel.reshape(B, g1 - g0, self.pc, -1)
                    acc += np.matmul(a_sel, b_sel.transpose(0, 1, 3, 2)).astype(np.int32)
                    xored = np.bitwise_xor.reduce(acc.view(np.uint32), axis=(2, 3))
                    tid = step % JACKPOT_SIZE
                    jackpot[:, g0:g1, tid] = (
                        _rotl32(jackpot[:, g0:g1, tid], LROT_PER_TILE) ^ xored)
                tiles_acc[:, g0:g1] = acc

        # Jackpot hashes: keyed BLAKE3 of the 16 u32 LE words per partition.
        block_words = jackpot.reshape(B * self.P, JACKPOT_SIZE)
        cv = np.repeat(a_key_words, self.P, axis=0)
        digest_words = _hash_many_messages(block_words, cv)          # (B*P, 8)
        digests = digest_words.astype("<u4").tobytes()
        digests = np.frombuffer(digests, dtype=np.uint8).reshape(B, self.P, 32)

        hit_mask = _le256_within(digests, self._bound)
        hits = []
        for b, p in np.argwhere(hit_mask):
            b, p = int(b), int(p)
            ai, bi = divmod(p, len(self.partitions_b))
            a_rows = self.partitions_a[ai]
            b_cols = self.partitions_b[bi]
            hits.append({
                "A": A[b],
                "Bt": Bt[b],
                "a_rows": list(a_rows),
                "b_cols": list(b_cols),
                "job_key": self.job_key,
                "jackpot": [int(w) for w in jackpot[b, p]],
                "jackpot_hash": digests[b, p].tobytes(),
            })
            if max_hits is not None and len(hits) >= max_hits:
                break
        return hits


def _words_to_values(words: np.ndarray) -> np.ndarray:
    """(..., 8) u32 words -> bytes -> (byte & 63) - 32 as int8."""
    raw = words.astype("<u4").tobytes()
    arr = np.frombuffer(raw, dtype=np.uint8)
    return ((arr & 63).astype(np.int8) - 32)


def _le256_within(digests: np.ndarray, bound: int) -> np.ndarray:
    """
    Vectorised ``int.from_bytes(digest, 'little') <= bound`` for 32-byte
    digests, compared as 4 little-endian u64 words from MSW to LSW.
    """
    bound = min(int(bound), U256_MAX)
    if bound < 0:
        return np.zeros(digests.shape[:2], dtype=bool)
    words = digests.view("<u8")                      # (..., 4) LE u64
    eq = np.ones(digests.shape[:2], dtype=bool)
    less = np.zeros(digests.shape[:2], dtype=bool)
    for i in range(3, -1, -1):
        bw = (bound >> (64 * i)) & 0xFFFFFFFFFFFFFFFF
        w = words[..., i].astype(np.uint64)
        less |= eq & (w < np.uint64(bw))
        eq &= (w == np.uint64(bw))
    return less | eq
