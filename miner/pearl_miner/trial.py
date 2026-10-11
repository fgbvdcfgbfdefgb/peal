"""
Reference single-trial mining loop - a direct port of
``zk_pow::ffi::mine::try_mine_one`` (non-MoE / dense path).

This module favours clarity over speed; the batched :mod:`engine` module is
what the miner actually runs.  ``selftest.py`` checks that both agree and that
both match the official Rust implementation.
"""

from __future__ import annotations

import numpy as np

from . import blake3k
from .core import (
    MiningConfiguration, IncompleteBlockHeader, compute_job_key,
    derive_noise_seeds, matrix_roots, compute_noise_for_indices,
    compute_jackpot_hash, hash_meets_bound, extract_difficulty_bound,
    derivation_for_cert_version,
    JACKPOT_SIZE, LROT_PER_TILE, SIGNAL_MAX, SIGNAL_MIN,
)
from .patterns import threads_partition


def _rotl32(x: int, r: int) -> int:
    return ((x << r) | (x >> (32 - r))) & 0xFFFFFFFF


def random_matrices(rng: np.random.Generator, m: int, n: int, k: int):
    """A (m, k) and B^T (n, k) int8 matrices with entries in [-64, 64]."""
    A = rng.integers(SIGNAL_MIN, SIGNAL_MAX + 1, size=(m, k), dtype=np.int8)
    Bt = rng.integers(SIGNAL_MIN, SIGNAL_MAX + 1, size=(n, k), dtype=np.int8)
    return A, Bt


def compute_jackpot_words(a_noised_i32: np.ndarray, bt_noised_i32: np.ndarray,
                          a_rows, b_cols, k: int, rank: int):
    """
    The 16 jackpot words for one (a_rows, b_cols) tile.

    Accumulates the tile over the full k in rank-sized steps; after every
    step the running tile is XOR-folded into ``jackpot[tid]`` with a
    rotate-left by ``LROT_PER_TILE``.
    """
    tile_h, tile_w = len(a_rows), len(b_cols)
    tile = np.zeros((tile_h, tile_w), dtype=np.int64)
    jackpot = [0] * JACKPOT_SIZE
    a_rows_arr = np.asarray(a_rows)
    b_cols_arr = np.asarray(b_cols)
    for step, ll in enumerate(range(rank, k + 1, rank)):
        lo = ll - rank
        tile += (a_noised_i32[np.ix_(a_rows_arr, np.arange(lo, ll))].astype(np.int64)
                 @ bt_noised_i32[np.ix_(b_cols_arr, np.arange(lo, ll))].astype(np.int64).T)
        xored = 0
        flat = tile.reshape(-1)
        for v in flat:
            xored ^= int(v) & 0xFFFFFFFF
        tid = step % JACKPOT_SIZE
        jackpot[tid] = _rotl32(jackpot[tid], LROT_PER_TILE) ^ xored
    return jackpot


def try_mine_one(rng: np.random.Generator, m: int, n: int, k: int,
                 header: IncompleteBlockHeader, config: MiningConfiguration,
                 cert_version: int):
    """
    One mining attempt.  Returns ``None`` or a dict describing the hit:

    ``{a_rows, b_cols, A, Bt, jackpot, jackpot_hash, a_noise_seed,
    b_noise_seed, job_key}``
    """
    rank = config.rank
    A, Bt = random_matrices(rng, m, n, k)

    job_key = compute_job_key(header, config)
    hash_a, hash_b = matrix_roots(A, Bt, job_key)
    derivation = derivation_for_cert_version(cert_version)
    b_noise_seed, a_noise_seed = derive_noise_seeds(
        job_key, hash_a, hash_b, m, n, derivation)

    noise_a, noise_b = compute_noise_for_indices(
        k, rank, (b_noise_seed, a_noise_seed), range(m), range(n))

    a_noised = A.astype(np.int32) + noise_a.astype(np.int32)
    # Bt_noised[c][l] = Bt[c][l] + noise_b[c][l]
    bt_noised = Bt.astype(np.int32) + noise_b.astype(np.int32)

    bound = extract_difficulty_bound(header.nbits, config)

    for a_rows in threads_partition(config.rows_pattern, m):
        for b_cols in threads_partition(config.cols_pattern, n):
            jackpot = compute_jackpot_words(a_noised, bt_noised, a_rows, b_cols, k, rank)
            jackpot_hash = compute_jackpot_hash(jackpot, a_noise_seed)
            if hash_meets_bound(jackpot_hash, bound):
                return {
                    "a_rows": list(a_rows),
                    "b_cols": list(b_cols),
                    "A": A,
                    "Bt": Bt,
                    "jackpot": jackpot,
                    "jackpot_hash": jackpot_hash,
                    "a_noise_seed": a_noise_seed,
                    "b_noise_seed": b_noise_seed,
                    "job_key": job_key,
                }
    return None
