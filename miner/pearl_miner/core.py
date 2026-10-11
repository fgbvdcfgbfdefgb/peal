"""
Core Pearl proof-of-work primitives.

Faithful Python ports of the consensus-critical pieces of the ``zk-pow`` and
``pearl-blake3`` crates:

  * :class:`IncompleteBlockHeader` / :class:`MiningConfiguration` (76 / 52
    byte serialisations, exactly as ``to_bytes`` does in Rust)
  * ``job_key = blake3(header || config)``
  * noise-seed derivation (Legacy for cert V1/V2, Salted for cert V3)
  * noise generation (``compute_noise_for_indices``)
  * jackpot hashing and the rank-penalised difficulty bounds

Every constant and byte layout here is pinned by ``miner/selftest.py``, which
cross-checks against the official ``pearl_mining`` Python module when it is
installed (built from the Pearl monorepo with maturin).
"""

from __future__ import annotations

import numpy as np

from . import blake3k
from .patterns import PeriodicPattern

# ---------------------------------------------------------------------------
# Optional fast paths: the ``blake3`` pip package implements exactly the same
# hash (keyed and unkeyed) and is much faster than pure numpy for the large
# commitment hashes.  It is used when available; otherwise blake3k is used.
# ---------------------------------------------------------------------------

try:  # pragma: no cover - availability depends on the environment
    import blake3 as _blake3_pkg

    def _keyed(data: bytes, key: bytes) -> bytes:
        return _blake3_pkg.blake3(data, key=key).digest()

    def _unkeyed(data: bytes) -> bytes:
        return _blake3_pkg.blake3(data).digest()

except Exception:  # pragma: no cover

    def _keyed(data: bytes, key: bytes) -> bytes:
        return blake3k.keyed_hash_root(data, key)

    def _unkeyed(data: bytes) -> bytes:
        return blake3k.unkeyed_hash(data)


def keyed_hash(data: bytes, key: bytes) -> bytes:
    """blake3(data, key=key) - 32-byte keyed digest."""
    return _keyed(data, key)


def unkeyed_hash(data: bytes) -> bytes:
    """blake3(data) - 32-byte digest."""
    return _unkeyed(data)


# ---------------------------------------------------------------------------
# Constants pinned from the Rust sources
# ---------------------------------------------------------------------------

HEADER_SIZE = 76            # IncompleteBlockHeader::SERIALIZED_SIZE
CONFIG_SIZE = 52            # MiningConfiguration::SERIALIZED_SIZE
PENALTY_BASE_RANK = 128     # sanity_checks::PENALTY_BASE_RANK
JACKPOT_SIZE = 16           # pearl_program::JACKPOT_SIZE
LROT_PER_TILE = 13          # pearl_program::LROT_PER_TILE
TILE_D = 16                 # pearl_program::TILE_D
TILE_H = 2                  # pearl_program::TILE_H
DWORD_SIZE = 4
BLAKE3_DIGEST_SIZE = 32
BLAKE3_CHUNK_LEN = 1024
SIGNAL_MIN, SIGNAL_MAX = -64, 64

# Noise generation constants (pearl_noise.rs)
NOISE_RANGE = 128
IDXS_PER_COL = 2
UNIFORM_NOISE_RANGE = NOISE_RANGE // IDXS_PER_COL          # 64
ZERO_POINT_TRANSLATION = UNIFORM_NOISE_RANGE // 2          # 32
RANGE_MASK = UNIFORM_NOISE_RANGE - 1                       # 63

SEED_LABEL_A = b"A_tensor".ljust(32, b"\x00")
SEED_LABEL_B = b"B_tensor".ljust(32, b"\x00")

# V3 domain-separation salts (seed.rs); recomputed below for parity checks.
SEED_SALT_A = unkeyed_hash(b"pearl/cert-v3/noise-seed/A")
SEED_SALT_B = unkeyed_hash(b"pearl/cert-v3/noise-seed/B")

# Certificate version -> noise-seed derivation (plain_proof.rs)
CERT_V1, CERT_V2, CERT_V3 = 1, 2, 3
DERIVATION_LEGACY = "legacy"    # V1 / V2
DERIVATION_SALTED = "salted"    # V3


def derivation_for_cert_version(cert_version: int) -> str:
    if cert_version in (CERT_V1, CERT_V2):
        return DERIVATION_LEGACY
    if cert_version == CERT_V3:
        return DERIVATION_SALTED
    raise ValueError(f"unknown certificate version: {cert_version}")


# ---------------------------------------------------------------------------
# Block header / mining configuration
# ---------------------------------------------------------------------------

class IncompleteBlockHeader:
    """The 76-byte header a Pearl proof commits to (no proof commitment)."""

    __slots__ = ("version", "prev_block", "merkle_root", "timestamp", "nbits")

    def __init__(self, version: int, prev_block: bytes, merkle_root: bytes,
                 timestamp: int, nbits: int):
        self.version = version
        self.prev_block = prev_block        # internal (little-endian) order
        self.merkle_root = merkle_root
        self.timestamp = timestamp
        self.nbits = nbits

    @classmethod
    def from_bytes(cls, data: bytes) -> "IncompleteBlockHeader":
        if len(data) != HEADER_SIZE:
            raise ValueError(f"header must be {HEADER_SIZE} bytes, got {len(data)}")
        return cls(
            version=int.from_bytes(data[0:4], "little"),
            prev_block=data[4:36][::-1],          # stored reversed
            merkle_root=data[36:68][::-1],
            timestamp=int.from_bytes(data[68:72], "little"),
            nbits=int.from_bytes(data[72:76], "little"),
        )

    def to_bytes(self) -> bytes:
        return (
            self.version.to_bytes(4, "little")
            + self.prev_block[::-1]
            + self.merkle_root[::-1]
            + self.timestamp.to_bytes(4, "little")
            + self.nbits.to_bytes(4, "little")
        )

    @property
    def prev_block_hex(self) -> str:
        """Display order (reversed back), like btcd chainhash.String()."""
        return self.prev_block[::-1].hex()

    def __repr__(self):
        return (f"IncompleteBlockHeader(version={self.version}, "
                f"prev_block={self.prev_block_hex}, "
                f"merkle_root={self.merkle_root[::-1].hex()}, "
                f"timestamp={self.timestamp}, nbits=0x{self.nbits:08x})")


class MiningConfiguration:
    """The 52-byte committed mining configuration."""

    __slots__ = ("common_dim", "rank", "rows_pattern", "cols_pattern")

    def __init__(self, common_dim: int, rank: int,
                 rows_pattern: PeriodicPattern, cols_pattern: PeriodicPattern):
        self.common_dim = common_dim          # k
        self.rank = rank
        self.rows_pattern = rows_pattern
        self.cols_pattern = cols_pattern

    def to_bytes(self) -> bytes:
        return (
            self.common_dim.to_bytes(4, "little")
            + self.rank.to_bytes(2, "little")
            + (0).to_bytes(2, "little")              # mma_type = Int7xInt7ToInt32
            + self.rows_pattern.to_bytes()
            + self.cols_pattern.to_bytes()
            + b"\x00" * 32                            # reserved / MoE trailer
        )

    @property
    def tile_size(self) -> int:
        return self.rows_pattern.size() * self.cols_pattern.size()

    @property
    def dot_product_length(self) -> int:
        return self.common_dim - self.common_dim % self.rank

    def sanity_check(self, m: int, n: int) -> None:
        """Mirror of the relevant parts of public_params_sanity_check()."""
        k, r = self.common_dim, self.rank
        h, w = self.rows_pattern.size(), self.cols_pattern.size()
        if not (isinstance(r, int) and r > 0 and (r & (r - 1)) == 0):
            raise ValueError(f"rank must be a power of two, got {r}")
        if not (32 <= r <= 1024):
            raise ValueError(f"rank must be in 2^5..2^10, got {r}")
        if r % TILE_D != 0:
            raise ValueError(f"rank must be divisible by {TILE_D}")
        if k > (1 << 16):
            raise ValueError("k must be <= 2^16")
        if k % 64 != 0:
            raise ValueError("k must be divisible by 64")
        if k > 4 * r * r:
            raise ValueError(f"k must be <= 4r^2 ({4 * r * r})")
        if k < 16 * r:
            raise ValueError(f"k must be >= 16r ({16 * r})")
        if k < 1024:
            raise ValueError("k must be >= 1024")
        if h % TILE_H or w % TILE_H:
            raise ValueError(f"tile dims must be divisible by {TILE_H}")
        if not (32 <= h * w <= 256):
            raise ValueError(f"tile size h*w must be in [32, 256], got {h * w}")
        if self.dot_product_length % DWORD_SIZE != 0:
            raise ValueError("dot_product_length must be divisible by 4")
        if m > (1 << 24) or n > (1 << 24):
            raise ValueError("m and n must be <= 2^24")
        if (h + w) * self.dot_product_length > (1 << 22):
            raise ValueError("worker input exceeds 4 MiB")
        if m % self.rows_pattern.period() != 0:
            raise ValueError("m must be divisible by the rows pattern period")
        if n % self.cols_pattern.period() != 0:
            raise ValueError("n must be divisible by the cols pattern period")


# ---------------------------------------------------------------------------
# Job key and noise seeds
# ---------------------------------------------------------------------------

def compute_job_key(header: IncompleteBlockHeader,
                    config: MiningConfiguration) -> bytes:
    """blake3(header.to_bytes() || config.to_bytes()) - the tree key."""
    return unkeyed_hash(header.to_bytes() + config.to_bytes())


def _bind_roots(hash_a: bytes, hash_b: bytes, m: int, n: int,
                derivation: str):
    """seed.rs SeedDerivation::bind_roots."""
    if derivation == DERIVATION_LEGACY:
        return hash_a, hash_b
    if derivation == DERIVATION_SALTED:
        a = keyed_hash(hash_a + m.to_bytes(4, "little") + b"\x00" * 28, SEED_SALT_A)
        b = keyed_hash(hash_b + n.to_bytes(4, "little") + b"\x00" * 28, SEED_SALT_B)
        return a, b
    raise ValueError(f"unknown derivation: {derivation}")


def derive_noise_seeds(job_key: bytes, hash_a: bytes, hash_b: bytes,
                       m: int, n: int, derivation: str):
    """
    Miner-side noise-seed derivation (mine.rs
    ``compute_commitment_hash_with_offsets``, non-MoE path).

    ``hash_a``/``hash_b`` are the keyed BLAKE3 Merkle roots of the padded A and
    B^T matrices.  Returns ``(b_noise_seed, a_noise_seed)``.
    """
    hash_a, hash_b = _bind_roots(hash_a, hash_b, m, n, derivation)
    b_noise_seed = unkeyed_hash(job_key + hash_b)
    a_noise_seed = unkeyed_hash(b_noise_seed + hash_a)
    return b_noise_seed, a_noise_seed


def matrix_roots(A: np.ndarray, Bt: np.ndarray, job_key: bytes):
    """
    Commitment hashes: keyed BLAKE3 of the chunk-padded, row-major matrix
    bytes.  ``A`` is (m, k) int8, ``Bt`` is (n, k) int8 (B transposed).
    """
    a_padded = blake3k.pad_to_chunk_boundary(np.ascontiguousarray(A, dtype=np.int8).tobytes())
    b_padded = blake3k.pad_to_chunk_boundary(np.ascontiguousarray(Bt, dtype=np.int8).tobytes())
    return keyed_hash(a_padded, job_key), keyed_hash(b_padded, job_key)


# ---------------------------------------------------------------------------
# Noise generation (pearl_noise.rs)
# ---------------------------------------------------------------------------

def _get_random_hash(index: int, seed: bytes, key: bytes, prepend_index: int) -> bytes:
    """64-byte message, single-block keyed hash (root output)."""
    msg = bytearray(64)
    value = (1 + index) & 0xFFFFFFFF
    msg[prepend_index * 4: prepend_index * 4 + 4] = value.to_bytes(4, "little")
    msg[32:64] = seed
    return keyed_hash(bytes(msg), key)


def generate_uniform_random_matrix(seed: bytes, key: bytes, row_indices,
                                   num_cols: int) -> np.ndarray:
    """
    Deterministic uniform matrix rows (values in [-32, 31]).

    Row ``r`` draws bytes from the hash stream at global positions
    ``[r*num_cols, (r+1)*num_cols)``; each byte maps to ``(b & 63) - 32``.
    """
    rows = list(row_indices)
    out = np.zeros((len(rows), num_cols), dtype=np.int8)
    for ri, row_idx in enumerate(rows):
        start = row_idx * num_cols
        end = start + num_cols
        first_block = start // BLAKE3_DIGEST_SIZE
        last_block = (end + BLAKE3_DIGEST_SIZE - 1) // BLAKE3_DIGEST_SIZE
        for block in range(first_block, last_block):
            digest = _get_random_hash(block, seed, key, prepend_index=0)
            base = block * BLAKE3_DIGEST_SIZE
            for j, byte in enumerate(digest):
                pos = base + j
                if start <= pos < end:
                    out[ri, pos - start] = (byte & RANGE_MASK) - ZERO_POINT_TRANSLATION
    return out


def generate_permutation_matrix(seed: bytes, key: bytes, k: int,
                                noise_rank: int) -> np.ndarray:
    """
    Sparse permutation pairs (k, 2): row j has +1 at pair[j][0] and -1 at
    pair[j][1].
    """
    lines_per_hash = BLAKE3_DIGEST_SIZE // 4
    rank_mask = noise_rank - 1
    out = np.zeros((k, 2), dtype=np.int64)
    for i in range((k + lines_per_hash - 1) // lines_per_hash):
        digest = _get_random_hash(i, seed, key, prepend_index=1)
        for j in range(min(lines_per_hash, k - i * lines_per_hash)):
            u32 = int.from_bytes(digest[j * 4: j * 4 + 4], "little")
            first = u32 & rank_mask
            second = first ^ (1 + (((noise_rank - 1) * u32) >> 32))
            out[i * lines_per_hash + j, 0] = first
            out[i * lines_per_hash + j, 1] = second
    return out


def compute_noise_for_indices(k: int, noise_rank: int,
                              commitment_hash, a_rows_indices,
                              b_cols_indices):
    """
    NOISE_A (len(a_rows), k) and NOISE_B^T (len(b_cols), k) as int8 arrays
    (port of ``compute_noise_for_indices``).
    """
    b_noise_seed, a_noise_seed = commitment_hash
    a_rows = list(a_rows_indices)
    b_cols = list(b_cols_indices)

    e_al = generate_uniform_random_matrix(SEED_LABEL_A, a_noise_seed, a_rows, noise_rank)
    e_ar = generate_permutation_matrix(SEED_LABEL_A, a_noise_seed, k, noise_rank)
    e_bl = generate_permutation_matrix(SEED_LABEL_B, b_noise_seed, k, noise_rank)
    e_br = generate_uniform_random_matrix(SEED_LABEL_B, b_noise_seed, b_cols, noise_rank)

    # NOISE_A[i][j] = E_AL[i][first_j] - E_AL[i][second_j]
    noise_a = e_al[:, e_ar[:, 0]] - e_al[:, e_ar[:, 1]]
    # NOISE_B^T[c][j] = E_BR[c][first_j] - E_BR[c][second_j]
    noise_b = e_br[:, e_bl[:, 0]] - e_br[:, e_bl[:, 1]]
    return noise_a.astype(np.int8), noise_b.astype(np.int8)


# ---------------------------------------------------------------------------
# Jackpot hashing and difficulty
# ---------------------------------------------------------------------------

def compute_jackpot_hash(jackpot, commitment_hash: bytes) -> bytes:
    """blake3(16 little-endian u32 words, key=a_noise_seed)."""
    words = np.asarray(jackpot, dtype=np.uint32).astype("<u4")
    return keyed_hash(words.tobytes(), commitment_hash)


def nbits_to_target(nbits: int) -> int:
    """Bitcoin compact encoding -> absolute target (U256 as Python int)."""
    exponent = (nbits >> 24) & 0xFF
    mantissa = nbits & 0x00FFFFFF
    if mantissa == 0 or exponent == 0:
        return 0
    if mantissa & 0x00800000:
        return 0
    if exponent <= 3:
        return mantissa >> (8 * (3 - exponent))
    return mantissa << (8 * (exponent - 3))


U256_MAX = (1 << 256) - 1


def _scale(base: int, adjustment_factor: int) -> int:
    if adjustment_factor <= 0:
        raise ValueError("adjustment factor must be > 0")
    if base > U256_MAX // adjustment_factor:
        return U256_MAX
    return base * adjustment_factor


def difficulty_adjustment_factor(config: MiningConfiguration) -> int:
    return config.tile_size * config.dot_product_length


def penalized_adjustment_factor(config: MiningConfiguration) -> int:
    return (config.tile_size
            * (config.dot_product_length // config.rank)
            * PENALTY_BASE_RANK)


def extract_difficulty_bound(nbits: int, config: MiningConfiguration) -> int:
    """Unpenalised bound (saturates at 2^256-1)."""
    return _scale(nbits_to_target(nbits), difficulty_adjustment_factor(config))


def penalized_target_bound(target: int, config: MiningConfiguration):
    """Rank-penalised bound, or None when degenerate / overflowing."""
    factor = penalized_adjustment_factor(config)
    if factor == 0 or target > U256_MAX // factor:
        return None
    return target * factor


def mining_bound(nbits: int, config: MiningConfiguration):
    """
    The bound a miner must satisfy: the rank-penalised bound (what consensus
    enforces at and after the rank-penalty fork).  At rank == 128 this is
    identical to the unpenalised bound.
    """
    bound = penalized_target_bound(nbits_to_target(nbits), config)
    if bound is None:
        raise ValueError("degenerate mining configuration: no usable target bound")
    return bound


def hash_meets_bound(digest: bytes, bound: int) -> bool:
    """Difficulty check: little-endian u256 interpretation of the digest."""
    return int.from_bytes(digest, "little") <= bound
