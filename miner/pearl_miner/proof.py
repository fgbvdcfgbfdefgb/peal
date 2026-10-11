"""
PlainProof construction, wire serialisation and local verification.

``PlainProof`` is the "share" a Pearl miner produces: Merkle membership proofs
for the rows of A and B^T that produced a jackpot hash meeting the target.
The gateway turns it into a ZK proof and a full block; this module produces
and validates the bytes that travel over GitHub.

Wire format (``bincode`` fixint, little-endian - must match
``PlainProof::to_base64`` in ``zk-pow`` byte-for-byte):

    m: u64 | n: u64 | k: u64 | noise_rank: u64
    a: MatrixMerkleProof | bt: MatrixMerkleProof
    moe: Option tag (0x00 = None, dense proofs only)

MatrixMerkleProof: proof: MerkleProof | row_indices: Vec<u64>
MerkleProof:        leaf_data (u64 count, then per leaf: u64 len + bytes)
                    | leaf_indices: Vec<u64> | total_leaves: u64
                    | root: [u8; 32] | siblings (u64 count + 32B each)
"""

from __future__ import annotations

import base64
from dataclasses import dataclass

import numpy as np

from . import blake3k
from .core import (
    MiningConfiguration, IncompleteBlockHeader, compute_job_key,
    derive_noise_seeds, compute_noise_for_indices, compute_jackpot_hash,
    extract_difficulty_bound, hash_meets_bound,
    JACKPOT_SIZE, LROT_PER_TILE, SIGNAL_MAX, SIGNAL_MIN,
)
from .patterns import PeriodicPattern


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MerkleProofData:
    leaf_data: list           # 1024-byte chunks
    leaf_indices: list        # sorted, unique
    total_leaves: int
    root: bytes
    siblings: list

    @classmethod
    def from_blake3k(cls, proof) -> "MerkleProofData":
        return cls(
            leaf_data=list(proof.leaf_data),
            leaf_indices=list(proof.leaf_indices),
            total_leaves=proof.total_leaves,
            root=proof.root,
            siblings=list(proof.siblings),
        )

    def to_blake3k(self, key: bytes):
        return blake3k.MerkleProof(
            leaf_data=self.leaf_data,
            leaf_indices=self.leaf_indices,
            total_leaves=self.total_leaves,
            root=self.root,
            siblings=self.siblings,
            key=blake3k.key_words(key),
        )


@dataclass
class MatrixMerkleProofData:
    proof: MerkleProofData
    row_indices: list


@dataclass
class PlainProof:
    m: int
    n: int
    k: int
    noise_rank: int
    a: MatrixMerkleProofData
    bt: MatrixMerkleProofData
    moe: object | None = None      # dense proofs only; MoE unsupported here

    # -- serialisation ------------------------------------------------------

    def to_bincode(self) -> bytes:
        if self.moe is not None:
            raise ValueError("MoE proofs are not supported by this miner")
        out = bytearray()
        for v in (self.m, self.n, self.k, self.noise_rank):
            out += v.to_bytes(8, "little")
        out += _ser_matrix_merkle_proof(self.a)
        out += _ser_matrix_merkle_proof(self.bt)
        out += b"\x00"                                   # Option::None
        return bytes(out)

    def to_base64(self) -> str:
        return base64.b64encode(self.to_bincode()).decode("ascii")

    @classmethod
    def from_bincode(cls, data: bytes) -> "PlainProof":
        pos = 0

        def take(n: int) -> bytes:
            nonlocal pos
            if pos + n > len(data):
                raise ValueError("truncated PlainProof")
            chunk = data[pos:pos + n]
            pos += n
            return chunk

        def u64() -> int:
            return int.from_bytes(take(8), "little")

        def usize_vec() -> list:
            count = u64()
            return [u64() for _ in range(count)]

        m, n, k, noise_rank = u64(), u64(), u64(), u64()
        a = _de_matrix_merkle_proof(take, u64, usize_vec)
        bt = _de_matrix_merkle_proof(take, u64, usize_vec)
        tag = take(1)
        if tag != b"\x00":
            raise ValueError("MoE proofs are not supported by this miner")
        if pos != len(data):
            raise ValueError(f"trailing bytes after PlainProof: {len(data) - pos}")
        return cls(m=m, n=n, k=k, noise_rank=noise_rank, a=a, bt=bt)

    @classmethod
    def from_base64(cls, data: str) -> "PlainProof":
        return cls.from_bincode(base64.b64decode(data))


def _ser_merkle_proof(p: MerkleProofData) -> bytes:
    out = bytearray()
    out += len(p.leaf_data).to_bytes(8, "little")
    for leaf in p.leaf_data:
        # serde_chunk_vec serialises Vec<[u8; 1024]> as Vec<&[u8]>:
        # each entry carries its own u64 length prefix.
        out += len(leaf).to_bytes(8, "little")
        out += leaf
    out += len(p.leaf_indices).to_bytes(8, "little")
    for i in p.leaf_indices:
        out += i.to_bytes(8, "little")
    out += p.total_leaves.to_bytes(8, "little")
    out += p.root
    out += len(p.siblings).to_bytes(8, "little")
    for sib in p.siblings:
        out += sib
    return bytes(out)


def _ser_matrix_merkle_proof(p: MatrixMerkleProofData) -> bytes:
    out = bytearray()
    out += _ser_merkle_proof(p.proof)
    out += len(p.row_indices).to_bytes(8, "little")
    for i in p.row_indices:
        out += i.to_bytes(8, "little")
    return bytes(out)


def _de_merkle_proof(take, u64, usize_vec) -> MerkleProofData:
    count = u64()
    leaf_data = []
    for _ in range(count):
        length = u64()
        leaf = take(length)
        if length != blake3k.CHUNK_LEN:
            raise ValueError(f"leaf data must be {blake3k.CHUNK_LEN} bytes")
        leaf_data.append(leaf)
    leaf_indices = usize_vec()
    total_leaves = u64()
    root = take(32)
    sib_count = u64()
    siblings = [take(32) for _ in range(sib_count)]
    return MerkleProofData(leaf_data, leaf_indices, total_leaves, root, siblings)


def _de_matrix_merkle_proof(take, u64, usize_vec) -> MatrixMerkleProofData:
    proof = _de_merkle_proof(take, u64, usize_vec)
    row_indices = usize_vec()
    return MatrixMerkleProofData(proof, row_indices)


# ---------------------------------------------------------------------------
# Construction from a mining hit
# ---------------------------------------------------------------------------

def build_matrix_proof(matrix: np.ndarray, job_key: bytes, row_indices,
                       num_cols: int) -> MatrixMerkleProofData:
    """
    Merkle membership proof for ``row_indices`` of a row-major int8 matrix
    (port of ``build_matrix_proof`` in mine.rs).
    """
    flat = np.ascontiguousarray(matrix, dtype=np.int8).tobytes()
    padded = blake3k.pad_to_chunk_boundary(flat)
    tree = blake3k.MerkleTree(padded, job_key)
    leaf_indices = blake3k.MerkleTree.compute_leaf_indices_from_rows(
        list(row_indices), (matrix.shape[0], num_cols))
    proof = tree.get_multileaf_proof(leaf_indices)
    return MatrixMerkleProofData(MerkleProofData.from_blake3k(proof),
                                 list(row_indices))


def plain_proof_from_hit(hit: dict, config: MiningConfiguration) -> PlainProof:
    """
    Build the PlainProof for a hit produced by the engine/trial code.

    ``hit`` must contain ``A`` (m, k), ``Bt`` (n, k), ``a_rows``, ``b_cols``
    and ``job_key``.
    """
    A, Bt = hit["A"], hit["Bt"]
    m, k = A.shape
    n = Bt.shape[0]
    a_proof = build_matrix_proof(A, hit["job_key"], hit["a_rows"], k)
    bt_proof = build_matrix_proof(Bt, hit["job_key"], hit["b_cols"], k)
    return PlainProof(m=m, n=n, k=k, noise_rank=config.rank, a=a_proof,
                      bt=bt_proof)


# ---------------------------------------------------------------------------
# Local verification (mirrors verify_plain_proof in zk-pow/src/api/verify.rs)
# ---------------------------------------------------------------------------

def _expected_merkle_leaves(m: int, k: int) -> int:
    return (m * k + blake3k.CHUNK_LEN - 1) // blake3k.CHUNK_LEN


def _list_to_pattern(indices: list):
    """Port of list_to_pattern: (pattern, offset)."""
    if not indices:
        raise ValueError("pattern parsing error: empty indices")
    if any(a >= b for a, b in zip(indices, indices[1:])):
        raise ValueError("pattern parsing error: indices not strictly increasing")
    offset = indices[0]
    normalized = [i - offset for i in indices]
    pattern = PeriodicPattern.from_list(normalized)
    if not pattern.offset_is_valid(offset):
        raise ValueError(f"pattern parsing error: offset {offset} is not valid")
    return pattern, offset


def verify_plain_proof(header: IncompleteBlockHeader, proof: PlainProof,
                       nbits: int, derivation: str):
    """
    Verify a PlainProof the way the node's verifier does (without the ZK
    circuit): reconstruct the public parameters, recompute noise and the
    jackpot from the proof's own strips, check the Merkle roots and the
    difficulty bound.

    Returns ``(True, "ok")`` or ``(False, reason)``.
    """
    try:
        return _verify_impl(header, proof, nbits, derivation)
    except Exception as exc:  # noqa: BLE001 - report, never raise
        return False, f"{type(exc).__name__}: {exc}"


def _verify_impl(header: IncompleteBlockHeader, proof: PlainProof,
                 nbits: int, derivation: str):
    m, n, k, rank = proof.m, proof.n, proof.k, proof.noise_rank

    if proof.a.proof.total_leaves != _expected_merkle_leaves(m, k):
        return False, "A Merkle tree declares wrong leaf count"
    if proof.bt.proof.total_leaves != _expected_merkle_leaves(n, k):
        return False, "B^T Merkle tree declares wrong leaf count"

    rows_pattern, t_rows = _list_to_pattern(proof.a.row_indices)
    cols_pattern, t_cols = _list_to_pattern(proof.bt.row_indices)

    config = MiningConfiguration(k, rank, rows_pattern, cols_pattern)
    config.sanity_check(m, n)

    if t_rows + rows_pattern.max() >= m:
        return False, "t_rows + pattern max >= m"
    if t_cols + cols_pattern.max() >= n:
        return False, "t_cols + pattern max >= n"

    dot_product_length = config.dot_product_length
    if dot_product_length == 0 or dot_product_length % 4 != 0:
        return False, "bad dot_product_length"

    # Merkle roots must reconstruct from the proof data.
    job_key = compute_job_key(header, config)
    a_tree = proof.a.proof.to_blake3k(job_key)
    b_tree = proof.bt.proof.to_blake3k(job_key)
    if a_tree.compute_root() != proof.a.proof.root:
        return False, "A Merkle proof does not reconstruct its root"
    if b_tree.compute_root() != proof.bt.proof.root:
        return False, "B^T Merkle proof does not reconstruct its root"

    # Strips: the pattern rows of A and B^T (dot_product_length bytes each).
    try:
        s_a = np.frombuffer(
            b"".join(a_tree.extract_row_bytes(r, dot_product_length)
                     for r in proof.a.row_indices), dtype=np.int8
        ).reshape(len(proof.a.row_indices), dot_product_length)
        s_b = np.frombuffer(
            b"".join(b_tree.extract_row_bytes(c, dot_product_length)
                     for c in proof.bt.row_indices), dtype=np.int8
        ).reshape(len(proof.bt.row_indices), dot_product_length)
    except ValueError as exc:
        return False, f"strip extraction failed: {exc}"

    for name, strips in (("A", s_a), ("B", s_b)):
        if strips.size and (int(strips.min()) < SIGNAL_MIN or int(strips.max()) > SIGNAL_MAX):
            return False, f"{name} strip values out of range [-64, 64]"

    # Noise for exactly the proven rows.
    hash_a, hash_b = proof.a.proof.root, proof.bt.proof.root
    b_noise_seed, a_noise_seed = derive_noise_seeds(
        job_key, hash_a, hash_b, m, n, derivation)
    noise_a, noise_b = compute_noise_for_indices(
        dot_product_length, rank, (b_noise_seed, a_noise_seed),
        proof.a.row_indices, proof.bt.row_indices)

    a_noised = s_a.astype(np.int32) + noise_a[:, :dot_product_length].astype(np.int32)
    bt_noised = s_b.astype(np.int32) + noise_b[:, :dot_product_length].astype(np.int32)

    # Jackpot over the tile.
    tile = np.zeros((a_noised.shape[0], bt_noised.shape[0]), dtype=np.int64)
    jackpot = [0] * JACKPOT_SIZE
    for step, ll in enumerate(range(rank, dot_product_length + 1, rank)):
        lo = ll - rank
        tile += (a_noised[:, lo:ll].astype(np.int64)
                 @ bt_noised[:, lo:ll].astype(np.int64).T)
        xored = 0
        for v in tile.reshape(-1):
            xored ^= int(v) & 0xFFFFFFFF
        tid = step % JACKPOT_SIZE
        jackpot[tid] = (((jackpot[tid] << LROT_PER_TILE)
                         | (jackpot[tid] >> (32 - LROT_PER_TILE))) & 0xFFFFFFFF) ^ xored

    jackpot_hash = compute_jackpot_hash(jackpot, a_noise_seed)
    bound = extract_difficulty_bound(nbits, config)
    if not hash_meets_bound(jackpot_hash, bound):
        return False, "jackpot hash does not meet difficulty target"
    return True, "mining solution verified successfully"
