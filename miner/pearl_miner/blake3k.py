"""
Keyed BLAKE3 compression + Pearl Merkle tree (pure Python / numpy).

This is a faithful, dependency-light reimplementation of the `pearl-blake3`
crate used by the Pearl network (pearld / pearl-gateway / py-pearl-mining):

  * ``blake3_digest(data, key)``          -> keyed BLAKE3 root digest
  * ``MerkleTree``                        -> chunk CVs + pairwise parent CVs
  * ``MerkleTree.get_multileaf_proof``    -> multi-leaf proof (sibling order
                                             matches the Rust implementation
                                             byte-for-byte)

Only ``numpy`` is required.  The compression function is vectorised so that
thousands of independent single-block hashes (one per mining trial) run in a
handful of numpy calls.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# BLAKE3 constants
# ---------------------------------------------------------------------------

IV = np.array(
    [
        0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
        0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
    ],
    dtype=np.uint32,
)

MSG_PERMUTATION = np.array(
    [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8], dtype=np.intp
)

CHUNK_START = 1
CHUNK_END = 2
PARENT = 4
ROOT = 8
KEYED_HASH = 16

CHUNK_LEN = 1024          # BLAKE3 chunk length
OUT_LEN = 32              # digest / CV length
BLOCK_LEN = 64            # compression input block length

_U32 = np.uint32


def _rotr(x: np.ndarray, n: int) -> np.ndarray:
    """Rotate-right for uint32 arrays (wraps like Rust's rotate_right)."""
    return (x >> _U32(n)) | (x << _U32(32 - n))


def _g(state: np.ndarray, m: np.ndarray, a: int, b: int, c: int, d: int,
       mx: int, my: int) -> None:
    """The BLAKE3 quarter-round mixing function, vectorised over rows."""
    va = state[:, a] + state[:, b] + m[:, mx]
    state[:, a] = va
    state[:, d] = _rotr(state[:, d] ^ va, 16)
    vc = state[:, c] + state[:, d]
    state[:, c] = vc
    state[:, b] = _rotr(state[:, b] ^ vc, 12)
    va2 = state[:, a] + state[:, b] + m[:, my]
    state[:, a] = va2
    state[:, d] = _rotr(state[:, d] ^ va2, 8)
    vc2 = state[:, c] + state[:, d]
    state[:, c] = vc2
    state[:, b] = _rotr(state[:, b] ^ vc2, 7)


def compress_batch(cv: np.ndarray, block_words: np.ndarray, counter: int,
                   block_len: int, flags: int) -> np.ndarray:
    """
    Run the BLAKE3 compression function on a batch of inputs.

    Parameters
    ----------
    cv : (N, 8) uint32
        Chaining value words (the key words in keyed mode, IV for unkeyed).
    block_words : (N, 16) uint32
        Message block words (little-endian u32 of the 64-byte block).
    counter : int
        64-bit chunk counter (same for every row in the batch).
    block_len : int
        Number of message bytes in the block (same for every row).
    flags : int
        Domain-separation flags (same for every row).

    Returns
    -------
    (N, 16) uint32
        The full 16-word state after the 7 rounds.  The chaining value is
        ``state[:, :8]``; the root output is ``state[:, :8] ^ state[:, 8:]``.
    """
    cv = np.ascontiguousarray(cv, dtype=np.uint32)
    block_words = np.ascontiguousarray(block_words, dtype=np.uint32)
    n = cv.shape[0]
    state = np.empty((n, 16), dtype=np.uint32)
    state[:, 0:8] = cv
    state[:, 8:12] = IV[0:4]
    state[:, 12] = _U32(counter & 0xFFFFFFFF)
    state[:, 13] = _U32((counter >> 32) & 0xFFFFFFFF)
    state[:, 14] = _U32(block_len)
    state[:, 15] = _U32(flags)

    m = block_words.copy()
    for _ in range(7):
        # Mix the columns.
        _g(state, m, 0, 4, 8, 12, 0, 1)
        _g(state, m, 1, 5, 9, 13, 2, 3)
        _g(state, m, 2, 6, 10, 14, 4, 5)
        _g(state, m, 3, 7, 11, 15, 6, 7)
        # Mix the diagonals.
        _g(state, m, 0, 5, 10, 15, 8, 9)
        _g(state, m, 1, 6, 11, 12, 10, 11)
        _g(state, m, 2, 7, 8, 13, 12, 13)
        _g(state, m, 3, 4, 9, 14, 14, 15)
        # Permute the message schedule.
        m = m[:, MSG_PERMUTATION]
    return state


def _hash_chunk(cv_words: np.ndarray, chunk: bytes, counter: int,
                flags: int, root: bool = False) -> np.ndarray:
    """
    Hash one chunk (<= 1024 bytes) and return the final 16-word state.

    The chunk is fed as 64-byte blocks: intermediate full blocks use
    ``block_len = 64`` and no CHUNK_END, the final block carries the chunk's
    remaining length, CHUNK_END and (when ``root``) the ROOT flag.
    """
    n = len(chunk)
    nblocks = max(1, (n + BLOCK_LEN - 1) // BLOCK_LEN)
    cv = np.ascontiguousarray(cv_words, dtype=np.uint32)
    state = None
    for i in range(nblocks):
        blk = chunk[i * BLOCK_LEN: (i + 1) * BLOCK_LEN]
        f = flags
        if i == 0:
            f |= CHUNK_START
        if i == nblocks - 1:
            f |= CHUNK_END
            blen = n - (nblocks - 1) * BLOCK_LEN
            if root:
                f |= ROOT
        else:
            blen = BLOCK_LEN
        state = compress_batch(cv[None, :], _words_of(blk)[None, :],
                               counter, blen, f)[0]
        # The chaining value is the XOR fold of the two state halves.
        cv = (state[:8] ^ state[8:]).copy()
    return state


def _words_of(block: bytes) -> np.ndarray:
    if len(block) > BLOCK_LEN:
        raise ValueError("block must be at most 64 bytes")
    if len(block) < BLOCK_LEN:
        block = block + b"\x00" * (BLOCK_LEN - len(block))
    return np.frombuffer(block, dtype="<u4")


def _cv_bytes(state_row: np.ndarray) -> bytes:
    """Chaining value bytes of one state row (state[i] ^ state[i+8])."""
    return (state_row[:8] ^ state_row[8:]).astype("<u4").tobytes()


def cv_output(state_row: np.ndarray) -> bytes:
    """Chaining value: state[i] ^ state[i+8] for the first 8 words.

    (For a 32-byte digest this is identical to the root output; the ROOT flag
    only changes the compression input, not the fold.)
    """
    return (state_row[:8] ^ state_row[8:]).astype("<u4").tobytes()


def root_output(state_row: np.ndarray) -> bytes:
    """Root output: state[i] ^ state[i+8] for the first 8 words."""
    return (state_row[:8] ^ state_row[8:]).astype("<u4").tobytes()


# ---------------------------------------------------------------------------
# Primitive hashes
# ---------------------------------------------------------------------------

def key_words(key: bytes) -> np.ndarray:
    """32-byte key -> 8 little-endian u32 words."""
    if len(key) != OUT_LEN:
        raise ValueError(f"key must be {OUT_LEN} bytes, got {len(key)}")
    return np.frombuffer(key, dtype="<u4").copy()


def _hash_message(key: bytes, data: bytes, flags_extra: int = 0) -> bytes:
    """Keyed BLAKE3 of arbitrary-length data (root output)."""
    kw = key_words(key)
    flags = KEYED_HASH | flags_extra
    if len(data) <= CHUNK_LEN:
        return root_output(_hash_chunk(kw, data, 0, flags, root=True))
    layers = _tree_layers(data, kw)
    return layers[-1][0]


def keyed_hash_root(data: bytes, key: bytes) -> bytes:
    """Keyed BLAKE3 digest of ``data`` (any length)."""
    return _hash_message(key, data)


def unkeyed_hash(data: bytes) -> bytes:
    """Plain BLAKE3 digest of ``data`` (any length)."""
    if len(data) <= CHUNK_LEN:
        return root_output(_hash_chunk(IV, data, 0, 0, root=True))
    return _tree_layers(data, IV, keyed=False)[-1][0]


def chunk_cv(chunk: bytes, key: np.ndarray, counter: int,
             keyed: bool = True) -> bytes:
    """Chunk chaining value (non-root) of one <=1024-byte chunk."""
    state = _hash_chunk(key, chunk, counter, KEYED_HASH if keyed else 0)
    return cv_output(state)


def _hash_parent(key: np.ndarray, left: bytes, right: bytes, root: bool,
                 keyed: bool = True) -> np.ndarray:
    """Compress a parent node (one 64-byte block, PARENT domain flags)."""
    flags = (KEYED_HASH if keyed else 0) | PARENT
    if root:
        flags |= ROOT
    block = _words_of(left + right)
    return compress_batch(key[None, :], block[None, :], 0, BLOCK_LEN, flags)[0]


def parent_cv(left: bytes, right: bytes, key: np.ndarray,
              keyed: bool = True) -> bytes:
    """Parent chaining value (non-root) of two child CVs."""
    return cv_output(_hash_parent(key, left, right, root=False, keyed=keyed))


def root_cv(left: bytes, right: bytes, key: np.ndarray,
            keyed: bool = True) -> bytes:
    """Root output of two child CVs (PARENT | ROOT)."""
    return root_output(_hash_parent(key, left, right, root=True, keyed=keyed))


# ---------------------------------------------------------------------------
# Merkle tree (mirrors pearl-blake3::MerkleTree)
# ---------------------------------------------------------------------------

def pad_to_chunk_boundary(data: bytes) -> bytes:
    """Zero-pad ``data`` up to the next multiple of 1024 bytes."""
    if len(data) % CHUNK_LEN == 0:
        return data
    return data + b"\x00" * (CHUNK_LEN - len(data) % CHUNK_LEN)


def _tree_layers(data: bytes, key: np.ndarray, keyed: bool = True) -> list:
    """
    Build the BLAKE3 Merkle tree layers exactly like ``MerkleTree::new``:

    * data <= 1024 bytes  -> single chunk root
    * otherwise           -> chunk CVs, then pairwise ``combine_layer`` until
      at most 2 nodes remain, then a final root CV.
    """
    if len(data) == 0:
        return [[b"\x00" * OUT_LEN]]
    if len(data) <= CHUNK_LEN:
        return [[root_output(_hash_chunk(key, data, 0,
                                         KEYED_HASH, root=True))]]
    base_flags = KEYED_HASH if keyed else 0
    chunks = [data[i:i + CHUNK_LEN] for i in range(0, len(data), CHUNK_LEN)]
    key_words = np.asarray(key, dtype=np.uint32)

    # Chunks are 1024 bytes = sixteen 64-byte blocks chained together.  All
    # full chunks are compressed in lock-step: one vectorised compression per
    # block position, carrying the chaining values across positions.
    n_full = sum(1 for c in chunks if len(c) == CHUNK_LEN)
    cvs: list = []
    if n_full:
        full = b"".join(c for c in chunks if len(c) == CHUNK_LEN)
        all_blocks = np.frombuffer(full, dtype="<u4").reshape(
            n_full, CHUNK_LEN // BLOCK_LEN, 16)
        counters = np.arange(n_full, dtype=np.uint64)
        keys = np.repeat(key_words[None, :], n_full, axis=0)
        cv = keys.copy()
        states = None
        nblocks = CHUNK_LEN // BLOCK_LEN
        for j in range(nblocks):
            flags = base_flags
            if j == 0:
                flags |= CHUNK_START
            if j == nblocks - 1:
                flags |= CHUNK_END
            states = compress_batch(cv, all_blocks[:, j, :], counters,
                                    BLOCK_LEN, flags)
            cv = states[:, :8] ^ states[:, 8:]
        cvs.extend(_cv_bytes(s) for s in states)
    # A trailing partial chunk (only when the data is not 1024-aligned).
    if len(chunks[-1]) < CHUNK_LEN:
        cvs.append(chunk_cv(chunks[-1], key_words, len(chunks) - 1, keyed))
    layers = [cvs]
    # Pairwise parent levels, one vectorised compression per level.
    while len(cvs) > 2:
        npairs = len(cvs) // 2
        carry = cvs[-1] if len(cvs) % 2 else None
        blocks = np.zeros((npairs, 16), dtype=np.uint32)
        for i in range(npairs):
            blocks[i, :8] = np.frombuffer(cvs[2 * i], dtype="<u4")
            blocks[i, 8:] = np.frombuffer(cvs[2 * i + 1], dtype="<u4")
        keys = np.repeat(key_words[None, :], npairs, axis=0)
        states = compress_batch(keys, blocks, np.zeros(npairs, dtype=np.uint64),
                                np.full(npairs, BLOCK_LEN, dtype=np.int64),
                                base_flags | PARENT)
        cvs = [_cv_bytes(s) for s in states]
        if carry is not None:
            cvs.append(carry)
        layers.append(cvs)
    if len(cvs) == 2:
        blocks = np.zeros((1, 16), dtype=np.uint32)
        blocks[0, :8] = np.frombuffer(cvs[0], dtype="<u4")
        blocks[0, 8:] = np.frombuffer(cvs[1], dtype="<u4")
        state = compress_batch(key_words[None, :], blocks, 0, BLOCK_LEN,
                               base_flags | PARENT | ROOT)[0]
        layers.append([root_output(state)])
    return layers


class MerkleTree:
    """BLAKE3 Merkle tree over chunk-padded data, keyed with a 32-byte key."""

    def __init__(self, data: bytes, key: bytes):
        self.key = key_words(key)
        self.data = data
        self.layers = _tree_layers(data, self.key)

    @property
    def root(self) -> bytes:
        return self.layers[-1][0]

    @property
    def num_leaves(self) -> int:
        return len(self.layers[0])

    @staticmethod
    def compute_leaf_indices_from_rows(row_indices, shape) -> list:
        """Leaf (chunk) indices covering the given matrix rows."""
        rows, cols = shape
        indices: set = set()
        for row in row_indices:
            first = (row * cols) // CHUNK_LEN
            last = ((row + 1) * cols - 1) // CHUNK_LEN
            indices.update(range(first, last + 1))
        return sorted(indices)

    def get_multileaf_proof(self, leaf_indices) -> "MerkleProof":
        """
        Multi-leaf proof.  The sibling ordering replicates the Rust
        implementation (``MerkleTree::get_multileaf_proof``) exactly, level by
        level, in ascending index order.
        """
        unique = sorted(set(leaf_indices))
        if not unique:
            raise ValueError("leaf_indices must be non-empty")
        total_leaves = self.num_leaves
        if unique[-1] >= total_leaves:
            raise ValueError("leaf index out of bounds")

        # Leaf data: the raw 1024-byte chunks (zero padded).
        leaf_data = []
        for i in unique:
            start = i * CHUNK_LEN
            end = min(start + CHUNK_LEN, len(self.data))
            chunk = bytearray(CHUNK_LEN)
            chunk[: end - start] = self.data[start:end]
            leaf_data.append(bytes(chunk))

        siblings: list = []
        current_set = set(unique)
        level_len = total_leaves
        level = 0
        while level_len > 1 and current_set:
            level_nodes = self.layers[level]
            for i in sorted(current_set):
                if i % 2 == 1:
                    if (i - 1) not in current_set:
                        siblings.append(level_nodes[i - 1])
                else:
                    if (i + 1) not in current_set and (i + 1) < level_len:
                        siblings.append(level_nodes[i + 1])
            current_set = {i // 2 for i in current_set}
            level_len = (level_len + 1) // 2
            level += 1

        return MerkleProof(
            leaf_data=leaf_data,
            leaf_indices=unique,
            total_leaves=total_leaves,
            root=self.root,
            siblings=siblings,
            key=self.key,
        )


class MerkleProof:
    """A multi-leaf BLAKE3 Merkle proof (``pearl_blake3::MerkleProof``)."""

    def __init__(self, leaf_data, leaf_indices, total_leaves, root, siblings,
                 key=None):
        self.leaf_data = leaf_data            # list of 1024-byte chunks
        self.leaf_indices = list(leaf_indices)  # sorted, unique
        self.total_leaves = total_leaves
        self.root = root
        self.siblings = siblings
        self.key = key

    def leaf_hashes(self, key=None) -> list:
        key = key if key is not None else self.key
        if key is None:
            raise ValueError("key required to recompute leaf hashes")
        return [chunk_cv(d, key, i)
                for i, d in zip(self.leaf_indices, self.leaf_data)]

    def compute_root(self, key=None):
        """Recompute the Merkle root from the proof (verification path)."""
        key = key if key is not None else self.key
        if key is None:
            raise ValueError("key required to compute root")
        if not self.leaf_indices:
            return None

        current = {i: h for i, h in zip(self.leaf_indices, self.leaf_hashes(key))}
        level_len = self.total_leaves
        if level_len == 1:
            return current[0] if not self.siblings else None

        sib_iter = iter(self.siblings)
        while level_len > 2:
            # Collect the (left, right) pairs of this level first, then run
            # one vectorised compression for all of them.
            pairs = []
            passthrough = []
            for i in sorted(current):
                cv = current[i]
                if i % 2 == 0:
                    if (i + 1) in current:
                        right = current[i + 1]
                    elif (i + 1) < level_len:
                        right = next(sib_iter)
                    else:
                        right = None
                    if right is not None:
                        pairs.append((i // 2, cv, right))
                    else:
                        passthrough.append((i // 2, cv))
                else:
                    if (i - 1) in current:
                        continue
                    left = next(sib_iter)
                    pairs.append((i // 2, left, cv))
            nxt: dict = {}
            if pairs:
                blocks = np.zeros((len(pairs), 16), dtype=np.uint32)
                for j, (_, left, right) in enumerate(pairs):
                    blocks[j, :8] = np.frombuffer(left, dtype="<u4")
                    blocks[j, 8:] = np.frombuffer(right, dtype="<u4")
                keys = np.repeat(np.asarray(key, dtype=np.uint32)[None, :],
                                 len(pairs), axis=0)
                states = compress_batch(keys, blocks, 0, BLOCK_LEN,
                                        KEYED_HASH | PARENT)
                for j, (idx, _, _) in enumerate(pairs):
                    nxt[idx] = _cv_bytes(states[j])
            for idx, cv in passthrough:
                nxt[idx] = cv
            current = nxt
            level_len = (level_len + 1) // 2

        left = current.get(0)
        if left is None:
            left = next(sib_iter)
        right = current.get(1)
        if right is None:
            right = next(sib_iter)
        return root_cv(left, right, key)

    def extract_row_bytes(self, row: int, num_cols: int) -> bytes:
        """Extract the ``num_cols`` bytes of matrix row ``row`` from the proof."""
        start = row * num_cols
        end = start + num_cols
        out = bytearray()
        for i, chunk in zip(self.leaf_indices, self.leaf_data):
            cstart = i * CHUNK_LEN
            cend = cstart + CHUNK_LEN
            lo = max(start, cstart)
            hi = min(end, cend)
            if lo < hi:
                out += chunk[lo - cstart: hi - cstart]
        if len(out) != num_cols:
            raise ValueError(
                f"row {row}: proof covers only {len(out)} of {num_cols} bytes")
        return bytes(out)
