"""
Pearl miner selftest.

Cross-checks this pure-Python implementation against:

  1. pinned vectors from the official Rust test suites (seed.rs / noise),
  2. the pip ``blake3`` package (independent BLAKE3 implementation),
  3. the readable per-trial reference in :mod:`trial`,
  4. the official ``pearl_mining`` Rust module (when installed - built from
     the Pearl monorepo with maturin; optional).

Run it with:

    python -m pearl_miner.selftest            # from the miner/ directory
    python miner/selftest.py                  # or directly
"""

from __future__ import annotations

import sys
import time

import numpy as np

from pearl_miner import blake3k
from pearl_miner.core import (
    IncompleteBlockHeader, MiningConfiguration, SEED_SALT_A, SEED_SALT_B,
    compute_job_key, compute_jackpot_hash, compute_noise_for_indices,
    derive_noise_seeds,
    derivation_for_cert_version, extract_difficulty_bound, hash_meets_bound,
    matrix_roots, mining_bound, nbits_to_target, unkeyed_hash,
    keyed_hash, U256_MAX,
)
from pearl_miner.patterns import PeriodicPattern
from pearl_miner.engine_np import NumpyEngine
from pearl_miner.trial import try_mine_one, compute_jackpot_words
from pearl_miner.proof import (
    PlainProof, plain_proof_from_hit, verify_plain_proof)

try:  # optional independent BLAKE3
    import blake3 as _pkg_blake3
except Exception:  # pragma: no cover
    _pkg_blake3 = None

try:  # optional official Rust module
    import pearl_mining as _rust
except Exception:  # pragma: no cover
    _rust = None

_PASS = 0
_FAIL = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global _PASS, _FAIL
    if ok:
        _PASS += 1
        print(f"  [ok] {name}")
    else:
        _FAIL += 1
        print(f"  [FAIL] {name}" + (f" - {detail}" if detail else ""))


def section(title: str) -> None:
    print(f"\n== {title} ==")


# ---------------------------------------------------------------------------
# 1. pinned vectors from the Rust test suites
# ---------------------------------------------------------------------------

def test_pinned_seed_vectors():
    section("pinned vectors (zk-pow/src/api/seed.rs)")
    check("SEED_SALT_A constant",
          SEED_SALT_A.hex() == "8249406ca0ed15169616f692fcf076f892dbdb2a70"
                           "23b852f0d47719c390017b",
          SEED_SALT_A.hex())
    check("SEED_SALT_B constant",
          SEED_SALT_B.hex() == "11300632ec6301ca2be2af718b3f4d4f1ae9c6398"
                           "8e8cc044844301d71b89aa9",
          SEED_SALT_B.hex())

    job_key = bytes([0x11]) * 32
    hash_a = bytes([0xAA]) * 32
    hash_b = bytes([0xBB]) * 32
    m, n = 192, 320  # PublicProofParams::new_for_tests(192, 320, 256)

    b_seed, a_seed = derive_noise_seeds(job_key, hash_a, hash_b, m, n, "legacy")
    check("legacy commitment_hash (b_noise_seed)",
          b_seed.hex() == "add6f7ea5feebf89c8a77e2ebfa0d82442e7dbb0046dbd48"
                          "971861d12fcb0177", b_seed.hex())
    check("legacy commitment_hash (a_noise_seed)",
          a_seed.hex() == "483b07b6f73105030b9482255f37723f3fed69ae916724e"
                          "e8291848b8c28794b", a_seed.hex())

    b_seed, a_seed = derive_noise_seeds(job_key, hash_a, hash_b, m, n, "salted")
    check("salted commitment_hash (b_noise_seed)",
          b_seed.hex() == "60ed9b73c5a9599b200b6cd563e7f0d5d9a67d2402d85fd4"
                          "ef966c580080d0e5", b_seed.hex())
    check("salted commitment_hash (a_noise_seed)",
          a_seed.hex() == "301784168005ec833ab0aa60006f7fe7faaa95307d8c1"
                          "fc6819b2ffdd717eccf", a_seed.hex())

    check("cert version mapping",
          derivation_for_cert_version(1) == "legacy"
          and derivation_for_cert_version(2) == "legacy"
          and derivation_for_cert_version(3) == "salted")


# ---------------------------------------------------------------------------
# 2. BLAKE3 vs the pip package
# ---------------------------------------------------------------------------

def test_blake3_against_pip():
    section("BLAKE3 vs pip blake3 package")
    if _pkg_blake3 is None:
        print("  (pip blake3 not installed - skipped)")
        return
    rng = np.random.default_rng(42)
    ok_h = True
    for size in (0, 1, 63, 64, 65, 127, 128, 1023, 1024, 1025, 2048, 3072,
                 5000, 100000):
        data = rng.integers(0, 256, size, dtype=np.uint8).tobytes()
        key = rng.integers(0, 256, 32, dtype=np.uint8).tobytes()
        if blake3k.unkeyed_hash(data) != _pkg_blake3.blake3(data).digest():
            ok_h = False
        if blake3k.keyed_hash_root(data, key) != _pkg_blake3.blake3(data, key=key).digest():
            ok_h = False
    check("hash roots (unkeyed + keyed, all sizes)", ok_h)

    ok_t = True
    for size in (1024, 2048, 3072, 4096, 5120, 65536, 131072):
        data = rng.integers(0, 256, size, dtype=np.uint8).tobytes()
        key = rng.integers(0, 256, 32, dtype=np.uint8).tobytes()
        if blake3k.MerkleTree(data, key).root != _pkg_blake3.blake3(data, key=key).digest():
            ok_t = False
    check("merkle roots equal the keyed hash", ok_t)

    data = rng.integers(0, 256, 128 * 4096, dtype=np.uint8).tobytes()
    key = rng.integers(0, 256, 32, dtype=np.uint8).tobytes()
    tree = blake3k.MerkleTree(blake3k.pad_to_chunk_boundary(data), key)
    li = blake3k.MerkleTree.compute_leaf_indices_from_rows([0, 1, 2, 3], (128, 4096))
    proof = tree.get_multileaf_proof(li)
    check("multileaf proof reconstructs its root",
          proof.compute_root() == proof.root)


# ---------------------------------------------------------------------------
# 3. engine vs the readable reference
# ---------------------------------------------------------------------------

SMALL_M = SMALL_N = 16
SMALL_K = 1024
SMALL_RANK = 32


def _small_config():
    return MiningConfiguration(
        SMALL_K, SMALL_RANK,
        PeriodicPattern.from_list([0, 1, 2, 3, 4, 5, 6, 7]),
        PeriodicPattern.from_list([0, 1, 2, 3, 4, 5, 6, 7]))


def _easy_header():
    # nbits 0x1e7fffff -> target ~2^239; with tile 64 and rank 32 the
    # unpenalised adjustment factor is 64*32 = 2048, so the per-partition hit
    # chance is about 2^-6 and hits arrive within a few trials.
    return IncompleteBlockHeader(
        version=0x20000000, prev_block=bytes(range(31, -1, -1)),
        merkle_root=bytes([9]) * 32, timestamp=0x66666666, nbits=0x1e7fffff)


def _saturated_header():
    """nbits whose bound saturates at 2^256-1: every jackpot hash is a hit.

    Used for the deterministic engine-vs-reference comparison.
    """
    return IncompleteBlockHeader(
        version=0x20000000, prev_block=bytes(range(31, -1, -1)),
        merkle_root=bytes([9]) * 32, timestamp=0x66666666, nbits=0x207fffff)


def test_engine_vs_reference():
    section("batched engine vs readable reference (trial.py)")
    cfg = _small_config()
    header = _saturated_header()

    rng = np.random.default_rng(7)
    A = rng.integers(-64, 65, (SMALL_M, SMALL_K), dtype=np.int8)
    Bt = rng.integers(-64, 65, (SMALL_N, SMALL_K), dtype=np.int8)

    job_key = compute_job_key(header, cfg)
    hash_a, hash_b = matrix_roots(A, Bt, job_key)
    b_seed, a_seed = derive_noise_seeds(job_key, hash_a, hash_b,
                                        SMALL_M, SMALL_N, "legacy")

    eng = NumpyEngine(SMALL_M, SMALL_N, SMALL_K, SMALL_RANK,
                      cfg.rows_pattern, cfg.cols_pattern,
                      batch_size=1, seed=1)
    eng.set_job(header, 2)
    noise_a_b, noise_b_b = eng._noise_matrices(
        np.frombuffer(a_seed, np.uint8)[None, :],
        np.frombuffer(b_seed, np.uint8)[None, :])
    noise_a_r, noise_b_r = compute_noise_for_indices(
        SMALL_K, SMALL_RANK, (b_seed, a_seed),
        range(SMALL_M), range(SMALL_N))
    check("noise A (batched == reference)",
          np.array_equal(noise_a_b[0], noise_a_r))
    check("noise B^T (batched == reference)",
          np.array_equal(noise_b_b[0], noise_b_r))

    # same jackpot words for the first partition
    a_noised = A.astype(np.int32) + noise_a_r.astype(np.int32)
    bt_noised = Bt.astype(np.int32) + noise_b_r.astype(np.int32)
    ref_words = compute_jackpot_words(a_noised, bt_noised,
                                      list(range(8)), list(range(8)),
                                      SMALL_K, SMALL_RANK)

    # Drive one engine batch with the same matrices.
    eng2 = NumpyEngine(SMALL_M, SMALL_N, SMALL_K, SMALL_RANK,
                       cfg.rows_pattern, cfg.cols_pattern,
                       batch_size=1, seed=1)
    eng2.set_job(header, 2)
    hits = _engine_batch_with_matrices(eng2, A, Bt)
    check("engine produces hits on easy target", len(hits) > 0,
          f"{len(hits)} hits")
    if hits:
        hit = hits[0]
        first_partition = (hit["a_rows"] == list(range(8))
                           and hit["b_cols"] == list(range(8)))
        check("engine jackpot words match reference",
              not first_partition
              or list(hit["jackpot"]) == [int(w) for w in ref_words])
        ref_hash = compute_jackpot_hash(ref_words, a_seed)
        check("jackpot hash matches reference",
              not first_partition or hit["jackpot_hash"] == ref_hash)
        bound = extract_difficulty_bound(header.nbits, cfg)
        check("hit meets the difficulty bound",
              hash_meets_bound(hit["jackpot_hash"], bound))


def _engine_batch_with_matrices(engine: NumpyEngine, A, Bt):
    """Run the engine's math on fixed matrices (bypasses its RNG)."""
    from pearl_miner.core import derive_noise_seeds as _dns
    B = 1
    m, n, k, rank = engine.m, engine.n, engine.k, engine.rank
    hashes_a, hashes_b = engine._commitment_roots(A[None, :], Bt[None, :])
    b_seed, a_seed = _dns(engine.job_key, hashes_a[0], hashes_b[0], m, n,
                          engine._derivation)
    a_seeds = np.frombuffer(a_seed, np.uint8)[None, :]
    b_seeds = np.frombuffer(b_seed, np.uint8)[None, :]
    noise_a, noise_b = engine._noise_matrices(a_seeds, b_seeds)

    jackpot = np.zeros((B, engine.P, 16), dtype=np.uint32)
    a_key_words = np.ascontiguousarray(a_seeds.view("<u4"))
    from pearl_miner.engine_np import _hash_many_messages, _rotl32, _le256_within
    from pearl_miner.core import LROT_PER_TILE
    tile = np.zeros((B, m, n), dtype=np.int32)
    for step, ll in enumerate(range(rank, k + 1, rank)):
        lo = ll - rank
        a_chunk = A[:, lo:ll].astype(np.float32) + noise_a[:, :, lo:ll]
        b_chunk = Bt[:, lo:ll].astype(np.float32) + noise_b[:, :, lo:ll]
        tile += np.matmul(a_chunk, b_chunk.transpose(0, 2, 1)).astype(np.int32)
        tiles = (tile.reshape(B, engine.P_a, engine.pr, engine.P_b, engine.pc)
                 .transpose(0, 1, 3, 2, 4)
                 .reshape(B, engine.P, engine.pr, engine.pc))
        xored = np.bitwise_xor.reduce(tiles.view(np.uint32), axis=(2, 3))
        tid = step % 16
        jackpot[:, :, tid] = _rotl32(jackpot[:, :, tid], LROT_PER_TILE) ^ xored
    block_words = jackpot.reshape(B * engine.P, 16)
    cv = np.repeat(a_key_words, engine.P, axis=0)
    digests = _hash_many_messages(block_words, cv).astype("<u4").tobytes()
    digests = np.frombuffer(digests, dtype=np.uint8).reshape(B, engine.P, 32)
    hit_mask = _le256_within(digests, engine._bound)
    hits = []
    for b, p in np.argwhere(hit_mask):
        b, p = int(b), int(p)
        ai, bi = divmod(p, len(engine.partitions_b))
        hits.append({
            "A": A, "Bt": Bt,
            "a_rows": list(engine.partitions_a[ai]),
            "b_cols": list(engine.partitions_b[bi]),
            "job_key": engine.job_key,
            "jackpot": [int(w) for w in jackpot[b, p]],
            "jackpot_hash": digests[b, p].tobytes(),
        })
    return hits


# ---------------------------------------------------------------------------
# 4. reference trial loop finds a verifiable share
# ---------------------------------------------------------------------------

def test_reference_trial_share():
    section("reference trial loop -> share -> local verify")
    cfg = _small_config()
    header = _easy_header()
    rng = np.random.default_rng(2024)
    hit = None
    t0 = time.time()
    for _ in range(2000):
        hit = try_mine_one(rng, SMALL_M, SMALL_N, SMALL_K, header, cfg, 2)
        if hit:
            break
    check("reference finds a hit", hit is not None,
          f"{time.time() - t0:.1f}s")
    if not hit:
        return
    proof = plain_proof_from_hit(hit, cfg)
    ok, msg = verify_plain_proof(header, proof, header.nbits, "legacy")
    check("locally verified share (reference)", ok, msg)
    check("serialisation round-trips",
          PlainProof.from_bincode(proof.to_bincode()).to_bincode()
          == proof.to_bincode())


# ---------------------------------------------------------------------------
# 5. engine share verified by the official Rust module
# ---------------------------------------------------------------------------

def test_against_rust_module():
    section("official pearl_mining Rust module")
    if _rust is None:
        print("  (pearl_mining not installed - skipped; build it from the")
        print("   Pearl monorepo with `maturin build --release` to enable)")
        return

    header = IncompleteBlockHeader(
        version=0x20000000, prev_block=bytes(range(31, -1, -1)),
        merkle_root=bytes([9]) * 32, timestamp=0x66666666, nbits=0x1d00ffff)
    rows = PeriodicPattern.from_list([0, 1, 2, 3])
    cols = PeriodicPattern.from_list([0, 1, 2, 3, 4, 5, 6, 7])
    cfg = MiningConfiguration(4096, 128, rows, cols)
    eng = NumpyEngine(128, 128, 4096, 128, rows, cols, batch_size=256,
                      seed=12345)
    eng.set_job(header, 2)
    hits = eng.run_batch()
    check("engine finds a hit (m=n=128, k=4096, rank=128)", len(hits) > 0)
    if not hits:
        return
    proof = plain_proof_from_hit(hits[0], cfg)
    ok, msg = verify_plain_proof(header, proof, header.nbits, "legacy")
    check("locally verified share (engine)", ok, msg)

    b64 = proof.to_base64()
    try:
        rust_proof = _rust.PlainProof.from_base64(b64)
        check("Rust parses the serialised proof", True)
    except Exception as exc:
        check("Rust parses the serialised proof", False, str(exc)[:120])
        return

    rust_header = _rust.IncompleteBlockHeader(
        0x20000000, bytes(range(31, -1, -1)), bytes([9]) * 32,
        0x66666666, 0x1d00ffff)
    ok2, msg2 = _rust.verify_plain_proof_v2(rust_header, rust_proof)
    check("Rust verifies the share", ok2, msg2)

    raw = bytearray(proof.to_bincode())
    raw[200] ^= 0xFF
    import base64
    bad = base64.b64encode(bytes(raw)).decode()
    try:
        bad_proof = _rust.PlainProof.from_base64(bad)
        ok3, _ = _rust.verify_plain_proof_v2(rust_header, bad_proof)
        check("Rust rejects a corrupted share", not ok3)
    except Exception:
        check("Rust rejects a corrupted share", True)

    # Certificate version 3 (salted seeds) - what current mainnet jobs use.
    eng_v3 = NumpyEngine(128, 128, 4096, 128, rows, cols, batch_size=256,
                         seed=999)
    eng_v3.set_job(header, 3)
    hits_v3 = eng_v3.run_batch()
    check("engine finds a V3 (salted) hit", len(hits_v3) > 0)
    if hits_v3:
        proof_v3 = plain_proof_from_hit(hits_v3[0], cfg)
        ok_v3, msg_v3 = verify_plain_proof(
            header, proof_v3, header.nbits,
            derivation_for_cert_version(3))
        check("locally verified V3 share", ok_v3, msg_v3)
        rp_v3 = _rust.PlainProof.from_base64(proof_v3.to_base64())
        ok_v3b, msg_v3b = _rust.verify_plain_proof_v3(rust_header, rp_v3)
        check("Rust verifies the V3 share (verify_plain_proof_v3)",
              ok_v3b, msg_v3b)
        ok_v3c, _ = _rust.verify_plain_proof_v2(rust_header, rp_v3)
        check("V3 share is rejected by the legacy verifier", not ok_v3c)


# ---------------------------------------------------------------------------
# 6. configuration sanity checks
# ---------------------------------------------------------------------------

def test_config_sanity():
    section("mining configuration sanity checks")
    good = MiningConfiguration(
        4096, 128,
        PeriodicPattern.from_list([0, 1, 2, 3]),
        PeriodicPattern.from_list([0, 1, 2, 3, 4, 5, 6, 7]))
    try:
        good.sanity_check(128, 128)
        check("default config accepted", True)
    except ValueError as exc:
        check("default config accepted", False, str(exc))

    bad_cases = [
        ("k not divisible by 64",
         MiningConfiguration(4000, 128, good.rows_pattern, good.cols_pattern)),
        ("rank not multiple of 16",
         MiningConfiguration(4096, 48, good.rows_pattern, good.cols_pattern)),
        ("k too small for rank",
         MiningConfiguration(512, 128, good.rows_pattern, good.cols_pattern)),
        ("tile too small",
         MiningConfiguration(4096, 128,
                             PeriodicPattern.from_list([0, 1]),
                             PeriodicPattern.from_list([0, 1]))),
    ]
    for name, cfg in bad_cases:
        try:
            cfg.sanity_check(128, 128)
            check(f"rejects {name}", False, "accepted!")
        except ValueError:
            check(f"rejects {name}", True)


def main() -> int:
    print("Pearl miner selftest")
    print(f"  python {sys.version.split()[0]}, numpy {np.__version__}")
    print(f"  pip blake3 : {'available' if _pkg_blake3 else 'not installed'}")
    print(f"  pearl_mining (Rust): {'available' if _rust else 'not installed'}")
    test_pinned_seed_vectors()
    test_blake3_against_pip()
    test_engine_vs_reference()
    test_reference_trial_share()
    test_config_sanity()
    test_against_rust_module()
    print(f"\n{_PASS} passed, {_FAIL} failed")
    return 1 if _FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
