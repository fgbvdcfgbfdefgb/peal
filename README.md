# Pearl solo miner (PRL) — source-only, GitHub-relayed

Solo Pearl mining setup where the **node runs on Windows** and the **miner
runs on an offline Linux machine** that has no internet except `pip` and
`git pull`/`git push`. The two machines talk through two files in this
repository:

```
                 Windows machine                         Linux machine (offline)
 ┌───────────────────────────────────────┐          ┌──────────────────────────┐
 │  pearld (release)                     │          │                          │
 │     │ getblocktemplate                │          │   python -m pearl_miner  │
 │     ▼                                 │          │     │ reads              │
 │  pearl-gateway  ◄── JSON-RPC ──┐      │          │     ▼                    │
 │     │ 127.0.0.1:8337           │      │          │   jobs.txt  ◄── git pull │
 │     ▼                          │      │          │     │ mines              │
 │  relay (this repo)  ───────────┘      │  GitHub  │     ▼                    │
 │     │ pushes jobs, pulls shares       │ ◄──────► │   share.txt ── git push ►│
 │     ▼                                 │          │                          │
 │  submitblock (full ZK proof + block)  │          │                          │
 └───────────────────────────────────────┘          └──────────────────────────┘
```

* The **miner never touches the network**: it watches `jobs.txt`, mines with
  its CuPy/NVRTC GPU engine (numpy fallback), verifies every share locally,
  and appends it to `share.txt`. You `git pull` jobs and `git push` shares
  yourself.
* The **relay never mines**: it asks the gateway for the current job,
  publishes it to `jobs.txt`, and submits every new share in `share.txt` to
  the gateway (which builds the ZK proof and submits the block to `pearld`).

Everything here runs **from source** — nothing is compiled into a release
binary.

---

## Repository layout

```
miner/                     the miner (runs on the offline Linux box)
  pearl_miner/
    blake3k.py             vectorised keyed BLAKE3 + Merkle tree (numpy)
    patterns.py            periodic patterns + thread partition
    core.py                header/config serialisation, job key, noise
                           seeds, noise generation, difficulty bounds
    trial.py               readable single-trial reference implementation
    engine_np.py           the batched numpy mining engine (CPU fallback)
    cuda_kernels.py        NVRTC CUDA kernels (compiled at runtime by CuPy)
    engine_cuda.py         the GPU mining engine (A10G / sm86 and newer)
    proof.py               PlainProof build/serialise/verify
    jobfile.py             jobs.txt / share.txt handling
    __main__.py            CLI:  python -m pearl_miner
  selftest.py              cross-checks against the official Rust module
  kernel_logic_test.py     kernel algorithms vs CPU reference (no GPU needed)
  nvrtc_compile_check.py   compile-checks the CUDA source (no GPU needed)
  gpu_selftest.py         GPU-vs-CPU cross-check, run it once per machine
relay/                     the relay (runs on Windows next to pearld)
  relay.py                 main loop: gateway <-> GitHub
  gateway_client.py        line-delimited JSON-RPC client
  github_sync.py           GitHub Contents API client (no git needed)
windows/                   PowerShell helpers for the Windows machine
run_miner.sh               Linux launcher (--install creates the venv)
requirements.txt           miner dependencies (numpy, optional blake3, CuPy)
requirements-relay.txt     relay dependencies (requests)
```

---

## 1. Windows machine (node + relay)

You need `pearld` (release), a wallet address, `pearl-gateway`, and this
relay.

### 1.1 Install the node (release build)

Follow the official installer from the Pearl monorepo — it puts
`pearld`/`prlctl`/`oyster` into `%LOCALAPPDATA%\Pearl\bin` and writes
`%LOCALAPPDATA%\Pearld\pearld.conf`:

```powershell
irm https://raw.githubusercontent.com/pearl-research-labs/pearl/main/install.ps1 | iex
```

Create/verify a wallet and a **Taproot mining address**, then start the node
with mining enabled, e.g.:

```powershell
pearld --miningaddr=<your taproot address>
```

### 1.2 Install the gateway

`pearl-gateway` is the Python service that turns mining shares into full
blocks. It is not on PyPI, so take it from the monorepo (the machine has
internet):

```powershell
git clone https://github.com/pearl-research-labs/pearl.git C:\pearl
cd C:\pearl
# Python 3.12 recommended; then install the workspace package:
pip install ./miner/pearl-gateway
```

Run the gateway next to the node (it reads `PEARLD_RPC_URL`,
`PEARLD_RPC_USER`, `PEARLD_RPC_PASSWORD`, `PEARLD_MINING_ADDRESS` and
exposes the miner RPC on TCP `127.0.0.1:8337` by default):

```powershell
$env:PEARLD_RPC_URL      = "http://127.0.0.1:44107"
$env:PEARLD_RPC_USER     = "user"
$env:PEARLD_RPC_PASSWORD = "pass"
$env:PEARLD_MINING_ADDRESS = "<your taproot address>"
$env:MINER_RPC_TRANSPORT = "tcp"
$env:MINER_RPC_PORT      = "8337"
python -m pearl_gateway
```

### 1.3 Install and run the relay

```powershell
cd <this repo>
.\windows\setup_relay.ps1        # creates .venv-relay, writes .env-relay.ps1
notepad .env-relay.ps1           # paste your GitHub token
.\windows\start_relay.ps1
```

The relay logs each published job and each submitted share. It keeps its
progress in `.relay-state.json` (which share bytes it already consumed), so
restarting it is safe.

Relay environment variables (all optional except the token):

| Variable                | Default                        | Meaning                        |
|-------------------------|--------------------------------|--------------------------------|
| `GITHUB_TOKEN`          | – (required)                   | token with `contents:write`    |
| `GITHUB_REPO`           | `fgbvdcfgbfdefgb/pearl-miner`  | `owner/name`                   |
| `GITHUB_BRANCH`         | `main`                         | branch to sync                 |
| `JOBS_FILE`             | `jobs.txt`                     | jobs file in the repo          |
| `SHARES_FILE`           | `share.txt`                    | shares file in the repo        |
| `RELAY_POLL_SECONDS`    | `1.0`                          | gateway poll interval          |
| `RELAY_SYNC_SECONDS`    | `5.0`                          | GitHub sync interval           |
| `RELAY_STATE_FILE`      | `.relay-state.json`            | consumed-offset memory         |
| `RELAY_DRY_RUN`         | `0`                            | `1` = log only, change nothing |

---

## 2. Linux mining machine (offline)

### 2.1 One-time setup

```bash
git clone https://github.com/fgbvdcfgbfdefgb/pearl-miner.git
cd pearl-miner
./run_miner.sh --install        # creates .venv, installs numpy + blake3
```

Optional but recommended — validate the miner against the official Rust
implementation (needs the `pearl_mining` wheel built from the monorepo with
`maturin build --release`; skipped automatically when absent):

```bash
PEARL_SELFTEST=1 ./run_miner.sh
```

### 2.2 GPU acceleration (4× A10G, sm86)

The monorepo's own GPU miner (`pearl-gemm`) is H100-only (`sm_90a`) and needs
`nvcc` + CUTLASS + PyTorch 2.11, so it cannot run on an A10G. This miner
ships its own engine instead: **CuPy + NVRTC**, i.e. `pip install` only —
the CUDA kernels are compiled at runtime by the driver, with no compiler
toolchain, no CUTLASS and no PyTorch. Compute capability 7.5+ is required
(Turing and newer; A10G is 8.6).

```bash
./run_miner.sh --install-gpu    # installs cupy-cuda12x (or -cuda11x)
./run_miner.sh --gpu-selftest   # GPU-vs-CPU cross-check, then exits
```

Two checks that need **no GPU** catch most problems before you ever touch
the mining machine:

```bash
./run_miner.sh --kernel-logic      # kernel algorithms vs the CPU reference
pip install nvidia-cuda-nvrtc-cu12 && python miner/nvrtc_compile_check.py
```

`--gpu-selftest` then compares the GPU pipeline against the numpy reference on
identical matrices — commitment roots, noise seeds, noise matrices, the
noised GEMM and the jackpot fold, the jackpot hashes, and the hit set under
both a saturating bound (every partition must hit) and a realistic bound
(the hit sets must match exactly) — and then verifies a mined share locally
and with the official Rust module when it is installed. **Run it once on
each mining machine before trusting the GPU.**

Then mine on all four GPUs (one worker thread per device, each with its own
RNG stream, all appending to the same `share.txt`):

```bash
./run_miner.sh --backend cuda --devices 0,1,2,3 --gpu-batch 512
```

`--backend auto` (the default) uses the GPU whenever CuPy is importable and
the configuration is supported by the GPU engine, and silently falls back to
the numpy engine otherwise. Use `--backend numpy` (or `PEARL_BACKEND=numpy`)
to force the CPU engine.

The GPU engine supports dense patterns with `m`/`n` multiples of 32, `k` a
multiple of `rank`, `rank` a multiple of 32, and matrices up to 1536 chunks
(1.5 MB) each; anything else falls back to numpy. The GEMM runs on `dp4a`
with exact int32 accumulation (fp32/fp16 accumulation is not exact for these
tiles), the commitment roots use a one-block-per-matrix BLAKE3 Merkle tree
kernel, and the matrices are counter-based splitmix64 streams, so a hit
carries its A/Bᵀ back to the host for CPU proof building.

Expect roughly **10⁵–10⁶ trials/s per A10G** for the default configuration
(m·n·dot_len ≈ 67 M MACs per trial); the exact number depends on the job's
`k`/`rank` and the driver version. Every share is still verified locally
before it is written.

### 2.3 Mining loop

Terminal 1 — the miner:

```bash
./run_miner.sh
```

It waits for a job in `jobs.txt`. Whenever the relay publishes a new job,
pull it and keep the miner running:

```bash
git pull            # picks up the new jobs.txt
```

When the miner finds a share it appends a line to `share.txt` and prints
`*** SHARE FOUND ... ***`. Push it:

```bash
git add share.txt && git commit -m "shares" && git push
```

The relay submits it within a few seconds; if the block template moved on in
the meantime the relay logs `skipping stale share` (the gateway would skip
it anyway — Plonky2 proving is expensive, so the relay pre-checks the
header).

### 2.4 Useful options

```
--backend auto|cuda|numpy   engine selection (default auto)
--devices 0,1,2,3           GPU ids, one worker thread each (default 0)
--gpu-batch N               trials per batch per GPU (default 512)
--batch N                   trials per batch, numpy engine (default 256)
--m/--n/--k/--rank/--rows/--cols   mining configuration (see below)
--max-shares N stop after N shares
--once N       mine N batches then exit
--stats S      stats print interval in seconds
```

The default configuration is a consensus-valid **dense** setup:

```
m = 128, n = 128, k = 4096, rank = 128
rows pattern [0,1,2,3]      (tile height 4)
cols pattern [0..7]         (tile width  8)   -> 32 outputs per tile
```

Current mainnet rules this satisfies: certificate version **V3** (salted
noise seeds — set automatically from the job's `cert_version`), **dense**
(non-MoE) proofs, and the **rank penalty** (rank 128 = neutral point).

---

## How the pieces fit together

| Step | Who | What |
|------|-----|------|
| 1 | gateway | refreshes the block template from `pearld` every second |
| 2 | relay | `getMiningInfo` → publishes `{job_id, incomplete_header_bytes, target, cert_version}` to `jobs.txt` |
| 3 | you | `git pull` on the mining machine |
| 4 | miner | mines trials: random A/Bᵀ → commitment roots → noise seeds → noised GEMM → jackpot hash ≤ target |
| 5 | miner | on a hit: builds the Merkle proofs, re-verifies the share locally, appends it to `share.txt` |
| 6 | you | `git push` from the mining machine |
| 7 | relay | reads new `share.txt` lines, skips stale headers, `submitPlainProof` |
| 8 | gateway | builds the ZK proof and submits the block to `pearld` |

See [ARCHITECTURE.md](ARCHITECTURE.md) for the protocol details (wire
formats, the mining algorithm, and what the selftest pins down).

## Troubleshooting

* **`jobs.txt` never appears** — is the relay running? Does the gateway
  respond to `getMiningInfo` (check `pearl-gateway` logs)? Is the token
  valid (`contents:write` on the repo)?
* **miner says "waiting for a job"** — you have not `git pull`ed yet, or
  `jobs.txt` is not in the working directory (pass `--jobs path`).
* **shares pile up unsubmitted** — the relay only submits shares whose
  `incomplete_header_bytes` matches the current job. If the template keeps
  changing (busy chain), shares can go stale before you push them; push
  promptly.
* **"job rejected (degenerate mining configuration)"** — the node published
  a target so easy that the difficulty bound overflows 256 bits. Nothing to
  do; wait for a normal job.
* **out of memory** — lower `--batch` (each trial costs m·k + n·k bytes plus
  a m·n int32 tile).
* **GPU engine not used** — `pip show cupy-cuda12x` in the venv, then
  `./run_miner.sh --gpu-selftest`. If the driver or the CUDA runtime version
  does not match the CuPy wheel, CuPy fails to import and the miner falls
  back to numpy (it prints the reason). `--backend cuda` turns that into a
  hard error so you can see it.
* **`kernel_logic_test.py` fails** — the kernel algorithms disagree with the
  CPU reference; nothing else about the GPU path can be trusted.
* **`nvrtc_compile_check.py` fails** — the CUDA source does not even compile;
  reinstall CuPy/NVRTC or report it.
* **`gpu_selftest.py` reports a mismatch** — do not mine with the GPU until
  it passes; the numpy engine is the reference and is always available with
  `--backend numpy`.
