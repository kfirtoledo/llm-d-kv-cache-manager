# fs_backend trace-mode timing instrumentation — handoff

Adds per-request timing breakdown for the vLLM `OffloadingConnector` storage
tier, so you can see exactly where time goes on a write→load cycle. All of it
is **gated on trace mode** (`STORAGE_LOG_LEVEL=TRACE`) and is a **no-op with
zero overhead otherwise**.

Status: implemented and run manually on `meta-llama/Meta-Llama-3.1-8B`
(single GPU). Needs an independent verification pass — see **How to test** and
**Open questions** below.

---

## What was added

Three independent timing signals, each surfaced as a segment on the per-request
line printed by the throughput harness:

| Signal | What it measures | Where it is emitted |
|--------|------------------|---------------------|
| **RT** (block round-trip lifetime) | wall-clock a KV block sat on storage between being **written** (`complete_store`) and **loaded back** (`prepare_load`) on a later request | `manager.py` (engine-core process) |
| **PHASE[W]/[S]** (connector phase time) | synchronous CPU time the connector spends per phase: scheduler `lookup/alloc/meta/finished`, worker `read/write/poll` | `phase_timing.py` monkey-patch (both processes) |
| **XFER** (engine read/copy split) | per-file I/O-thread time inside `transfer_async`: GET `read`(storage→CPU) vs `copy`(CPU→GPU); PUT `copy`(GPU→CPU) vs `write`(CPU→storage) | already emitted by `storage_offload.so` as `[TIME]` lines — harness just parses them |

Example per-request line (128K tokens, hot request):

```
[2] 2.140s | GET 1jobs 7168files 14336MB | PUT 1jobs 832files 1664MB, 1/1 done in 0.317s
   | LOOKUP 0 | RT 7168blk lifetime avg=7.303s min=5.805s max=11.001s
   | PHASE[W] read=76.4ms write=0.1ms poll=1.0ms Σ=77.6ms
   | PHASE[S] lookup=35.9ms alloc=3.1ms meta=2.4ms Σ=41.4ms
   | XFER GET read=4707.5ms copy=820.4ms (7148f) | PUT copy=12.8ms write=70.4ms (26f)
   | Δ+69 files ...
```

---

## Files changed

- **`llmd_fs_backend/manager.py`** — RT (block round-trip lifetime).
  - Module flag `_TRACE_ENABLED` (true when `STORAGE_LOG_LEVEL` ∈ {TRACE, DEBUG}).
  - `__init__`: `self._block_write_times: dict[bytes, float]`.
  - `complete_store()`: on success, record `time.monotonic()` per block hash.
  - `prepare_load()`: calls `_record_roundtrips()`, which emits
    `block_roundtrip: blocks=N lifetime_avg=.. lifetime_min=.. lifetime_max=.. [s]`.

- **`llmd_fs_backend/phase_timing.py`** — NEW. Connector phase timing.
  - `install_connector_phase_timing_patch()` monkey-patches the inner classes
    `OffloadingConnectorScheduler` and `OffloadingConnectorWorker` (NOT the
    `OffloadingConnector` facade — patching the inner classes is what lets us
    split read/write/poll, which the facade lumps into one `get_finished`).
  - Accumulates per-phase delta time, flushes one
    `connector_phase: role=<sched|worker> <phase>=<ms>:<calls> ... total=<ms>`
    line per engine step (flush hooked on `build_connector_meta` for sched,
    `get_finished` for worker — the last connector call of each step).
  - Follows the existing monkey-patch pattern in `metrics.py`.

- **`llmd_fs_backend/__init__.py`** — calls
  `install_connector_phase_timing_patch()` at import (next to the existing
  `install_offload_metric_suffix_patch()`).

- **`test_cold_hot_req.py`** — harness parsing + display (the run script).
  - New `--trace` flag: sets `STORAGE_LOG_LEVEL=TRACE` BEFORE any
    `LLM()`/worker/engine-core process spawns (so subprocesses inherit it at
    import time).
  - `TransferTally` extended to parse + aggregate per request:
    `block_roundtrip:` (RT), `connector_phase:` (PHASE), and the engine
    `[TIME] read/write phase N` lines (XFER).
  - Per-request print gains `RT`, `PHASE[W]`, `PHASE[S]`, `XFER` segments.

No C++ / `.so` changes — the read/copy split was already emitted by the engine
at TRACE level; we only surface it.

---

## How to run

Single GPU, storage tier, trace on:

```bash
cd kv_connectors/llmd_fs_backend
CUDA_VISIBLE_DEVICES=<free_gpu> python test_cold_hot_req.py \
  --test storage --model meta-llama/Meta-Llama-3.1-8B \
  --tp-size 1 --trace --num-req 10 --num-tokens 10000
```

For 128K context use `--num-tokens 128000` (8B supports 128K; needs ~16 GB KV
+ ~16 GB weights, fits on one 80 GB GPU). Drop `--trace` for real throughput.

---

## How to test (verification checklist)

1. **No-op when off**: import `llmd_fs_backend` without `STORAGE_LOG_LEVEL` set;
   confirm `phase_timing._TRACE_ENABLED is False` and the connector methods are
   NOT wrapped (`getattr(OffloadingConnectorWorker.get_finished,
   'llmd_fs_phase_timed', False) is False`).
2. **Patch installs in trace**:
   ```bash
   STORAGE_LOG_LEVEL=TRACE python -c "import llmd_fs_backend.phase_timing as pt; \
     pt.install_connector_phase_timing_patch(); \
     from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import OffloadingConnectorWorker as W; \
     print(getattr(W.get_finished, pt.PATCHED_FLAG, False))"   # -> True
   ```
3. **End-to-end**: run the command above; confirm every hot request shows
   non-empty `RT`, `PHASE[W]`, `PHASE[S]`, `XFER` segments.
4. **Sanity of numbers**:
   - `PHASE[*] Σ` should be small (ms) and scale UP with block count
     (≈1–2 ms at 10K tokens → ≈40–90 ms at 128K).
   - `XFER` GET `read ≫ copy`; PUT `write ≫ copy`.
   - RT `min` → ~0s (just-written block reused); `max` grows over the run
     (oldest cold blocks age).
5. **Regression**: a non-trace run must produce the SAME throughput as before
   this change (instrumentation is fully gated).

---

## Key findings (manual runs, 8B, 1× H100, storage tier, fs flavor)

- **Connector sync cost is tiny.** `PHASE[*] Σ` is single-digit-to-tens of ms
  out of a ~200 ms (10K) / ~2 s (128K) request. `write=0.0ms` because
  `prepare_store_kv` only *queues* the async offload.
- **Storage I/O dominates, GPU staging copy is cheap.** read ≈ 4–6× copy on
  GET; write ≈ 15× copy on PUT (128K cold: write 11.6 s vs copy 0.77 s of
  aggregate thread-time).
- **`--trace` perturbs throughput ~9×** (see caveat). 128K example:
  - no-trace baseline: hot ≈ 0.39 s, ≈ 42.5 GB/s.
  - with `--trace`: hot ≈ 1.9 s, ≈ 4.85 GB/s.

---

## Caveats — read before trusting numbers

- **Do NOT read absolute throughput off a `--trace` run.** Trace emits ~22k
  per-file `[TIME]` lines PER REQUEST (a 6-request 128K run = 45 MB / 182k
  log lines, 132k of them `[TIME]`). Each is formatted in C++, written to
  stderr, then regex-matched by the harness stderr-tail thread, which stalls
  the I/O worker threads. Trace measures logging, not storage. Use no-trace
  for perf; use trace only for the proportional breakdown.
- **XFER times are aggregate I/O-thread time**, summed across
  `threads_per_gpu` workers and overlapped with compute — NOT request
  wall-clock. (e.g. `read=4707ms` overlapped into a 2 s request.) Per-file:
  divide by the `(Nf)` file count.
- **PHASE times are per-step deltas summed per request** = connector
  synchronous CPU time only. The request's wall-clock is dominated by GPU
  prefill compute + overlapped async transfer, so `Σ ≠ request latency`.
- **`last_10_avg` includes the cold request when `--num-req < 10`** — not
  comparable to a 40-request steady-state average.

---

## Open questions / possible next steps

- Add a **wall-clock** GET/PUT transfer duration (from the existing
  `Transfer finished ... time=` DEBUG lines) next to the aggregate XFER
  thread-time, to show overlap efficiency directly.
- A **`STORAGE_LOG_LEVEL=DEBUG`** middle ground keeps `Transfer finished`,
  `connector_phase`, `block_roundtrip` (all DEBUG) but drops the 132k per-file
  `[TRACE]` lines → near-full speed, but loses the read/copy split. Consider
  making XFER opt-in separately so PHASE/RT can run at low overhead.
- `manager._block_write_times` grows unbounded across a run (fine for benches;
  ~1600 entries for a 40-req/10K run). Add an eviction/cap if used long-lived.
- The two `_TRACE_ENABLED` flags (`manager.py`, `phase_timing.py`) read the env
  at import; consider centralizing.

---

## Environment note

Runs need a free GPU. During this work the shared box's 8 GPUs were
intermittently fully reserved by other workloads (≈78/81 GB used) — if the
worker dies with `AssertionError: local rank 0 is out of bounds`, it means
`CUDA_VISIBLE_DEVICES` resolved to empty (no free GPU), not a code fault. Pick a
card showing <5 GB used.
