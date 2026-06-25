# HMA (Hybrid Memory Architecture) in vLLM Offloading

**Status:** draft for blog post
**Audience:** infra/perf engineers familiar with vLLM offloading

This is a working summary of what HMA is, why hybrid models need it, how
the `llmd_fs_backend` connector implements it, and what we measured. Use
it as a source pool when writing the public post — drop anything
internal-only before publishing.

---

## TL;DR

For a hybrid model like **gpt-oss-120b** (interleaved sliding-window +
full-attention layers), enabling **HMA** (Hybrid Memory Architecture in
vLLM's KV cache manager) and a HMA-aware offload connector gives:

| Result | Value |
|---|---|
| GPU KV cache capacity | **1.77×** more tokens fit (2.66M → 4.71M @ tp=4, 128k seq) |
| Hot read latency, CPU tier | **1.80×–1.99×** faster (20b / 120b) |
| Hot read latency, Storage tier | **1.79×–1.87×** faster |
| Throughput, prod (900 users, conc=80) | **+18%** total tok/s (286k → 337k) |
| Cold write time | essentially unchanged |

The wins come from one simple structural change: HMA splits one logical
KV cache into **N groups by attention type**. Each group has its own
buffer sized to its actual needs (SWA layers don't need full-sequence
storage), and the offload backend serves the groups in parallel.

---

## Background: hybrid attention models

Modern open models like **gpt-oss** (20B and 120B) and **gemma-3** use
**interleaved attention layers**: every other layer is sliding-window
attention (SWA) with a small fixed window (typically 128 or 1024
tokens), and the rest are standard full-attention layers.

The motivation is well-known: SWA layers are cheap (compute and KV
cache both scale with the *window*, not full sequence length), so
mixing them in cuts model cost without much quality loss.

But the optimization only pays off if the **runtime takes it
seriously**. If the KV cache layer allocates a full-sequence buffer
for every layer (full-attention *and* SWA), you've thrown away the SWA
win for memory — even though the compute is still saved.

This is exactly what happens when the **hybrid KV cache manager is
disabled** in vLLM. From the warning vLLM prints:

```
Hybrid KV cache manager is disabled for this hybrid model.
This means we do not enable any optimizations for saving KV cache
memory (e.g., dropping the KV cache outside the sliding window).
The compute of layers like sliding window is still saved.
```

So no-HMA = full-attention buffer for everything, even SWA.
HMA = separate buffer per attention type, sized to actual need.

---

## What HMA is, structurally

In vLLM V1 with HMA enabled, the KV cache becomes a set of **KV cache
groups**, one per distinct attention pattern:

- gpt-oss-120b at tp=4 → **2 groups**: 18 full-attn layers + 18 SWA layers.
- gemma-3-27b → **7 groups** (multiple SWA window sizes + a global pattern).

Each group:
- Gets its own GPU memory budget (vLLM sizes it to whatever the
  attention pattern actually needs — SWA layers get a small window
  buffer, full layers get full-sequence).
- Has its own "logical block index" — `gpu_spec.block_indices[g]` —
  that tracks where the group's first block lives in the current request.
- Stores its own subset of canonical KV tensors. The connector sees a
  per-group list of tensor indices.

When HMA is disabled, vLLM collapses everything into a single group
sized to full-attention worst case. Correctness is maintained but
memory is wasted.

---

## What HMA gives you on the GPU side (before any offload)

Real numbers from our setup, gpt-oss-120b @ tp=4, bf16, mxfp4 weights,
`max_model_len=129000`, `gpu_memory_utilization=0.85`:

| Setup | GPU KV cache size | Max concurrency @ 128k tok/req |
|---|---:|---:|
| HMA off | 2,656,155 tokens | 20.59× |
| **HMA on** | **4,709,024 tokens** | **36.50×** |

So with HMA, **77% more requests fit in memory** at full context.
That alone is reason enough to enable HMA when running hybrid models.

vLLM logs this at startup (`kv_cache_utils.py:1733`).

---

## The offload connector side: making HMA correct end-to-end

The GPU savings are nice, but they're only the first half. To
actually offload these caches to CPU or shared storage, the
**connector** has to know about the group structure too — otherwise it
either crashes, mishandles the layout, or silently performs poorly.

In the `llmd_fs_backend` connector we ship as part of `llm-d`, HMA
support touches four layers:

### 1. Python: `worker.py`

The handler receives `CanonicalKVCaches` from vLLM — a dataclass
wrapping a flat tensor list plus per-group typed refs. From it, we
derive three flat views the C++ engine consumes:

```python
tensors              = [ct.tensor for ct in kv_caches.tensors]
group_tensor_indices = [[ref.tensor_idx for ref in g]
                        for g in kv_caches.group_data_refs]
per_group_block_bytes = [sum(ref.page_size_bytes for ref in g)
                         for g in kv_caches.group_data_refs]
```

Per-group block bytes (summed from each group's
`CanonicalKVCacheRef.page_size_bytes`) drives the per-group CPU
staging slot size — so a SWA group gets a small buffer, a full-attn
group gets a large one.

### 2. File mapping: alignment-aware

For each group, we split its GPU blocks into files of
`gpu_blocks_per_file` blocks. A group whose first block lands
mid-file (because the SWA window has slid past the start of the
request) is **head-partial** — we have to handle it correctly:

```python
file_logical_lo = start_file_idx * gpu_blocks_per_file  # rounded down
slice_lo = max(start_block_idx, file_logical_lo)  # head-partial fix
head_offset = slice_lo - file_logical_lo           # 0 for aligned, >0 for partial
```

The `max()` is necessary for the first file when
`start_block_idx % gpu_blocks_per_file != 0`. SWA groups in
long-context requests routinely have non-zero, non-aligned
`block_indices[g]` because vLLM doesn't align block indices to the
connector's chunk size — it has no reason to.

### 3. `head_offset` propagation

The `head_offset` (slot offset in GPU-blocks where the group's data
starts in a file) is plumbed from Python all the way through:

```
Python (worker.py) → C++ engine (storage_offload.cpp)
       → StorageHandler (per-backend) → POSIX file_io / GDS gds_file_io
                                       → TensorCopier
```

In the POSIX backend, this means writes start at
`head_offset × bytes_per_block` into the staging buffer; reads
pull from the same slot. In the GDS backend, the file offset
becomes `head_offset × tensors × block_size`. The on-disk layout
stays packed (no padding); `head_offset` only controls the **CPU
staging slot**.

### 4. C++ engine: per-group bookkeeping

The `StorageOffloadEngine` takes:
- `group_tensor_indices`: per-group tensor positions
- `per_group_block_bytes`: per-group block byte size
- `head_offsets`: per-file slot offset

Per-job operations include the group index, so the engine knows
which tensor subset to copy and what staging slot to use.

---

## Benchmark methodology

### Local test (`test_cold_hot_req.py`)

The connector ships a benchmark script that times batched
`llm.generate()` calls.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 test_cold_hot_req.py \
  --test storage \             # tier: gpu / cpu / storage / multi-connector
  --model openai/gpt-oss-120b \
  --tp-size 4 \
  --block-size 256 \           # offload block size
  --num-tokens 128000 \        # input length per prompt
  --concurrent 10 \            # distinct prompts batched per iter
  --num-req 40 \               # iterations
  --hma                        # or --no-hma
```

What `--concurrent 10` means: build **10 distinct prompts** at startup
(each with a different fill_id, so each hashes to a different cache
key), and every iteration calls `llm.generate(prompts)` with **all 10**
in one batched submission. vLLM schedules them in parallel; the call
returns when **all 10 finish**.

Iteration 1 is "cold" (writes all 10 prompts' KV to cache).
Iterations 2–40 are "hot" (reads all 10 from cache). We report
`last_10_avg` — the mean wall-clock of iters 31–40.

**Important caveat:** this measures **batch completion time**, not
TTFT. We're measuring "how fast can the offload engine serve 10
concurrent hot prompts," not per-request latency. For TTFT use
`vllm bench serve` (what the prod chart is based on).

### Workload pattern: `prefix_repetition`

Both local and prod benchmarks build prompts where each request has
a long, distinct prefix. This is the worst case for the offload
tier — every request's prefix is a unique cache key, so cold-pass
fills the cache with N×prefix_len tokens and hot-pass reads them
back. Realistic for shared-prefix serving (system prompts,
multi-turn) where you want offload to actually do work.

---

## Results

### Local sweep, 10 reps per config, conc=10, 128k tokens

These numbers are from the most reliable dataset we have — 80 runs
total, averaged across 10 reps per config:

| Model | Tier | HMA | last_10 mean (s) | stdev | tput mean (GB/s) | HMA speedup |
|---|---|:---:|---:|---:|---:|---:|
| 20b | cpu | no | 0.721 | 0.060 | 4.12 | — |
| 20b | cpu | **yes** | **0.400** | 0.012 | 7.32 | **1.80×** |
| 20b | storage | no | 1.337 | 0.010 | 2.19 | — |
| 20b | storage | **yes** | **0.748** | 0.004 | 3.92 | **1.79×** |
| 120b | cpu | no | 0.791 | 0.074 | 5.62 | — |
| 120b | cpu | **yes** | **0.397** | 0.059 | 11.36 | **1.99×** |
| 120b | storage | no | 1.887 | 0.067 | 2.33 | — |
| 120b | storage | **yes** | **1.011** | 0.005 | 4.36 | **1.87×** |

(Plots: `results/repeats10_c10_20260606_0831/latency.png`,
`throughput.png`, `speedup.png`.)

Observations:
- HMA hot reads are **~1.8–2.0× faster** across the board.
- HMA configs are extremely stable (stdev 0.004–0.060 s); no-HMA configs
  have wider spread (transient contention occasionally drops a run faster).
- **Cold time is identical** within 1% across HMA on/off on every tier —
  single-PUT cold-write is bandwidth-limited, not parallelism-limited.

### Why HMA helps storage I/O specifically

HMA splits a single group's KV bytes into **N groups, each with its
own files**. With gpu_blocks_per_file=64 and our gpt-oss test:
- no-HMA: 1 group → 1 file per chunk, **full size** (e.g., 18 MB)
- HMA: 2 groups → 2 files per chunk, **half size each** (9 MB each)

That doubled file count + halved file size lets the I/O thread pool
serve more parallel requests on the read path: each worker can pick
up a smaller transfer and complete it faster, instead of a few
workers being tied up on big serial reads.

On **fast local NVMe** the per-file metadata cost is sub-ms, so the
parallelism win dominates. On **shared GPFS in prod** the metadata
cost is higher (network round-trips), but **at modest concurrency the
parallelism win still wins** (see prod results below).

### Concurrency scaling sweep, gpt-oss-120b storage tier

| conc | no-HMA last_10 (s) | HMA last_10 (s) | HMA speedup |
|---:|---:|---:|---:|
| 1 | 0.138 | 0.083 | 1.66× |
| 5 | 0.967 | 0.518 | 1.87× |
| 10 | 1.900 | 1.000 | 1.90× |
| 20 | 3.760 | 2.040 | 1.84× |

(Same pattern for gpt-oss-20b: 1.54× → 1.68× → 1.77× → 1.77×.)

The HMA speedup grows from concurrent=1 → 10 and then plateaus
because the I/O thread pool (24 workers/GPU) saturates.

### Prod (vllm bench serve, gpt-oss-120b multi-connector, 900 users)

Real serving workload, not the local synthetic test. CPU tier
500 GiB + Storage tier (GPFS), rate=80 req/s, max-concurrency=80:

| Config | Total tok/s mean (5-run) | TTFT mean | TTFT p99 |
|---|---:|---:|---:|
| HMA | **337,333** | 3,636 ms | 4,683 ms |
| no-HMA | 286,786 (stdev 0.3%) | 4,343 ms | 5,110 ms |

**+18% throughput, 16% lower TTFT mean, 8% lower TTFT p99.**

The chart from an older internal benchmark (`exp22`) suggested HMA
was *slower* than non-HMA in prod — that turned out to be an
artifact: the "no-HMA" yaml was missing `--disable-hybrid-kv-cache-manager`,
so vLLM auto-enabled HMA anyway. Once the comparison is apples-to-apples
on the current fs_backend, HMA decisively wins.

---

## Gotcha: `disable_hybrid_kv_cache_manager` default

vLLM's default for `disable_hybrid_kv_cache_manager` is `None`,
which resolves to **`False` (HMA on)** in most modern configurations.
Specifically, if your `kv_transfer_config` connector subclasses
`SupportsHMA`, vLLM auto-enables HMA. `OffloadingConnector` does;
`MultiConnector` does only if every sub-connector does.

So if you set up an offload pod and *don't* explicitly disable HMA,
you're probably running HMA on the GPU side — good if your offload
connector handles it, bad if it doesn't. To **definitely** turn HMA
off:

```yaml
--disable-hybrid-kv-cache-manager
```

To **definitely** turn it on (will error if unsupported):

```yaml
--no-disable-hybrid-kv-cache-manager
```

---

## Commits that landed HMA support in `llmd_fs_backend`

Branch: `hma` (PR #476 against `llm-d/llm-d-kv-cache-manager`).

```
ed9e080  fs_backend: default max_write_queued_seconds 10s -> 30s
dbaad02  fs_backend: exact bytes per job in PUT/GET metrics
8ec05aa  fs_backend: alignment-aware file mapping via block_indices
0b1ff60  csrc: take per_group_block_bytes from canonical refs
1982780  fs_backend: use OffloadingSpec.hash_block_size (DSv4 compat)
9794f30  csrc: batch KV block copies via cudaMemcpyBatchAsync
b2d701e  feat: Add HMA support to fs connector
27b3c86  fs_backend: extract _num_files_for_group + drop unused per_block_bytes
9175ab9  fs_backend: pass CanonicalKVCaches to _create_engine directly
b7483d8  fs_backend: propagate head_offset for partial file transfers (POSIX + GDS)
```

Open review item: OR's comment on `worker.py:228` (`max()` redundancy).
The `max()` is required for SWA groups in long-context requests where
the sliding window has advanced past the start.

---

## Limitations / future work

- Concurrency sweep plateaus around 1.8–2.0× — to push further we'd
  need a wider I/O thread pool or a non-blocking storage backend.
- The TensorCopier **kernel path** doesn't honor `head_offset` yet;
  it's gated with `TORCH_CHECK(head_offset == 0)`. Off by default
  (`USE_KERNEL_COPY_*=0`), so practical impact is nil today.
- GDS backend supports head_offset but leaves slots before the
  offset unwritten (OS-zeroed on read). Fine for correctness in
  our setup; worth documenting.
- `gemma-3-27b-it` reports `num_groups=7` (multiple SWA windows +
  full attention). Tested, works, no observable regression. Worth a
  dedicated benchmark if blog needs a non-gpt-oss data point.
- HMA TTFT win in prod (16% mean, 8% p99) is consistent with the
  local cache-read speedup but not as dramatic. Prod is GPU-compute
  bound at 900 users × 5 output tokens; the offload speedup is partly
  diluted by prefill+decode cost.

---

## What to highlight in the public blog post

If I were writing the public version, the punchy story is:

1. **"Hybrid models leave performance on the table without HMA"** —
   77% more KV cache capacity from a config flag, before any offload.
2. **"HMA also makes offloading 2× faster"** — concrete latency
   reduction on both CPU and storage tiers.
3. **"It's not just GPU-side: the connector has to participate"** —
   show the file split, the `head_offset` plumbing, the per-group
   staging. Most connectors don't have this and silently drop to
   no-HMA mode.
4. **"It works in prod, not just synthetic"** — +18% on 900 users with
   real `vllm bench serve` streaming workload.
5. **"Watch the default flag"** — `disable_hybrid_kv_cache_manager`
   auto-resolution is easy to get wrong; show how to verify with
   `num_groups=N` in the connector startup log.

---

## References

- vLLM PR #37885: canonical KV caches for HMA (cherry-picked in our
  fork)
- vLLM PR #42212: in-progress upstream HMA work
- `llmd_fs_backend` results dirs:
  - `results/repeats10_c10_20260606_0831/` — 10-rep mean dataset
  - `results/c10_sweep_20260606/` — single-shot 3-tier sweep
- `gpt-oss-multi-hma-v2.yaml` — k8s prod deployment used for the
  900-user benchmark
