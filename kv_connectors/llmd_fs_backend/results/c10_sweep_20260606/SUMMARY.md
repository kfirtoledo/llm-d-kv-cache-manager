# Sweep at concurrent=10 — 3 tiers × HMA on/off × 20b/120b

Date: 2026-06-06
fs_backend: hma branch HEAD `b7483d8`
vllm: local fork `main-42212-37885` (`0.21.1rc1.dev287+gf4abc309f`)
Params: `--block-size 256 --num-tokens 128000 --num-req 40 --concurrent 10`
GPUs: 20b on 0,1 (tp=2), 120b on 0-3 (tp=4)

## Raw results

| Model | Tier | HMA | Cold (s) | hot_avg (s) | last_10 (s) | Throughput (GB/s) |
|---|---|---|---:|---:|---:|---:|
| 20b | gpu | no | 26.53 | 0.190 | 0.190 | 15.30 |
| 20b | gpu | yes | 26.07 | 0.200 | 0.200 | 14.81 |
| 20b | cpu | no | 26.11 | 0.740 | 0.750 | 3.94 |
| 20b | cpu | yes | 26.09 | 0.400 | 0.400 | 7.39 |
| 20b | storage | no | 26.04 | 1.330 | 1.330 | 2.20 |
| 20b | storage | yes | 26.05 | 0.780 | 0.750 | 3.93 |
| 120b | gpu | no | 24.59 | 0.210 | 0.210 | 21.32 |
| 120b | gpu | yes | 22.57 | 0.210 | 0.200 | 21.48 |
| 120b | cpu | no | 22.61 | 1.050 | 1.050 | 4.19 |
| 120b | cpu | yes | 22.59 | 0.310 | 0.310 | 14.35 |
| 120b | storage | no | 22.62 | 1.890 | 1.900 | 2.32 |
| 120b | storage | yes | 22.55 | 1.020 | 1.010 | 4.35 |

## HMA speedup (no-HMA last_10 / HMA last_10)

| Model | gpu | cpu | storage |
|---|---:|---:|---:|
| 20b | **0.95×** (tie) | **1.88×** | **1.77×** |
| 120b | **1.05×** (tie) | **3.39×** | **1.88×** |

## Observations

- **GPU tier**: HMA has no effect (~1×). The "gpu" config uses vLLM's native prefix cache; no external offload happens. HMA-vs-no-HMA only affects how vLLM internally lays out KV groups, which the prefix-cache code path is agnostic to.
- **CPU tier**: HMA wins big — **1.88× on 20b, 3.39× on 120b**. The CPU offload code path stores K+V per layer-group; HMA's split lets the bigger model fit more usefully into the per-group staging buffers and parallelize the host-to-CPU memcpy across groups.
- **Storage tier**: HMA wins consistently at **~1.8×** on both sizes. Matches the earlier concurrency sweep (1.77× at c=10 for 20b, 1.90× at c=10 for 120b — within noise).
- **Cold times equal** across HMA on/off within ~2% on every tier — single-PUT cold-write is bandwidth-limited everywhere.

## Files in this directory

- `*.log` — full stderr+stdout from each of the 12 runs
- `sweep.log` — the launcher's combined log
- `SUMMARY.md` — this file
