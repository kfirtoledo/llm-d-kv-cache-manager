# 10-rep sweep at concurrent=10 — CPU + Storage offloading × HMA on/off × 20b/120b

Date: 2026-06-06
fs_backend: hma branch HEAD `b7483d8`
vllm: local fork `main-42212-37885` (`0.21.1rc1.dev287+gf4abc309f`)
Params: `--block-size 256 --num-tokens 128000 --num-req 40 --concurrent 10`
GPUs: 20b on 0,1 (tp=2); 120b on 0-3 (tp=4)

80 runs total (8 configs × 10 reps each).

## Means across 10 reps per config

| Model | Tier | HMA | reps | cold mean (s) | last_10 mean (s) | last_10 stdev | last_10 min/max | tput mean (GB/s) |
|---|---|:---:|:---:|---:|---:|---:|---:|---:|
| 20b | cpu | no | 10 | 25.884 | 0.721 | 0.060 | 0.60 / 0.79 | 4.12 |
| 20b | cpu | **yes** | 10 | 25.990 | **0.400** | 0.012 | 0.39 / 0.43 | 7.32 |
| 20b | storage | no | 10 | 26.012 | 1.337 | 0.010 | 1.33 / 1.36 | 2.19 |
| 20b | storage | **yes** | 10 | 26.018 | **0.748** | 0.004 | 0.74 / 0.75 | 3.92 |
| 120b | cpu | no | 10 | 22.598 | 0.791 | 0.074 | 0.57 / 0.82 | 5.62 |
| 120b | cpu | **yes** | 10 | 22.607 | **0.397** | 0.059 | 0.30 / 0.44 | 11.36 |
| 120b | storage | no | 10 | 22.608 | 1.887 | 0.067 | 1.69 / 1.94 | 2.33 |
| 120b | storage | **yes** | 10 | 22.574 | **1.011** | 0.005 | 1.00 / 1.02 | 4.36 |

## HMA speedup (no-HMA last_10 / HMA last_10)

| Model | cpu | storage |
|---|---:|---:|
| 20b | **1.80×** | **1.79×** |
| 120b | **1.99×** | **1.87×** |

## Notes

- **Variance pattern**: HMA configs are highly stable (stdev 0.004–0.060 s). no-HMA cpu and 120b storage configs have wider spread (some runs land notably faster — likely transient GPU/GPFS contention dropping briefly).
- **Cold time is identical across HMA on/off within ~1%** on every tier — single-PUT bandwidth-limited.
- **120b CPU HMA hits a peak win** (1.99×) — splitting the 500 GiB CPU tier across 2 attention groups lets the host-to-CPU memcpy parallelize across groups, doubling effective bandwidth.
- **HMA storage win (~1.8×) is consistent across both models** — the doubled-file-count layout parallelizes better through the 24-threads/GPU pool.
- The earlier single-shot 120b CPU 3.39× was an outlier; the 10-rep mean lands at 1.99×.

## Files

- `<model>_<hma>_<tier>_c10/run01..run10.log` — full stdout/stderr per rep
- `sweep.log` — overall driver log with start/end timestamps per run
- `SUMMARY.md` — this file
- `plot.py`, `*.png` — comparison plots (generate via `python3 plot.py`)
