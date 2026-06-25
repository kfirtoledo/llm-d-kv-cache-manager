import argparse
import os
import re
import threading
import time
import logging
import gc

# Enable DEBUG-level logging on the llmd_fs_backend logger so each
# "Transfer finished: ... type=s->g | g->s" line prints to stderr. The
# worker subprocesses inherit this env var and configure their logger
# the same way at import time. See llmd_fs_backend/__init__.py.
#os.environ.setdefault("STORAGE_LOG_LEVEL", "DEBUG")

import torch
from vllm import LLM, SamplingParams, TokensPrompt
from vllm.config import KVTransferConfig
from transformers import AutoTokenizer
from tests.test_utils import cleanup_test_dirs, get_test_configs, prepare_lmcache_env, warmup_req, del_llm_and_cleanup, resolve_hma


# Matches the synchronous start-of-transfer log emitted by worker.py:
#   "PUT started: job_id=N files=M blocks=B size=X.XX [MB]"
#   "GET started: job_id=N files=M blocks=B size=X.XX [MB]"
# These fire on every transfer initiation, unlike "Transfer finished" which
# is only logged when get_finished() is polled (sparse).
_STARTED_RE = re.compile(
    r"(GET|PUT) started: job_id=\d+ files=(\d+) blocks=(\d+) size=([\d.]+) \[MB\]"
)
# Per-completion line (DEBUG only, sparse): we use it for transfer TIME.
_FINISHED_RE = re.compile(
    r"Transfer finished:.*?size=([\d.]+) \[MB\].*?time=([\d.]+) \[s\].*?type=([A-Z_]+->[A-Z_]+)"
)
_GET_TYPE = "SHARED_STORAGE->GPU"
_PUT_TYPE = "GPU->SHARED_STORAGE"
# Per-request block round-trip lifetime, emitted by manager.py in trace mode:
#   "block_roundtrip: blocks=N lifetime_avg=X lifetime_min=Y lifetime_max=Z [s]"
# Lifetime = wall-clock a KV block sat on storage between being WRITTEN
# (complete_store) and being LOADED back (prepare_load) on a later request.
_ROUNDTRIP_RE = re.compile(
    r"block_roundtrip: blocks=(\d+) lifetime_avg=([\d.]+) "
    r"lifetime_min=([\d.]+) lifetime_max=([\d.]+) \[s\]"
)
# Per-step connector phase breakdown, emitted by phase_timing.py in trace mode:
#   "connector_phase: role=worker read=1.234:1 write=5.678:1 poll=0.012:1 total=6.924"
# Each <phase>=<ms>:<calls> is the time the connector spent in that function
# since the previous flush; summed over a request the parts add up to the
# total time the request spent inside the OffloadingConnector.
_PHASE_LINE_RE = re.compile(r"connector_phase: role=(\w+) (.+?) total=([\d.]+)")
_PHASE_PAIR_RE = re.compile(r"(\w+)=([\d.]+):(\d+)")
# Per-file engine transfer sub-phases (storage_offload .so, TRACE only). Each
# transfer_async job splits into READ vs COPY (GET) and COPY vs WRITE (PUT):
#   GET  read phase 1: read_buffer_from_file        -> storage -> CPU buffer  (READ)
#   GET  read phase 2: copy_cpu_tensor_to_gpu_tensors-> CPU buffer -> GPU      (COPY)
#   PUT  write phase 1: copy_blocks                  -> GPU -> CPU buffer       (COPY)
#   PUT  write phase 2: write_buffer_to_file         -> CPU buffer -> storage   (WRITE)
# Emitted once per file from background I/O threads, so the summed times are
# aggregate thread-time (overlapped in wall-clock), not request latency.
_XFER_RE = {
    "get_read": re.compile(r"read phase 1: read_buffer_from_file took ([\d.]+) ms"),
    "get_copy": re.compile(r"read phase 2: copy_cpu_tensor_to_gpu_tensors took ([\d.]+) ms"),
    "put_copy": re.compile(r"write phase 1: copy_blocks\s+took ([\d.]+) ms"),
    "put_write": re.compile(r"write phase 2: write_buffer_to_file took ([\d.]+) ms"),
}


class TransferTally:
    """Thread-safe per-iteration accumulator of fs_backend transfers.

    Tracks two kinds of events:
      - started:  every transfer's initiation (PUT/GET, files, bytes). Always
                  emitted at INFO from worker.transfer_async().
      - finished: per-transfer completion line (DEBUG, sparse) — used for the
                  per-transfer time when available.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._started = []   # list of (kind, files, blocks, size_mb)
        self._finished = []  # list of (size_mb, time_s, type_str)
        self._lookups = []   # list of (kind, calls, hits, time_s)
        self._roundtrips = []  # list of (blocks, avg_s, min_s, max_s)
        # role -> {phase: [time_ms, calls]} accumulated this iteration
        self._phases = {"sched": {}, "worker": {}}
        # engine read/copy sub-phase: key -> [time_ms, files]
        self._xfer = {k: [0.0, 0] for k in _XFER_RE}

    def reset(self):
        with self._lock:
            self._started.clear()
            self._finished.clear()
            self._lookups.clear()
            self._roundtrips.clear()
            self._phases = {"sched": {}, "worker": {}}
            self._xfer = {k: [0.0, 0] for k in _XFER_RE}

    def feed(self, line):
        m = _STARTED_RE.search(line)
        if m:
            with self._lock:
                self._started.append((m.group(1), int(m.group(2)), int(m.group(3)), float(m.group(4))))
            return
        m = _FINISHED_RE.search(line)
        if m:
            with self._lock:
                self._finished.append((float(m.group(1)), float(m.group(2)), m.group(3)))
            return
        m = _LOOKUP_BATCH_RE.search(line)
        if m:
            with self._lock:
                self._lookups.append((m.group(1), int(m.group(2)), int(m.group(3)), float(m.group(4))))
            return
        m = _ROUNDTRIP_RE.search(line)
        if m:
            with self._lock:
                self._roundtrips.append((int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))))
            return
        m = _PHASE_LINE_RE.search(line)
        if m:
            role = m.group(1)
            with self._lock:
                bucket = self._phases.setdefault(role, {})
                for ph, ms, calls in _PHASE_PAIR_RE.findall(m.group(2)):
                    slot = bucket.setdefault(ph, [0.0, 0])
                    slot[0] += float(ms)
                    slot[1] += int(calls)
            return
        if "[TIME]" in line:
            for key, rex in _XFER_RE.items():
                mm = rex.search(line)
                if mm:
                    with self._lock:
                        self._xfer[key][0] += float(mm.group(1))
                        self._xfer[key][1] += 1
                    return

    def snapshot(self):
        with self._lock:
            started = list(self._started)
            finished = list(self._finished)
            lookups = list(self._lookups)
            roundtrips = list(self._roundtrips)
            phases = {r: {p: list(v) for p, v in d.items()}
                      for r, d in self._phases.items()}
            xfer = {k: list(v) for k, v in self._xfer.items()}
        gets = [s for s in started if s[0] == "GET"]
        puts = [s for s in started if s[0] == "PUT"]
        fin_gets = [(s, t) for s, t, ty in finished if ty == _GET_TYPE]
        fin_puts = [(s, t) for s, t, ty in finished if ty == _PUT_TYPE]
        # Block round-trip lifetime: aggregate across every prepare_load batch
        # this iteration. Weight the average by block count so a batch that
        # reused 100 blocks counts more than one that reused 2.
        rt_blocks = sum(r[0] for r in roundtrips)
        rt_avg = (
            sum(r[0] * r[1] for r in roundtrips) / rt_blocks if rt_blocks else 0.0
        )
        rt_min = min((r[2] for r in roundtrips), default=0.0)
        rt_max = max((r[3] for r in roundtrips), default=0.0)
        return {
            "get_jobs": len(gets),
            "get_files": sum(g[1] for g in gets),
            "get_mb": sum(g[3] for g in gets),
            "put_jobs": len(puts),
            "put_files": sum(p[1] for p in puts),
            "put_mb": sum(p[3] for p in puts),
            # Completion-only fields (may be partial when get_finished() is sparse):
            "get_done": len(fin_gets),
            "get_done_time": sum(t for _, t in fin_gets),
            "put_done": len(fin_puts),
            "put_done_time": sum(t for _, t in fin_puts),
            # Per-request lookup totals from manager.py flush
            "lookup_calls": sum(l[1] for l in lookups),
            "lookup_hits": sum(l[2] for l in lookups),
            "lookup_time_s": sum(l[3] for l in lookups),
            # Block round-trip lifetime (write->reuse-load), trace mode only:
            "rt_blocks": rt_blocks,
            "rt_avg_s": rt_avg,
            "rt_min_s": rt_min,
            "rt_max_s": rt_max,
            # Per-phase connector time, trace mode only:
            # {role: {phase: [time_ms, calls]}}
            "phases": phases,
            # Engine read/copy/write sub-phase, trace mode only:
            # {key: [time_ms, files]} (aggregate thread-time)
            "xfer": xfer,
        }


_TALLY = TransferTally()
_STDERR_TAIL_INSTALLED = False


_LOOKUP_BATCH_RE = re.compile(
    r"lookup_batch: kind=(\w+) calls=(\d+) hits=(\d+) time=([\d.]+) \[s\]"
)


def install_stderr_tail():
    """Redirect this process's fd 2 through a pipe so we can intercept every
    line written to stderr (by us OR by child workers that inherited fd 2).
    A daemon thread reads the pipe, tees back to the real stderr, and feeds
    matching lines to the global TransferTally.

    Must run BEFORE LLM() so child workers spawn with fd 2 = our pipe.
    Idempotent."""
    global _STDERR_TAIL_INSTALLED
    if _STDERR_TAIL_INSTALLED:
        return
    pipe_r, pipe_w = os.pipe()
    orig_fd = os.dup(2)
    os.dup2(pipe_w, 2)
    os.close(pipe_w)

    def reader():
        with os.fdopen(pipe_r, "rb", buffering=0) as r, os.fdopen(orig_fd, "wb", buffering=0) as orig:
            buf = b""
            while True:
                chunk = r.read(65536)
                if not chunk:
                    break
                try:
                    orig.write(chunk)
                except Exception:
                    pass
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    try:
                        _TALLY.feed(line.decode("utf-8", errors="replace"))
                    except Exception:
                        pass

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    _STDERR_TAIL_INSTALLED = True

def build_prompt_exact_tokens(model_name: str, target_tokens: int, seed_text: str) -> str:
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    buf = []
    ids = []
    # grow until we reach at least target_tokens (no specials)
    while len(ids) < target_tokens:
        buf.append(seed_text)
        ids = tok("".join(buf), add_special_tokens=False).input_ids
    ids = ids[:target_tokens]
    prompt = tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    # verify exact
    assert len(tok(prompt, add_special_tokens=False).input_ids) == target_tokens
    return prompt

def dir_stats(path):
    """Walk `path` and return per-file sizes.
    Returns a list[int] of sizes in bytes (empty if path missing)."""
    if not path or not os.path.isdir(path):
        return []
    sizes = []
    for root, _, names in os.walk(path):
        for n in names:
            try:
                sizes.append(os.path.getsize(os.path.join(root, n)))
            except OSError:
                pass
    return sizes


def fmt_size(b):
    return f"{b / (1 << 20):.1f}MB" if b >= (1 << 20) else f"{b / 1024:.1f}KB"


def size_summary(sizes):
    """Return 'N files / X MB (avg Y, min Z, max W)' summary string."""
    if not sizes:
        return "0 files"
    n = len(sizes)
    total = sum(sizes)
    avg = total / n
    return (f"{n} files / {fmt_size(total)} "
            f"(avg {fmt_size(avg)}, min {fmt_size(min(sizes))}, max {fmt_size(max(sizes))})")


def size_histogram(sizes, bins=None):
    """Bucket file sizes and return a list of (label, count) for printing."""
    if not sizes:
        return []
    if bins is None:
        bins = [0, 64 * 1024, 256 * 1024, 1 << 20, 4 << 20, 16 << 20, 64 << 20, 256 << 20, 1 << 30]
    buckets = [0] * (len(bins) + 1)
    for s in sizes:
        placed = False
        for i, b in enumerate(bins):
            if s <= b:
                buckets[i] += 1
                placed = True
                break
        if not placed:
            buckets[-1] += 1
    labels = []
    prev = 0
    for b in bins:
        labels.append(f"≤{fmt_size(b)}")
        prev = b
    labels.append(f">{fmt_size(prev)}")
    return [(lbl, cnt) for lbl, cnt in zip(labels, buckets) if cnt]


def build_tokens_prompt(num_tokens: int, prefix_id: int = 1, fill_id: int = 2) -> TokensPrompt:
    """
    Build a TokensPrompt directly from token IDs without using tokenizer.
    
    Args:
        num_tokens: Total number of tokens in the prompt
        prefix_id: Token ID for the first token (default: 1)
        fill_id: Token ID to fill the rest of the prompt (default: 2)
    
    Returns:
        TokensPrompt object with the specified token IDs
    
    Example:
        For num_tokens=10000, creates: [1, 2, 2, 2, ..., 2] (10000 tokens total)
    """
    prompt_token_ids = [prefix_id] + [fill_id] * (num_tokens - 1)
    return TokensPrompt(prompt_token_ids=prompt_token_ids)

def run_generation_test(name: str,
                        model_name: str,
                        gpu_block_size: int,
                        tensor_parallel_size: int = 4,
                        kv_transfer_config=None,
                        enable_prefix_caching=False,
                        temperature=1.8,
                        top_p=0.95,
                        seed=42,
                        num_req=4,
                        num_tokens=10000,
                        distributed_executor_backend=None,
                        use_token_ids=False,
                        hma_override=None,
                        test_dir=None,
                        concurrent=1,
                        max_num_seqs=None,
                        **kwargs):
    print(f"\n===== Running test: {name} =====")
    print(f"[CONFIG] enable_prefix_caching={enable_prefix_caching} | gpu_cache_block_size={gpu_block_size} | kv_transfer={'ON' if kv_transfer_config else 'OFF'}")

    # Build prompts - either from text or directly from token IDs.
    # When concurrent>1 we build that many DISTINCT prompts (differ by fill_id)
    # so each request hashes to its own block range and the storage tier sees
    # independent file traffic, not N hits on the same file.
    if use_token_ids:
        prompts = [
            build_tokens_prompt(num_tokens, prefix_id=1, fill_id=2 + j)
            for j in range(concurrent)
        ]
        print(f"[INFO] Using {concurrent} TokensPrompt(s) with {num_tokens} token IDs each")
    else:
        base_sentence = "Once upon a time there was a cat. The cat was big. It was blue. And then suddenly it"
        # Use distinct base sentences to get distinct hashes when concurrent>1
        prompts = [
            build_prompt_exact_tokens(
                model_name,
                num_tokens,
                base_sentence + f" run-{j}.",
            )
            for j in range(concurrent)
        ]
        print(f"[INFO] Using {concurrent} text prompt(s) with {num_tokens} tokens each")
    max_model_len = max(num_tokens + 1000, 34000)

    enable_hma, is_pure_hma = resolve_hma(model_name, hma_override)

    llm_kwargs = dict(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        kv_transfer_config=kv_transfer_config,
        enable_prefix_caching=enable_prefix_caching,
        gpu_memory_utilization=0.85,
        seed=seed,
        distributed_executor_backend=distributed_executor_backend,
    )
    # On vLLM main the hybrid KV cache manager is on by default even with
    # kv_transfer_config set. Explicitly toggle from our --hma flag so
    # --no-hma actually collapses interleaved-attention models (gpt-oss)
    # to a single unified KV group (compatible with pre-HMA fs_backend).
    llm_kwargs["disable_hybrid_kv_cache_manager"] = not enable_hma
    # Set the GPU block_size explicitly unless the model has no fixed block
    # concept (Mamba/Qwen). Interleaved sliding-window models DO have fixed
    # 16-token blocks, so they still need this.
    if not is_pure_hma:
        llm_kwargs["block_size"] = gpu_block_size
    if max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = max_num_seqs
    # Tap stderr BEFORE LLM() so the mp-spawned worker subprocesses inherit
    # our pipe and we see their fs_backend "Transfer finished" lines.
    install_stderr_tail()
    llm = LLM(**llm_kwargs)

    # Warm up the model with an initial request
    warmup_req(llm, temperature=temperature, top_p=top_p, seed=seed)

    # Main test params
    sampling_params = SamplingParams(
        #temperature=temperature,
        #top_p=top_p,
        detokenize=False,
        ignore_eos=True,
        seed=seed,
        max_tokens=1
    )
    # # half prompt for checking
    # half_prompt = prompt[:len(prompt) // 2]
    # outputs = llm.generate([half_prompt], sampling_params)
    # print(f" [INFO] generate half prompt")
    
    logging.getLogger("vllm").setLevel(logging.WARNING)
    logging.getLogger("vllm.engine").setLevel(logging.WARNING)
    logging.getLogger("vllm.worker").setLevel(logging.WARNING)
    prev_sizes = dir_stats(test_dir)
    if test_dir:
        print(f"[INFO] Watching storage dir: {test_dir}")
        print(f"[INFO] Start state: {size_summary(prev_sizes)}")
    times = []
    for i in range(num_req):
        _TALLY.reset()
        t0 = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        dt = time.perf_counter() - t0

        times.append(dt)
        # On the cold storage request, async writes are still in flight when
        # generate() returns. Wait before snapshotting so the [1] line reports
        # the real cold offload footprint, not just whatever landed early.
        # The transfer tally also catches late "Transfer finished" lines that
        # the worker emits during the flush.
        if i == 0 and name in ("Storage Offloading","GDS-Storage Offloading","GDS-BB-Storage Offloading"):
            time.sleep(5)

        text = outputs[0].outputs[0].text.strip()
        tally = _TALLY.snapshot()

        def fmt(kind, jobs, files, mb, done, done_time):
            if not jobs:
                return f"{kind} 0"
            time_part = f", {done}/{jobs} done in {done_time:.3f}s" if done else ""
            return f"{kind} {jobs}jobs {files}files {mb:.0f}MB{time_part}"

        get_line = fmt("GET", tally["get_jobs"], tally["get_files"], tally["get_mb"],
                       tally["get_done"], tally["get_done_time"])
        put_line = fmt("PUT", tally["put_jobs"], tally["put_files"], tally["put_mb"],
                       tally["put_done"], tally["put_done_time"])

        lk_calls = tally["lookup_calls"]
        lk_hits = tally["lookup_hits"]
        lk_s = tally["lookup_time_s"]
        if lk_calls:
            lk_line = (f"LOOKUP {lk_calls}calls {lk_hits}hits "
                       f"time={lk_s*1000:.1f}ms ({lk_s/dt*100:.1f}% of req)")
        else:
            lk_line = "LOOKUP 0"

        # Block round-trip lifetime = write->reuse-load latency per KV block,
        # as seen at the offloading connector (trace mode only).
        rt_blocks = tally["rt_blocks"]
        if rt_blocks:
            rt_line = (f"RT {rt_blocks}blk lifetime avg={tally['rt_avg_s']:.3f}s "
                       f"min={tally['rt_min_s']:.3f}s max={tally['rt_max_s']:.3f}s")
        else:
            rt_line = "RT 0"

        # Per-phase connector time decomposition (trace mode only). Σ is the
        # sum of the parts = total time the request spent in the connector,
        # split per process (W=worker: read/write/poll, S=scheduler:
        # lookup/alloc/meta/finished).
        def fmt_phase(role, tag, order):
            d = tally["phases"].get(role, {})
            if not d:
                return None
            names = [p for p in order if p in d] + [p for p in d if p not in order]
            parts = " ".join(f"{p}={d[p][0]:.1f}ms" for p in names)
            total = sum(v[0] for v in d.values())
            return f"PHASE[{tag}] {parts} Σ={total:.1f}ms"

        ph_w = fmt_phase("worker", "W", ["read", "write", "poll"])
        ph_s = fmt_phase("sched", "S", ["lookup", "alloc", "meta", "finished"])
        ph_line = " | ".join(x for x in (ph_w, ph_s) if x) or "PHASE 0"

        # Engine transfer_async read/copy/write split (aggregate I/O-thread time
        # summed over all files this request; overlapped, not wall-clock).
        xf = tally["xfer"]
        xf_parts = []
        if xf["get_read"][1] or xf["get_copy"][1]:
            xf_parts.append(
                f"GET read={xf['get_read'][0]:.1f}ms copy={xf['get_copy'][0]:.1f}ms "
                f"({xf['get_read'][1]}f)")
        if xf["put_copy"][1] or xf["put_write"][1]:
            xf_parts.append(
                f"PUT copy={xf['put_copy'][0]:.1f}ms write={xf['put_write'][0]:.1f}ms "
                f"({xf['put_write'][1]}f)")
        xf_line = "XFER " + " | ".join(xf_parts) if xf_parts else "XFER 0"

        total_input_tokens = sum(len(o.prompt_token_ids) for o in outputs)
        tput_toks = total_input_tokens / dt if dt > 0 else 0.0
        conc_part = (f"conc={concurrent} tok/s={tput_toks:,.0f} | "
                     if concurrent > 1 else "")
        if test_dir:
            new_sizes = dir_stats(test_dir)
            d_files = len(new_sizes) - len(prev_sizes)
            d_bytes = sum(new_sizes) - sum(prev_sizes)
            avg_mb = (sum(new_sizes) / len(new_sizes) / (1 << 20)) if new_sizes else 0.0
            print(f"[{i+1}] {dt:.3f}s | {conc_part}{get_line} | {put_line} | {lk_line} | {rt_line} | {ph_line} | {xf_line} | Δ{d_files:+d} files ({d_bytes / (1 << 20):+.1f} MB) → {len(new_sizes)} files / {sum(new_sizes) / (1 << 20):.1f} MB (avg {avg_mb:.2f} MB/file)")
            prev_sizes = new_sizes
        else:
            print(f"[{i+1}] {dt:.3f}s | {conc_part}{get_line} | {put_line} | {lk_line} | {rt_line} | {ph_line} | {xf_line} | {text[:80].replace(chr(10),' ')}")

    if test_dir:
        final_sizes = dir_stats(test_dir)
        print(f"\n[INFO] Final storage state: {size_summary(final_sizes)}")
        hist = size_histogram(final_sizes)
        if hist:
            print("[INFO] File size distribution:")
            for label, count in hist:
                bar = "█" * min(50, count)
                print(f"  {label:>8s}: {count:5d}  {bar}")
        

    cold = times[0]
    hot_avg = sum(times[1:]) / (num_req - 1)
    total = sum(times)
    input_tokens = len(outputs[0].prompt_token_ids)

    print(f"\n[INFO] Cold time (req 1) [{input_tokens} input tokens: {cold:.3f}s")
    print(f"[INFO] Hot average (req 2-{num_req}) [{input_tokens} input tokens: {hot_avg:.3f}s")
    print(f"[INFO] Total for {num_req} requests: {total:.3f}s")
    
    # Print last 10 requests and their average
    last_10 = times[-10:] if len(times) >= 10 else times
    last_10_avg = sum(last_10) / len(last_10)
    print(f"[INFO] Average of last {len(last_10)} requests: {last_10_avg:.3f}s")

    del_llm_and_cleanup(llm)
    return cold, hot_avg, total, last_10_avg

def calculate_throughput(model_name: str, num_tokens: int, gpu_block_size: int, avg_time: float) -> float:
    """
    Calculate throughput in GB/s based on model KV cache size.
    
    Args:
        model_name: Name of the model
        num_tokens: Number of input tokens
        block_size: Token block size
        avg_time: Average time in seconds
    
    Returns:
        Throughput in GB/s
    """
    # mb_per_block: MB per 16-token block for the KV that GROWS with seq len
    # (for sliding-window models this is full-attention layers only).
    # sliding_window: tokens of sliding-attention KV that stays constant past
    # the window. 0 means the model has no sliding attention.
    sliding_window = 0
    if "meta-llama/Meta-Llama-3.1-70B" == model_name:
        mb_per_block = 5.0
    elif "meta-llama/Meta-Llama-3.1-8B" == model_name:
        mb_per_block = 2.0
    elif "Qwen/Qwen3-30B-A3B-Instruct-2507" == model_name:
        mb_per_block = 0.75
    elif "Qwen/Qwen3.5-27B" == model_name:
        mb_per_block = 13.0
    elif "state-spaces/mamba-130m-hf" == model_name:
        mb_per_block = 1.41  # fixed recurrent state (conv+ssm) per block
    elif "openai/gpt-oss-20b" == model_name:
        mb_per_block = 0.375  # 12 full layers × 2 KB/tok × 16 / 1024
        sliding_window = 128
    elif "openai/gpt-oss-120b" == model_name:
        mb_per_block = 0.5625  # 18 full layers × 2 KB/tok × 16 / 1024
        sliding_window = 128
    else:
        return 0

    num_blocks = num_tokens / 16
    # Full-attention (or uniform) KV grows with num_tokens.
    total_mb = num_blocks * mb_per_block
    # Sliding-attention KV is capped at sliding_window tokens.
    # Same per-block footprint assumed as full (true for gpt-oss: equal counts).
    if sliding_window > 0:
        sliding_blocks = min(num_tokens, sliding_window) / 16
        total_mb += sliding_blocks * mb_per_block
    total_gb = total_mb / 1024
    
    # Throughput = data size / time
    throughput = total_gb / avg_time if avg_time > 0 else 0.0
    
    return throughput

def main():
    parser = argparse.ArgumentParser(description="Run LLM generation tests.")
    parser.add_argument(
        "--test", type=str, default="all",
        choices=["all", "no", "gpu", "cpu", "simple-cpu", "lmcache-cpu", "storage","gds-storage", "gds-bb-storage", "lmcache-storage", "multi-connector"],
        help="Specify which test to run: all, no, gpu, cpu, simple-cpu, lmcache-cpu, storage, gds-storage, gds-bb-storage, lmcache-storage, multi-connector"
    )
    parser.add_argument("--num-req", type=int, default=40,
                        help="Number of identical requests to run per test (default: 4)")
    parser.add_argument("--block-size", type=int, default=16,
                        help="Token block size (default: 16)")
    parser.add_argument("--gpu-block-size", type=int, default=16,
                        help="Token gpu block size (default: 16)")
    parser.add_argument("--num-tokens", type=int, default=10000,
                        help="Approx input token count for the prompt (default: 10000)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable DEBUG logging for vLLM")
    parser.add_argument("--trace", action="store_true",
                        help="Enable fs_backend trace mode (STORAGE_LOG_LEVEL=TRACE). "
                             "Turns on the worker PUT/GET transfer lines AND the "
                             "manager block_roundtrip lifetime line (write->reuse-load "
                             "latency per KV block), surfaced as 'RT' per request.")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-70B",
                        help="Model name to use for tests (default: meta-llama/Meta-Llama-3.1-70B). Supported: Qwen/Qwen2.5-32B, Qwen/Qwen3-32B, meta-llama/Meta-Llama-3.1-70B, etc.")
    parser.add_argument("--tp-size", type=int, default=4,help="Tensor parallel size (default: 4)")
    parser.add_argument("--use-token-ids", action="store_true", default=True,
                        help="Use TokensPrompt with token IDs instead of text prompts (faster, no tokenizer needed)")
    parser.add_argument("--hma", action=argparse.BooleanOptionalAction, default=None,
                        help="Force HMA on/off (default: auto from model classification). Use --no-hma to disable.")
    parser.add_argument("--concurrent", type=int, default=1,
                        help="Number of distinct prompts to issue concurrently per iteration (default: 1). "
                             "Each prompt has a different fill_id so it hashes to its own block range.")
    parser.add_argument("--max-num-seqs", type=int, default=None,
                        help="Cap on in-flight sequences in the vLLM scheduler (default: vLLM default). "
                             "Set to match vllm-bench --max-concurrency for serving-like comparisons.")
    parser.add_argument("--max-write-queued-seconds", type=float, default=None,
                        help="Override SharedStorageOffloadingSpec max_write_queued_seconds. "
                             "Larger value = bigger write queue (fewer drops, more memory). "
                             "0 disables the limit. Default: use fs_backend default (10s).")
    args = parser.parse_args()

    # Enable fs_backend trace mode BEFORE any LLM()/worker/engine-core process
    # is spawned, so the connector subprocesses read STORAGE_LOG_LEVEL=TRACE at
    # import time and emit the block_roundtrip lifetime line.
    if args.trace:
        os.environ["STORAGE_LOG_LEVEL"] = "TRACE"
        print("[INFO] fs_backend trace mode ON (STORAGE_LOG_LEVEL=TRACE)")

      # Set up debug logging if requested
    if args.debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logging.getLogger("vllm").setLevel(logging.DEBUG)

    # Define test configurations
    test_configs = get_test_configs(args.test, block_size=args.block_size, num_cpu_blocks=10000)
    if not test_configs:
        return

    # Propagate --max-write-queued-seconds into any SharedStorageOffloadingSpec
    # extra_config (storage, gds-*, multi-connector). Walk known shapes.
    if args.max_write_queued_seconds is not None:
        for cfg in test_configs:
            kv = cfg.get("kv_transfer_config")
            if kv is None or not getattr(kv, "kv_connector_extra_config", None):
                continue
            extra = kv.kv_connector_extra_config
            # Single OffloadingConnector form
            if extra.get("spec_name") == "SharedStorageOffloadingSpec":
                extra["max_write_queued_seconds"] = str(args.max_write_queued_seconds)
            # MultiConnector wraps a list of inner connectors
            for inner in extra.get("connectors", []):
                inner_extra = inner.get("kv_connector_extra_config", {})
                if inner_extra.get("spec_name") == "SharedStorageOffloadingSpec":
                    inner_extra["max_write_queued_seconds"] = str(args.max_write_queued_seconds)
        print(f"[INFO] max_write_queued_seconds override = {args.max_write_queued_seconds}")
    #os.environ["OMP_NUM_THREADS"]= "32"
    # Run tests and collect results
    results = []
    for config in test_configs:
        try:
            prepare_lmcache_env(config["name"], config.get("test_dir"), block_size=args.block_size)
            cold, hot_avg, total, last_10_avg = run_generation_test(
                num_req=args.num_req,
                num_tokens=args.num_tokens,
                model_name=args.model,
                gpu_block_size= args.gpu_block_size,
                tensor_parallel_size=args.tp_size,
                use_token_ids=args.use_token_ids,
                hma_override=args.hma,
                concurrent=args.concurrent,
                max_num_seqs=args.max_num_seqs,
                **config
            )
            results.append((config["name"], (cold, hot_avg, total, last_10_avg)))
        except Exception as e:
            print(f"Error running test '{config['name']}': {e}")
            results.append((config["name"], None))

    # Cleanup
    if os.environ.get("SKIP_CLEANUP") != "1":
        cleanup_test_dirs(test_configs)
    else:
        print("[INFO] SKIP_CLEANUP=1 — leaving test dirs for inspection")

    # Print final summary
    prompt_method = "TokensPrompt (token IDs)" if args.use_token_ids else "Text prompt (tokenizer)"
    print(f"\n===== Test Summary (offloading_block_size: {args.block_size}, gpu_block_size: {args.gpu_block_size}, prompt: {prompt_method}) =====")
    for name, r in results:
        if r is not None:
            cold, hot_avg, total, last_10_avg = r
            # Calculate throughput based on last 10 average
            throughput = calculate_throughput(args.model, args.num_tokens, args.gpu_block_size, last_10_avg)
            print(
                f"{name:<40} | "
                f"cold: {cold:.2f}s  "
                f"hot_avg(2-{args.num_req}): {hot_avg:.2f}s  "
                f"last_10_avg: {last_10_avg:.2f}s  "
                f"throughput: {throughput:.2f} GB/s  "
                f"total: {total:.2f}s  "
                f"[{args.num_tokens} input tokens]"
            )
        else:
            print(f"{name:<40} | FAILED")


if __name__ == "__main__":
    main()
