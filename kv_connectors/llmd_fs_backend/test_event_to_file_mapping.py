"""
Run vLLM with the fs_backend storage offloader, capture GPU KV events,
then walk the storage directory and verify that every GPU-event block hash
appears as a file on disk (and vice versa).

Usage:
    python test_event_to_file_mapping.py
"""
import argparse
import asyncio
import os
import shutil
import time
import uuid
from pathlib import Path

import zmq
import zmq.asyncio
from msgspec.msgpack import Decoder

from vllm import LLM, SamplingParams, TokensPrompt
from vllm.config import KVTransferConfig
from vllm.config.kv_events import KVEventsConfig
from vllm.distributed.kv_events import (
    BlockRemoved,
    BlockStored,
    KVEventBatch,
)

GPU_ZMQ_PORT = 5557
STORAGE_ZMQ_PORT = 5559


async def listen_gpu_events(zmq_port, zmq_topic, stop_event):
    decoder = Decoder(type=KVEventBatch)
    ctx = zmq.asyncio.Context()
    sub = ctx.socket(zmq.SUB)
    sub.connect(f"tcp://localhost:{zmq_port}")
    sub.setsockopt_string(zmq.SUBSCRIBE, zmq_topic)
    print(f"[GPU EVT] listening on tcp://localhost:{zmq_port} topic={zmq_topic!r}")

    stored = []
    removed = []
    batches = 0
    try:
        while not stop_event.is_set():
            try:
                parts = await asyncio.wait_for(sub.recv_multipart(), timeout=0.3)
            except asyncio.TimeoutError:
                continue
            if len(parts) < 3:
                continue
            batches += 1
            payload = parts[2]
            batch = decoder.decode(payload)
            for ev in batch.events:
                if isinstance(ev, BlockStored):
                    stored.extend(ev.block_hashes)
                elif isinstance(ev, BlockRemoved):
                    removed.extend(ev.block_hashes)
    finally:
        sub.close()
        ctx.term()
    print(f"[GPU EVT] received {batches} batches, stored={len(stored)} removed={len(removed)}")
    return stored, removed


async def listen_storage_events(zmq_port, zmq_topic, stop_event):
    """Listen to fs_backend storage events (msgpack positional array)."""
    import msgpack
    ctx = zmq.asyncio.Context()
    sub = ctx.socket(zmq.SUB)
    sub.connect(f"tcp://localhost:{zmq_port}")
    sub.setsockopt_string(zmq.SUBSCRIBE, zmq_topic)
    print(f"[STO EVT] listening on tcp://localhost:{zmq_port} topic={zmq_topic!r}")

    stored = []
    batches = 0
    try:
        while not stop_event.is_set():
            try:
                parts = await asyncio.wait_for(sub.recv_multipart(), timeout=0.3)
            except asyncio.TimeoutError:
                continue
            if len(parts) < 3:
                continue
            batches += 1
            payload = msgpack.unpackb(parts[2], raw=False)
            # payload = [ts, [packed_event, ...]]
            _ts, packed_events = payload
            for pe in packed_events:
                ev = msgpack.unpackb(pe, raw=False)
                # ev = [tag, block_hashes, parent, token_ids, block_size, lora, medium]
                if ev and ev[0] == "BlockStored":
                    stored.extend(ev[1])
    finally:
        sub.close()
        ctx.term()
    print(f"[STO EVT] received {batches} batches, stored={len(stored)}")
    return stored


def scan_storage_dir(root):
    """Walk root and return {hex_stem: full_path} for every *.bin file."""
    found = {}
    root = Path(root)
    if not root.exists():
        return found
    for p in root.rglob("*.bin"):
        found[p.stem] = str(p)
    return found


def build_tokens_prompt(num_tokens, prefix_id=1, fill_id=2):
    return TokensPrompt(prompt_token_ids=[prefix_id] + [fill_id] * (num_tokens - 1))


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--num-tokens", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-req", type=int, default=2)
    args = parser.parse_args()

    run_id = uuid.uuid4().hex[:12]
    storage_root = f"/tmp/fs_event_test/{run_id}"
    os.makedirs(storage_root, exist_ok=True)

    extra = {
        "spec_name": "SharedStorageOffloadingSpec",
        "block_size": str(args.block_size),
        "threads_per_gpu": "32",
        "shared_storage_path": storage_root,
        "spec_module_path": "llmd_fs_backend.spec",
        "enable_events": "true",
        "storage_events_endpoint": f"tcp://*:{STORAGE_ZMQ_PORT}",
        "storage_medium": "SHARED_STORAGE",
    }
    kv_transfer_config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config=extra,
    )

    gpu_topic = f"kv@localhost@{args.model}"
    storage_topic = f"kv@SHARED_STORAGE@{args.model}"
    kv_events_config = KVEventsConfig(
        enable_kv_cache_events=True,
        publisher="zmq",
        endpoint=f"tcp://*:{GPU_ZMQ_PORT}",
        topic=gpu_topic,
    )

    stop_event = asyncio.Event()
    gpu_listener = asyncio.create_task(
        listen_gpu_events(GPU_ZMQ_PORT, gpu_topic, stop_event)
    )
    storage_listener = asyncio.create_task(
        listen_storage_events(STORAGE_ZMQ_PORT, storage_topic, stop_event)
    )
    await asyncio.sleep(1)

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp_size,
        kv_transfer_config=kv_transfer_config,
        kv_events_config=kv_events_config,
        enable_prefix_caching=True,
        max_model_len=args.num_tokens + 256,
        gpu_memory_utilization=0.85,
        block_size=args.block_size,
        seed=42,
        distributed_executor_backend="mp",
    )

    sampling = SamplingParams(detokenize=False, ignore_eos=True, seed=42, max_tokens=1)
    for i in range(args.num_req):
        # Vary the prefix token to force new blocks each request
        prompt = build_tokens_prompt(args.num_tokens, prefix_id=10 + i, fill_id=2)
        t0 = time.perf_counter()
        llm.generate([prompt], sampling, use_tqdm=False)
        dt = time.perf_counter() - t0
        print(f"[gen] req {i+1}: {dt:.2f}s")
        # Let publisher thread flush + storage workers complete writes
        await asyncio.sleep(5)

    # Extra drain time
    await asyncio.sleep(5)
    stop_event.set()
    stored_ints, removed_ints = await gpu_listener
    storage_stored = await storage_listener

    del llm

    print("\n========== MAPPING REPORT ==========")
    print(f"storage root: {storage_root}")

    files = scan_storage_dir(storage_root)
    file_stems = set(files.keys())
    print(f"files on disk:        {len(file_stems)}")

    def to_hex(h):
        if isinstance(h, (bytes, bytearray)):
            return bytes(h).hex()
        return f"{int(h):016x}"

    gpu_stored_hex = {to_hex(h) for h in stored_ints}
    gpu_removed_hex = {to_hex(h) for h in removed_ints}
    print(f"GPU BlockStored evts:  {len(stored_ints)} ({len(gpu_stored_hex)} unique)")
    print(f"GPU BlockRemoved evts: {len(removed_ints)} ({len(gpu_removed_hex)} unique)")

    if stored_ints:
        sample_h = stored_ints[0]
        print(f"GPU hash sample: type={type(sample_h).__name__} value={to_hex(sample_h)}")

    matched_full = file_stems & gpu_stored_hex
    matched_suffix = set()
    for stem in file_stems - matched_full:
        suffix = stem[-16:]
        if suffix in gpu_stored_hex:
            matched_suffix.add(stem)

    print(f"\nfile stems == full GPU hash (exact):           {len(matched_full)}")
    print(f"file stems whose last 16 hex == GPU hash:      {len(matched_suffix)}")

    unmatched_files = file_stems - matched_full - matched_suffix
    matched_gpu_hex = matched_full | {s[-16:] for s in matched_suffix}
    unmatched_gpu = gpu_stored_hex - matched_gpu_hex

    print(f"files with NO matching GPU event:              {len(unmatched_files)}")
    print(f"GPU events with NO matching file:              {len(unmatched_gpu)}")

    # --- Storage events sanity check (should be exact 1:1 with files) ---
    storage_hex = {to_hex(h) for h in storage_stored}
    print(f"\nstorage events received:        {len(storage_stored)} ({len(storage_hex)} unique)")
    if storage_stored:
        h = storage_stored[0]
        print(f"storage hash sample: type={type(h).__name__} value={to_hex(h)}")
        # If type is int, show its uint64 hex
        if isinstance(h, int):
            print(f"  uint64 hex: {h:016x}")
    storage_full_match = file_stems & storage_hex
    storage_suffix_match = set()
    for stem in file_stems - storage_full_match:
        if stem[-16:] in storage_hex:
            storage_suffix_match.add(stem)
    print(f"storage event == file stem full:     {len(storage_full_match)}")
    print(f"storage event == file stem suffix16: {len(storage_suffix_match)}")
    storage_matched_hex = storage_full_match | {s[-16:] for s in storage_suffix_match}
    print(f"storage events with NO matching file: {len(storage_hex - storage_matched_hex)}")
    print(f"files with NO matching storage event: {len(file_stems - storage_full_match - storage_suffix_match)}")

    # --- Pretty-print the GPU/Storage event ↔ file table ---
    suffix_to_file = {stem[-16:]: stem for stem in file_stems}
    gpu_set = {to_hex(h) for h in stored_ints}
    sto_set = {to_hex(h) for h in storage_stored}

    print("\n" + "=" * 130)
    print("EVENT ↔ FILE TABLE")
    print("=" * 130)
    header = f"{'#':>3}  {'event hash (uint64 hex)':24}  {'GPU':3}  {'STO':3}  {'full file hash (sha256)':64}  file path tail"
    print(header)
    print("-" * 130)

    rows = []
    all_event_hashes = gpu_set | sto_set
    for hx in sorted(all_event_hashes):
        stem = suffix_to_file.get(hx)
        if stem is None:
            continue
        rows.append(
            (hx, "✓" if hx in gpu_set else "·", "✓" if hx in sto_set else "·", stem, files[stem])
        )

    for i, (hx, gpu, sto, stem, path) in enumerate(rows[:20], 1):
        tail = "/".join(path.split("/")[-3:])
        print(f"{i:3d}  {hx:24}  {gpu:3}  {sto:3}  {stem:64}  …/{tail}")

    if len(rows) > 20:
        print(f"... ({len(rows) - 20} more matched rows)")

    print("-" * 130)
    print(f"total matched event↔file rows: {len(rows)}")
    print(f"  events seen in BOTH GPU and storage: {len(gpu_set & sto_set)}")
    print(f"  events only in GPU stream:           {len(gpu_set - sto_set)}")
    print(f"  events only in storage stream:       {len(sto_set - gpu_set)}")
    print(f"  files with no matching event (either stream): {len(file_stems - {s for _, _, _, s, _ in rows})}")

    sample = next(iter(file_stems), None)
    if sample:
        print(f"\nexample file stem: {sample!r} (len={len(sample)})")
        print(f"example file:      {files[sample]}")

    if unmatched_files:
        print("\nsample unmatched file stems (up to 5):")
        for s in list(unmatched_files)[:5]:
            print(f"  {s} -> {files[s]}")
    if unmatched_gpu:
        print("\nsample unmatched GPU hash hex (up to 5):")
        for s in list(unmatched_gpu)[:5]:
            print(f"  {s}")

    try:
        shutil.rmtree(storage_root)
    except Exception:
        pass


if __name__ == "__main__":
    asyncio.run(main())
