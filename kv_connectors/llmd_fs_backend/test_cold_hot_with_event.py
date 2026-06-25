import argparse
import os
import time
import logging
import gc,torch
import asyncio
import zmq
import zmq.asyncio
from msgspec.msgpack import Decoder
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.config.kv_events import KVEventsConfig
from vllm.distributed.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVEventBatch,
)
from transformers import AutoTokenizer
from tests.test_utils import cleanup_test_dirs, get_test_configs, prepare_lmcache_env, warmup_req, del_llm_and_cleanup

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

async def listen_for_kv_event(zmq_port: int, zmq_topic: str, duration: float = 120.0) -> list:
    """
    Listens for KV cache events using a ZMQ SUB socket for a specified duration.
    Collects all events during that time period.
    """
    decoder = Decoder(type=KVEventBatch)
    context = zmq.asyncio.Context()
    sub = context.socket(zmq.SUB)
    sub.connect(f"tcp://localhost:{zmq_port}")
    sub.setsockopt_string(zmq.SUBSCRIBE, zmq_topic)

    print(f"[ZMQ] Listener started on port {zmq_port}, topic: {zmq_topic}")

    events = []
    start_time = time.time()

    try:
        while time.time() - start_time < duration:
            try:
                # Use await with recv_multipart for async socket
                parts = await asyncio.wait_for(sub.recv_multipart(), timeout=0.5)
                print(f"\n[ZMQ DEBUG] Received message with {len(parts)} parts")
                if len(parts) >= 3:
                    _, seq_bytes, payload = parts[0], parts[1], parts[2]
                    event_batch = decoder.decode(payload)
                    
                    # Print each event immediately as it's received
                    for event in event_batch.events:
                        print(f"\n[KV EVENT RECEIVED] {event.__class__.__name__}:")
                        if hasattr(event, 'block_hashes') and event.block_hashes:
                            print(f"  Block hashes (decimal): {event.block_hashes}")
                            hex_hashes = [f"{h:016x}" for h in event.block_hashes]
                            print(f"  Block hashes (hex):     {hex_hashes}")
                        if hasattr(event, 'parent_block_hash') and event.parent_block_hash:
                            print(f"  Parent hash (decimal):  {event.parent_block_hash}")
                            print(f"  Parent hash (hex):      {event.parent_block_hash:016x}")
                        if hasattr(event, 'token_ids') and event.token_ids:
                            print(f"  Token IDs ({len(event.token_ids)}): {event.token_ids[:10]}{'...' if len(event.token_ids) > 10 else ''}")
                        if hasattr(event, 'block_size'):
                            print(f"  Block size: {event.block_size}")
                        if hasattr(event, 'medium'):
                            print(f"  Medium: {event.medium}")
                    
                    events.extend(event_batch.events)
                    print(f"[ZMQ] Batch complete: {len(event_batch.events)} events in this batch (total: {len(events)})")
                else:
                    print(f"[ZMQ DEBUG] Message has only {len(parts)} parts, expected >= 3")
                    for i, part in enumerate(parts):
                        print(f"[ZMQ DEBUG]   Part {i}: {part[:50] if len(part) > 50 else part}")
            except asyncio.TimeoutError:
                # No message available within timeout, continue waiting
                await asyncio.sleep(0.1)
            except Exception as e:
                print(f"[ZMQ] Listener error: {e}")
                import traceback
                traceback.print_exc()
                break
    except asyncio.CancelledError:
        print(f"[ZMQ] Listener cancelled. Total events collected: {len(events)}")
    finally:
        sub.close()
        context.term()

    return events

async def run_generation_test(name: str,
                        model_name: str,
                        tensor_parallel_size: int = 4,
                        kv_transfer_config=None,
                        enable_prefix_caching=False,
                        temperature=1.8,
                        top_p=0.95,
                        seed=42,
                        num_req=4,
                        num_tokens=10000,
                        distributed_executor_backend=None,
                        **kwargs):
    print(f"\n===== Running test: {name} =====")

    # Build an approx num_tokens input prompt
    # Assuming ~4 chars/token average for English
    base_sentence = "Once upon a time there was a cat. The cat was big. It was blue. And then suddenly it"
    prompt = build_prompt_exact_tokens(model_name, num_tokens, base_sentence)
    max_model_len=max(num_tokens+1000,64000)
    
    # Configure KV events
    ZMQ_PORT = 5557
    ZMQ_TOPIC = f"kv@localhost@{model_name}"
    
    # Enable debug mode via environment variables
    os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"
    
    kv_events_config = KVEventsConfig(
        enable_kv_cache_events=True,
        publisher="zmq",
        endpoint=f"tcp://*:{ZMQ_PORT}",
        topic=ZMQ_TOPIC,
    )

    # Start KV event listener in background - it will run for the entire test duration
    event_task = asyncio.create_task(listen_for_kv_event(ZMQ_PORT, ZMQ_TOPIC, duration=300.0))
    
    # Give listener time to start and connect
    await asyncio.sleep(1)

    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        kv_transfer_config=kv_transfer_config,
        kv_events_config=kv_events_config,
        enable_prefix_caching=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.85,
        block_size=16,
        seed=seed,
        distributed_executor_backend=distributed_executor_backend,
    )

    # Warm up the model with an initial request
    # warmup_req(llm, temperature=temperature, top_p=top_p, seed=seed)

    # Main test params
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        max_tokens=10
    )
    # # half prompt for checking
    # half_pompt = prompt[:len(prompt) // 2]
    # outputs = llm.generate([half_pompt], sampling_params)
    # print(f" [INFO] generate half prompt")
    
    times = []
    for i in range(num_req):
        t0 = time.time()
        outputs = llm.generate([prompt], sampling_params)
        dt = time.time() - t0

        times.append(dt)
        text = outputs[0].outputs[0].text.strip()
        
        print(f"[{i+1}] {dt:.3f}s | {text[:120].replace('\n',' ')}")
        if i == 0 and name =="Storage Offloading":
            time.sleep(5) # wait a bit for storage to settle
        

    cold = times[0]
    hot_avg = sum(times[1:]) / (num_req - 1)
    total = sum(times)
    input_tokens = len(outputs[0].prompt_token_ids)

    print(f"\n[INFO] Cold time (req 1) [{input_tokens} input tokens: {cold:.3f}s")
    print(f"[INFO] Hot average (req 2-{num_req}) [{input_tokens} input tokens: {hot_avg:.3f}s")
    print(f"[INFO] Total for {num_req} requests: {total:.3f}s")

    # Give a moment for any final events to be published
    await asyncio.sleep(10)
    
    # Cancel the listener task and collect events
    event_task.cancel()
    events = []
    try:
        events = await event_task
    except asyncio.CancelledError:
        # Task was cancelled, but it should have returned events before cancellation
        print("[KV Events] Listener task cancelled")
    except Exception as e:
        print(f"[KV Events] Error collecting events: {e}")
    
    if events:
        print(f"\n[KV Events] Received {len(events)} KV cache events:")
        for i, event in enumerate(events[:5]):  # Show first 5 events
            print(f"  [{i+1}] {event.__class__.__name__}: {event}")
        if len(events) > 5:
            print(f"  ... and {len(events) - 5} more events")
    else:
        print("\n[KV Events] No KV cache events received. Possible reasons:")
        print("  - KV events may not be supported with the current kv_transfer_config")
        print("  - ZMQ publisher may not be initialized correctly")
        print("  - Events may be published on a different topic/port")
        print(f"  - Listening on: tcp://localhost:{ZMQ_PORT}, topic: {ZMQ_TOPIC}")

    del_llm_and_cleanup(llm)
    return cold, hot_avg, total

def main():
    parser = argparse.ArgumentParser(description="Run LLM generation tests.")
    parser.add_argument(
        "--test", type=str, default="all",
        choices=["all", "no", "gpu", "cpu", "lmcache-cpu", "kvbm-cpu", "storage", "lmcache-storage", "multi-connector"],
        help="Specify which test to run: all, no, gpu, cpu, kvbm-cpu, storage, lmcache-storage, multi-connector"
    )
    parser.add_argument("--num-req", type=int, default=2,
                        help="Number of identical requests to run per test (default: 4)")
    parser.add_argument("--block-size", type=int, default=16,
                        help="Token block size (default: 16)")
    parser.add_argument("--num-tokens", type=int, default=10000,
                        help="Approx input token count for the prompt (default: 10000)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable DEBUG logging for vLLM")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-70B",
                        help="Model name to use for tests (default: meta-llama/Meta-Llama-3.1-8B)")
    parser.add_argument("--tp-size", type=int, default=4,help="Tensor parallel size (default: 4)")
    args = parser.parse_args()

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
    #os.environ["OMP_NUM_THREADS"]= "32"
    # Run tests and collect results
    results = []
    for config in test_configs:
        try:
            prepare_lmcache_env(config["name"], config.get("test_dir"), block_size=args.block_size)
            cold, hot_avg, total = asyncio.run(run_generation_test(
                num_req=args.num_req,
                num_tokens=args.num_tokens,
                model_name=args.model,
                tensor_parallel_size=args.tp_size,
                **config
            ))
            results.append((config["name"], (cold, hot_avg, total)))
        except Exception as e:
            print(f"Error running test '{config['name']}': {e}")
            results.append((config["name"], None))

    # Cleanup
    cleanup_test_dirs(test_configs)

    # Print final summary
    print(f"\n===== Test Summary (block_size: {args.block_size}) =====")
    for name, r in results:
        if r is not None:
            cold, hot_avg, total = r
            print(
                f"{name:<40} | "
                f"cold: {cold:.2f}s  "
                f"hot_avg(2-{args.num_req}): {hot_avg:.2f}s  "
                f"total: {total:.2f}s  "
                f"[{args.num_tokens} input tokens]"
            )
        else:
            print(f"{name:<40} | FAILED")


if __name__ == "__main__":
    main()
