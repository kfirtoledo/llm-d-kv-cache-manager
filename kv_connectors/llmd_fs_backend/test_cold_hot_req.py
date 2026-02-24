import argparse
import os
import time
import logging
import gc,torch
from vllm import LLM, SamplingParams, TokensPrompt
from vllm.config import KVTransferConfig
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
                        **kwargs):
    print(f"\n===== Running test: {name} =====")

    # Build prompt - either from text or directly from token IDs
    if use_token_ids:
        # Simple token ID-based prompt (faster, no tokenizer needed)
        prompt = build_tokens_prompt(num_tokens, prefix_id=1, fill_id=2)
        print(f"[INFO] Using TokensPrompt with {num_tokens} token IDs")
    else:
        # Text-based prompt using tokenizer (original method)
        base_sentence = "Once upon a time there was a cat. The cat was big. It was blue. And then suddenly it"
        prompt = build_prompt_exact_tokens(model_name, num_tokens, base_sentence)
        print(f"[INFO] Using text prompt with {num_tokens} tokens")
    max_model_len=max(num_tokens+1000,64000)
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        kv_transfer_config=kv_transfer_config,
        enable_prefix_caching=enable_prefix_caching,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.85,
        block_size=gpu_block_size,
        seed=seed,
        distributed_executor_backend=distributed_executor_backend,
    )

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
    # half_pompt = prompt[:len(prompt) // 2]
    # outputs = llm.generate([half_pompt], sampling_params)
    # print(f" [INFO] generate half prompt")
    
    logging.getLogger("vllm").setLevel(logging.WARNING)
    logging.getLogger("vllm.engine").setLevel(logging.WARNING)
    logging.getLogger("vllm.worker").setLevel(logging.WARNING)
    times = []
    for i in range(num_req):
        t0 = time.perf_counter()
        outputs = llm.generate([prompt], sampling_params,use_tqdm=False)
        dt = time.perf_counter() - t0

        times.append(dt)
        text = outputs[0].outputs[0].text.strip()
        print(f"[{i+1}] {dt:.3f}s | {text[:120].replace('\n',' ')}")
        if i == 0 and name in ("Storage Offloading","GDS-Storage Offloading"):
            time.sleep(5) # wait a bit for storage to settle
        

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
    # KV cache size per block (16 tokens) in GB
    # llama-70b: 5GB per 16 tokens
    # llama-8b: 2GB per 16 tokens
    
    if "meta-llama/Meta-Llama-3.1-70B" == model_name:
        mb_per_block = 5.0
    elif "meta-llama/Meta-Llama-3.1-8B" == model_name:
        mb_per_block = 2.0
    else:
        return 0
        
    
    # Calculate number of blocks
    num_blocks = num_tokens / 16
    
    # Total data size in GB
    total_gb = num_blocks * mb_per_block / 1024
    
    # Throughput = data size / time
    throughput = total_gb / avg_time if avg_time > 0 else 0.0
    
    return throughput

def main():
    parser = argparse.ArgumentParser(description="Run LLM generation tests.")
    parser.add_argument(
        "--test", type=str, default="all",
        choices=["all", "no", "gpu", "cpu", "lmcache-cpu", "storage","gds-storage", "lmcache-storage", "multi-connector"],
        help="Specify which test to run: all, no, gpu, cpu, lmcache-cpu, storage, lmcache-storage, multi-connector"
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
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-70B",
                        help="Model name to use for tests (default: meta-llama/Meta-Llama-3.1-8B)")
    parser.add_argument("--tp-size", type=int, default=4,help="Tensor parallel size (default: 4)")
    parser.add_argument("--use-token-ids",  type=bool, default=True,
                        help="Use TokensPrompt with token IDs instead of text prompts (faster, no tokenizer needed)")
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
            cold, hot_avg, total, last_10_avg = run_generation_test(
                num_req=args.num_req,
                num_tokens=args.num_tokens,
                model_name=args.model,
                gpu_block_size= args.gpu_block_size,
                tensor_parallel_size=args.tp_size,
                use_token_ids=args.use_token_ids,
                **config
            )
            results.append((config["name"], (cold, hot_avg, total, last_10_avg)))
        except Exception as e:
            print(f"Error running test '{config['name']}': {e}")
            results.append((config["name"], None))

    # Cleanup
    cleanup_test_dirs(test_configs)

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
