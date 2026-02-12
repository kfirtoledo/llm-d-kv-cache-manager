import argparse
import os
import time
import logging
import gc,torch
from vllm import LLM, SamplingParams
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
                        **kwargs):
    print(f"\n===== Running test: {name} =====")

    # Build an approx num_tokens input prompt
    # Assuming ~4 chars/token average for English
    base_sentence = "Once upon a time there was a cat. The cat was big. It was blue. And then suddenly it"
    prompt = build_prompt_exact_tokens(model_name, num_tokens, base_sentence)
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
        if i == 0 and name in ("Storage Offloading","GDS-Storage Offloading"):
            time.sleep(5) # wait a bit for storage to settle
        

    cold = times[0]
    hot_avg = sum(times[1:]) / (num_req - 1)
    total = sum(times)
    input_tokens = len(outputs[0].prompt_token_ids)

    print(f"\n[INFO] Cold time (req 1) [{input_tokens} input tokens: {cold:.3f}s")
    print(f"[INFO] Hot average (req 2-{num_req}) [{input_tokens} input tokens: {hot_avg:.3f}s")
    print(f"[INFO] Total for {num_req} requests: {total:.3f}s")

    del_llm_and_cleanup(llm)
    return cold, hot_avg, total

def main():
    parser = argparse.ArgumentParser(description="Run LLM generation tests.")
    parser.add_argument(
        "--test", type=str, default="all",
        choices=["all", "no", "gpu", "cpu", "lmcache-cpu", "storage","gds-storage", "lmcache-storage", "multi-connector"],
        help="Specify which test to run: all, no, gpu, cpu, lmcache-cpu, storage, lmcache-storage, multi-connector"
    )
    parser.add_argument("--num-req", type=int, default=6,
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
            cold, hot_avg, total = run_generation_test(
                num_req=args.num_req,
                num_tokens=args.num_tokens,
                model_name=args.model,
                gpu_block_size= args.gpu_block_size,
                tensor_parallel_size=args.tp_size,
                **config
            )
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
