import uuid
import shutil
import os
from vllm import  LLM,SamplingParams
from vllm.config import KVTransferConfig
import gc
import torch
import torch.distributed as dist


def prepare_lmcache_env(test_name, folder, block_size):
    """Set LMCache environment variables based on the test name."""
    os.environ["PYTHONHASHSEED"] = "0"
    if "lmcache-cpu" in test_name.lower():
        setup_lmcache_cpu_offloading(block_size=block_size)
    elif "lmcache-storage" in test_name.lower():
        setup_lmcache_storage_offloading(block_size=block_size, folder=folder)

def setup_lmcache_cpu_offloading(block_size=16, max_local_cpu_gb="120"):
    # LMCache-related environment variables
    # Use experimental features in LMCache
    os.environ["LMCACHE_USE_EXPERIMENTAL"] = "True"
    # LMCache is set to use 256 tokens per chunk
    os.environ["LMCACHE_CHUNK_SIZE"] = str(block_size)
    # Enable local CPU backend in LMCache
    os.environ["LMCACHE_LOCAL_CPU"] = "True"
    # Set local CPU memory limit to 5.0 GB
    os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = max_local_cpu_gb


def setup_lmcache_storage_offloading(block_size=16, folder="/tmp/lmcache/", max_local_disk_gb="150"):
    unset_lmcache_env()
    os.environ["LMCACHE_USE_EXPERIMENTAL"] = "True"
    os.environ["LMCACHE_CHUNK_SIZE"] = str(block_size)
    os.environ["LMCACHE_LOCAL_CPU"] = "False"
    os.environ["LMCACHE_LOCAL_DISK"] = folder
    os.environ["LMCACHE_LOCAL_DISK"] = folder
    os.environ["LMCACHE_ENABLE_ASYNC_LOADING"] = "False"
    os.environ["LMCACHE_MAX_LOCAL_DISK_SIZE"] = max_local_disk_gb
    #os.environ["LMCACHE_EXTRA_CONFIG"] = '{"use_odirect": true}'


def unset_lmcache_env():
    for var in [
        "LMCACHE_LOCAL_CPU",
        "LMCACHE_MAX_LOCAL_CPU_SIZE",
        "LMCACHE_LOCAL_DISK",
        "LMCACHE_MAX_LOCAL_DISK_SIZE",
        "LMCACHE_EXTRA_CONFIG"
    ]:
        os.environ.pop(var, None)

def warmup_req(llm: LLM, temperature=0.9, top_p=0.95, seed=42):
    """Run a small generation to warm up the model."""
    # Warmup request with short prompt
    warmup_prompt = "Warmup test sentence. " * 30
    warmup_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        max_tokens=10
    )
    print("[INFO] Running warmup request (10 tokens prompt)...")
    _ = llm.generate([warmup_prompt], warmup_params)
    print("[INFO] Finish warmup request")


def get_test_configs(selected_test="all", block_size=16, num_cpu_blocks=50000):
    """Return the filtered test configs based on the selected test name."""
    random_id=uuid.uuid4().hex[:16]
    shared_storage_path=f"/ibm/fs1-remote/kfir/native-fs/{random_id}"
    configs = [
        {
            "name": "No Offloading | Prefix Caching = False",
            "kv_transfer_config": None,
            "enable_prefix_caching": False
        },
        {
            "name": "GPU Offloading | Prefix Caching = True",
            "kv_transfer_config": None,
            "enable_prefix_caching": True
        },
        {
            "name": "LMcache-CPU Offloading",
            "kv_transfer_config": KVTransferConfig(
                kv_connector="LMCacheConnectorV1",
                kv_role="kv_both",
                kv_connector_extra_config={"num_cpu_blocks": num_cpu_blocks}
            ),
            "enable_prefix_caching": False
        },
        {
            "name": "CPU Offloading",
            "kv_transfer_config": KVTransferConfig(
                kv_connector="OffloadingConnector",
                kv_role="kv_both",
                kv_connector_extra_config={"spec_name": "CPUOffloadingSpec","block_size": f"{block_size}","cpu_bytes_to_use": 100 *1024 * 1024 * 1024}
            ),
            "enable_prefix_caching": False,
        },
        {
            "name": "LMcache-Storage Offloading",
            "kv_transfer_config": KVTransferConfig(
                kv_connector="LMCacheConnectorV1",
                kv_role="kv_both"

            ),
            "enable_prefix_caching": False,
            "test_dir": f"/mnt/files-storage/lmcache/{random_id}/"
        },
        {
            "name": "Storage Offloading",
            "kv_transfer_config": KVTransferConfig(
                kv_connector="OffloadingConnector",
                kv_role="kv_both",
                kv_connector_extra_config={"spec_name": "SharedStorageOffloadingSpec",
                                           "block_size": f"{block_size}",
                                           "threads_per_gpu": "64",
                                           "shared_storage_path": shared_storage_path,
                                           "spec_module_path":"llmd_fs_backend.spec"}
            ),
            "enable_prefix_caching": False,
            "test_dir": shared_storage_path,
            # "distributed_executor_backend": "mp"
        },
        {
            "name": "GDS-Storage Offloading",
            "kv_transfer_config": KVTransferConfig(
                kv_connector="OffloadingConnector",
                kv_role="kv_both",
                kv_connector_extra_config={"spec_name": "SharedStorageOffloadingSpec",
                                           "block_size": f"{block_size}",
                                           "threads_per_gpu": "64",
                                           "shared_storage_path": shared_storage_path,
                                           "spec_module_path":"llmd_fs_backend.spec",
                                           "enable_gds": True} 
            ),
            "enable_prefix_caching": False,
            "test_dir": shared_storage_path,
            # "distributed_executor_backend": "mp"
        },
        {
            "name": "multi-connector",
            "kv_transfer_config": KVTransferConfig(
                kv_connector="MultiConnector",
                kv_role="kv_both",
                kv_connector_extra_config={
                    "connectors": [
                          {
                            "kv_connector": "OffloadingConnector",
                            "kv_role": "kv_both",
                            "kv_connector_extra_config": {
                                "spec_name": "CPUOffloadingSpec",
                                "block_size": f"{block_size}",
                                "cpu_bytes_to_use": 800* 2  * 1024 * 1024, 
                            },
                        },
                        {
                            "kv_connector": "OffloadingConnector",
                            "kv_role": "kv_both",
                            "kv_connector_extra_config": {
                                "spec_name": "SharedStorageOffloadingSpec",
                                "block_size": f"{block_size}",
                                "threads_per_gpu": "64",
                                "shared_storage_path": shared_storage_path,
                                "spec_module_path": "llmd_fs_backend.spec",
                            },
                        },
                    ],
                }
            ),
            "enable_prefix_caching": False,
            "test_dir": shared_storage_path,
            # "distributed_executor_backend": "mp",
        }
    ]

    if selected_test != "all":
        configs = [cfg for cfg in configs if cfg["name"].lower().startswith(selected_test.lower())]

    if not configs:
        print(f"No tests found matching '{selected_test}'")
        return []

    return configs

def cleanup_test_dirs(configs):
    """Delete test_dir folders from configs if they exist."""
    for cfg in configs:
        test_dir = cfg.get("test_dir")
        if test_dir and os.path.exists(test_dir):
            try:
                shutil.rmtree(test_dir)
                print(f"[CLEANUP] Finish Removed {test_dir}")
            except Exception as e:
                print(f"[CLEANUP] Failed to remove {test_dir}: {e}")

def del_llm_and_cleanup(llm: LLM):
    try:
        # Delete model and trigger garbage collection
        del llm
        gc.collect()

        # Release GPU memory
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Clean up torch distributed group if initialized
        if dist.is_initialized():
            dist.destroy_process_group()
            print("[INFO] torch.distributed process group destroyed.")
    except Exception as e:
        print(f"[WARN] Cleanup failed: {e}")
