from vllm.logger import init_logger
from vllm.v1.kv_offload.factory import OffloadingSpecFactory

logger = init_logger(__name__)

# Register SharedStorageOffloadingSpec to offloading connector
OffloadingSpecFactory.register_spec("SharedStorageOffloadingSpec",
                                    "vllm.v1.kv_offload.shared_storage",
                                    "SharedStorageOffloadingSpec")
