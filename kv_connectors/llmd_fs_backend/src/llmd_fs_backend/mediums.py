import numpy as np
from typing import Iterable
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.abstract import LoadStoreSpec

class SharedStorageLoadStoreSpec(LoadStoreSpec):
    """
    Spec for loading and storing KV blocks on shared storage.
    Stores block hashes internally as a numpy array.
    """

    def __init__(self, block_hashes: Iterable[BlockHash]):
        # Validate all items are bytes (BlockHash)
        block_hashes = list(block_hashes)
        for h in block_hashes:
            if not isinstance(h, (bytes, bytearray)):
                raise TypeError(f"Expected BlockHash (bytes-like), got {type(h).__name__}")

        # Store directly as object array of bytes
        self.block_hashes = np.array(block_hashes, dtype=object)

    def __repr__(self) -> str:
        return repr(self.block_hashes)

    @staticmethod
    def medium() -> str:
        return "SHARED_STORAGE"
