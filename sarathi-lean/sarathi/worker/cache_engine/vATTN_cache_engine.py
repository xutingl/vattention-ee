"""CacheEngine class for managing the KV cache."""
import traceback
from typing import List, Tuple, Union
from sarathi.core.datatypes.sequence import Sequence
import torch
from sarathi.core.datatypes.sequence import SequenceMetadata
from sarathi.config import CacheConfig, ModelConfig, ParallelConfig
from sarathi.logger import init_logger
from sarathi.model_executor.attention import get_attention_wrapper
from sarathi.utils import in_wsl
from sarathi.worker.cache_engine.base_cache_engine import BaseCacheEngine
import vattention
from sarathi.model_executor.attention import get_attention_wrapper
logger = init_logger(__name__)
import os
import time

# Per-step timing instrumentation is appended to unbounded lists that are only
# read by (currently commented-out) debug prints. Off by default; enable at launch
# (no source edit) with DREX_EE_PROFILE=1 / run_ee.py --ee_profile.
EE_PROFILE = os.environ.get("DREX_EE_PROFILE", "0") == "1"

KVCache = Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]

class vATTNCacheEngine(BaseCacheEngine):
    """Manages the KV cache.

    This class is responsible for initializing and managing the GPU KV cache.
    """
    _instance = None

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        mem_alloc_backend: str,
    ) -> None:
        self.max_batch_size = cache_config.max_batch_size * 2 + 1 # 1 batch and 1 buffer
        self.device = torch.empty(1).cuda().device if not in_wsl() else torch.device("cuda")
        self.device_idx = int(str(self.device).split(":")[-1])
        self.max_model_seq_len = model_config.max_model_len
        self.curr_seq_lens = [0 for i in range(self.max_batch_size)]
        self.seq_to_batch_idx = {}
        self.page_size = cache_config.page_size
        self.vattn_async = True if mem_alloc_backend == "async" else False
        self.vattn_mega_cache = True if "megacache" in model_config.attention_backend.lower() else False
        self.cache_mem_size = cache_config.memory_for_gpu
        # Full contiguous KV tensors for the megacache layout, shaped
        # [max_batch_size, max_seq_len, num_layers, num_kv_heads, head_size].
        # Set in allocate_gpu_cache(); used by copy_kv_cache_starting_at_layer to
        # fill deep-layer KV for early-exited tokens in a single broadcast write.
        self.megacache_k = None
        self.megacache_v = None
        super().__init__(cache_config, model_config, parallel_config)

        self.step_times = []

    def num_free_blocks(self) -> int:
        return vattention.num_free_kvblocks()

    def allocate_gpu_cache(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        print(f"[vATTNCacheEngine] Allocating GPU cache with size: {self.cache_mem_size}. page_size: {self.page_size}.")
        kv_cache = vattention.init_kvcache(
                                    self.num_layers,
                                    self.num_heads,
                                    self.head_size,
                                    self.max_batch_size,
                                    self.max_model_seq_len,
                                    self.device_idx,
                                    self.dtype,
                                    self.page_size,
                                    self.vattn_mega_cache)
        if self.vattn_mega_cache:
            k_cache = kv_cache[0]
            v_cache = kv_cache[1]
            # Keep handles to the full [batch, seq, layers, heads, head] tensors so
            # the deep-layer KV copy can be done as one vectorized scatter.
            self.megacache_k = k_cache
            self.megacache_v = v_cache
            assert k_cache.device == self.device, \
                        "k_cache device mismatch. expected: {}, got: {}".format(self.device, self.k_cache.device)
            assert v_cache.device == self.device, \
                        "v_cache device mismatch expected: {}, got: {}".format(self.device, self.v_cache.device) 

            cache_list = []
            for i in range(self.num_layers):
                cache_list.append((k_cache[:,:,i], v_cache[:,:,i]))
            
            print("[vATTNCacheEngine] Allocated Mega Cache")
        else:
            print("[vATTNCacheEngine] Allocated Normal Cache")
            k_cache = kv_cache[:self.num_layers]
            v_cache = kv_cache[self.num_layers:]
            for i in range(self.num_layers):
                assert k_cache[i].device == self.device, \
                            "k_cache device mismatch. expected: {}, got: {}".format(self.device, self.k_cache[i].device)
                assert v_cache[i].device == self.device, \
                            "v_cache device mismatch expected: {}, got: {}".format(self.device, self.v_cache[i].device)
            cache_list = list(zip(k_cache, v_cache))
        vattention.reserve_physical_pages(self.cache_mem_size)

        # Before: return cache_list of shape <num_layers, 2(k and v), batch_size, num_heads, head_size>
        # New: return Tuple of shape <2, num_layers, batch_size, num_heads, head_size>
        return cache_list
        #return (k_cache, v_cache)
    def preempt_requests(self, preempted_seq: List[int]) -> None:
        for seq in preempted_seq:
            self.free_request(seq.seq_id)

    def get_k_cache(self, layer_idx: int) -> torch.Tensor:
        #return self.gpu_cache[0][layer_idx]
        return self.gpu_cache[layer_idx][0]

    def get_v_cache(self, layer_idx: int) -> torch.Tensor:
        #return self.gpu_cache[1][layer_idx]
        return self.gpu_cache[layer_idx][1]
    
    """
    Copy the KV cache for the given request indices and token indices, from the source layer to all the layers after it.
    """
    def copy_kv_cache(self, src_layer_idx: int, seq_ids_to_copy: List[int], token_indices: torch.Tensor) -> None:
        

        target_cache_idx = [self.seq_to_batch_idx[seq_id] for seq_id in seq_ids_to_copy]
        src_k = self.gpu_cache[src_layer_idx][0][target_cache_idx, token_indices] 
        src_v = self.gpu_cache[src_layer_idx][1][target_cache_idx, token_indices]

        for layer in self.gpu_cache[src_layer_idx + 1:]:
            layer[0][target_cache_idx, token_indices] = src_k
            layer[1][target_cache_idx, token_indices] = src_v
    
    def copy_kv_cache_starting_at_layer(self, src_layer_idx: int, token_indices: torch.Tensor, exited_req_indices: torch.Tensor = None) -> None:
        """Fill the missing deep-layer KV for early-exited tokens.

        When a token early-exits at the shallow layer, layers
        ``src_layer_idx + 1 .. num_layers - 1`` never compute its K/V, so future
        decode steps would attend over uninitialized (stale, cross-sequence)
        cache slots. As a cheap approximation we copy the K/V from the last
        computed layer (``src_layer_idx``) into every deeper layer at the
        exited tokens' ``(batch_idx, position)`` slots.

        ``token_indices``/``exited_req_indices`` are aligned with the current
        batch order set by the most recent ``step()`` (see get_batch_idx()).
        """
        target_cache_idx = self.get_batch_idx()
        if target_cache_idx is None or target_cache_idx.numel() == 0:
            return

        if exited_req_indices is not None:
            target_cache_idx = target_cache_idx[exited_req_indices]
        if target_cache_idx.numel() == 0:
            return

        # Nothing to fill if the exit layer is (one before) the last layer.
        if src_layer_idx + 1 >= self.num_layers:
            return

        # The KV-cache seq dimension is allocated for max_model_len; a sequence that
        # has reached its length limit presents position == max_model_len (one past
        # the last storable slot). That token is the sequence's last — it terminates,
        # so its deep-layer KV is never read again. PyTorch advanced indexing
        # bounds-checks (unlike vAttention's raw-pointer kernel), so drop any such
        # out-of-range positions to avoid a device-side index assert.
        seq_dim = (self.megacache_k.shape[1]
                   if (self.vattn_mega_cache and self.megacache_k is not None)
                   else self.gpu_cache[src_layer_idx][0].shape[1])
        valid = (token_indices >= 0) & (token_indices < seq_dim)
        target_cache_idx = target_cache_idx[valid]
        token_indices = token_indices[valid]
        if target_cache_idx.numel() == 0:
            return

        if self.vattn_mega_cache and self.megacache_k is not None:
            # megacache: layers are a contiguous dim (dim 2), so the copy into all
            # deeper layers is a SINGLE broadcast scatter (2 kernels total: K, V)
            # instead of a Python loop launching 2*(num_layers - src) tiny kernels.
            # shape: [batch, seq, num_layers, num_kv_heads, head_size]
            src_k = self.megacache_k[target_cache_idx, token_indices, src_layer_idx]      # [N, H, D]
            src_v = self.megacache_v[target_cache_idx, token_indices, src_layer_idx]      # [N, H, D]
            self.megacache_k[target_cache_idx, token_indices, src_layer_idx + 1:] = src_k.unsqueeze(1)
            self.megacache_v[target_cache_idx, token_indices, src_layer_idx + 1:] = src_v.unsqueeze(1)
        else:
            # Non-megacache layout: each layer is a separate tensor; copy per layer.
            src_k = self.gpu_cache[src_layer_idx][0][target_cache_idx, token_indices]
            src_v = self.gpu_cache[src_layer_idx][1][target_cache_idx, token_indices]
            for layer in self.gpu_cache[src_layer_idx + 1:]:
                layer[0][target_cache_idx, token_indices] = src_k
                layer[1][target_cache_idx, token_indices] = src_v
    
    def copy_k_cache_between_layers(self, src_layer_idx: int, dest_layer_idx: int, seq_ids_to_copy: List[int], token_indices: torch.Tensor) -> None:
        target_cache_idx = [self.seq_to_batch_idx[seq_id] for seq_id in seq_ids_to_copy]
        src_k = self.gpu_cache[src_layer_idx][0][target_cache_idx, token_indices]
        
        self.gpu_cache[dest_layer_idx][0][target_cache_idx, token_indices] = src_k

    def copy_v_cache_between_layers(self, src_layer_idx: int, dest_layer_idx: int, seq_ids_to_copy: List[int], token_indices: torch.Tensor) -> None:
        target_cache_idx = [self.seq_to_batch_idx[seq_id] for seq_id in seq_ids_to_copy]
        src_v = self.gpu_cache[src_layer_idx][1][target_cache_idx, token_indices]

        self.gpu_cache[dest_layer_idx][1][target_cache_idx, token_indices] = src_v
    
    def step(self, seq_metadata_list: List[SequenceMetadata]) -> None:
        # print(f"[vATTNCacheEngine] Stepping with seq_metadata_list: {[metadata.seq.seq_id for metadata in seq_metadata_list]}")
        # print(f"[vATTNCacheEngine] curr_seq_lens: {self.curr_seq_lens}")
        # print(f"[vATTNCacheEngine] seq_to_batch_idx: {self.seq_to_batch_idx}")
        b_idx_prompt = []
        b_idx_gen = []
        for seq_metadata in seq_metadata_list:
            
            if seq_metadata.is_prompt:
                seq_id = seq_metadata.seq.seq_id
                prompt_chunk_len = seq_metadata.prompt_chunk_len
                current_prompt_chunk_len = seq_metadata.seq.get_next_prompt_chunk_len(
                prompt_chunk_len
                )
                processed_prompt_len = seq_metadata.seq.get_num_prompt_tokens_processed()

                context_len = processed_prompt_len + current_prompt_chunk_len
                new_batch_idx = self.get_req_batch_idx(seq_id, context_len)
                self.curr_seq_lens[new_batch_idx] = context_len
                # b_idx.append(new_batch_idx)
                b_idx_prompt.append(new_batch_idx)
            
            else:
                context_len = seq_metadata.seq.get_len()
                seq_id = seq_metadata.seq.seq_id
                new_batch_idx = self.get_req_batch_idx(seq_id, context_len)
                self.curr_seq_lens[new_batch_idx] = context_len 
                # b_idx.append(new_batch_idx)
                b_idx_gen.append(new_batch_idx)
        
        # seq_ids = [seq_metadata.seq.seq_id for seq_metadata in seq_metadata_list]
        # if 1 in seq_ids:
        #     print(f"[vATTNCacheEngine] Stepping with curr_seq_lens: {self.curr_seq_lens}. b_idx_gen: {b_idx_gen}. seq_ids: {seq_ids}")

        start_time = time.perf_counter()

        if self.vattn_async:
            # print(f"[vATTNCacheEngine] Stepping async with curr_seq_lens: {self.curr_seq_lens}")
            vattention.step_async(self.curr_seq_lens)
        else:
            # print(f"[vATTNCacheEngine] Stepping sync with curr_seq_lens: {self.curr_seq_lens}")
            vattention.step(self.curr_seq_lens, False)
        
        end_time = time.perf_counter()

        self.curr_batch_idx = torch.tensor(b_idx_prompt+b_idx_gen, dtype=torch.int32, device=self.device)

        # print(f"[vATTNCacheEngine] curr_batch_idx: {self.curr_batch_idx}")
        get_attention_wrapper().set_batch_idx(self.curr_batch_idx, torch.tensor(b_idx_gen, dtype=torch.int32, device=self.device))

        if EE_PROFILE:
            self.step_times.append(end_time - start_time)
        # print(f"[vATTNCacheEngine] num step times: {len(self.step_times)}. total time: {sum(self.step_times)}. avg time: {sum(self.step_times) / len(self.step_times)}")

    def on_step_completion(self, seq_metadata_list: List[SequenceMetadata]) -> None:
        for seq_metadata in seq_metadata_list:
            if seq_metadata.seq.is_finished():
                self.free_request(seq_metadata.seq.seq_id)

    def get_req_batch_idx(self, seq_id: int, seq_len: int) -> int:
        if seq_id in self.seq_to_batch_idx:
            return self.seq_to_batch_idx[seq_id]

        return self.alloc_new_batch_idx(seq_id, seq_len)

    def alloc_new_batch_idx(self, seq_id: int, seq_len: int) -> int:
        new_batch_idx = vattention.alloc_new_batch_idx(seq_len)
        if new_batch_idx == -1:
            print(self.curr_seq_lens)
        assert new_batch_idx != -1, "Failed to allocate new batch idx. This is not expected..."
        self.seq_to_batch_idx[seq_id] = new_batch_idx
        return new_batch_idx

    def free_request(self, seq_id: int) -> None:
        if seq_id in self.seq_to_batch_idx:
            batch_idx = self.seq_to_batch_idx[seq_id]
            vattention.free_batch_idx(batch_idx)
            self.seq_to_batch_idx.pop(seq_id)
            self.curr_seq_lens[batch_idx] = 0
            return
        raise Exception(f"seq_id {seq_id} not found in req_table")

    def reclaim_req_ids(self) -> None:
        for seq_id in list(self.seq_to_batch_idx.keys()):
            self.free_request(seq_id)

    def get_batch_idx(self) -> torch.Tensor:
        return self.curr_batch_idx

    def clear_batch_index(self) -> None:
        self.curr_batch_idx = None

    def release_kvcache_physical(self):
        vattention.release_kvcache_physical()

    def disable_deferred_reclamation(self):
        vattention.set_deferred_reclamation(False)

    def get_attention_context_lens(self):
        return self.attn_context_lens

    @staticmethod
    def get_cache_block_size(
        block_size: int,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        head_size = model_config.get_head_size()
        num_heads = model_config.get_num_kv_heads(parallel_config)
        num_layers = model_config.get_num_layers(parallel_config)

        key_cache_block = block_size * num_heads * head_size
        value_cache_block = key_cache_block
        total = num_layers * (key_cache_block + value_cache_block)
        dtype_size = _get_dtype_size(model_config.dtype)
        return dtype_size * total

    def cleanup_kvcache(self):
        # this is required to ensure UVM module is not holding on to the memory
        vattention.cleanup()


def _get_dtype_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()
