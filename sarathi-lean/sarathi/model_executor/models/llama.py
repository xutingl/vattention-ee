# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The Sarathi team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only LLaMA model compatible with HuggingFace weights.

The input of the model is flattened to a 1D tensor of tokens.
"""
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from transformers import LlamaConfig
import math

from sarathi.metrics.constants import OperationMetrics
from sarathi.metrics.cuda_timer import CudaTimer
from sarathi.model_executor.attention import get_attention_wrapper
from sarathi.model_executor.layers.activation import SiluAndMul
from sarathi.model_executor.layers.layernorm import RMSNorm
from sarathi.model_executor.layers.rotary_embedding import get_rope
from sarathi.model_executor.layers.sampler import Sampler, _get_logits
from sarathi.model_executor.parallel_utils.parallel_state import (
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    is_pipeline_first_stage,
    is_pipeline_last_stage,
)
from sarathi.model_executor.parallel_utils.pipeline_parallel.mappings import recv, send
from sarathi.model_executor.parallel_utils.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from sarathi.model_executor.weight_utils import (
    hf_model_weights_iterator,
    load_padded_tensor_parallel_vocab,
    load_tensor_parallel_weights,
)
from sarathi.worker.cache_engine import KVCache
from sarathi.worker.cache_engine.vATTN_cache_engine import vATTNCacheEngine
from sarathi.core.datatypes.sequence import Sequence, SequenceMetadata
from sarathi.core.sequence_manager.base_sequence_manager import BaseSequenceManager

from collections import defaultdict

import random
import time
# from torch.profiler import profile, record_function, ProfilerActivity
from decimal import Decimal


class LlamaMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        layer_id: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = ColumnParallelLinear(
            hidden_size,
            2 * intermediate_size,
            bias=False,
            gather_output=False,
            perform_initialization=False,
            linear_metric_name=OperationMetrics.MLP_UP_PROJ,
            communication_metric_name=OperationMetrics.MLP_UP_PROJ_ALL_GATHER,
            layer_id=layer_id,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            perform_initialization=False,
            linear_metric_name=OperationMetrics.MLP_DOWN_PROJ,
            communication_metric_name=OperationMetrics.MLP_DOWN_PROJ_ALL_REDUCE,
            layer_id=layer_id,
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

        self._mlp_activation_timer = CudaTimer(
            OperationMetrics.MLP_ACTIVATION, layer_id=layer_id
        )

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        with self._mlp_activation_timer:
            x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class LlamaAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        layer_id: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.layer_id = layer_id

        self.qkv_proj = ColumnParallelLinear(
            hidden_size,
            (self.total_num_heads + 2 * self.total_num_kv_heads) * self.head_dim,
            bias=False,
            gather_output=False,
            perform_initialization=False,
            linear_metric_name=OperationMetrics.ATTN_PRE_PROJ,
            communication_metric_name=OperationMetrics.ATTN_PRE_PROJ_ALL_GATHER,
            layer_id=layer_id,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            perform_initialization=False,
            linear_metric_name=OperationMetrics.ATTN_POST_PROJ,
            communication_metric_name=OperationMetrics.ATTN_POST_PROJ_ALL_REDUCE,
            layer_id=layer_id,
        )
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position_embeddings,
            base=self.rope_theta,
            is_neox_style=True,
            rope_scaling=rope_scaling,
        )
        self._attn_rope_timer = CudaTimer(OperationMetrics.ATTN_ROPE)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # with self._attn_rope_timer:
        #     q, k = self.rotary_emb(positions, q, k)
        q, k = self.rotary_emb(positions, q, k)
        
        try:
            attn_output = get_attention_wrapper().forward(
                q,
                k,
                v,
                kv_cache,
                self.scaling,
                self.layer_id,
            )
        except Exception as e:
            print(f"[LlamaAttention.forward] Error in layer {self.layer_id}. q.shape: {q.shape}, k.shape: {k.shape}, v.shape: {v.shape}, positions.shape: {positions.shape}. kv_cache.shape: {kv_cache[0][0].shape}")
            raise e

        output, _ = self.o_proj(attn_output)
        return output


class LlamaDecoderLayer(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        layer_id: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        # Requires transformers > 4.32.0
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        self.self_attn = LlamaAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            layer_id=layer_id,
        )
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            layer_id=layer_id,
        )
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            norm_name=OperationMetrics.INPUT_LAYERNORM,
            layer_id=layer_id,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            norm_name=OperationMetrics.POST_ATTENTION_LAYERNORM,
            layer_id=layer_id,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

class HiddenStatesBuffer():
    """
    A buffer that stores hidden states
    """

    def __init__(self, batch_size: int, capacity: int, hidden_state_length: int=5120): # 4096 for llama-3-8b, 5120 for llama-2-13b, 8192 for llama-2-70b
        self.batch_size = batch_size
        self.capacity = capacity
        # [WARNING!] hard code device
        self.hidden_states = torch.zeros(self.capacity, hidden_state_length, device='cuda:0') # [capacity, hidden_state_length]
        self.hidden_states = self.hidden_states.to(torch.float16) # bfloat16 for llama3
        self.positions = torch.zeros(self.capacity, device='cuda:0') # [capacity]
        self.positions = self.positions.to(torch.int64)
        self.available_slots = set(range(self.capacity))
        self.hidden_states_map = dict() # keys: req_ids, values: indices in hidden_states.
        self.dtype = torch.bfloat16
        self.time_spent_adding = 0
        self.time_spent_taking = 0
    
        
    def add_hidden_states(self, hidden_states: torch.Tensor, req_ids: List[int], positions: torch.Tensor) -> None:
        #start_time = time.perf_counter()
        num_hidden_states = hidden_states.size(0)
        slots = []
        for i in range(num_hidden_states):
            slot = self.available_slots.pop()
            slots.append(slot)
            self.hidden_states_map[req_ids[i]] = slot
        self.hidden_states[slots] = hidden_states
        self.positions[slots] = positions

        #self.time_spent_adding += time.perf_counter() - start_time

        
            
    
    """
    Args:
        num: number of hidden states to take. If 0, take all hidden states in the buffer.
    Returns:
        output_hidden_states: Tensor. hidden states taken from the buffer. <num, hidden_state_length>
        output_req_ids: List[int]. req_ids corresponding to the hidden states. <num>
        output_positions: Tensor. positions corresponding to the hidden states. <num>
    """
    def take_hidden_states(self, num: int=-1) -> Tuple[torch.Tensor, List[int], torch.Tensor]: 
        #start_time = time.perf_counter()
        if num == -1:
            num = self.batch_size
        # assert num <= len(self.hidden_states_map), f"Not enough hidden states in buffer. num: {num}, len(hidden_states_map): {len(self.hidden_states_map)}"

        
        output_req_ids = []
        
        # FIFO order: take hidden states from the left of the hidden_states_map
        slots = []
        for req_id, slot in list(self.hidden_states_map.items())[:num]:
            slots.append(slot)
            self.available_slots.add(slot)
            self.hidden_states_map.pop(req_id)
            output_req_ids.append(req_id)
            
        output_hidden_states = self.hidden_states[slots]
        output_positions = self.positions[slots]

        #self.time_spent_taking += time.perf_counter() - start_time
        return output_hidden_states, output_req_ids, output_positions

    def __len__(self):
        return len(self.hidden_states_map)


class LlamaModel(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = None
        if is_pipeline_first_stage():
            vocab_size = ((config.vocab_size + 63) // 64) * 64
            self.embed_tokens = VocabParallelEmbedding(
                vocab_size,
                config.hidden_size,
                perform_initialization=False,
                linear_metric_name=OperationMetrics.EMBED_LINEAR,
                communication_metric_name=OperationMetrics.EMBED_ALL_REDUCE,
            )

        num_layers = (
            config.num_hidden_layers // get_pipeline_model_parallel_world_size()
        )
        layer_offset = get_pipeline_model_parallel_rank() * num_layers
        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(config, layer_id=layer_id + layer_offset)
                for layer_id in range(num_layers)
            ]
        )

        self.norm = None
        if is_pipeline_last_stage():
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.ee_policy = config.ee_policy
        self.shallow_exit_layer_1 = config.shallow_exit_layer_1
        self.shallow_exit_layer_2 = config.shallow_exit_layer_2
        self.conf_threshold_1 = config.conf_threshold_1
        self.conf_threshold_2 = config.conf_threshold_2
        self.exited_rates = [0, 1] # [0]: early-exited, [1]: not early-exited. Initialized `not early-exited` to 1 to avoid division by 0.

        self.max_batch_size = config.max_num_seqs

        self.deep_buffer_1 = HiddenStatesBuffer(self.max_batch_size, self.max_batch_size * 2 + 1) # Buffers the hidden states that EE'ed
        self.deep_buffer_2 = HiddenStatesBuffer(self.max_batch_size, self.max_batch_size * 2 + 1) # Buffers the hidden states that EE'ed
        self.seq_metadata_map: Dict[int, SequenceMetadata] = {} # keys: seq_ids, values: SequenceMetadata. Used to update kv cache with updated sequences in the current batch.

        self.batch_size_lst = [0]

        self.avg_exited_conf = 0.0
        self.conf_sum = 0.0
        self.exited_cnt = 0

        self.update_kvcache_time_cnt = 0



        self.prefill_batch_size_limit = 64 # If emprical batch size is larger than this, we will not use EE. This is to avoid overhead of EE in prefill.

        self.sampler: Sampler = None

        self.early_exit_head = None
        self.rebatching_time = 0
        self.num_ee_threshold = getattr(config, 'num_ee_threshold', -1)

        self.kv_method = config.kv_method # "postfill" or "copy"
        self.recompute_seq_id_to_hidden_states = defaultdict(list) # seq_id -> a list of hidden states. This hidden states is the output of EE'ed layer and will be used for recomputing kv cache.
        self.recompute_seq_id_to_positions = defaultdict(list) # seq_id -> a list of positions. This positions is the output of EE'ed layer and will be used for recomputing kv cache.
        self.recompute_seq_id_to_input_hidden_states = dict() # seq_id -> this request's input hidden states.
    
    def softmax_confidence(
        self,
        logits: torch.Tensor,
    ):
        probs = torch.softmax(logits, dim=-1)
        top_2 = torch.topk(probs, dim=-1, k=2)[0]
        return (top_2[..., 0] - top_2[..., 1]).squeeze()
    
    def get_skip_mask(
        self,
        logits: torch.Tensor,
        hidden_states: torch.Tensor,
        conf_threshold: float,
        ee_policy: str = "eager",
        return_conf=False,
        rebatching_ee_factor: float = 0,
    ):
        # assert ee_policy != "off", "Turn off EE by setting self.use_shallow_deep = False. Set policy to 'off' incurrs unnecessary overhead."
        # if hidden_states.size(0) > self.prefill_batch_size_limit:
        #     # Heuristic to avoid using EE for prefilling
        #     mask = torch.tensor(0.0, device=hidden_states.device).bool()
        #     conf = torch.tensor(0.0, device=hidden_states.device)
        #     if not return_conf:
        #         return mask, False
        #     else:
        #         return mask, conf, False
        # logits = logits[~torch.any(logits.isnan(),dim=1)]
        conf = self.softmax_confidence(logits)
        # conf = conf[~torch.isnan(conf)]
        mask = torch.where(conf <= conf_threshold, 0.0, 1.0).bool() # <batch_size>, False: not EE, True: EE

        num_ee = torch.sum(mask).item()

        if self.num_ee_threshold == -1:
            # Auto mode
            if rebatching_ee_factor > 0: # Valid rebatching_ee_factor
                num_ee_threshold = hidden_states.size(0) * rebatching_ee_factor
            else: # Rebatching factor is not set, use default value
                num_ee_threshold = hidden_states.size(0) // 2
        else:
            # Manual mode
            num_ee_threshold = self.num_ee_threshold

        # For latency-only mode: process individual EE just like rebatching; no num_ee_threshold needed.
        if ee_policy == "latency-only":
            num_ee_threshold = 0

        need_skip = num_ee > num_ee_threshold

        if not (ee_policy == "rebatching" or ee_policy == "latency-only"):

            conf_median = torch.median(conf)
            conf = torch.mean(conf).item()
            if ee_policy == "eager":
                need_skip = torch.any(mask)
            elif ee_policy == "lazy":
                need_skip = torch.all(mask)
            elif ee_policy == "average":
                val = 0.0 if conf <= self.conf_threshold else 1.0
                need_skip = torch.tensor(val, device=hidden_states.device).bool()
            elif ee_policy == "median":
                need_skip = conf_median >= self.conf_threshold
            else:
                raise ValueError("Invalid EE policy: {}".format(ee_policy))
        else:
            exited_conf = torch.masked_select(conf, mask)
            conf = exited_conf.mean().item()
        

        if not return_conf:
            return mask, need_skip
        else:
            return mask, conf, need_skip
    
    def measure_batch_size(self, hidden_states: torch.Tensor, seq_ids_in_batch: List[int]=[]):
        if len(seq_ids_in_batch) > 0:
            #print(f"[LlamaModel.measure_batch_size] executing batch of size {len(seq_ids_in_batch)}")
            self.batch_size_lst.append(len(seq_ids_in_batch))
        else:
            batch_size = hidden_states.size(0)
            if batch_size <= self.max_batch_size:
                self.batch_size_lst.append(batch_size)
    
    def check_req_for_kv_recompute(self, hidden_states: torch.Tensor, positions: torch.Tensor, seq_ids_in_batch: List[int]) -> Tuple[dict, torch.Tensor, torch.Tensor, List[int]]:
        if not seq_ids_in_batch:
            return {}, hidden_states, positions, seq_ids_in_batch
        
        recompute_dict = {}

        recompute_req_to_idx = {}
        non_recompute_req_to_idx = {}

        # Check which sequences need recomputation
        for idx, ee_req_id in enumerate(seq_ids_in_batch):
            seq_metadata = self.seq_metadata_map[ee_req_id]
            
            if seq_metadata.seq.recompute_length > 0:
                recompute_req_to_idx[ee_req_id] = idx

                recompute_length = seq_metadata.seq.recompute_length + 1
                recompute_dict[ee_req_id] = recompute_length
                
                # Treat the EE'ed sequence as a prefill step, and the number of tokens to be prefilled is the recompute length.
                seq_metadata.seq.prompt_token_ids = seq_metadata.seq.prompt_token_ids + seq_metadata.seq.output_token_ids
                seq_metadata.seq.prompt_tokens_processed = len(seq_metadata.seq.prompt_token_ids) - recompute_length
                seq_metadata.seq.recompute_length = 0
                seq_metadata.seq.prompt_processing_finished = False
                seq_metadata.prompt_chunk_len = recompute_length
                seq_metadata.seq.state._prompt_processing_completed_at = None
            else:
                non_recompute_req_to_idx[ee_req_id] = idx

        # If nothing needs to be recomputed
        if not recompute_dict:
            return {}, hidden_states, positions, seq_ids_in_batch

        assert len(recompute_req_to_idx) + len(non_recompute_req_to_idx) == len(seq_ids_in_batch), f"recompute_req_to_idx: {recompute_req_to_idx}. non_recompute_req_to_idx: {non_recompute_req_to_idx}. seq_ids_in_batch: {seq_ids_in_batch}"
        
        
        
        
        # Collect stored recomputed data
        all_recompute_hidden_states_lst = []
        all_recompute_positions_lst = []


        
        for seq_id, recompute_length in recompute_dict.items():
            # Get stored hidden states and positions for this sequence
            seq_hidden_states = self.recompute_seq_id_to_hidden_states[seq_id]
            seq_positions = self.recompute_seq_id_to_positions[seq_id]


            
            # Add hidden states and positions to be recomputed to the list
            all_recompute_hidden_states_lst.extend(seq_hidden_states)
            all_recompute_positions_lst.extend(seq_positions)

            # Clear the stored data
            self.recompute_seq_id_to_hidden_states[seq_id].clear()
            self.recompute_seq_id_to_positions[seq_id].clear()

            # Add the current hidden states and positions to the list
            all_recompute_hidden_states_lst.append(hidden_states[recompute_req_to_idx[seq_id]])
            all_recompute_positions_lst.append(positions[recompute_req_to_idx[seq_id]])
            
        all_recompute_hidden_states_tensor = torch.stack(all_recompute_hidden_states_lst)
        all_recompute_positions_tensor = torch.stack(all_recompute_positions_lst)
        
        non_recompute_hidden_states = hidden_states[list(non_recompute_req_to_idx.values())]
        non_recompute_positions = positions[list(non_recompute_req_to_idx.values())]
        
        final_hidden_states = torch.cat([all_recompute_hidden_states_tensor, non_recompute_hidden_states], dim=0)
        final_positions = torch.cat([all_recompute_positions_tensor, non_recompute_positions], dim=0)
        final_seq_ids = list(recompute_req_to_idx.keys()) + list(non_recompute_req_to_idx.keys())
        print(f"[LlamaModel.check_req_for_kv_recompute] recompute_dict: {recompute_dict}. final_seq_ids: {final_seq_ids}")
        return recompute_dict, final_hidden_states, final_positions, final_seq_ids
    
    """
    For non-rebatching policies, requests information for the current batch is passed to cache_engine in `base_worker.py` and `model_runner.py`.
    For Rebatching, becuase we are updating request in current batch, we notify the cache_engine to update the kv cache using this function (and skip the update in the above 2 files).
    """
    def update_seqs_in_kvcache(
        self,
        seq_ids_in_batch: List[int],
        cache_engine: vATTNCacheEngine,
    ) -> None:  
        #start_time = time.perf_counter()
        # assert self.ee_policy == "rebatching", "update_seqs_in_kvcache is only used in rebatching mode."
        # updated_seq_metadata_list = [self.seq_metadata_map[seq_id] for seq_id in seq_ids_in_batch] 
        updated_seq_metadata_list = []
        for seq_id in seq_ids_in_batch:
            
            updated_seq_metadata_list.append(self.seq_metadata_map[seq_id])

        cache_engine.step(updated_seq_metadata_list) # Originally in base_worker
        get_attention_wrapper().begin_forward(updated_seq_metadata_list) # Originally in model_runner

        #self.update_kvcache_time_cnt += time.perf_counter() - start_time
    
    def fill_missing_kvcache_with_copy(self, exited_layer: int, cache_engine: vATTNCacheEngine, token_indices: torch.Tensor, exited_req_indices: Optional[torch.Tensor] = None, seq_ids_to_copy: Optional[List[int]] = None):

        # Copy method 1
        # seq_ids_to_copy = seq_ids_in_batch
        # for l in range(exited_layer, len(self.layers)):
        #     cache_engine.copy_k_cache_between_layers(exited_layer, l, seq_ids_to_copy, token_indices)
        #     cache_engine.copy_v_cache_between_layers(exited_layer, l, seq_ids_to_copy, token_indices)
        
        # Copy method 2
        cache_engine.copy_kv_cache_starting_at_layer(exited_layer, token_indices, exited_req_indices) # If exited_req_indices is None,

        # Copy method 3
        # cache_engine.copy_kv_cache(exited_layer, seq_ids_to_copy, token_indices)




    """
    Returns:
    - hidden_states: <batch_size, hidden_size>
    - seq_ids_in_batch: <batch_size>
    - exited_rates: List[int]: [exited cnt, not exited cnt]
    - lm_logits: Tensor: logits of the last layer, only returned if this forward is EE.
    """
    def forward_without_rebatching(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        lm_head,
        cache_engine: Optional[vATTNCacheEngine] = None,
        seq_ids_in_batch: Optional[List[int]] = None,
        seq_metadata_list: Optional[List[SequenceMetadata]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int], Optional[torch.Tensor]]:
        in_model_iter_start_time = time.perf_counter()
        self.measure_batch_size(hidden_states)

        # Update seq_metadasta_map
        if seq_metadata_list:
            for seq_metadata in seq_metadata_list:
                self.seq_metadata_map[seq_metadata.seq.seq_id] = seq_metadata
        
        if self.embed_tokens:
            hidden_states = self.embed_tokens(hidden_states)
        
        # Store input hidden states for recomputing kv cache
        input_hidden_states = hidden_states

        if self.kv_method == "postfill" and seq_ids_in_batch is not None and self.ee_policy != "rebatching":
            recompute_dict, hidden_states, positions, seq_ids_in_batch = self.check_req_for_kv_recompute(hidden_states, positions, seq_ids_in_batch)
            if len(recompute_dict) > 0:
                self.update_seqs_in_kvcache(seq_ids_in_batch, cache_engine)
        else:
            recompute_dict = {}

        
        if seq_ids_in_batch is not None:
            for seq_metadata in seq_metadata_list:
                self.seq_metadata_map[seq_metadata.seq.seq_id] = seq_metadata
            
            if self.ee_policy == "rebatching":
                # This is the prefilling of a rebatching run
                # print(f"[LlamaModel.forward_without_rebatching] prefilling with seq_ids_in_batch: {seq_ids_in_batch}. batch size: {hidden_states.size(0)}")
                self.update_seqs_in_kvcache(seq_ids_in_batch, cache_engine)
        
        check_for_ee = cache_engine is not None and self.ee_policy != "off" and self.shallow_exit_layer_1 is not None and hidden_states.size(0) <= self.max_batch_size and len(recompute_dict) == 0

        conf = None

        has_ee = False
        latency_only_ee_iter_time = None

        exited_ramp_id = -1
        for i in range(len(self.layers)):
            layer = self.layers[i]
            if check_for_ee and (i == self.shallow_exit_layer_1 or i == self.shallow_exit_layer_2):

                if self.early_exit_head:
                    lm_logits = self.early_exit_head(self.norm(hidden_states))
                else:
                    lm_logits = _get_logits(self.norm(hidden_states), self.sampler.embedding, self.sampler.vocab_size)

                skip_mask, conf, need_skip = self.get_skip_mask(
                    logits=lm_logits,
                    hidden_states=hidden_states,
                    ee_policy=self.ee_policy,
                    return_conf=True
                )

                if need_skip:
                    if i == self.shallow_exit_layer_1:
                        exited_ramp_id = 1
                    elif i == self.shallow_exit_layer_2:
                        exited_ramp_id = 2
                    else:
                        raise ValueError(f"exited from wrong layer: {i}")

                    if hidden_states.size(0) <= self.max_batch_size: # Only count decoding requests
                            self.exited_rates[0] += hidden_states.size(0)

                    if self.ee_policy == "latency-only" and not torch.all(skip_mask):
                        latency_only_exited_req_indices = torch.where(skip_mask)[0]
                        latency_only_exited_hidden_states = hidden_states[latency_only_exited_req_indices].clone()

                        latency_only_ee_iter_time = time.perf_counter() - in_model_iter_start_time


                    else:
                        has_ee = True

                        # Copy layer i-1's kv cache for the prev token to layer i - last layer.
                        # [TODO] i-2 seems to give better results.
                        if self.kv_method == "copy":
                            self.fill_missing_kvcache_with_copy(i-1, cache_engine, positions, exited_req_indices=None, seq_ids_to_copy=seq_ids_in_batch)
                        elif self.kv_method == "postfill":
                            normed_hidden_states = self.norm(hidden_states)
                            for idx, ee_req_id in enumerate(seq_ids_in_batch):
                                seq_metadata = self.seq_metadata_map[ee_req_id]
                                seq_metadata.seq.recompute_length += 1

                                # Store the hidden states and positions for recomputing kv cache
                                self.recompute_seq_id_to_hidden_states[ee_req_id].append(input_hidden_states[idx])
                                self.recompute_seq_id_to_positions[ee_req_id].append(positions[idx])
                            recompute_dict = {} # When EE, we don't need to recompute kv cache. Postfill will possiblly happen in the next forward.

                        break
                else:
                    if hidden_states.size(0) <= self.max_batch_size: # Only count decoding requests
                        self.exited_rates[1] += hidden_states.size(0)

                    
                
            hidden_states = layer(
                positions,
                hidden_states,
                kv_caches[i],
            )
        
        if latency_only_ee_iter_time is not None:
            hidden_states[latency_only_exited_req_indices] = latency_only_exited_hidden_states

        if self.norm:
            hidden_states = self.norm(hidden_states)

        # print(f"[LlamaModel.forward_without_rebatching] ee_rates: {self.exited_rates}")
        # print(f"[LlamaModel.forward] ee_rates: {self.exited_rates}={(self.exited_rates[0]/sum(self.exited_rates)):.2f}. Avg batch size: {sum(self.batch_size_lst)/len(self.batch_size_lst)}.\n number of batchsize=1,2,3,4: {self.batch_size_lst.count(1)}, {self.batch_size_lst.count(2)}, {self.batch_size_lst.count(3)}, {self.batch_size_lst.count(4)}")

        # print(f"[LlamaModel.forward_without_rebatching] returning seq_ids_in_batch: {seq_ids_in_batch}\n")

        if has_ee:
            return hidden_states, seq_ids_in_batch, self.exited_rates, lm_logits, exited_ramp_id, False, False, recompute_dict, conf, None
        return hidden_states, seq_ids_in_batch, self.exited_rates, None, -1, False, False, recompute_dict, None, latency_only_ee_iter_time
    
    """
    When seq_ids_in_batch is provided, rebatching based on early exit status is enabled

    Returns:
    - hidden_states: <batch_size, hidden_size>
    - seq_ids_in_batch: <batch_size>
    - exited_rates: List[int]: [exited cnt, not exited cnt]
    - lm_logits: Tensor: logits of the last layer, only returned if this forward is EE.
    """
    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache], # List of (k_cache, v_cache), where each k_cache and v_cache is a <batch_size, max_seq_len, num_heads(8), head_dim(128)>
        lm_head,
        cache_engine: Optional[vATTNCacheEngine] = None,
        seq_ids_in_batch: Optional[List[int]] = None, # <batch_size> # The seq_id of the seqences in the batch
        seq_metadata_list: Optional[List[SequenceMetadata]] = None,
        rebatching_ee_factor: float = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int], Optional[torch.Tensor]]:
        
        # print(f"=========== start iter =============\n[LlamaModel.forward] seq_ids_in_batch: {seq_ids_in_batch}. hidden_states.shape: {hidden_states.shape}. positions.shape: {positions.shape}")
        
        if self.ee_policy != "rebatching" or hidden_states.size(0) > self.prefill_batch_size_limit: # Rebatching disabled
            return self.forward_without_rebatching(hidden_states, positions, kv_caches, lm_head, cache_engine, seq_ids_in_batch, seq_metadata_list=seq_metadata_list)
        

        # Rebatching enabled

        # Update seq_metadasta_map
        for seq_metadata in seq_metadata_list:
            self.seq_metadata_map[seq_metadata.seq.seq_id] = seq_metadata


        if self.embed_tokens:
            hidden_states = self.embed_tokens(hidden_states)


        incoming_batch_size = len(seq_ids_in_batch)

        # Store input hidden states for recomputing kv cache
        for i, seq_id in enumerate(seq_ids_in_batch):
            self.recompute_seq_id_to_input_hidden_states[seq_id] = hidden_states[i]


        # 0. Flush: If we receive an empty batch, we process any leftover hidden states in the buffer. Scheduler will send a flush request if deep_buffer is full or starving.
        # Multi-exits: check deep_buffer_2 (the deeper buffer) first, then deep_buffer_1.
        flush_buffer = incoming_batch_size == 0
        start_at_exit_layer_1 = False
        if flush_buffer:
            if len(self.deep_buffer_2) >= self.max_batch_size or len(self.deep_buffer_2) > len(self.deep_buffer_1):
                # Take hidden states from `deep_buffer`
                hidden_states, seq_ids_in_batch, positions = self.deep_buffer_2.take_hidden_states(len(self.deep_buffer_2))
                # print(f"[LlamaModel.forward] [1] updating kvcache with seq_ids_in_batch: {seq_ids_in_batch}")

                self.update_seqs_in_kvcache(seq_ids_in_batch, cache_engine)

                #self.measure_batch_size(hidden_states, seq_ids_in_batch)

                for i in range(self.shallow_exit_layer_2, len(self.layers)):
                    layer = self.layers[i]
                    hidden_states = layer(
                        positions,
                        hidden_states,
                        kv_caches[i],
                    )
                if self.norm:
                    hidden_states = self.norm(hidden_states)
                # print(f"[LlamaModel.forward] returning flush_buffer 1: deep_buffer. seq_ids_in_batch: {seq_ids_in_batch}")
                #print(f"[LlamaModel.forward] ee_rates: {self.exited_rates}={(self.exited_rates[0]/sum(self.exited_rates)):.2f}. Avg batch size: {sum(self.batch_size_lst)/len(self.batch_size_lst)}")
                return hidden_states, seq_ids_in_batch, self.exited_rates, None, -1, False, True, {}, None, None
            
            elif len(self.deep_buffer_1) > 0:
                hidden_states, seq_ids_in_batch, positions = self.deep_buffer_1.take_hidden_states(len(self.deep_buffer_1))
                
                start_at_exit_layer_1 = True

            else:
                #print(f"[LlamaModel.forward] flush_buffer: no buffer. seq_ids_in_batch: {seq_ids_in_batch}")
                return None, None, self.exited_rates, None, -1, False,False, {}, None, None

        # 1. Process normal requests.
        if self.kv_method == "postfill":
            recompute_dict, hidden_states, positions, seq_ids_in_batch = self.check_req_for_kv_recompute(hidden_states, positions, seq_ids_in_batch)
        else:
            recompute_dict = {}

        self.update_seqs_in_kvcache(seq_ids_in_batch, cache_engine)

        #self.measure_batch_size(hidden_states, seq_ids_in_batch)

        conf = None
        has_ee = False

        # Layer [0, shallow_exit_layer_1 - 1]
        if not start_at_exit_layer_1:
            for i in range(self.shallow_exit_layer_1):
                layer = self.layers[i]
                hidden_states = layer(positions, hidden_states, kv_caches[i])
        
            # Check shallow_exit_layer_1
            if cache_engine and hidden_states.size(0) <= self.max_batch_size and not recompute_dict:
                if self.early_exit_head:
                    lm_logits = self.early_exit_head(self.norm(hidden_states))
                else:
                    lm_logits = _get_logits(self.norm(hidden_states), self.sampler.embedding, self.sampler.vocab_size)

                skip_mask, conf, need_skip = self.get_skip_mask(
                    logits=lm_logits,
                    hidden_states=hidden_states,
                    conf_threshold=self.conf_threshold_1,
                    ee_policy=self.ee_policy,
                    return_conf=True,
                    rebatching_ee_factor=rebatching_ee_factor
                )

                if need_skip:
                    # print(f"Exiting with confidence {conf}. exited rates: {self.exited_rates}", flush=True)
                    rebatching_start_time = time.perf_counter()

                    has_ee = True

                    if torch.all(skip_mask):
                        self.exited_rates[0] += len(seq_ids_in_batch)
                        # 3.1 If all requests want to EE, no rebatching is done.

                        if self.kv_method == "copy":
                            self.fill_missing_kvcache_with_copy(i-1, cache_engine, positions, exited_req_indices=None, seq_ids_to_copy=seq_ids_in_batch)
                        elif self.kv_method == "postfill":
                            for idx, ee_req_id in enumerate(seq_ids_in_batch):
                                seq_metadata = self.seq_metadata_map[ee_req_id]
                                seq_metadata.seq.recompute_length += 1

                                # Store the hidden states and positions for recomputing kv cache
                                self.recompute_seq_id_to_hidden_states[ee_req_id].append(self.recompute_seq_id_to_input_hidden_states[ee_req_id])
                                self.recompute_seq_id_to_positions[ee_req_id].append(positions[idx])

                    else:
                        exited_req_indices = torch.where(skip_mask)[0]
                        if self.kv_method == "copy":
                            # Need to copy the KV cache for the requests that EE i.e. skip_mask[i] is True.
                            token_indices = positions[exited_req_indices]
                            seq_ids_to_copy = [seq_ids_in_batch[i] for i in exited_req_indices]

                            self.fill_missing_kvcache_with_copy(i-1, cache_engine, token_indices, exited_req_indices=exited_req_indices, seq_ids_to_copy=seq_ids_to_copy)

                        self.exited_rates[0] += len(exited_req_indices)
                        self.exited_rates[1] += (len(seq_ids_in_batch) - len(exited_req_indices))

                        # print(f"[LlamaModel.forward] partial {seq_ids_in_batch}. exited rates: {self.exited_rates}")

                        lm_logits = lm_logits[exited_req_indices]
                        # assert lm_logits.size(0) == len(req_indices), f"lm_logits size: {lm_logits.size()}, req_indices size: {len(req_indices)}"


                        req_idx_in_batch_to_buffer = []
                        seq_ids_in_batch_to_buffer = []

                        for req_idx, skip in enumerate(skip_mask):
                            if skip:
                                pass
                                
                            else:
                                # 3.2 Put requests that don't EE into `deep_buffer`.
                                req_idx_in_batch_to_buffer.append(req_idx)
                                seq_ids_in_batch_to_buffer.append(seq_ids_in_batch[req_idx])
                            
                        self.deep_buffer_1.add_hidden_states(hidden_states[req_idx_in_batch_to_buffer], seq_ids_in_batch_to_buffer, positions[req_idx_in_batch_to_buffer])


                        # Keep requests EE
                        hidden_states = hidden_states[skip_mask]
                        seq_ids_in_batch = [seq_ids_in_batch[i] for i in range(len(seq_ids_in_batch)) if skip_mask[i]] #seq_ids_in_batch[skip_mask]
                        positions = positions[skip_mask]
                        # print(f"EE'ed seq_ids: {seq_ids_in_batch}")

                        if self.kv_method == "postfill":
                            for idx, ee_req_id in enumerate(seq_ids_in_batch):
                                seq_metadata = self.seq_metadata_map[ee_req_id]
                                seq_metadata.seq.recompute_length += 1

                                # Store the hidden states and positions for recomputing kv cache
                                self.recompute_seq_id_to_hidden_states[ee_req_id].append(self.recompute_seq_id_to_input_hidden_states[ee_req_id])
                                self.recompute_seq_id_to_positions[ee_req_id].append(positions[idx])

                        self.rebatching_time += time.perf_counter() - rebatching_start_time
                    
                    # EE from shallow_exit_layer_1
                    if self.norm:
                        hidden_states = self.norm(hidden_states)
                    
                    # returns: hidden_states, seq_ids_in_batch, exited_rates, lm_logits, ee_from_layer (-1 for no ee), is_flush_1, is_flush_2, recompute_dict, conf, latency_only_ee_iter_time
                    return hidden_states, seq_ids_in_batch, self.exited_rates, lm_logits, 1, False, False, recompute_dict, conf, None

                        
                else:
                    # print(f"[LlamaModel.forward] no ee {seq_ids_in_batch}. exited rates: {self.exited_rates}")
                    self.exited_rates[1] += len(seq_ids_in_batch)
        
        # Layer [shallow_exit_layer_1, last_layer]
        for i in range(self.shallow_exit_layer_1, len(self.layers)):
            layer = self.layers[i]

            # Check shallow_exit_layer_2
            if cache_engine and hidden_states.size(0) <= self.max_batch_size and not recompute_dict:
                if self.early_exit_head:
                    lm_logits = self.early_exit_head(self.norm(hidden_states))
                else:
                    lm_logits = _get_logits(self.norm(hidden_states), self.sampler.embedding, self.sampler.vocab_size)
                
                skip_mask, conf, need_skip = self.get_skip_mask(
                    logits=lm_logits,
                    hidden_states=hidden_states,
                    conf_threshold=self.conf_threshold_2,
                    ee_policy=self.ee_policy,
                    return_conf=True,
                    rebatching_ee_factor=rebatching_ee_factor
                )

                if need_skip:
                    # print(f"Exiting with confidence {conf}. exited rates: {self.exited_rates}", flush=True)
                    rebatching_start_time = time.perf_counter()

                    has_ee = True

                    if torch.all(skip_mask):

                        self.exited_rates[0] += len(seq_ids_in_batch)
                        # 3.1 If all requests want to EE, no rebatching is done.

                        if self.kv_method == "copy":
                            self.fill_missing_kvcache_with_copy(i-1, cache_engine, positions, exited_req_indices=None, seq_ids_to_copy=seq_ids_in_batch)
                        elif self.kv_method == "postfill":
                            for idx, ee_req_id in enumerate(seq_ids_in_batch):
                                seq_metadata = self.seq_metadata_map[ee_req_id]
                                seq_metadata.seq.recompute_length += 1

                                # Store the hidden states and positions for recomputing kv cache
                                self.recompute_seq_id_to_hidden_states[ee_req_id].append(self.recompute_seq_id_to_input_hidden_states[ee_req_id])
                                self.recompute_seq_id_to_positions[ee_req_id].append(positions[idx])
                    else:
                        exited_req_indices = torch.where(skip_mask)[0]
                        if self.kv_method == "copy":
                            # Need to copy the KV cache for the requests that EE i.e. skip_mask[i] is True.
                            token_indices = positions[exited_req_indices]
                            seq_ids_to_copy = [seq_ids_in_batch[i] for i in exited_req_indices]

                            self.fill_missing_kvcache_with_copy(i-1, cache_engine, token_indices, exited_req_indices=exited_req_indices, seq_ids_to_copy=seq_ids_to_copy)

                        self.exited_rates[0] += len(exited_req_indices)
                        self.exited_rates[1] += (len(seq_ids_in_batch) - len(exited_req_indices))

                        lm_logits = lm_logits[exited_req_indices]



                        req_idx_in_batch_to_buffer = []
                        seq_ids_in_batch_to_buffer = []

                        for req_idx, skip in enumerate(skip_mask):
                            if skip:
                                pass
                                
                            else:
                                # 3.2 Put requests that don't EE into `deep_buffer`.
                                req_idx_in_batch_to_buffer.append(req_idx)
                                seq_ids_in_batch_to_buffer.append(seq_ids_in_batch[req_idx])
                            
                        self.deep_buffer_2.add_hidden_states(hidden_states[req_idx_in_batch_to_buffer], seq_ids_in_batch_to_buffer, positions[req_idx_in_batch_to_buffer])


                        # Keep requests EE
                        hidden_states = hidden_states[skip_mask]
                        seq_ids_in_batch = [seq_ids_in_batch[i] for i in range(len(seq_ids_in_batch)) if skip_mask[i]] #seq_ids_in_batch[skip_mask]
                        positions = positions[skip_mask]
                        # print(f"EE'ed seq_ids: {seq_ids_in_batch}")

                        if self.kv_method == "postfill":
                            for idx, ee_req_id in enumerate(seq_ids_in_batch):
                                seq_metadata = self.seq_metadata_map[ee_req_id]
                                seq_metadata.seq.recompute_length += 1

                                # Store the hidden states and positions for recomputing kv cache
                                self.recompute_seq_id_to_hidden_states[ee_req_id].append(self.recompute_seq_id_to_input_hidden_states[ee_req_id])
                                self.recompute_seq_id_to_positions[ee_req_id].append(positions[idx])



                        
                    self.rebatching_time += time.perf_counter() - rebatching_start_time
                    break
                else:
                    # print(f"[LlamaModel.forward] no ee {seq_ids_in_batch}. exited rates: {self.exited_rates}")
                    self.exited_rates[1] += len(seq_ids_in_batch)
                    

            hidden_states = layer(positions, hidden_states, kv_caches[i])
        
        if self.norm:
            hidden_states = self.norm(hidden_states)

        is_flush_1 = start_at_exit_layer_1

        if has_ee:
            return hidden_states, seq_ids_in_batch, self.exited_rates, lm_logits, 2, is_flush_1, False, recompute_dict, conf, None
        else:
            return hidden_states, seq_ids_in_batch, self.exited_rates, None, -1, is_flush_1, False, recompute_dict, None, None


class LlamaForCausalLM(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        vocab_size = ((config.vocab_size + 63) // 64) * 64

        self.is_pipeline_first_stage = is_pipeline_first_stage()
        self.is_pipeline_last_stage = is_pipeline_last_stage()

        self.lm_head = None
        if self.is_pipeline_last_stage:
            self.lm_head = ColumnParallelLinear(
                config.hidden_size,
                vocab_size,
                bias=False,
                gather_output=False,
                perform_initialization=False,
            )
        self.early_exit_head = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        cache_engine: Optional[vATTNCacheEngine] = None,
        seq_ids_in_batch: Optional[List[int]] = None,
        seq_metadata_list: Optional[List[SequenceMetadata]] = None,
        rebatching_ee_factor: float = 0,
    ) -> torch.Tensor:
        if not self.is_pipeline_first_stage:
            # hidden_states_shape: num_tokens x hidden_size
            hidden_states = torch.empty(
                (positions.shape[0], self.config.hidden_size),
                dtype=self.config.dtype,
                device=hidden_states.device,
            )
            hidden_states = recv(hidden_states)

        hidden_states, output_seq_ids, exited_rates, lm_logits, ee_from_layer, is_flush_1, is_flush_2, recompute_dict, conf, latency_only_ee_iter_time = self.model(hidden_states, positions, kv_caches, self.lm_head, cache_engine=cache_engine, seq_ids_in_batch=seq_ids_in_batch, seq_metadata_list=seq_metadata_list, rebatching_ee_factor=rebatching_ee_factor)

        if not self.is_pipeline_last_stage:
            send(hidden_states)

        return hidden_states, output_seq_ids, exited_rates, lm_logits, ee_from_layer, is_flush_1, is_flush_2, recompute_dict, conf, latency_only_ee_iter_time

    _column_parallel_layers = []
    _row_parallel_layers = ["o_proj", "down_proj"]

    def load_weights(
        self,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        load_format: str = "auto",
        revision: Optional[str] = None,
    ):
        weight_suffixes = ["weight"]

        column_parallel_weights: List[str] = []
        for layer in self._column_parallel_layers:
            for suffix in weight_suffixes:
                column_parallel_weights.append(f"{layer}.{suffix}")
        row_parallel_weights: List[str] = []
        for layer in self._row_parallel_layers:
            for suffix in weight_suffixes:
                row_parallel_weights.append(f"{layer}.{suffix}")

        tp_size = get_tensor_model_parallel_world_size()
        pp_size = get_pipeline_model_parallel_world_size()
        tensor_model_parallel_rank = get_tensor_model_parallel_rank()
        pp_model_parallel_rank = get_pipeline_model_parallel_rank()

        assert self.config.num_hidden_layers % pp_size == 0
        layers_per_stage = self.config.num_hidden_layers // pp_size

        first_layer_id = layers_per_stage * pp_model_parallel_rank
        last_layer_id = layers_per_stage * (pp_model_parallel_rank + 1) - 1

        q_proj_shard_size = self.config.hidden_size // tp_size
        kv_proj_shard_size = (
            self.config.hidden_size
            // self.config.num_attention_heads
            * self.config.num_key_value_heads
            // tp_size
        )
        attention_weight_specs = [
            # (weight_name, shard_size, offset)
            ("q_proj", q_proj_shard_size, 0),
            ("k_proj", kv_proj_shard_size, q_proj_shard_size),
            ("v_proj", kv_proj_shard_size, q_proj_shard_size + kv_proj_shard_size),
        ]
        state_dict = self.state_dict()

        for name, loaded_weight in hf_model_weights_iterator(
            model_name_or_path, cache_dir, load_format, revision
        ):
            if "rotary_emb.inv_freq" in name:
                continue

            if pp_model_parallel_rank != 0 and "embed_tokens" in name:
                continue

            if pp_model_parallel_rank != pp_size - 1 and (
                "lm_head" in name or name == "model.norm.weight"
            ):
                continue

            if "model.layers" in name:
                layer_id = int(name.split(".")[2])
                if layer_id < first_layer_id or layer_id > last_layer_id:
                    continue

                new_layer_id = layer_id - first_layer_id
                name = name.replace(str(layer_id), str(new_layer_id))

            is_attention_weight = False
            for weight_name, shard_size, offset in attention_weight_specs:
                if weight_name not in name:
                    continue
                param = state_dict[name.replace(weight_name, "qkv_proj")]

                loaded_weight = loaded_weight[
                    shard_size
                    * tensor_model_parallel_rank : shard_size
                    * (tensor_model_parallel_rank + 1)
                ]
                param_slice = param.data[offset : offset + shard_size]
                assert param_slice.shape == loaded_weight.shape

                param_slice.copy_(loaded_weight)
                is_attention_weight = True
                break
            if is_attention_weight:
                continue

            is_gate_up_weight = False
            for stride_id, weight_name in enumerate(["gate_proj", "up_proj"]):
                if weight_name not in name:
                    continue
                param = state_dict[name.replace(weight_name, "gate_up_proj")]

                shard_size = param.shape[0] // 2
                loaded_weight = loaded_weight[
                    shard_size
                    * tensor_model_parallel_rank : shard_size
                    * (tensor_model_parallel_rank + 1)
                ]
                param_slice = param.data[
                    shard_size * stride_id : shard_size * (stride_id + 1)
                ]
                assert param_slice.shape == loaded_weight.shape
                param_slice.copy_(loaded_weight)
                is_gate_up_weight = True
                break
            if is_gate_up_weight:
                continue

            param = state_dict[name]

            # ---------- llama2 ----------
            # if "embed_tokens" in name or "lm_head" in name:
            #     load_padded_tensor_parallel_vocab(
            #         param, loaded_weight, tensor_model_parallel_rank
            #     )
            #     continue
            # ---------- llama2 ----------
            
            # ---------- llama3 ----------
            if "embed_tokens" in name:
                # print(f"[DEBUG] Loading {name} into {param.shape}")
                load_padded_tensor_parallel_vocab(
                    param, loaded_weight, tensor_model_parallel_rank
                )
                # print(f"[DEBUG] embed_tokens weight sum after copy: {param.sum().item()}")
                continue

            if "lm_head" in name:
                # print(f"[DEBUG] Loading {name} into {param.shape}")
                load_padded_tensor_parallel_vocab(
                    param, loaded_weight, tensor_model_parallel_rank
                )
                continue
            # ---------- llama3 ----------


            load_tensor_parallel_weights(
                param,
                loaded_weight,
                name,
                column_parallel_weights,
                row_parallel_weights,
                tensor_model_parallel_rank,
            )
            
            # ---------- For small models that tie word embeddings (i.e. self.config.tie_word_embeddings == True) ----------
            # if self.config.tie_word_embeddings and self.lm_head is not None and self.model.embed_tokens is not None:
            #     self.lm_head.weight = self.model.embed_tokens.weight
        
        # Load early exit head
        if self.config.early_exit_head_path:
            print(f"[LlamaForCausalLM.load_weights] Loading early exit head from {self.config.early_exit_head_path}")
            # early_exit_head_path = "/workspace/xutingl/finetune-ee/output/early_exit_head.pt"
            early_exit_head_path = self.config.early_exit_head_path
            checkpoint = torch.load(early_exit_head_path)
            early_exit_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
            early_exit_head.load_state_dict(checkpoint['early_exit_head_state_dict'])
            self.early_exit_head = early_exit_head
            self.early_exit_head.to("cuda:0")
            self.model.early_exit_head = self.early_exit_head
        else:
            print(f"[LlamaForCausalLM.load_weights] No early exit head path provided. Using default head.")
    
    def set_sampler(self, sampler: Optional[Sampler] = None):
        self.model.sampler = sampler