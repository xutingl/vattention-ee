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
        attn_output = get_attention_wrapper().forward(
            q,
            k,
            v,
            kv_cache,
            self.scaling,
            self.layer_id,
        )
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
        self.hidden_states = self.hidden_states.to(torch.float16)
        self.positions = torch.zeros(self.capacity, device='cuda:0') # [capacity]
        self.positions = self.positions.to(torch.int64)
        self.available_slots = set(range(self.capacity))
        self.hidden_states_map = dict() # keys: req_ids, values: indices in hidden_states.
        self.dtype = torch.float16
        self.time_spent_adding = 0
        self.time_spent_taking = 0
    
        
    def add_hidden_states(self, hidden_states: torch.Tensor, req_ids: List[int], positions: torch.Tensor) -> None:
        start_time = time.time()
        num_hidden_states = hidden_states.size(0)
        slots = []
        for i in range(num_hidden_states):
            slot = self.available_slots.pop()
            slots.append(slot)
            self.hidden_states_map[req_ids[i]] = slot
        self.hidden_states[slots] = hidden_states
        self.positions[slots] = positions

        self.time_spent_adding += time.time() - start_time

        
            
    
    """
    Args:
        num: number of hidden states to take. If 0, take all hidden states in the buffer.
    Returns:
        output_hidden_states: Tensor. hidden states taken from the buffer. <num, hidden_state_length>
        output_req_ids: List[int]. req_ids corresponding to the hidden states. <num>
        output_positions: Tensor. positions corresponding to the hidden states. <num>
    """
    def take_hidden_states(self, num: int=-1) -> Tuple[torch.Tensor, List[int], torch.Tensor]: 
        start_time = time.time()
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

        self.time_spent_taking += time.time() - start_time
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
        self.shallow_exit_layer = config.shallow_exit_layer
        self.conf_threshold = config.conf_threshold
        self.exited_rates = [0, 1] # [0]: exited, [1]: not exited. Initialized `not exited` to 1 to avoid division by 0.

        self.max_batch_size = config.max_num_seqs
        # self.start_buffer = HiddenStatesBuffer(self.max_batch_size, self.max_batch_size * 3 + 1) # Buffers the hidden states of the token arrived at first layer
        self.start_buffer = [] # Start buffer will no loger be used. Scheduler will send flush request (an empty schedule) to flush deep buffer. So no request will be moved to start buffer.
        self.deep_buffer = HiddenStatesBuffer(self.max_batch_size, self.max_batch_size * 2 + 1) # Buffers the hidden states that EE'ed
        self.seq_metadata_map: Dict[int, SequenceMetadata] = {} # keys: seq_ids, values: SequenceMetadata. Used to update kv cache with updated sequences in the current batch.

        self.batch_size_lst = [0]

        self.avg_exited_conf = 0.0
        self.conf_sum = 0.0
        self.exited_cnt = 0

        self.update_kvcache_time_cnt = 0

        self.process_ee_time = 0

        self.sampler: Sampler = None
    
    def softmax_confidence(
        self,
        logits: torch.Tensor,
    ):
        probs = torch.softmax(logits, dim=-1)
        top_2 = torch.topk(probs, dim=-1, k=2)[0]
        return (top_2[..., 0] - top_2[..., 1]).squeeze()
    
    def get_skip_mask(
        self,
        logits: torch.Tensor = None,
        hidden_states: torch.Tensor = None,
        ee_policy: str = "eager",
        return_conf=False,
    ):
        # assert ee_policy != "off", "Turn off EE by setting self.use_shallow_deep = False. Set policy to 'off' incurrs unnecessary overhead."
        if hidden_states.size(0) > 16:
            # Heuristic to avoid using EE for prefilling
            mask = torch.tensor(0.0, device=hidden_states.device).bool()
            conf = torch.tensor(0.0, device=hidden_states.device)
            if not return_conf:
                return mask, False
            else:
                return mask, conf, False
        logits = logits[~torch.any(logits.isnan(),dim=1)]
        conf = self.softmax_confidence(logits)
        conf = conf[~torch.isnan(conf)]
        mask = torch.where(conf <= self.conf_threshold, 0.0, 1.0).bool()

        need_skip = torch.any(mask)

        if ee_policy != "rebatching":
            
            conf = torch.mean(conf)
            if ee_policy == "eager":
                need_skip = torch.any(mask)
            elif ee_policy == "lazy":
                need_skip = torch.all(mask)
            elif ee_policy == "average":
                val = 0.0 if conf <= self.conf_threshold else 1.0
                need_skip = torch.tensor(val, device=hidden_states.device).bool()
            else:
                raise ValueError("Invalid EE policy: {}".format(ee_policy))
        

        if not return_conf:
            return mask, need_skip
        else:
            return mask, conf, need_skip
    
    def measure_batch_size(self, hidden_states: torch.Tensor):
        batch_size = hidden_states.size(0)
        if batch_size <= self.max_batch_size:
            self.batch_size_lst.append(batch_size)
    
    """
    For non-rebatching policies, requests information for the current batch is passed to cache_engine in `base_worker.py` and `model_runner.py`.
    For Rebatching, becuase we are updating request in current batch, we notify the cache_engine to update the kv cache using this function (and skip the update in the above 2 files).
    """
    def update_seqs_in_kvcache(
        self,
        seq_ids_in_batch: List[int],
        cache_engine: vATTNCacheEngine,
    ) -> None:  
        start_time = time.time()
        # assert self.ee_policy == "rebatching", "update_seqs_in_kvcache is only used in rebatching mode."
        # updated_seq_metadata_list = [self.seq_metadata_map[seq_id] for seq_id in seq_ids_in_batch] 
        updated_seq_metadata_list = []
        for seq_id in seq_ids_in_batch:
            
            updated_seq_metadata_list.append(self.seq_metadata_map[seq_id])

        cache_engine.step(updated_seq_metadata_list) # Originally in base_worker
        get_attention_wrapper().begin_forward(updated_seq_metadata_list) # Originally in model_runner

        self.update_kvcache_time_cnt += time.time() - start_time

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
        #self.measure_batch_size(hidden_states)

        batch_size = hidden_states.size(0) if hidden_states.size(0) <= self.max_batch_size else self.max_batch_size

        if self.embed_tokens:
            hidden_states = self.embed_tokens(hidden_states)
        
        if seq_ids_in_batch is not None:
            for seq_metadata in seq_metadata_list:
                self.seq_metadata_map[seq_metadata.seq.seq_id] = seq_metadata
            
            if self.ee_policy == "rebatching":
                # This is the prefilling of a rebatching run
                # print(f"[LlamaModel.forward_without_rebatching] prefilling with seq_ids_in_batch: {seq_ids_in_batch}. batch size: {hidden_states.size(0)}")
                self.update_seqs_in_kvcache(seq_ids_in_batch, cache_engine)
        
        check_for_ee = cache_engine is not None and self.ee_policy != "off" and self.shallow_exit_layer is not None and hidden_states.size(0) == self.max_batch_size

        has_ee = False
        for i in range(len(self.layers)):
            layer = self.layers[i]
            if check_for_ee and i == self.shallow_exit_layer:
                #need_skip = random.random() < 0.4
                # lm_logits, _ = lm_head(self.norm(hidden_states))

                lm_logits = _get_logits(self.norm(hidden_states), self.sampler.embedding, self.sampler.vocab_size)

                skip_mask, conf, need_skip = self.get_skip_mask(
                    logits=lm_logits,
                    hidden_states=hidden_states,
                    ee_policy=self.ee_policy,
                    return_conf=True
                )
                # need_skip = True
                #need_skip=False

                if need_skip:
                    has_ee = True
                    # print(f"[LlamaModel.forward_without_rebatching] trying to exit with confidence {conf}.")
                    self.exited_rates[0] += batch_size
                    # print(f"Exiting with confidence {conf}. exited rates: {self.exited_rates}", flush=True)

                    # Copy layer i-1's kv cache for the prev token to layer i - last layer.
                    # [TODO] i-2 seems to give better results.\

                    # Copy mothod 1
                    # for batch_idx, token_idx in enumerate(positions):
                    #     for l in range(i, len(self.layers)):
                    #         cache_engine.copy_k_cache_between_layers(i-1, l, batch_idx, token_idx)
                    #         cache_engine.copy_v_cache_between_layers(i-1, l, batch_idx, token_idx)


                    # Copy method 2
                    # for req_idx, token_idx in enumerate(positions):
                    #     cache_engine.copy_kv_cache_starting_at_layer(i-1, req_idx, token_idx)

                    # Copy method 3
                    req_indices = torch.arange(len(positions), device=positions.device)
                    token_indices = positions
                    cache_engine.copy_kv_cache(i-1, req_indices, token_indices)

                    # print(f"[LlamaModel.forward_without_rebatching] Exited with confidence {conf}.")
                    # print(f"[LlamaModel.forward_without_rebatching] Exited with confidence {conf}. positions: {positions}. req_ids: {seq_ids_in_batch}")
                    
                    
                    # conf = torch.mean(conf).item()
                    # self.avg_exited_conf = (Decimal(self.avg_exited_conf) * Decimal(self.exited_cnt) + Decimal(conf) * Decimal(batch_size)) / (Decimal(self.exited_cnt) + Decimal(batch_size))
                    # self.exited_cnt += batch_size
                    # print(f"[LLamaModel.forward_without_rebatching] Exited with confidence {conf}. avg exited conf: {self.avg_exited_conf}. exited rates: {self.exited_rates}, exited_cnt: {self.exited_cnt}", flush=True)
                    
                    break
                else:
                    # print(f"[LlamaModel.forward_without_rebatching] Exited without EE.")
                    
                    self.exited_rates[1] += batch_size
                
            hidden_states = layer(
                positions,
                hidden_states,
                kv_caches[i],
            )

        if self.norm:
            hidden_states = self.norm(hidden_states)

        # print(f"[LlamaModel.forward_without_rebatching] ee_rates: {self.exited_rates}")
        # print(f"[LlamaModel.forward] ee_rates: {self.exited_rates}={(self.exited_rates[0]/sum(self.exited_rates)):.2f}. Avg batch size: {sum(self.batch_size_lst)/len(self.batch_size_lst)}")

        # print(f"[LlamaModel.forward_without_rebatching] returning seq_ids_in_batch: {seq_ids_in_batch}\n")

        if has_ee:
            return hidden_states, seq_ids_in_batch, self.exited_rates, lm_logits
        return hidden_states, seq_ids_in_batch, self.exited_rates, None
    
    """
    When seq_ids_in_batch is provided, rebatching based on early exit status is enabled:
    - We check `deep_buffer` first to see if there are enough hidden states to form a batch. If there are, we will process and return them. The incoming hidden states are added to `start_buffer`. Return.
    - Any requests in the `start_buffer` has a higher priority than incoming requests. Pop requests from `start_buffer` and swap incoming requests to `start_buffer`.
        Process the requests:
        - If all requests want to EE, no rebatching is done.
        - If some requests want to EE, they are returned immediately, and the rest are put into `deep_buffer`.

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
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int], Optional[torch.Tensor]]:
        
        if seq_ids_in_batch is None or self.ee_policy != "rebatching" or hidden_states.size(0) > self.max_batch_size: # Rebatching disabled
            return self.forward_without_rebatching(hidden_states, positions, kv_caches, lm_head, cache_engine, seq_ids_in_batch, seq_metadata_list=seq_metadata_list)
        
        # print(f"\n[LlamaModel.forward] Incoming seq_ids_in_batch: {seq_ids_in_batch}. positions: {positions}")
        

        # Rebatching enabled

        # Update seq_metadasta_map
        for seq_metadata in seq_metadata_list:
            self.seq_metadata_map[seq_metadata.seq.seq_id] = seq_metadata


        if self.embed_tokens:
            hidden_states = self.embed_tokens(hidden_states)


        incoming_batch_size = len(seq_ids_in_batch)


        # 0. Flush: If we receive an empty batch, we process any leftover hidden states in the buffer.
        flush_buffer = incoming_batch_size == 0
        if flush_buffer:
            if len(self.deep_buffer) > 0:
                # Take hidden states from `deep_buffer`
                hidden_states, seq_ids_in_batch, positions = self.deep_buffer.take_hidden_states(min(self.max_batch_size, len(self.deep_buffer)))
                # print(f"[LlamaModel.forward] [1] updating kvcache with seq_ids_in_batch: {seq_ids_in_batch}")
                self.update_seqs_in_kvcache(seq_ids_in_batch, cache_engine)

                #self.measure_batch_size(hidden_states)

                for i in range(self.shallow_exit_layer, len(self.layers)):
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
                return hidden_states, seq_ids_in_batch, self.exited_rates, None
            elif len(self.start_buffer) > 0:
                # Take hidden states from `start_buffer`
                hidden_states, seq_ids_in_batch, positions = self.start_buffer.take_hidden_states(min(self.max_batch_size, len(self.start_buffer)))
                # print(f"[LlamaModel.forward] [2]updating kvcache with seq_ids_in_batch: {seq_ids_in_batch}")
                self.update_seqs_in_kvcache(seq_ids_in_batch, cache_engine)

                #self.measure_batch_size(hidden_states)

                for i in range(len(self.layers)):
                    layer = self.layers[i]
                    hidden_states = layer(
                        positions,
                        hidden_states,
                        kv_caches[i],
                    )
                if self.norm:
                    hidden_states = self.norm(hidden_states)
                # print(f"[LlamaModel.forward] returning flush_buffer 2: start_buffer. seq_ids_in_batch: {seq_ids_in_batch}")
                #print(f"[LlamaModel.forward] ee_rates: {self.exited_rates}={(self.exited_rates[0]/sum(self.exited_rates)):.2f}. Avg batch size: {sum(self.batch_size_lst)/len(self.batch_size_lst)}")
                return hidden_states, seq_ids_in_batch, self.exited_rates, None
            else:
                #print(f"[LlamaModel.forward] flush_buffer: no buffer. seq_ids_in_batch: {seq_ids_in_batch}")
                return None, None, self.exited_rates, None

        # 1. We check `deep_buffer` first to see if there are enough hidden states to form a batch. If there are, we will process and return them. The incoming hidden states are added to `start_buffer`.
        if len(self.deep_buffer) >= self.max_batch_size:
            # 1.1 Put incoming hidden states into `start_buffer`
            # print(f"[LlamaModel.forward] incoming hidden states size: {hidden_states.size()}. hidden states: {hidden_states}")
            # print(f"[LlamaModel.forward] incoming seq_ids_in_batch: {seq_ids_in_batch}")
            # print(f"[LlamaModel.forward] deep_buffer map: {self.deep_buffer.hidden_states_map}")
            
            self.start_buffer.add_hidden_states(hidden_states, seq_ids_in_batch, positions)
            # 1.2 Take hidden states from `deep_buffer`
            hidden_states, seq_ids_in_batch, positions = self.deep_buffer.take_hidden_states(self.max_batch_size)

            # self.measure_batch_size(hidden_states)

            # if 1 in seq_ids_in_batch:
            #     print(f"[LlamaModel.forward] [3] Ther are taken out from deep buffer. updating kvcache with seq_ids_in_batch: {seq_ids_in_batch}")
            self.update_seqs_in_kvcache(seq_ids_in_batch, cache_engine)

            # 1.3 Process hidden states starting from the EE layer
            for i in range(self.shallow_exit_layer, len(self.layers)):
                layer = self.layers[i]
                hidden_states = layer(
                    positions,
                    hidden_states,
                    kv_caches[i],
                )
            if self.norm:
                hidden_states = self.norm(hidden_states)

            # print(f"case 1 seq_ids_in_batch: {seq_ids_in_batch}")
            # print(f"seq ids in deep buffer: {self.deep_buffer.hidden_states_map.keys()}")
            # print(f"seq ids in start buffer: {self.start_buffer.hidden_states_map.keys()}\n")
            #print(f"[LlamaModel.forward] ee_rates: {self.exited_rates}. Avg batch size: {sum(self.batch_size_lst)/len(self.batch_size_lst)}")
            
            return hidden_states, seq_ids_in_batch, self.exited_rates, None

        # 2. Requests in the `start_buffer` has a higher priority than incoming requests. Pop requests from `start_buffer` and swap incoming requests to `start_buffer`.
        num_req_in_start_buffer = len(self.start_buffer)
        if num_req_in_start_buffer > 0:
            num_req_to_take = min(num_req_in_start_buffer, self.max_batch_size)

            if incoming_batch_size + num_req_to_take <= self.max_batch_size:
                # Take hidden states from `start_buffer` and add them to incoming hidden states. No need to add anything back to `start_buffer`
                # Concat hidden states in `start_buffer` and incoming hidden states
                taking_hidden_states, taking_seq_ids_in_batch, taking_positions = self.start_buffer.take_hidden_states(num_req_to_take)
                
                hidden_states = torch.cat([hidden_states, taking_hidden_states], dim=0)
                # seq_ids_in_batch = torch.cat([seq_ids_in_batch, taking_seq_ids_in_batch], dim=0)
                seq_ids_in_batch.extend(taking_seq_ids_in_batch)
                positions = torch.cat([positions, taking_positions], dim=0)
            else: # incoming_batch_size + num_req_to_take > max batch size. 
                if num_req_to_take > incoming_batch_size:
                    # Swap whole incoming hidden states with `start_buffer` hidden states
                    self.start_buffer.add_hidden_states(hidden_states, seq_ids_in_batch, positions)
                    hidden_states, seq_ids_in_batch, positions = self.start_buffer.take_hidden_states(num_req_to_take)
                else:
                    # Swap partial incoming hidden states with `start_buffer` hidden states
                    self.start_buffer.add_hidden_states(hidden_states[:num_req_to_take], seq_ids_in_batch[:num_req_to_take], positions[:num_req_to_take])
                    hidden_states[:num_req_to_take], seq_ids_in_batch[:num_req_to_take], positions[:num_req_to_take] = self.start_buffer.take_hidden_states(num_req_to_take)
                

        self.update_seqs_in_kvcache(seq_ids_in_batch, cache_engine)

        # self.measure_batch_size(hidden_states)

        has_ee = False

        for i in range(len(self.layers)):
            layer = self.layers[i]
            if cache_engine and i == self.shallow_exit_layer and hidden_states.size(0) <= self.max_batch_size: # Avoid EE in profiling (cache_engine is None) and prefilling (batch size is too large)
                # lm_logits, _ = lm_head(self.norm(hidden_states))
                lm_logits = _get_logits(self.norm(hidden_states), self.sampler.embedding, self.sampler.vocab_size)
                skip_mask, conf, need_skip = self.get_skip_mask(
                    logits=lm_logits,
                    hidden_states=hidden_states,
                    ee_policy=self.ee_policy,
                    return_conf=True
                )
                
                if need_skip:
                    # print(f"Exiting with confidence {conf}. exited rates: {self.exited_rates}", flush=True)

                    has_ee = True

                    if torch.all(skip_mask):
                        self.exited_rates[0] += incoming_batch_size
                        # 3.1 If all requests want to EE, no rebatching is done.
                        # k_cache dimention: <batch_size, max_seq_len, num_heads(8), head_dim(128)>
                        # Copy layer i-1's kv cache for the prev token to layer i - last layer.
                        # curr_batch_size = len(seq_ids_in_batch)
                        # for batch_idx, token_pos_idx in enumerate(positions[:curr_batch_size]):
                        #     for l in range(i, len(self.layers)):
                        #         cache_engine.copy_k_cache_between_layers(i-1, l, batch_idx, token_pos_idx)
                        #         cache_engine.copy_v_cache_between_layers(i-1, l, batch_idx, token_pos_idx)
                        

                        # Copy method 3
                        req_indices = torch.arange(len(positions), device=positions.device)
                        token_indices = positions
                        cache_engine.copy_kv_cache(i-1, req_indices, token_indices)


                        
                        # self.conf_sum += sum(conf)
                        # conf = torch.mean(conf)
                        # conf = conf.item()
                        # self.avg_exited_conf = ((Decimal(self.avg_exited_conf) * Decimal(self.exited_cnt)) + Decimal(conf) * Decimal(incoming_batch_size)) / Decimal((self.exited_cnt + incoming_batch_size))

                        # self.exited_cnt += incoming_batch_size
                        # print(f"[LlamaModel.forward] All need to EE with confidence {conf}. incoming_batch_size:{incoming_batch_size}. avg exited conf: {self.avg_exited_conf}; {self.conf_sum / self.exited_cnt}. exited rates: {self.exited_rates}, exit_cnt: {self.exited_cnt}", flush=True)
                    else:

                        # Need to copy the KV cache for the requests that EE i.e. skip_mask[i] is True.
                        req_indices = torch.where(skip_mask)[0]
                        # print(f"req_indices: {req_indices}. skip_mask: {skip_mask}")
                        token_indices = positions[req_indices]
                        cache_engine.copy_kv_cache(i-1, req_indices, token_indices)

                        self.exited_rates[0] += len(req_indices)
                        self.exited_rates[1] += (incoming_batch_size - len(req_indices))

                        lm_logits = lm_logits[req_indices]
                        # assert lm_logits.size(0) == len(req_indices), f"lm_logits size: {lm_logits.size()}, req_indices size: {len(req_indices)}"


                        req_idx_in_batch_to_buffer = []
                        seq_ids_in_batch_to_buffer = []

                        for req_idx, skip in enumerate(skip_mask):
                            if skip:
                                pass
                                # conf_i = conf[req_idx]
                                # conf_i = conf_i.item()
                                # print(f"[LlamaModel.forward] prev avg_exitecd_conf:{self.avg_exited_conf}, conf_i: {conf_i}, exited_cnt: {self.exited_cnt}")
                                # self.avg_exited_conf = ((Decimal(self.avg_exited_conf) * Decimal(self.exited_cnt)) + Decimal(conf_i)) / Decimal((self.exited_cnt + 1))
                                # self.conf_sum += conf_i
                                # self.exited_cnt += 1
                                # print(f"[LlamaModel.forward] Need to EE with confidence {conf_i}. avg exited conf: {self.avg_exited_conf}; {self.conf_sum / self.exited_cnt}. exited rates: {self.exited_rates}. exit_cnt: {self.exited_cnt}", flush=True)
                                # continue # Skip because already handled above
                                # token_pos_idx = positions[req_idx]
                                # for l in range(i, len(self.layers)):
                                #     print(f"[LlamaModel.forward] copying k and v cache between layers {i-1} and {l} for req {req_idx} and token pos {token_pos_idx}. req_ids_in_batch: {seq_ids_in_batch}, current seq_ids_in_batch: {seq_ids_in_batch[req_idx]}")
                                #     cache_engine.copy_k_cache_between_layers(i-1, l, req_idx, token_pos_idx)
                                #     cache_engine.copy_v_cache_between_layers(i-1, l, req_idx, token_pos_idx)
                                
                                
                                # Method 2
                                # cache_engine.copy_kv_cache_starting_at_layer(i-1, req_idx, token_pos_idx)
                                
                            else:
                                # 3.2 Put requests that don't EE into `deep_buffer`.
                                # if seq_ids_in_batch[req_idx] == 1:
                                #     print(f"[LlamaModel.forward] [6] of req 1:Dont't want to EE. adding to deep buffer: seq_ids_in_batch[req_idx]: {seq_ids_in_batch[req_idx]}")

                                # self.deep_buffer.add_hidden_states(hidden_states[req_idx].unsqueeze(0), [seq_ids_in_batch[req_idx]], positions[req_idx].unsqueeze(0))


                                req_idx_in_batch_to_buffer.append(req_idx)
                                seq_ids_in_batch_to_buffer.append(seq_ids_in_batch[req_idx])



                                # print(f"req_id {seq_ids_in_batch[req_idx].item()} don't want EE and is put into deep buffer")
                            
                        self.deep_buffer.add_hidden_states(hidden_states[req_idx_in_batch_to_buffer], seq_ids_in_batch_to_buffer, positions[req_idx_in_batch_to_buffer])


                        # Keep requests EE
                        hidden_states = hidden_states[skip_mask]
                        seq_ids_in_batch = [seq_ids_in_batch[i] for i in range(len(seq_ids_in_batch)) if skip_mask[i]] #seq_ids_in_batch[skip_mask]
                        positions = positions[skip_mask]
                        # print(f"EE'ed seq_ids: {seq_ids_in_batch}")

                        
                    
                    break
                else:
                    self.exited_rates[1] += incoming_batch_size
                    # pass
                
            hidden_states = layer(
                positions,
                hidden_states,
                kv_caches[i],
            )

        if self.norm:
            hidden_states = self.norm(hidden_states)

        # print("================================================")
        # print(f"seq_ids_in_batch: {seq_ids_in_batch}")
        # print(f"seq ids in deep buffer: {self.deep_buffer.hidden_states_map.keys()}")
        # print(f"seq ids in start buffer: {self.start_buffer.hidden_states_map.keys()}")
        # print(f"[LlamaModel.forward] ee_rates: {self.exited_rates}={(self.exited_rates[0]/sum(self.exited_rates)):.2f}. Avg batch size: {sum(self.batch_size_lst)/len(self.batch_size_lst)}")
        # print("================================================")

        # print(f"[LlamaModel.forward] hidden states buffer spent time (adding, taking): ({self.start_buffer.time_spent_adding:.2f}, {self.start_buffer.time_spent_taking:.2f}). deep buffer spent time (adding, taking): ({self.deep_buffer.time_spent_adding:.2f}, {self.deep_buffer.time_spent_taking:.2f}). update kvcache spent time: {self.update_kvcache_time_cnt:.2f}", flush=True)

        if has_ee:
            return hidden_states, seq_ids_in_batch, self.exited_rates, lm_logits
        return hidden_states, seq_ids_in_batch, self.exited_rates, None


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

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        cache_engine: Optional[vATTNCacheEngine] = None,
        seq_ids_in_batch: Optional[List[int]] = None,
        seq_metadata_list: Optional[List[SequenceMetadata]] = None,
    ) -> torch.Tensor:
        if not self.is_pipeline_first_stage:
            # hidden_states_shape: num_tokens x hidden_size
            hidden_states = torch.empty(
                (positions.shape[0], self.config.hidden_size),
                dtype=self.config.dtype,
                device=hidden_states.device,
            )
            hidden_states = recv(hidden_states)

        hidden_states, output_seq_ids, exited_rates, lm_logits = self.model(hidden_states, positions, kv_caches, self.lm_head, cache_engine=cache_engine, seq_ids_in_batch=seq_ids_in_batch, seq_metadata_list=seq_metadata_list)

        if not self.is_pipeline_last_stage:
            send(hidden_states)

        return hidden_states, output_seq_ids, exited_rates, lm_logits

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

            if "embed_tokens" in name or "lm_head" in name:
                load_padded_tensor_parallel_vocab(
                    param, loaded_weight, tensor_model_parallel_rank
                )
                continue

            load_tensor_parallel_weights(
                param,
                loaded_weight,
                name,
                column_parallel_weights,
                row_parallel_weights,
                tensor_model_parallel_rank,
            )
    
    def set_sampler(self, sampler: Optional[Sampler] = None):
        self.model.sampler = sampler
