# coding=utf-8
# Adapted from
# https://huggingface.co/Qwen/Qwen-7B/blob/main/modeling_qwen.py
# Copyright (c) Alibaba Cloud.
# LICENSE: https://huggingface.co/Qwen/Qwen-7B/blob/main/LICENSE
"""Inference-only QWen model compatible with HuggingFace weights.

The input of the model is flattened to a 1D tensor of tokens.
"""
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
import math
import time
from collections import defaultdict

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
    convert_pyslice_to_tensor,
    hf_model_weights_iterator,
    load_padded_tensor_parallel_vocab,
    load_tensor_parallel_weights,
)
from sarathi.transformers_utils.configs.qwen import QWenConfig
from sarathi.worker.cache_engine import KVCache
from sarathi.worker.cache_engine.vATTN_cache_engine import vATTNCacheEngine
from sarathi.core.datatypes.sequence import Sequence, SequenceMetadata
from sarathi.core.sequence_manager.base_sequence_manager import BaseSequenceManager

from .ee_utils import HiddenStatesBuffer, softmax_confidence, get_adaptive_rebatching_threshold, update_seqs_in_kvcache, fill_missing_kvcache_with_copy, get_skip_mask


class QWenMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
        layer_id: Optional[int] = None,
    ):
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
        self.c_proj = RowParallelLinear(
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
        x, _ = self.c_proj(x)
        return x


class QWenAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        max_position_embeddings: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        layer_id: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        tensor_model_parallel_world_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tensor_model_parallel_world_size == 0
        self.num_heads = self.total_num_heads // tensor_model_parallel_world_size
        self.head_dim = hidden_size // self.total_num_heads
        self.layer_id = layer_id

        # pylint: disable=invalid-name
        self.c_attn = ColumnParallelLinear(
            hidden_size,
            3 * hidden_size,
            bias=True,
            gather_output=False,
            perform_initialization=False,
            linear_metric_name=OperationMetrics.ATTN_PRE_PROJ,
            communication_metric_name=OperationMetrics.ATTN_PRE_PROJ_ALL_GATHER,
            layer_id=layer_id,
        )
        self.c_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            perform_initialization=False,
            linear_metric_name=OperationMetrics.ATTN_POST_PROJ,
            communication_metric_name=OperationMetrics.ATTN_POST_PROJ_ALL_REDUCE,
            layer_id=layer_id,
        )
        self.scaling = self.head_dim**-0.5

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
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
        qkv, _ = self.c_attn(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)

        with self._attn_rope_timer:
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
            print(f"[QWenAttention.forward] Error in layer {self.layer_id}. q.shape: {q.shape}, k.shape: {k.shape}, v.shape: {v.shape}, positions.shape: {positions.shape}. kv_cache.shape: {kv_cache[0][0].shape}")
            raise e

        output, _ = self.c_proj(attn_output)
        return output


class QWenBlock(nn.Module):

    def __init__(self, config: QWenConfig, layer_id: Optional[int] = None):
        super().__init__()
        self.ln_1 = RMSNorm(
            config.hidden_size, 
            eps=config.layer_norm_epsilon,
            norm_name=OperationMetrics.INPUT_LAYERNORM,
            layer_id=layer_id,
        )

        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        self.attn = QWenAttention(
            config.hidden_size,
            config.num_attention_heads,
            config.max_position_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            layer_id=layer_id,
        )

        self.ln_2 = RMSNorm(
            config.hidden_size, 
            eps=config.layer_norm_epsilon,
            norm_name=OperationMetrics.POST_ATTENTION_LAYERNORM,
            layer_id=layer_id,
        )

        self.mlp = QWenMLP(
            config.hidden_size,
            config.intermediate_size // 2,
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
        hidden_states = self.ln_1(hidden_states)
        hidden_states = self.attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class QWenModel(nn.Module):

    def __init__(self, config: QWenConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.wte = None
        if is_pipeline_first_stage():
            vocab_size = ((config.vocab_size + 63) // 64) * 64
            self.wte = VocabParallelEmbedding(
                vocab_size, config.hidden_size, perform_initialization=False
            )

        num_layers = (
            config.num_hidden_layers // get_pipeline_model_parallel_world_size()
        )
        layer_offset = get_pipeline_model_parallel_rank() * num_layers
        self.h = nn.ModuleList(
            [
                QWenBlock(config, layer_id=layer_id + layer_offset)
                for layer_id in range(num_layers)
            ]
        )

        self.ln_f = None
        if is_pipeline_last_stage():
            self.ln_f = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        
        # EE-related attributes
        self.ee_policy = getattr(config, 'ee_policy', 'off')
        self.shallow_exit_layer = getattr(config, 'shallow_exit_layer', None)
        self.conf_threshold = getattr(config, 'conf_threshold', 0.5)
        self.exited_rates = [0, 1]  # [0]: early-exited, [1]: not early-exited

        self.max_batch_size = getattr(config, 'max_num_seqs', 32)
        # Width/dtype from config so bf16 Qwen checkpoints aren't silently downcast
        # to fp16 (which also dtype-mismatch-crashes the rebatching torch.cat merge).
        self.deep_buffer = HiddenStatesBuffer(self.max_batch_size, self.max_batch_size * 2 + 1, hidden_state_length=config.hidden_size, dtype=config.dtype)
        self.seq_metadata_map: Dict[int, SequenceMetadata] = {}

        self.batch_size_lst = [0]

        self.avg_exited_conf = 0.0
        self.conf_sum = 0.0
        self.exited_cnt = 0

        self.update_kvcache_time_lst = []
        self.ee_overhead_time_lst = []
        self.fill_kvcache_time_lst = []

        self.prefill_batch_size_limit = 64

        self.sampler: Sampler = None

        self.early_exit_head = None
        self.rebatching_time = 0
        self.num_ee_threshold = getattr(config, 'num_ee_threshold', -1)

        self.kv_method = getattr(config, 'kv_method', 'postfill')
        self.recompute_seq_id_to_hidden_states = defaultdict(list)
        self.recompute_seq_id_to_positions = defaultdict(list)
        self.recompute_seq_id_to_input_hidden_states = dict()

    def measure_batch_size(self, hidden_states: torch.Tensor, seq_ids_in_batch: List[int]=[]):
        if len(seq_ids_in_batch) > 0:
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
                
                # Treat the EE'ed sequence as a prefill step
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

        # Collect stored recomputed data
        all_recompute_hidden_states_lst = []
        all_recompute_positions_lst = []

        for seq_id, recompute_length in recompute_dict.items():
            seq_hidden_states = self.recompute_seq_id_to_hidden_states[seq_id]
            seq_positions = self.recompute_seq_id_to_positions[seq_id]

            all_recompute_hidden_states_lst.extend(seq_hidden_states)
            all_recompute_positions_lst.extend(seq_positions)

            self.recompute_seq_id_to_hidden_states[seq_id].clear()
            self.recompute_seq_id_to_positions[seq_id].clear()

            all_recompute_hidden_states_lst.append(hidden_states[recompute_req_to_idx[seq_id]])
            all_recompute_positions_lst.append(positions[recompute_req_to_idx[seq_id]])
            
        all_recompute_hidden_states_tensor = torch.stack(all_recompute_hidden_states_lst)
        all_recompute_positions_tensor = torch.stack(all_recompute_positions_lst)
        
        non_recompute_hidden_states = hidden_states[list(non_recompute_req_to_idx.values())]
        non_recompute_positions = positions[list(non_recompute_req_to_idx.values())]
        
        final_hidden_states = torch.cat([all_recompute_hidden_states_tensor, non_recompute_hidden_states], dim=0)
        final_positions = torch.cat([all_recompute_positions_tensor, non_recompute_positions], dim=0)
        final_seq_ids = list(recompute_req_to_idx.keys()) + list(non_recompute_req_to_idx.keys())
        
        return recompute_dict, final_hidden_states, final_positions, final_seq_ids

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

        # Update seq_metadata_map
        if seq_metadata_list:
            for seq_metadata in seq_metadata_list:
                self.seq_metadata_map[seq_metadata.seq.seq_id] = seq_metadata
        
        if self.wte:
            hidden_states = self.wte(hidden_states)
        
        # Store input hidden states for recomputing kv cache
        input_hidden_states = hidden_states

        if self.kv_method == "postfill" and seq_ids_in_batch is not None and self.ee_policy != "rebatching":
            recompute_dict, hidden_states, positions, seq_ids_in_batch = self.check_req_for_kv_recompute(hidden_states, positions, seq_ids_in_batch)
            if len(recompute_dict) > 0:
                update_seqs_in_kvcache(self.seq_metadata_map, seq_ids_in_batch, cache_engine)
        else:
            recompute_dict = {}

        if seq_ids_in_batch is not None:
            for seq_metadata in seq_metadata_list:
                self.seq_metadata_map[seq_metadata.seq.seq_id] = seq_metadata
            
            if self.ee_policy == "rebatching":
                update_seqs_in_kvcache(self.seq_metadata_map, seq_ids_in_batch, cache_engine)
        
        check_for_ee = cache_engine is not None and self.ee_policy != "off" and self.shallow_exit_layer is not None and hidden_states.size(0) <= self.max_batch_size and len(recompute_dict) == 0

        conf = None
        exited_conf_lst = None
        has_ee = False
        latency_only_ee_iter_time = None
        num_seq_would_ee_but_stay, num_seq_would_not_ee_but_ee = 0, 0
        
        for i in range(len(self.h)):
            layer = self.h[i]
            if check_for_ee and i == self.shallow_exit_layer:
                if self.early_exit_head:
                    lm_logits = self.early_exit_head(self.ln_f(hidden_states))
                else:
                    lm_logits, _ = lm_head(self.ln_f(hidden_states))

                conf = softmax_confidence(lm_logits)
                skip_mask, conf, exited_conf_lst, need_skip, num_seq_would_ee_but_stay, num_seq_would_not_ee_but_ee = get_skip_mask(
                    conf=conf,
                    conf_threshold=self.conf_threshold,
                    logits=lm_logits,
                    hidden_states=hidden_states,
                    num_ee_threshold=self.num_ee_threshold,
                    ee_policy=self.ee_policy,
                    return_conf=True
                )

                if need_skip:
                    if hidden_states.size(0) <= self.max_batch_size:
                        self.exited_rates[0] += hidden_states.size(0)

                    if self.ee_policy == "latency-only" and not torch.all(skip_mask):
                        latency_only_exited_req_indices = torch.where(skip_mask)[0]
                        latency_only_exited_hidden_states = hidden_states[latency_only_exited_req_indices].clone()
                        latency_only_ee_iter_time = time.perf_counter() - in_model_iter_start_time
                    else:
                        has_ee = True

                        if self.kv_method == "copy":
                            fill_missing_kvcache_with_copy(i-1, cache_engine, positions, exited_req_indices=None, seq_ids_to_copy=seq_ids_in_batch)
                        elif self.kv_method == "postfill":
                            for idx, ee_req_id in enumerate(seq_ids_in_batch):
                                seq_metadata = self.seq_metadata_map[ee_req_id]
                                seq_metadata.seq.recompute_length += 1
                                self.recompute_seq_id_to_hidden_states[ee_req_id].append(input_hidden_states[idx])
                                self.recompute_seq_id_to_positions[ee_req_id].append(positions[idx])
                            recompute_dict = {}
                        break
                else:
                    if hidden_states.size(0) <= self.max_batch_size:
                        self.exited_rates[1] += hidden_states.size(0)
            
            hidden_states = layer(
                positions,
                hidden_states,
                kv_caches[i],
            )
        
        if latency_only_ee_iter_time is not None:
            hidden_states[latency_only_exited_req_indices] = latency_only_exited_hidden_states

        if self.ln_f:
            hidden_states = self.ln_f(hidden_states)

        if has_ee:
            return hidden_states, seq_ids_in_batch, self.exited_rates, lm_logits, False, recompute_dict, conf, exited_conf_lst, None, num_seq_would_ee_but_stay, num_seq_would_not_ee_but_ee
        return hidden_states, seq_ids_in_batch, self.exited_rates, None, False, recompute_dict, conf, exited_conf_lst, latency_only_ee_iter_time, num_seq_would_ee_but_stay, num_seq_would_not_ee_but_ee

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        lm_head,
        cache_engine: Optional[vATTNCacheEngine] = None,
        seq_ids_in_batch: Optional[List[int]] = None,
        seq_metadata_list: Optional[List[SequenceMetadata]] = None,
        rebatching_ee_factor: float = 0,
        priority_reqs: List[int] = [],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int], Optional[torch.Tensor]]:
        
        if self.ee_policy != "rebatching" or hidden_states.size(0) > self.prefill_batch_size_limit:
            return self.forward_without_rebatching(hidden_states, positions, kv_caches, lm_head, cache_engine, seq_ids_in_batch, seq_metadata_list=seq_metadata_list)

        # Rebatching enabled
        for seq_metadata in seq_metadata_list:
            self.seq_metadata_map[seq_metadata.seq.seq_id] = seq_metadata

        if self.wte:
            hidden_states = self.wte(hidden_states)

        incoming_batch_size = len(seq_ids_in_batch)

        # Store input hidden states for recomputing kv cache
        if self.kv_method == "postfill":
            for i, seq_id in enumerate(seq_ids_in_batch):
                self.recompute_seq_id_to_input_hidden_states[seq_id] = hidden_states[i]

        # 0. Flush: If we receive an empty batch, we process any leftover hidden states in the buffer
        flush_buffer = incoming_batch_size == 0
        if flush_buffer:
            if len(self.deep_buffer) > 0:
                hidden_states, seq_ids_in_batch, positions = self.deep_buffer.take_hidden_states(len(self.deep_buffer))
                update_seqs_in_kvcache(self.seq_metadata_map, seq_ids_in_batch, cache_engine)

                for i in range(self.shallow_exit_layer, len(self.h)):
                    layer = self.h[i]
                    hidden_states = layer(
                        positions,
                        hidden_states,
                        kv_caches[i],
                    )
                if self.ln_f:
                    hidden_states = self.ln_f(hidden_states)
                return hidden_states, seq_ids_in_batch, self.exited_rates, None, True, {}, None, None, None, 0, 0
            else:
                return None, None, self.exited_rates, None, False, {}, None, None, None, 0, 0

        # 1. Process normal requests
        if self.kv_method == "postfill":
            recompute_dict, hidden_states, positions, seq_ids_in_batch = self.check_req_for_kv_recompute(hidden_states, positions, seq_ids_in_batch)
        else:
            recompute_dict = {}

        update_seqs_in_kvcache(self.seq_metadata_map, seq_ids_in_batch, cache_engine)

        conf = None
        has_ee = False
        num_seq_would_ee_but_stay, num_seq_would_not_ee_but_ee = 0, 0

        for i in range(len(self.h)):
            layer = self.h[i]
            if cache_engine and i == self.shallow_exit_layer and hidden_states.size(0) <= self.max_batch_size and not recompute_dict:
                if self.early_exit_head:
                    lm_logits = self.early_exit_head(self.ln_f(hidden_states))
                else:
                    lm_logits, _ = lm_head(self.ln_f(hidden_states))

                ee_check_start_time = time.perf_counter()
                conf = softmax_confidence(lm_logits)
                skip_mask, conf, exited_conf_lst, need_skip, num_seq_would_ee_but_stay, num_seq_would_not_ee_but_ee = get_skip_mask(
                    conf=conf,
                    conf_threshold=self.conf_threshold,
                    logits=lm_logits,
                    hidden_states=hidden_states,
                    num_ee_threshold=self.num_ee_threshold,
                    ee_policy=self.ee_policy,
                    return_conf=True,
                    rebatching_ee_factor=rebatching_ee_factor,
                    seq_ids_in_batch=seq_ids_in_batch,
                    priority_reqs=priority_reqs
                )
                self.ee_overhead_time_lst.append(time.perf_counter() - ee_check_start_time)

                if need_skip:
                    rebatching_start_time = time.perf_counter()
                    has_ee = True

                    if torch.all(skip_mask):
                        self.exited_rates[0] += len(seq_ids_in_batch)

                        if self.kv_method == "copy":
                            fill_missing_kvcache_with_copy(i-1, cache_engine, positions, exited_req_indices=None, seq_ids_to_copy=seq_ids_in_batch)
                        elif self.kv_method == "postfill":
                            for idx, ee_req_id in enumerate(seq_ids_in_batch):
                                seq_metadata = self.seq_metadata_map[ee_req_id]
                                seq_metadata.seq.recompute_length += 1
                                self.recompute_seq_id_to_hidden_states[ee_req_id].append(self.recompute_seq_id_to_input_hidden_states[ee_req_id])
                                self.recompute_seq_id_to_positions[ee_req_id].append(positions[idx])
                    else:
                        exited_req_indices = torch.where(skip_mask)[0]
                        if self.kv_method == "copy":
                            token_indices = positions[exited_req_indices]
                            seq_ids_to_copy = [seq_ids_in_batch[i] for i in exited_req_indices]
                            fill_missing_kvcache_with_copy(i-1, cache_engine, token_indices, exited_req_indices=exited_req_indices, seq_ids_to_copy=seq_ids_to_copy)

                        self.exited_rates[0] += len(exited_req_indices)
                        self.exited_rates[1] += (len(seq_ids_in_batch) - len(exited_req_indices))

                        lm_logits = lm_logits[exited_req_indices]

                        req_idx_in_batch_to_buffer = []
                        seq_ids_in_batch_to_buffer = []

                        for req_idx, skip in enumerate(skip_mask):
                            if not skip:
                                req_idx_in_batch_to_buffer.append(req_idx)
                                seq_ids_in_batch_to_buffer.append(seq_ids_in_batch[req_idx])
                            
                        self.deep_buffer.add_hidden_states(hidden_states[req_idx_in_batch_to_buffer], seq_ids_in_batch_to_buffer, positions[req_idx_in_batch_to_buffer])

                        hidden_states = hidden_states[skip_mask]
                        seq_ids_in_batch = [seq_ids_in_batch[i] for i in range(len(seq_ids_in_batch)) if skip_mask[i]]
                        positions = positions[skip_mask]

                        if self.kv_method == "postfill":
                            for idx, ee_req_id in enumerate(seq_ids_in_batch):
                                seq_metadata = self.seq_metadata_map[ee_req_id]
                                seq_metadata.seq.recompute_length += 1
                                self.recompute_seq_id_to_hidden_states[ee_req_id].append(self.recompute_seq_id_to_input_hidden_states[ee_req_id])
                                self.recompute_seq_id_to_positions[ee_req_id].append(positions[idx])
                        
                    self.rebatching_time += time.perf_counter() - rebatching_start_time
                    break
                else:
                    self.exited_rates[1] += len(seq_ids_in_batch)

                    # if len(self.deep_buffer) >= max(self.max_batch_size//2, get_adaptive_rebatching_threshold(self.num_ee_threshold, self.max_batch_size, rebatching_ee_factor)):
                    if len(self.deep_buffer) >= 2:
                        deep_buffer_hidden_states, deep_buffer_seq_ids, deep_buffer_positions = self.deep_buffer.take_hidden_states()

                        hidden_states = torch.cat([hidden_states, deep_buffer_hidden_states], dim=0)
                        seq_ids_in_batch.extend(deep_buffer_seq_ids)
                        positions = torch.cat([positions, deep_buffer_positions], dim=0)

                        update_seqs_in_kvcache(self.seq_metadata_map, seq_ids_in_batch, cache_engine)
            
            hidden_states = layer(
                positions,
                hidden_states,
                kv_caches[i],
            )

        if self.ln_f:
            hidden_states = self.ln_f(hidden_states)

        if has_ee:
            return hidden_states, seq_ids_in_batch, self.exited_rates, lm_logits, False, recompute_dict, conf, exited_conf_lst, None, num_seq_would_ee_but_stay, num_seq_would_not_ee_but_ee
        return hidden_states, seq_ids_in_batch, self.exited_rates, None, False, recompute_dict, None, None, None, num_seq_would_ee_but_stay, num_seq_would_not_ee_but_ee


class QWenLMHeadModel(nn.Module):

    def __init__(self, config: QWenConfig):
        super().__init__()
        self.config = config
        self.transformer = QWenModel(config)

        self.is_pipeline_first_stage = is_pipeline_first_stage()
        self.is_pipeline_last_stage = is_pipeline_last_stage()

        self.lm_head = None
        if self.is_pipeline_last_stage:
            vocab_size = ((config.vocab_size + 63) // 64) * 64

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
        priority_reqs: List[int] = [],
    ) -> torch.Tensor:
        if not self.is_pipeline_first_stage:
            # hidden_states_shape: num_tokens x hidden_size
            hidden_states = torch.empty(
                (positions.shape[0], self.config.hidden_size),
                dtype=self.config.dtype,
                device=hidden_states.device,
            )
            hidden_states = recv(hidden_states)
        
        hidden_states, output_seq_ids, exited_rates, lm_logits, is_flush, recompute_dict, conf, exited_conf_lst, latency_only_ee_iter_time, num_seq_would_ee_but_stay, num_seq_would_not_ee_but_ee = self.transformer(
            hidden_states, 
            positions, 
            kv_caches, 
            self.lm_head, 
            cache_engine=cache_engine, 
            seq_ids_in_batch=seq_ids_in_batch, 
            seq_metadata_list=seq_metadata_list, 
            rebatching_ee_factor=rebatching_ee_factor, 
            priority_reqs=priority_reqs
        )

        if not self.is_pipeline_last_stage:
            send(hidden_states)

        return hidden_states, output_seq_ids, exited_rates, lm_logits, is_flush, recompute_dict, conf, exited_conf_lst, latency_only_ee_iter_time, num_seq_would_ee_but_stay, num_seq_would_not_ee_but_ee

    def set_sampler(self, sampler: Optional[Sampler] = None):
        self.transformer.sampler = sampler

    _column_parallel_weights = []
    _row_parallel_weights = ["c_proj.weight"]

    def load_weights(
        self,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        load_format: str = "auto",
        revision: Optional[str] = None,
    ):
        tp_world_size = get_tensor_model_parallel_world_size()
        pp_world_size = get_pipeline_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        pp_rank = get_pipeline_model_parallel_rank()
        state_dict = self.state_dict()

        assert self.config.num_hidden_layers % pp_world_size == 0
        layers_per_stage = self.config.num_hidden_layers // pp_world_size

        first_layer_id = layers_per_stage * pp_rank
        last_layer_id = layers_per_stage * (pp_rank + 1) - 1

        for name, loaded_weight in hf_model_weights_iterator(
            model_name_or_path, cache_dir, load_format, revision
        ):
            if "rotary_emb.inv_freq" in name:
                continue

            if pp_rank != 0 and "wte" in name:
                continue

            if pp_rank != pp_world_size - 1 and ("lm_head" in name or "ln_f" in name):
                continue

            loaded_weight = convert_pyslice_to_tensor(loaded_weight)

            if "model.h." in name:
                layer_id = int(name.split(".")[2])
                if layer_id < first_layer_id or layer_id > last_layer_id:
                    continue

                new_layer_id = layer_id - first_layer_id
                name = name.replace(str(layer_id), str(new_layer_id))

            if "c_attn" in name:
                total_num_heads = self.config.num_attention_heads
                hidden_size = self.config.hidden_size
                head_size = hidden_size // total_num_heads
                num_heads = total_num_heads // tp_world_size
                head_start = tp_rank * num_heads
                head_end = (tp_rank + 1) * num_heads

                if "weight" in name:
                    loaded_weight = loaded_weight.view(
                        3, total_num_heads, head_size, hidden_size
                    )
                    loaded_weight = loaded_weight[:, head_start:head_end, :, :]
                    loaded_weight = loaded_weight.reshape(-1, hidden_size)
                elif "bias" in name:
                    loaded_weight = loaded_weight.view(3, total_num_heads, head_size)
                    loaded_weight = loaded_weight[:, head_start:head_end, :]
                    loaded_weight = loaded_weight.reshape(-1)

            is_gate_up_weight = False
            for stride_id, weight_name in enumerate(["w2", "w1"]):
                if weight_name not in name:
                    continue
                param = state_dict[name.replace(weight_name, "gate_up_proj")]
                shard_size = param.shape[0] // 2
                loaded_weight = loaded_weight[
                    shard_size * tp_rank : shard_size * (tp_rank + 1)
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

            if "wte" in name or "lm_head" in name:
                load_padded_tensor_parallel_vocab(param, loaded_weight, tp_rank)
                continue

            load_tensor_parallel_weights(
                param,
                loaded_weight,
                name,
                self._column_parallel_weights,
                self._row_parallel_weights,
                tp_rank,
            )
        
        # Load early exit head
        if hasattr(self.config, 'early_exit_head_path') and self.config.early_exit_head_path:
            print(f"[QWenLMHeadModel.load_weights] Loading early exit head from {self.config.early_exit_head_path}")
            early_exit_head_path = self.config.early_exit_head_path
            checkpoint = torch.load(early_exit_head_path)
            early_exit_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
            early_exit_head.load_state_dict(checkpoint['early_exit_head_state_dict'])
            self.early_exit_head = early_exit_head
            self.early_exit_head.to("cuda:0")
            self.transformer.early_exit_head = self.early_exit_head
        else:
            print(f"[QWenLMHeadModel.load_weights] No early exit head path provided. Using default head.")
