# coding=utf-8
"""Drop-in NestedLlama model for vattention-ee.

This is a pragmatic adapter derived from the current llama.py, intended to load
Balcony / NestedLlama-style checkpoints with fixed exit ramps (e.g. 15, 18, 21)
and reuse vattention-ee's existing exit-policy logic.

Important notes:
- This is a starting point for integration and has not been executed in this environment.
- For the fastest first run, set:
    config.exit_layer_indices = [15, 18, 21]
    config.output_exit_layers = [15, 18, 21]
    config.tie_exit_lm_head = True
    config.exit_decoder_layer = False
- Once that works, enable config.exit_decoder_layer = True and verify KV-cache behavior
  for the extra exit decoder layer.
"""

from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import time

import torch
from torch import nn
from transformers import LlamaConfig

from sarathi.metrics.constants import OperationMetrics
from sarathi.metrics.cuda_timer import CudaTimer
from sarathi.model_executor.attention import get_attention_wrapper
from sarathi.model_executor.layers.activation import SiluAndMul
from sarathi.model_executor.layers.layernorm import RMSNorm
from sarathi.model_executor.layers.rotary_embedding import get_rope
from sarathi.model_executor.layers.sampler import Sampler
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
from sarathi.core.datatypes.sequence import SequenceMetadata


class NestedLlamaMLP(nn.Module):
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
            raise ValueError(f"Unsupported activation: {hidden_act}. Only silu is supported for now.")
        self.act_fn = SiluAndMul()
        self._mlp_activation_timer = CudaTimer(OperationMetrics.MLP_ACTIVATION, layer_id=layer_id)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        with self._mlp_activation_timer:
            x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class NestedLlamaAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 5120,
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

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
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


class NestedLlamaDecoderLayer(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        layer_id: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 5120)
        self.self_attn = NestedLlamaAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            layer_id=layer_id,
        )
        self.mlp = NestedLlamaMLP(
            hidden_size=config.hidden_size,
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
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states, kv_cache=kv_cache)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class BalconyExitModule(nn.Module):
    def __init__(self, config: LlamaConfig, layer_id: Optional[int] = None) -> None:
        super().__init__()
        self.use_decoder_layer = getattr(config, "exit_decoder_layer", True)
        self.tie_exit_lm_head = getattr(config, "tie_exit_lm_head", True)

        self.decoder = (
            NestedLlamaDecoderLayer(config, layer_id=layer_id)
            if self.use_decoder_layer else None
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.lm_head = None
        if not self.tie_exit_lm_head:
            vocab_size = ((config.vocab_size + 63) // 64) * 64
            self.lm_head = ColumnParallelLinear(
                config.hidden_size,
                vocab_size,
                bias=False,
                gather_output=False,
                perform_initialization=False,
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: Optional[KVCache],
        shared_lm_head,
    ) -> torch.Tensor:
        x = hidden_states
        if self.decoder is not None:
            if kv_cache is None:
                raise ValueError("Exit decoder layer requires a kv_cache.")
            x = self.decoder(positions, x, kv_cache)
        x = self.norm(x)

        lm_head = shared_lm_head if self.tie_exit_lm_head else self.lm_head
        if lm_head is None:
            raise ValueError("No LM head available for exit module.")

        logits = lm_head(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits


class HiddenStatesBuffer:
    def __init__(self, batch_size: int, capacity: int, hidden_state_length: int = 5120):
        self.batch_size = batch_size
        self.capacity = capacity
        self.hidden_states = torch.zeros(self.capacity, hidden_state_length, device="cuda:0", dtype=torch.float16)
        self.positions = torch.zeros(self.capacity, device="cuda:0", dtype=torch.int64)
        self.available_slots = set(range(self.capacity))
        self.hidden_states_map: Dict[int, int] = {}
        self.time_spent_adding: List[float] = []
        self.time_spent_taking: List[float] = []

    def add_hidden_states(self, hidden_states: torch.Tensor, req_ids: List[int], positions: torch.Tensor) -> None:
        start_time = time.perf_counter()
        slots: List[int] = []
        for req_id in req_ids:
            slot = self.available_slots.pop()
            slots.append(slot)
            self.hidden_states_map[req_id] = slot
        self.hidden_states[slots] = hidden_states
        self.positions[slots] = positions
        self.time_spent_adding.append(time.perf_counter() - start_time)

    def take_hidden_states(self, num: int = -1) -> Tuple[torch.Tensor, List[int], torch.Tensor]:
        start_time = time.perf_counter()
        if num == -1:
            num = self.batch_size

        output_req_ids: List[int] = []
        slots: List[int] = []
        for req_id, slot in list(self.hidden_states_map.items())[:num]:
            slots.append(slot)
            self.available_slots.add(slot)
            self.hidden_states_map.pop(req_id)
            output_req_ids.append(req_id)

        output_hidden_states = self.hidden_states[slots]
        output_positions = self.positions[slots]
        self.time_spent_taking.append(time.perf_counter() - start_time)
        return output_hidden_states, output_req_ids, output_positions

    def __len__(self) -> int:
        return len(self.hidden_states_map)


class NestedLlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        if not hasattr(config, "exit_layer_indices") or config.exit_layer_indices is None:
            shallow = getattr(config, "shallow_exit_layer", None)
            config.exit_layer_indices = [shallow] if shallow is not None else []
        if not hasattr(config, "output_exit_layers") or config.output_exit_layers is None:
            config.output_exit_layers = list(config.exit_layer_indices)
        if isinstance(config.output_exit_layers, int):
            config.output_exit_layers = [config.output_exit_layers]
        if not hasattr(config, "tie_exit_lm_head"):
            config.tie_exit_lm_head = True
        if not hasattr(config, "exit_decoder_layer"):
            config.exit_decoder_layer = True
        if not hasattr(config, "output_full_model"):
            config.output_full_model = True

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

        num_layers = config.num_hidden_layers // get_pipeline_model_parallel_world_size()
        layer_offset = get_pipeline_model_parallel_rank() * num_layers
        self.layers = nn.ModuleList(
            [
                NestedLlamaDecoderLayer(config, layer_id=layer_id + layer_offset)
                for layer_id in range(num_layers)
            ]
        )

        self.exit_layer_indices: List[int] = list(config.exit_layer_indices)
        self.output_exit_layers: List[int] = list(config.output_exit_layers)
        self.exit_layer_set = set(self.exit_layer_indices)
        self.exit_layer_to_idx = {layer: idx for idx, layer in enumerate(self.exit_layer_indices)}

        self.shallow_exit_layer = getattr(config, "shallow_exit_layer", None)
        if self.shallow_exit_layer is not None and self.exit_layer_set and self.shallow_exit_layer not in self.exit_layer_set:
            raise ValueError(
                f"shallow_exit_layer={self.shallow_exit_layer} is not one of the checkpoint's "
                f"exit layers {sorted(self.exit_layer_set)}"
            )

        self.exit_modules = nn.ModuleDict()
        for exit_layer in self.exit_layer_indices:
            self.exit_modules[str(exit_layer)] = BalconyExitModule(
                config,
                layer_id=(num_layers + layer_offset + self.exit_layer_to_idx[exit_layer]),
            )

        self.norm = None
        if is_pipeline_last_stage():
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.ee_policy = config.ee_policy
        self.conf_threshold = config.conf_threshold
        self.exited_rates = [0, 1]

        self.max_batch_size = config.max_num_seqs
        self.start_buffer: List = []
        self.deep_buffer = HiddenStatesBuffer(self.max_batch_size, self.max_batch_size * 2 + 1, hidden_state_length=config.hidden_size)
        self.seq_metadata_map: Dict[int, SequenceMetadata] = {}

        self.batch_size_lst = [0]
        self.avg_exited_conf = 0.0
        self.conf_sum = 0.0
        self.exited_cnt = 0

        self.update_kvcache_time_lst: List[float] = []
        self.ee_overhead_time_lst: List[float] = []
        self.fill_kvcache_time_lst: List[float] = []

        self.prefill_batch_size_limit = 64
        self.sampler: Optional[Sampler] = None
        self.rebatching_time = 0
        self.num_ee_threshold = getattr(config, "num_ee_threshold", -1)

        self.kv_method = config.kv_method
        self.recompute_seq_id_to_hidden_states = defaultdict(list)
        self.recompute_seq_id_to_positions = defaultdict(list)
        self.recompute_seq_id_to_input_hidden_states: Dict[int, torch.Tensor] = {}

    def softmax_confidence(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        top_2 = torch.topk(probs, dim=-1, k=2)[0]
        return (top_2[..., 0] - top_2[..., 1]).squeeze()

    def get_adaptive_rebatching_threshold(self, batch_size: int, rebatching_ee_factor: float) -> float:
        if self.num_ee_threshold == -1:
            if rebatching_ee_factor > 0:
                return batch_size * rebatching_ee_factor
            return batch_size // 2
        return self.num_ee_threshold

    def get_skip_mask(
        self,
        logits: torch.Tensor,
        hidden_states: torch.Tensor,
        ee_policy: str = "eager",
        return_conf: bool = False,
        rebatching_ee_factor: float = 0,
        seq_ids_in_batch: List[int] = [],
        priority_reqs: List[int] = [],
    ):
        conf = self.softmax_confidence(logits)
        mask = torch.where(conf <= self.conf_threshold, 0.0, 1.0).bool()

        num_ee = torch.sum(mask).item()
        num_ee_threshold = self.get_adaptive_rebatching_threshold(self.max_batch_size, rebatching_ee_factor)

        if ee_policy == "latency-only":
            num_ee_threshold = 0

        need_skip = num_ee > num_ee_threshold

        if not (ee_policy == "rebatching" or ee_policy == "latency-only"):
            exited_conf_lst = conf.tolist()
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
                raise ValueError(f"Invalid EE policy: {ee_policy}")
        else:
            exited_conf = torch.masked_select(conf, mask)
            exited_conf_lst = exited_conf.tolist()
            conf = exited_conf.mean().item() if exited_conf.numel() > 0 else 0.0

        batch_size = hidden_states.size(0)
        if need_skip:
            num_seq_would_ee_but_stay = 0
            if ee_policy == "rebatching":
                num_seq_would_not_ee_but_ee = 0
            else:
                num_seq_would_not_ee_but_ee = batch_size - num_ee
        else:
            num_seq_would_not_ee_but_ee = 0
            num_seq_would_ee_but_stay = num_ee

        if not return_conf:
            return mask, need_skip
        return (
            mask,
            conf,
            exited_conf_lst,
            need_skip,
            num_seq_would_ee_but_stay,
            num_seq_would_not_ee_but_ee,
        )

    def get_exit_logits(
        self,
        layer_idx: int,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_caches: List[KVCache],
        lm_head,
    ) -> torch.Tensor:
        exit_module = self.exit_modules[str(layer_idx)]
        exit_kv_cache = None
        if getattr(self.config, "exit_decoder_layer", True):
            exit_module_slot = len(self.layers) + self.exit_layer_to_idx[layer_idx]
            exit_kv_cache = kv_caches[exit_module_slot]
        return exit_module(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=exit_kv_cache,
            shared_lm_head=lm_head,
        )

    def measure_batch_size(self, hidden_states: torch.Tensor, seq_ids_in_batch: List[int] = []):
        if len(seq_ids_in_batch) > 0:
            self.batch_size_lst.append(len(seq_ids_in_batch))
        else:
            batch_size = hidden_states.size(0)
            if batch_size <= self.max_batch_size:
                self.batch_size_lst.append(batch_size)

    def check_req_for_kv_recompute(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        seq_ids_in_batch: List[int],
    ) -> Tuple[dict, torch.Tensor, torch.Tensor, List[int]]:
        if not seq_ids_in_batch:
            return {}, hidden_states, positions, seq_ids_in_batch

        recompute_dict = {}
        recompute_req_to_idx = {}
        non_recompute_req_to_idx = {}

        for idx, ee_req_id in enumerate(seq_ids_in_batch):
            seq_metadata = self.seq_metadata_map[ee_req_id]
            if seq_metadata.seq.recompute_length > 0:
                recompute_req_to_idx[ee_req_id] = idx
                recompute_length = seq_metadata.seq.recompute_length + 1
                recompute_dict[ee_req_id] = recompute_length

                seq_metadata.seq.prompt_token_ids = seq_metadata.seq.prompt_token_ids + seq_metadata.seq.output_token_ids
                seq_metadata.seq.prompt_tokens_processed = len(seq_metadata.seq.prompt_token_ids) - recompute_length
                seq_metadata.seq.recompute_length = 0
                seq_metadata.seq.prompt_processing_finished = False
                seq_metadata.prompt_chunk_len = recompute_length
                seq_metadata.seq.state._prompt_processing_completed_at = None
            else:
                non_recompute_req_to_idx[ee_req_id] = idx

        if not recompute_dict:
            return {}, hidden_states, positions, seq_ids_in_batch

        all_recompute_hidden_states_lst = []
        all_recompute_positions_lst = []

        for seq_id, _recompute_length in recompute_dict.items():
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

    def update_seqs_in_kvcache(self, seq_ids_in_batch: List[int], cache_engine: vATTNCacheEngine) -> None:
        start_time = time.perf_counter()
        updated_seq_metadata_list = [self.seq_metadata_map[seq_id] for seq_id in seq_ids_in_batch]
        cache_engine.step(updated_seq_metadata_list)
        get_attention_wrapper().begin_forward(updated_seq_metadata_list)
        self.update_kvcache_time_lst.append(time.perf_counter() - start_time)

    def fill_missing_kvcache_with_copy(
        self,
        exited_layer: int,
        cache_engine: vATTNCacheEngine,
        token_indices: torch.Tensor,
        exited_req_indices: Optional[torch.Tensor] = None,
        seq_ids_to_copy: Optional[List[int]] = None,
    ):
        start_time = time.perf_counter()
        cache_engine.copy_kv_cache_starting_at_layer(exited_layer, token_indices, exited_req_indices)
        self.fill_kvcache_time_lst.append(time.perf_counter() - start_time)

    def forward_without_rebatching(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        lm_head,
        cache_engine: Optional[vATTNCacheEngine] = None,
        seq_ids_in_batch: Optional[List[int]] = None,
        seq_metadata_list: Optional[List[SequenceMetadata]] = None,
    ):
        in_model_iter_start_time = time.perf_counter()
        self.measure_batch_size(hidden_states)

        if seq_metadata_list:
            for seq_metadata in seq_metadata_list:
                self.seq_metadata_map[seq_metadata.seq.seq_id] = seq_metadata

        if self.embed_tokens:
            hidden_states = self.embed_tokens(hidden_states)

        input_hidden_states = hidden_states

        if self.kv_method == "postfill" and seq_ids_in_batch is not None and self.ee_policy != "rebatching":
            recompute_dict, hidden_states, positions, seq_ids_in_batch = self.check_req_for_kv_recompute(hidden_states, positions, seq_ids_in_batch)
            if len(recompute_dict) > 0:
                self.update_seqs_in_kvcache(seq_ids_in_batch, cache_engine)
        else:
            recompute_dict = {}

        if seq_ids_in_batch is not None and self.ee_policy == "rebatching":
            self.update_seqs_in_kvcache(seq_ids_in_batch, cache_engine)

        check_for_ee = (
            cache_engine is not None
            and self.ee_policy != "off"
            and len(self.exit_layer_indices) > 0
            and hidden_states.size(0) <= self.max_batch_size
            and len(recompute_dict) == 0
        )

        conf = None
        exited_conf_lst = None
        has_ee = False
        latency_only_ee_iter_time = None
        num_seq_would_ee_but_stay, num_seq_would_not_ee_but_ee = 0, 0

        # During prefill (batch too large for EE check), we still need to warm up
        # the exit-module decoder's KV cache so it has valid K/V entries for all
        # prompt positions.  Collect hidden states at each exit point here.
        is_prefill_warmup = (
            not check_for_ee
            and self.ee_policy != "off"
            and getattr(self.config, "exit_decoder_layer", True)
            and len(self.exit_layer_indices) > 0
            and cache_engine is not None
            and len(recompute_dict) == 0
            and hidden_states.size(0) > self.max_batch_size
        )
        prefill_exit_hidden_states: Dict[int, torch.Tensor] = {}

        for i in range(len(self.layers)):
            # Exit check fires BEFORE applying layer i — matching native Balcony where
            # all_hidden_states[i] is saved before the i-th decoder layer runs.
            if (
                check_for_ee
                and i in self.exit_layer_set
                and (self.shallow_exit_layer is None or i == self.shallow_exit_layer)
            ):
                lm_logits = self.get_exit_logits(
                    layer_idx=i,
                    positions=positions,
                    hidden_states=hidden_states,
                    kv_caches=kv_caches,
                    lm_head=lm_head,
                )

                skip_mask, conf, exited_conf_lst, need_skip, num_seq_would_ee_but_stay, num_seq_would_not_ee_but_ee = self.get_skip_mask(
                    logits=lm_logits,
                    hidden_states=hidden_states,
                    ee_policy=self.ee_policy,
                    return_conf=True,
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
                            self.fill_missing_kvcache_with_copy(i - 1, cache_engine, positions, exited_req_indices=None, seq_ids_to_copy=seq_ids_in_batch)
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

            # Save pre-exit hidden states during prefill for KV warmup below.
            if is_prefill_warmup and i in self.exit_layer_set:
                prefill_exit_hidden_states[i] = hidden_states

            hidden_states = self.layers[i](positions, hidden_states, kv_caches[i])

        # Warm up exit-module decoder KV caches for all prompt tokens so that
        # subsequent decode steps find valid (not zero) K/V entries in those slots.
        if prefill_exit_hidden_states:
            for exit_layer_idx, hs_at_exit in prefill_exit_hidden_states.items():
                exit_module = self.exit_modules[str(exit_layer_idx)]
                if exit_module.decoder is not None:
                    exit_module_slot = len(self.layers) + self.exit_layer_to_idx[exit_layer_idx]
                    exit_kv_cache = kv_caches[exit_module_slot]
                    exit_module.decoder(positions, hs_at_exit, exit_kv_cache)

        if latency_only_ee_iter_time is not None:
            hidden_states[latency_only_exited_req_indices] = latency_only_exited_hidden_states

        if self.norm:
            hidden_states = self.norm(hidden_states)

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
    ):
        if self.ee_policy != "rebatching" or hidden_states.size(0) > self.prefill_batch_size_limit:
            return self.forward_without_rebatching(hidden_states, positions, kv_caches, lm_head, cache_engine, seq_ids_in_batch, seq_metadata_list=seq_metadata_list)

        for seq_metadata in seq_metadata_list:
            self.seq_metadata_map[seq_metadata.seq.seq_id] = seq_metadata

        if self.embed_tokens:
            hidden_states = self.embed_tokens(hidden_states)

        incoming_batch_size = len(seq_ids_in_batch)

        if self.kv_method == "postfill":
            for i, seq_id in enumerate(seq_ids_in_batch):
                self.recompute_seq_id_to_input_hidden_states[seq_id] = hidden_states[i]

        flush_buffer = incoming_batch_size == 0
        if flush_buffer:
            if len(self.deep_buffer) > 0:
                hidden_states, seq_ids_in_batch, positions = self.deep_buffer.take_hidden_states(len(self.deep_buffer))
                self.update_seqs_in_kvcache(seq_ids_in_batch, cache_engine)

                start_layer = 0 if len(self.exit_layer_indices) == 0 else (min(self.exit_layer_indices) + 1)
                for i in range(start_layer, len(self.layers)):
                    hidden_states = self.layers[i](positions, hidden_states, kv_caches[i])

                if self.norm:
                    hidden_states = self.norm(hidden_states)
                return hidden_states, seq_ids_in_batch, self.exited_rates, None, True, {}, None, None, None, 0, 0
            return None, [], self.exited_rates, None, False, {}, None, None, None, 0, 0

        if self.kv_method == "postfill":
            recompute_dict, hidden_states, positions, seq_ids_in_batch = self.check_req_for_kv_recompute(hidden_states, positions, seq_ids_in_batch)
        else:
            recompute_dict = {}

        self.update_seqs_in_kvcache(seq_ids_in_batch, cache_engine)

        conf = None
        has_ee = False
        exited_conf_lst = None
        num_seq_would_ee_but_stay, num_seq_would_not_ee_but_ee = 0, 0

        for i in range(len(self.layers)):
            # Exit check fires BEFORE applying layer i — matching native Balcony indexing.
            if (
                cache_engine
                and i in self.exit_layer_set
                and (self.shallow_exit_layer is None or i == self.shallow_exit_layer)
                and hidden_states.size(0) <= self.max_batch_size
                and not recompute_dict
            ):
                lm_logits = self.get_exit_logits(
                    layer_idx=i,
                    positions=positions,
                    hidden_states=hidden_states,
                    kv_caches=kv_caches,
                    lm_head=lm_head,
                )

                ee_check_start_time = time.perf_counter()
                skip_mask, conf, exited_conf_lst, need_skip, num_seq_would_ee_but_stay, num_seq_would_not_ee_but_ee = self.get_skip_mask(
                    logits=lm_logits,
                    hidden_states=hidden_states,
                    ee_policy=self.ee_policy,
                    return_conf=True,
                    rebatching_ee_factor=rebatching_ee_factor,
                    seq_ids_in_batch=seq_ids_in_batch,
                    priority_reqs=priority_reqs,
                )
                self.ee_overhead_time_lst.append(time.perf_counter() - ee_check_start_time)

                if need_skip:
                    rebatching_start_time = time.perf_counter()
                    has_ee = True

                    if torch.all(skip_mask):
                        self.exited_rates[0] += len(seq_ids_in_batch)
                        if self.kv_method == "copy":
                            self.fill_missing_kvcache_with_copy(i - 1, cache_engine, positions, exited_req_indices=None, seq_ids_to_copy=seq_ids_in_batch)
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
                            seq_ids_to_copy = [seq_ids_in_batch[idx] for idx in exited_req_indices.tolist()]
                            self.fill_missing_kvcache_with_copy(i - 1, cache_engine, token_indices, exited_req_indices=exited_req_indices, seq_ids_to_copy=seq_ids_to_copy)

                        self.exited_rates[0] += len(exited_req_indices)
                        self.exited_rates[1] += len(seq_ids_in_batch) - len(exited_req_indices)

                        lm_logits = lm_logits[exited_req_indices]

                        req_idx_in_batch_to_buffer = []
                        seq_ids_in_batch_to_buffer = []
                        for req_idx, skip in enumerate(skip_mask):
                            if not skip:
                                req_idx_in_batch_to_buffer.append(req_idx)
                                seq_ids_in_batch_to_buffer.append(seq_ids_in_batch[req_idx])

                        self.deep_buffer.add_hidden_states(
                            hidden_states[req_idx_in_batch_to_buffer],
                            seq_ids_in_batch_to_buffer,
                            positions[req_idx_in_batch_to_buffer],
                        )

                        hidden_states = hidden_states[skip_mask]
                        seq_ids_in_batch = [seq_ids_in_batch[idx] for idx in range(len(seq_ids_in_batch)) if skip_mask[idx]]
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
                    if len(self.deep_buffer) >= 2:
                        deep_buffer_hidden_states, deep_buffer_seq_ids, deep_buffer_positions = self.deep_buffer.take_hidden_states()
                        hidden_states = torch.cat([hidden_states, deep_buffer_hidden_states], dim=0)
                        seq_ids_in_batch.extend(deep_buffer_seq_ids)
                        positions = torch.cat([positions, deep_buffer_positions], dim=0)
                        self.update_seqs_in_kvcache(seq_ids_in_batch, cache_engine)

            hidden_states = self.layers[i](positions, hidden_states, kv_caches[i])

        if self.norm:
            hidden_states = self.norm(hidden_states)

        if has_ee:
            return hidden_states, seq_ids_in_batch, self.exited_rates, lm_logits, False, recompute_dict, conf, exited_conf_lst, None, num_seq_would_ee_but_stay, num_seq_would_not_ee_but_ee
        return hidden_states, seq_ids_in_batch, self.exited_rates, None, False, recompute_dict, None, None, None, num_seq_would_ee_but_stay, num_seq_would_not_ee_but_ee


class NestedLlamaForCausalLM(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.config = config
        self.model = NestedLlamaModel(config)
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
        rebatching_ee_factor: float = 0,
        priority_reqs: List[int] = [],
    ):
        if not self.is_pipeline_first_stage:
            hidden_states = torch.empty(
                (positions.shape[0], self.config.hidden_size),
                dtype=self.config.dtype,
                device=hidden_states.device,
            )
            hidden_states = recv(hidden_states)

        outputs = self.model(
            hidden_states,
            positions,
            kv_caches,
            self.lm_head,
            cache_engine=cache_engine,
            seq_ids_in_batch=seq_ids_in_batch,
            seq_metadata_list=seq_metadata_list,
            rebatching_ee_factor=rebatching_ee_factor,
            priority_reqs=priority_reqs,
        )

        hidden_states = outputs[0]
        if not self.is_pipeline_last_stage and hidden_states is not None:
            send(hidden_states)

        return outputs

    _column_parallel_layers = ["qkv_proj", "gate_up_proj", "lm_head"]
    _row_parallel_layers = ["o_proj", "down_proj"]

    def load_weights(
        self,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        load_format: str = "auto",
        revision: Optional[str] = None,
    ):
        weight_suffixes = ["weight"]
        
        def _safe_shape(x):
            try:
                return tuple(x.shape)
            except Exception:
                try:
                    return tuple(x.size())
                except Exception:
                    return f"<no-shape:{type(x).__name__}>"

        loaded_names = []
        skipped_names = []
        missing_names = []
        remapped_names = []

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
            ("q_proj", q_proj_shard_size, 0),
            ("k_proj", kv_proj_shard_size, q_proj_shard_size),
            ("v_proj", kv_proj_shard_size, q_proj_shard_size + kv_proj_shard_size),
        ]

        state_dict = self.state_dict()
        exit_layer_indices = getattr(self.config, "exit_layer_indices", []) or []

        for name, loaded_weight in hf_model_weights_iterator(
            model_name_or_path, cache_dir, load_format, revision
        ):
            original_name = name

            # Skip RoPE buffers
            if "rotary_emb.inv_freq" in name:
                skipped_names.append((original_name, "rotary_emb.inv_freq buffer"))
                continue

            # Pipeline-stage filtering
            if pp_model_parallel_rank != 0 and "embed_tokens" in name:
                skipped_names.append((original_name, "embed_tokens not on this PP stage"))
                continue

            if pp_model_parallel_rank != pp_size - 1 and (
                "lm_head" in name or name == "model.norm.weight" or "model.exit_modules" in name
            ):
                skipped_names.append((original_name, "final-stage-only tensor not on this PP stage"))
                continue

            # Remap base model layer ids from global HF ids to local PP-stage ids
            if "model.layers" in name:
                layer_id = int(name.split(".")[2])
                if layer_id < first_layer_id or layer_id > last_layer_id:
                    skipped_names.append((original_name, f"layer {layer_id} not on this PP stage"))
                    continue
                new_layer_id = layer_id - first_layer_id
                name = name.replace(f".{layer_id}.", f".{new_layer_id}.")
                if name != original_name:
                    remapped_names.append((original_name, name))

            # Remap Balcony / NestedLlama exit modules
            if "model.exit_modules" in name:
                exit_original_name = name
                parts = name.split(".")
                hf_exit_idx = int(parts[2])
                subpath = ".".join(parts[3:])

                if hf_exit_idx >= len(exit_layer_indices):
                    skipped_names.append((original_name, f"exit module index {hf_exit_idx} >= configured exits"))
                    continue

                exit_layer = exit_layer_indices[hf_exit_idx]
                local_prefix = f"model.exit_modules.{exit_layer}"

                if subpath.startswith("0.") and getattr(self.config, "exit_decoder_layer", True):
                    name = f"{local_prefix}.decoder.{subpath[2:]}"
                elif subpath == "1.weight":
                    name = f"{local_prefix}.norm.weight"
                elif subpath == "2.weight":
                    name = f"{local_prefix}.lm_head.weight"
                else:
                    skipped_names.append((original_name, f"unsupported exit subpath: {subpath}"))
                    continue

                if name != exit_original_name:
                    remapped_names.append((exit_original_name, name))

            # Packed q/k/v -> qkv_proj
            is_attention_weight = False
            for weight_name, shard_size, offset in attention_weight_specs:
                if weight_name not in name:
                    continue

                packed_name = name.replace(weight_name, "qkv_proj")
                if packed_name not in state_dict:
                    missing_names.append((original_name, packed_name, "packed attention target missing"))
                    continue

                param = state_dict[packed_name]
                shard = loaded_weight[
                    shard_size * tensor_model_parallel_rank : shard_size * (tensor_model_parallel_rank + 1)
                ]
                param_slice = param.data[offset : offset + shard_size]

                if param_slice.shape != shard.shape:
                    raise ValueError(
                        f"Shape mismatch for {original_name} -> {packed_name}: "
                        f"{param_slice.shape} vs {shard.shape}"
                    )

                param_slice.copy_(shard)
                loaded_names.append((original_name, packed_name, _safe_shape(shard), "packed_attention"))
                is_attention_weight = True
                break

            if is_attention_weight:
                continue

            # Packed gate/up -> gate_up_proj
            is_gate_up_weight = False
            for stride_id, weight_name in enumerate(["gate_proj", "up_proj"]):
                if weight_name not in name:
                    continue

                packed_name = name.replace(weight_name, "gate_up_proj")
                if packed_name not in state_dict:
                    missing_names.append((original_name, packed_name, "packed mlp target missing"))
                    continue

                param = state_dict[packed_name]
                shard_size = param.shape[0] // 2
                shard = loaded_weight[
                    shard_size * tensor_model_parallel_rank : shard_size * (tensor_model_parallel_rank + 1)
                ]
                param_slice = param.data[shard_size * stride_id : shard_size * (stride_id + 1)]

                if param_slice.shape != shard.shape:
                    raise ValueError(
                        f"Shape mismatch for {original_name} -> {packed_name}: "
                        f"{param_slice.shape} vs {shard.shape}"
                    )

                param_slice.copy_(shard)
                loaded_names.append((original_name, packed_name, _safe_shape(shard), f"packed_{weight_name}"))
                is_gate_up_weight = True
                break

            if is_gate_up_weight:
                continue

            # Direct lookup
            if name not in state_dict:
                missing_names.append((original_name, name, "target tensor missing in local state_dict"))
                continue

            param = state_dict[name]

            # Vocab-parallel embedding
            if "embed_tokens" in name:
                load_padded_tensor_parallel_vocab(param, loaded_weight, tensor_model_parallel_rank)
                loaded_names.append((original_name, name, _safe_shape(loaded_weight), "embed_tokens"))
                continue

            # Vocab-parallel LM heads (shared or untied exit lm_head)
            if "lm_head" in name:
                load_padded_tensor_parallel_vocab(param, loaded_weight, tensor_model_parallel_rank)
                loaded_names.append((original_name, name, _safe_shape(loaded_weight), "lm_head"))
                continue

            # Generic TP loader
            load_tensor_parallel_weights(
                param,
                loaded_weight,
                name,
                column_parallel_weights,
                row_parallel_weights,
                tensor_model_parallel_rank,
            )
            loaded_names.append((original_name, name, _safe_shape(loaded_weight), "generic"))

        # Summaries
        print(
            f"[load_weights] loaded={len(loaded_names)} "
            f"skipped={len(skipped_names)} "
            f"missing={len(missing_names)} "
            f"remapped={len(remapped_names)}"
        )

        print("[load_weights] first 30 loaded:")
        for item in loaded_names[:30]:
            print("  ", item)

        print("[load_weights] first 30 missing:")
        for item in missing_names[:30]:
            print("  ", item)

        print("[load_weights] first 30 remaps:")
        for item in remapped_names[:30]:
            print("  ", item)

        print("[load_weights] first 30 skipped:")
        for item in skipped_names[:30]:
            print("  ", item)

        # Focused exit-module coverage report
        for exit_layer in exit_layer_indices:
            matched = [x for x in loaded_names if f"model.exit_modules.{exit_layer}" in x[1]]
            print(f"[load_weights] exit layer {exit_layer}: loaded {len(matched)} tensors")

        # Optional sanity checks for common important tensors
        sanity_keys = [
            "model.norm.weight",
            "lm_head.weight",
        ]
        for key in sanity_keys:
            if key in state_dict:
                t = state_dict[key].float()
                print(
                    f"[load_weights] sanity {key}: "
                    f"mean={t.mean().item():.6f} std={t.std().item():.6f} absmax={t.abs().max().item():.6f}"
                )

        for exit_layer in exit_layer_indices:
            key = f"model.exit_modules.{exit_layer}.norm.weight"
            if key in state_dict:
                t = state_dict[key].float()
                print(
                    f"[load_weights] sanity {key}: "
                    f"mean={t.mean().item():.6f} std={t.std().item():.6f} absmax={t.abs().max().item():.6f}"
                )

    def set_sampler(self, sampler: Optional[Sampler] = None):
        self.model.sampler = sampler