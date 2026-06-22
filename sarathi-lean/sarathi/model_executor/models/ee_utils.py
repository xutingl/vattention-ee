import os
import torch
import time
from typing import List, Tuple, Dict, Optional
from sarathi.worker.cache_engine.vATTN_cache_engine import vATTNCacheEngine
from sarathi.core.datatypes.sequence import SequenceMetadata
from sarathi.model_executor.attention import get_attention_wrapper

# Launch-time toggles (set via env var, NOT by editing source). run_ee.py sets
# these from its --collect_conf / --ee_profile CLI flags; you can also export them.
#   DREX_EE_PROFILE=1   -> record per-step buffer/copy/kvcache timing (default off)
#   DREX_COLLECT_CONF=0 -> drop per-step confidence host syncs for throughput (default on)
EE_PROFILE = os.environ.get("DREX_EE_PROFILE", "0") == "1"
COLLECT_CONF = os.environ.get("DREX_COLLECT_CONF", "1") != "0"


class HiddenStatesBuffer():
    """
    A buffer that stores hidden states
    """

    def __init__(self, batch_size: int, capacity: int, hidden_state_length: int=5120, dtype: torch.dtype=torch.float16, device: str='cuda:0'):
        self.batch_size = batch_size
        self.capacity = capacity
        # Storage dtype/width MUST match the model's activations (config.dtype /
        # config.hidden_size). A hard-coded fp16/5120 buffer silently downcasts (and
        # then dtype-mismatch-crashes at the torch.cat merge) bf16 models such as
        # Llama-3 / Qwen, and mis-sizes any model whose hidden size != 5120.
        self.dtype = dtype
        self.hidden_states = torch.zeros(self.capacity, hidden_state_length, dtype=dtype, device=device) # [capacity, hidden_state_length]
        self.positions = torch.zeros(self.capacity, dtype=torch.int64, device=device) # [capacity]
        self.available_slots = set(range(self.capacity))
        self.hidden_states_map = dict() # keys: req_ids, values: indices in hidden_states.
        self.time_spent_adding = []
        self.time_spent_taking = []
    
        
    def add_hidden_states(self, hidden_states: torch.Tensor, req_ids: List[int], positions: torch.Tensor) -> None:
        start_time = time.perf_counter()
        num_hidden_states = hidden_states.size(0)
        slots = []
        for i in range(num_hidden_states):
            slot = self.available_slots.pop()
            slots.append(slot)
            self.hidden_states_map[req_ids[i]] = slot
        self.hidden_states[slots] = hidden_states
        self.positions[slots] = positions

        if EE_PROFILE:
            self.time_spent_adding.append(time.perf_counter() - start_time)

        
            
    
    """
    Args:
        num: number of hidden states to take. If 0, take all hidden states in the buffer.
    Returns:
        output_hidden_states: Tensor. hidden states taken from the buffer. <num, hidden_state_length>
        output_req_ids: List[int]. req_ids corresponding to the hidden states. <num>
        output_positions: Tensor. positions corresponding to the hidden states. <num>
    """
    def take_hidden_states(self, num: int=-1) -> Tuple[torch.Tensor, List[int], torch.Tensor]: 
        start_time = time.perf_counter()
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

        if EE_PROFILE:
            self.time_spent_taking.append(time.perf_counter() - start_time)
        return output_hidden_states, output_req_ids, output_positions

    def __len__(self):
        return len(self.hidden_states_map)

def softmax_confidence(
        logits: torch.Tensor,
    ):
        probs = torch.softmax(logits, dim=-1)
        top_2 = torch.topk(probs, dim=-1, k=2)[0]
        return (top_2[..., 0] - top_2[..., 1]).squeeze()
    
def get_adaptive_rebatching_threshold(num_ee_threshold: int, batch_size: int, rebatching_ee_factor: float) -> float:
    if num_ee_threshold == -1:
        # Auto mode
        if rebatching_ee_factor > 0: # Valid rebatching_ee_factor
            adaptive_rebatching_threshold = batch_size * rebatching_ee_factor
        else: # Rebatching factor is not set, use default value
            adaptive_rebatching_threshold = batch_size // 2
    else:
        # Manual mode
        adaptive_rebatching_threshold = num_ee_threshold
    return adaptive_rebatching_threshold

def get_skip_mask(
    conf: torch.Tensor,
    conf_threshold: float,
    logits: torch.Tensor,
    hidden_states: torch.Tensor,
    num_ee_threshold: int,
    ee_policy: str = "eager",
    return_conf=False,
    rebatching_ee_factor: float = 0,
    seq_ids_in_batch: List[int] = [],
    priority_reqs: List[int] = [],
):
    # conf = conf[~torch.isnan(conf)]
    mask = torch.where(conf <= conf_threshold, 0.0, 1.0).bool() # <batch_size>, False: not EE, True: EE

    num_ee = torch.sum(mask).item()

    # For latency-only mode: process individual EE just like rebatching; no num_ee_threshold needed.
    if ee_policy == "latency-only":
        num_ee_threshold = 0

    need_skip = num_ee > num_ee_threshold

    
    # This is only needed for priority experiments
    # if need_skip and ee_policy == "rebatching":
    #     for seq_id in priority_reqs:
    #         for idx, seq_id_in_batch in enumerate(seq_ids_in_batch):
    #             if seq_id_in_batch == seq_id:
    #                 if not mask[idx]: # If priority request doesn't want to EE, the batch does not EE
    #                     need_skip = False 
    #                     break


    if not (ee_policy == "rebatching" or ee_policy == "latency-only"):

        exited_conf_lst = conf.tolist() if COLLECT_CONF else []

        conf_median = torch.median(conf)
        conf = torch.mean(conf).item()
        if ee_policy == "eager":
            need_skip = torch.any(mask)
        elif ee_policy == "lazy":
            need_skip = torch.all(mask)
        elif ee_policy == "average":
            val = 0.0 if conf <= conf_threshold else 1.0
            need_skip = torch.tensor(val, device=hidden_states.device).bool()
        elif ee_policy == "median":
            need_skip = conf_median >= conf_threshold
        else:
            raise ValueError("Invalid EE policy: {}".format(ee_policy))
    else:
        # rebatching / latency-only: need_skip is already decided above from
        # num_ee; conf/exited_conf are only used for logging, so when confidence
        # collection is off we skip the masked_select + host syncs entirely.
        if COLLECT_CONF:
            exited_conf = torch.masked_select(conf, mask)
            exited_conf_lst = exited_conf.tolist()
            conf = exited_conf.mean().item()
        else:
            exited_conf_lst = []
            conf = None

    batch_size = hidden_states.size(0)
    if need_skip:
        # Choose to EE: some sequences may be forced to EE.
        num_seq_would_ee_but_stay = 0
        if ee_policy == "rebatching":
            num_seq_would_not_ee_but_ee = 0 # No sequences is ever forced to EE.
        else:
            num_seq_would_not_ee_but_ee = batch_size - num_ee # Out of total <batch_size> sequences, num_ee sequences want to EE themselves. The rest are forced to EE.
    else:
        # Choose not to EE: some sequences may be forced to not EE.
        num_seq_would_not_ee_but_ee = 0

        num_seq_would_ee_but_stay = num_ee # num_ee sequences want to EE, but did not EE
    

    if not return_conf:
        return mask, need_skip
    else:
        return mask, conf, exited_conf_lst, need_skip, num_seq_would_ee_but_stay, num_seq_would_not_ee_but_ee

"""
For non-rebatching policies, requests information for the current batch is passed to cache_engine in `base_worker.py` and `model_runner.py`.
For Rebatching, becuase we are updating request in current batch, we notify the cache_engine to update the kv cache using this function (and skip the update in the above 2 files).
"""
def update_seqs_in_kvcache(
    seq_metadata_map: Dict[int, SequenceMetadata],
    seq_ids_in_batch: List[int],
    cache_engine: vATTNCacheEngine,
) -> None:  

    # assert self.ee_policy == "rebatching", "update_seqs_in_kvcache is only used in rebatching mode."
    # updated_seq_metadata_list = [self.seq_metadata_map[seq_id] for seq_id in seq_ids_in_batch] 
    updated_seq_metadata_list = []
    for seq_id in seq_ids_in_batch:
        
        updated_seq_metadata_list.append(seq_metadata_map[seq_id])

    cache_engine.step(updated_seq_metadata_list) # Originally in base_worker
    get_attention_wrapper().begin_forward(updated_seq_metadata_list) # Originally in model_runner

def fill_missing_kvcache_with_copy(exited_layer: int, cache_engine: vATTNCacheEngine, token_indices: torch.Tensor, exited_req_indices: Optional[torch.Tensor] = None, seq_ids_to_copy: Optional[List[int]] = None):

    # Copy method 1
    # seq_ids_to_copy = seq_ids_in_batch
    # for l in range(exited_layer, len(self.layers)):
    #     cache_engine.copy_k_cache_between_layers(exited_layer, l, seq_ids_to_copy, token_indices)
    #     cache_engine.copy_v_cache_between_layers(exited_layer, l, seq_ids_to_copy, token_indices)
    
    # Copy method 2
    cache_engine.copy_kv_cache_starting_at_layer(exited_layer, token_indices, exited_req_indices) # If exited_req_indices is None,

    # Copy method 3
    # cache_engine.copy_kv_cache(exited_layer, seq_ids_to_copy, token_indices)



