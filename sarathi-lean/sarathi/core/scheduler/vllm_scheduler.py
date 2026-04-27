import time
from typing import List

from sarathi.config import CacheConfig, VLLMSchedulerConfig
from sarathi.core.block_space_manager.vllm_block_space_manager import (
    VLLMBlockSpaceManager,
)
from sarathi.core.datatypes.scheduler_output import SchedulerOutputs
from sarathi.core.datatypes.sequence import Sequence, SequenceScheduleMetadata, SequenceStatus, SequenceMetadata
from sarathi.core.scheduler.base_scheduler import BaseScheduler
from sarathi.logger import init_logger
from sarathi.model_executor.attention import is_vattention_backend
from sarathi.core.block_space_manager.vattention_block_space_manager import (
    vAttentionBlockSpaceManager
)

logger = init_logger(__name__)


class VLLMScheduler(BaseScheduler):

    def __init__(
        self,
        scheduler_config: VLLMSchedulerConfig,
        cache_config: CacheConfig,
    ) -> None:
        super().__init__(scheduler_config, cache_config)

        self.prompt_limit = min(
            self.scheduler_config.max_model_len,
            self.scheduler_config.max_num_batched_tokens,
        )

        self.buffer_age = 0
        self.buffer_age_threshold = 100
        # self.buffer_age_factor = scheduler_config.buffer_age_factor

        # self.request_age_threshold = 70

        self.buffer_age_factor = 0

        # Minimum buffer occupancy before triggering a dedicated flush iteration.
        # Prevents wasting a full iteration to process just 1-2 buffered sequences.
        self.min_flush_size = max(self.scheduler_config.max_num_seqs // 2, 1)
        self.min_flush_size = 8

        self.request_age_threshold = scheduler_config.buffer_age_factor

        self.router_meta_map = {}  # seq_id -> {"conf": float, "pred_cost": float, "is_shallow": bool}

        # cost_flush_threshold: total predicted compute that must accumulate in the
        # rebatching buffer before a dedicated flush is triggered.
        #
        # At pred_cost = 1.0 (no router signal / worst case), this equals min_flush_size,
        # so behaviour is identical to the old count-based scheduler.
        #
        # At typical confidence values (conf ≈ 0.45, pred_cost ≈ 0.55), a full buffer
        # of 8 requests sums to ~4.4, well below min_flush_size=8 — so cost flush would
        # never trigger.  We therefore set the threshold to half of min_flush_size, which
        # corresponds to flushing when the buffer holds the compute equivalent of
        # min_flush_size/2 worst-case (pred_cost=1.0) requests, or ~min_flush_size
        # typical requests.  Tune this for your workload.
        self.cost_flush_threshold = self.min_flush_size * 0.5

        # Ablation knob: set to False to revert to count-only flush logic (for comparison).
        self.use_router_aware = True

    def get_block_space_manager_class(self):
        return vAttentionBlockSpaceManager if is_vattention_backend() else VLLMBlockSpaceManager 

    def _schedule(self) -> SchedulerOutputs:
        # Fix the current time.
        now = time.monotonic()

        priority_reqs = []

        # Disable request age-based priority
        # for seq_id in self.request_age_map:
        #     if self.request_age_map[seq_id] > self.request_age_threshold:
        #         priority_reqs.append(seq_id)

        ignored_seq_ids: List[int] = []
        preempted_seq_ids: List[int] = []
        scheduled_seq_metadata_list: List[SequenceScheduleMetadata] = []

        if type(self.block_manager) == vAttentionBlockSpaceManager:
            self.block_manager.clear_promised_blocks()
            
        # The total number of sequences on the fly, including the
        # requests in the generation phase.
        num_batched_tokens = 0
        # Optimization: We do not sort the waiting queue since the preempted
        # sequence groups are added to the front and the new sequence groups
        # are added to the back.

        # print(f"[VLLMScheduler._schedule] Start of iteration: {self._iteration_id}:")
        # print(f"[VLLMScheduler._schedule] running list: {self.running}")
        
        if len(self.rebatching_buffer) > 0:
            self.buffer_age += 1

        age_adjusted_buffer_size = len(self.rebatching_buffer) * (1 + self.buffer_age * self.buffer_age_factor)

        flush_due_to_full = age_adjusted_buffer_size >= self.scheduler_config.max_num_seqs
        flush_due_to_age  = self.buffer_age >= self.buffer_age_threshold

        if self.use_router_aware:
            # Router-aware flush: accumulate predicted compute cost instead of raw
            # count, so low-confidence (expensive) requests trigger a flush sooner.
            # When router metadata is absent, pred_cost=1.0 per request, so this
            # degrades gracefully to the original count-based behaviour.
            buffer_pred_cost   = self._buffer_pred_cost_sum()
            cost_adjusted_buffer = buffer_pred_cost * (1 + self.buffer_age * self.buffer_age_factor)
            flush_due_to_cost  = cost_adjusted_buffer >= self.cost_flush_threshold
        else:
            # Baseline: original count-only flush (used for ablation / comparison).
            buffer_pred_cost     = float(len(self.rebatching_buffer))
            cost_adjusted_buffer = buffer_pred_cost
            flush_due_to_cost    = len(self.rebatching_buffer) >= self.min_flush_size

        if len(self.rebatching_buffer) > 0:
            print(f"[RouterAware] buffer_cost={buffer_pred_cost:.2f}, "
                  f"cost_adj={cost_adjusted_buffer:.2f}, "
                  f"len={len(self.rebatching_buffer)}, "
                  f"age={self.buffer_age}, "
                  f"router_aware={self.use_router_aware}")

        if flush_due_to_full or flush_due_to_cost or flush_due_to_age:
            print(f"[RouterAware] flush triggered — "
                  f"cost={flush_due_to_cost}(sum={buffer_pred_cost:.2f} >= thr={self.cost_flush_threshold}), "
                  f"full={flush_due_to_full}, "
                  f"age={flush_due_to_age}(age={self.buffer_age})")
            self.buffer_age = 0
            return SchedulerOutputs(id=self._iteration_id,
                                    ignored_seq_ids=[],
                                    preempted_seq_ids=[],
                                    scheduled_seq_metadata_list=[]), priority_reqs


       
        while self.waiting:
            seq = self.waiting[0]
            # This is required to handle benchmarking where
            # we set request arrival time ahead of time
            if seq.arrival_time > now:
                break

            num_prompt_tokens = seq.get_len()
            if not self._check_request_prompt_length(seq):
                ignored_seq_ids.append(seq.seq_id)
                continue

            # If the sequence group cannot be allocated, stop.
            
            if not self.block_manager.can_allocate(seq):
                break

            # If the number of batched tokens exceeds the limit, stop.
            if (
                num_batched_tokens + num_prompt_tokens
                > self.scheduler_config.max_num_batched_tokens
            ):
                break
            # print(f"[VLLMScheduler._schedule] len(self.running): {len(self.running)}. max_num_seqs: {self.scheduler_config.max_num_seqs}")
            if len(self.running) + 1 > self.scheduler_config.max_num_seqs:
                break

            if len(self.rebatching_buffer) > 2 * self.scheduler_config.max_num_seqs + 1:
                break

            seq = self.waiting.pop(0)
            self._allocate(seq)
            num_batched_tokens += num_prompt_tokens
            scheduled_seq_metadata_list.append(
                SequenceScheduleMetadata.from_sequence(seq)
            )
            self.running.append(seq)
            self.request_age_map[seq.seq_id] = 0

        if scheduled_seq_metadata_list or ignored_seq_ids:
            # print(f"[VLLMScheduler._schedule] returning scheduled list: {[metadata.seq_id for metadata in scheduled_seq_metadata_list]}")
            return SchedulerOutputs(
                id=self._iteration_id,
                ignored_seq_ids=ignored_seq_ids,
                preempted_seq_ids=[],
                scheduled_seq_metadata_list=scheduled_seq_metadata_list,
            ), priority_reqs

        # NOTE(woosuk): Preemption happens only when there is no available slot
        # to keep all the sequence groups in the RUNNING state.
        # In this case, the policy is responsible for deciding which sequence
        # groups to preempt.
        self.running = self.policy.sort_by_priority(now, self.running)

        # Reserve new token slots for the running sequence groups.
        running: List[Sequence] = []

        #print(f"[VLLMScheduler._schedule] running list: {running}")

        while self.running:
            seq = self.running.pop(0)
            if len(scheduled_seq_metadata_list) + 1 > self.scheduler_config.max_num_seqs:
                running.append(seq) # Don't schedule this sequence, but keep it in the running list
                continue

            if not seq.is_paused():
                # The sequence group is already in the RUNNING state.
                running.append(seq)
                continue

            assert seq.prompt_processing_finished


            while not self.block_manager.can_append_slot():
                if self.running:
                    # Preempt the lowest-priority sequence groups.
                    victim_seq = self.running.pop(-1)
                    self._preempt(victim_seq)
                    preempted_seq_ids.append(victim_seq.seq_id)
                else:
                    # No other sequence groups can be preempted.
                    # Preempt the current sequence group.
                    self._preempt(seq)
                    preempted_seq_ids.append(seq.seq_id)
                    break
            else:
                # Append new slots to the sequence group.
                self._append_slot(seq)
                running.append(seq)
                scheduled_seq_metadata_list.append(
                    SequenceScheduleMetadata.from_sequence(seq)
                )

        self.running = running

        # for seq in self.running:
        #     print(f"[VLLMScheduler._schedule] returning running list: {seq.seq_id}, status: {seq.get_status()}")
        # for seq in scheduled_seq_metadata_list:
        #     print(f"[VLLMScheduler._schedule] returning scheduled list: {seq.seq_id}")

        # print(f"[VLLMScheduler._schedule] returning scheduled list: {[metadata.seq_id for metadata in scheduled_seq_metadata_list]}")

        return SchedulerOutputs(
            id=self._iteration_id,
            ignored_seq_ids=[],
            preempted_seq_ids=preempted_seq_ids,
            scheduled_seq_metadata_list=scheduled_seq_metadata_list,
        ), priority_reqs
    

    def update_router_meta(self, router_meta_map: dict):
        self.router_meta_map.update(router_meta_map)

    def _buffer_pred_cost_sum(self) -> float:
        total = 0.0
        for seq_id in self.rebatching_buffer:
            meta = self.router_meta_map.get(seq_id)
            if meta is not None:
                pred_cost = meta["pred_cost"]
            else:
                # No router signal yet — assume worst-case cost so the system
                # degrades gracefully to count-based scheduling.
                pred_cost = 1.0
            total += pred_cost
        return total

    def _buffer_avg_pred_cost(self) -> float:
        if not self.rebatching_buffer:
            return 0.0
        return self._buffer_pred_cost_sum() / len(self.rebatching_buffer)

    def on_rebatching(self, scheduled_seq_metadata_list: List[SequenceMetadata], output_seqs: List[Sequence], is_ee: bool, is_flush: bool):
        # 1. Handle the case where requests come out from the rebatching buffer.
        # Loop through outputs. If a request is in the rebatching buffer, and is outputted, then it is moved to the running list.
        # If this happens, we need to move all existing sequences in the running list to `waiting`
        # print(f"[VLLMScheduler.on_rebatching] output_seqs: {output_seqs}. running list: {self.running}")

        output_seq_ids = []
        for output_seq in output_seqs:
            output_seq_ids.append(output_seq.seq_id)

        if is_flush or len(output_seqs) > len(scheduled_seq_metadata_list):

            for output_seq in output_seqs:
                if output_seq.seq_id in self.rebatching_buffer:
                    assert output_seq not in self.running, f"seq_id: {output_seq.seq_id}, status: {output_seq.get_status()}"
                    # output_seq_metadata.seq.set_status(SequenceStatus.RUNNING) # meant to set the status of IN_BUFFER to RUNNING
                    self.running.insert(0, output_seq)
                    self.rebatching_buffer.remove(output_seq.seq_id)
            return
        

        # 2. Handle the case where requests are scheduled, but not outputted. The missing requests are moved to the rebatching buffer.
        # Loop through the scheduled list (input list): if a request is in the input list but not in the output list, then it is moved to the rebatching buffer.
        if is_ee:
            for input_seq_metadata in scheduled_seq_metadata_list:
                if input_seq_metadata.seq.seq_id not in output_seq_ids:
                    # This sequence is in the input but not in the output --> it is in the rebatching buffer
                    input_seq_metadata.seq.set_status(SequenceStatus.IN_BUFFER)
                    self.rebatching_buffer.append(input_seq_metadata.seq.seq_id)
                    # remove the sequence from the running list
                    if input_seq_metadata.seq in self.running:
                        self.running.remove(input_seq_metadata.seq)

                # print(f"[VLLMScheduler.on_rebatching] added to rebatching buffer: {input_seq_metadata.seq.seq_id}, status: {input_seq_metadata.seq.get_status()}!!!!!")
