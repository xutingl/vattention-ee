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
        self.buffer_age_threshold = 1000
        self.buffer_age_factor = scheduler_config.buffer_age_factor

    def get_block_space_manager_class(self):
        return vAttentionBlockSpaceManager if is_vattention_backend() else VLLMBlockSpaceManager 

    def _schedule(self) -> SchedulerOutputs:
        # Fix the current time.
        now = time.monotonic()

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
        
        if len(self.rebatching_buffer_1) > 0 or len(self.rebatching_buffer_2) > 0:
            self.buffer_age += 1

        age_adjusted_buffer_size_1 = len(self.rebatching_buffer_1) * (1 + self.buffer_age * self.buffer_age_factor)
        age_adjusted_buffer_size_2 = len(self.rebatching_buffer_2) * (1 + self.buffer_age * self.buffer_age_factor)

        # Need to run requests in the rebatching buffer first
        buffer_1_full = age_adjusted_buffer_size_1 >= self.scheduler_config.max_num_seqs or (len(self.rebatching_buffer_1) >= len(self.waiting) and len(self.rebatching_buffer_1) > 0)
        buffer_2_full = age_adjusted_buffer_size_2 >= self.scheduler_config.max_num_seqs or (len(self.rebatching_buffer_2) >= len(self.waiting) and len(self.rebatching_buffer_2) > 0)
        if buffer_1_full or buffer_2_full or self.buffer_age > self.buffer_age_threshold:
            # print(f"[VLLMScheduler._schedule] rebatching buffer is full: {self.rebatching_buffer}. returning empty scheduler outputs.")
            self.buffer_age = 0
            return SchedulerOutputs(id=self._iteration_id,
                                    ignored_seq_ids=[],
                                    preempted_seq_ids=[],
                                    scheduled_seq_metadata_list=[])


       
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

            if len(self.rebatching_buffer_1) > 2 * self.scheduler_config.max_num_seqs + 1 or len(self.rebatching_buffer_2) > 2 * self.scheduler_config.max_num_seqs + 1:
                break

            seq = self.waiting.pop(0)
            self._allocate(seq)
            num_batched_tokens += num_prompt_tokens
            scheduled_seq_metadata_list.append(
                SequenceScheduleMetadata.from_sequence(seq)
            )
            self.running.append(seq)

        if scheduled_seq_metadata_list or ignored_seq_ids:
            # print(f"[VLLMScheduler._schedule] returning scheduled list: {[metadata.seq_id for metadata in scheduled_seq_metadata_list]}")
            return SchedulerOutputs(
                id=self._iteration_id,
                ignored_seq_ids=ignored_seq_ids,
                preempted_seq_ids=[],
                scheduled_seq_metadata_list=scheduled_seq_metadata_list,
            )

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
        )
    

    def on_rebatching(self, scheduled_seq_metadata_list: List[SequenceMetadata], output_seqs: List[Sequence], ee_from_layer: int, is_flush_1: bool, is_flush_2: bool):
        # 1. Handle the case where requests come out from the rebatching buffer.
        # Loop through outputs. If a request is in the rebatching buffer, and is outputted, then it is moved to the running list.
        # If this happens, we need to move all existing sequences in the running list to `waiting`

        output_seq_ids = []
        for output_seq in output_seqs:
            output_seq_ids.append(output_seq.seq_id)




        if is_flush_2:
            # flush 2 --> no EE
            assert ee_from_layer == -1
            for output_seq in output_seqs:
                assert output_seq not in self.running, f"seq_id: {output_seq.seq_id}, status: {output_seq.get_status()}"
                # output_seq_metadata.seq.set_status(SequenceStatus.RUNNING) # meant to set the status of IN_BUFFER to RUNNING
                self.running.insert(0, output_seq)
                self.rebatching_buffer_2.remove(output_seq.seq_id)
            assert len(self.rebatching_buffer_2) == 0
            return
        elif is_flush_1:
            # flush 1 -->  1) no EE; 2) EE from 2;
            if ee_from_layer == -1:
                for output_seq in output_seqs:
                    assert output_seq not in self.running, f"seq_id: {output_seq.seq_id}, status: {output_seq.get_status()}"
                    # output_seq_metadata.seq.set_status(SequenceStatus.RUNNING) # meant to set the status of IN_BUFFER to RUNNING
                    self.running.insert(0, output_seq)
                    self.rebatching_buffer_1.remove(output_seq.seq_id)
                assert len(self.rebatching_buffer_1) == 0
                return
            elif ee_from_layer == 2:
                for output_seq in output_seqs:
                    assert output_seq not in self.running, f"seq_id: {output_seq.seq_id}, status: {output_seq.get_status()}"
                    # output_seq_metadata.seq.set_status(SequenceStatus.RUNNING) # meant to set the status of IN_BUFFER to RUNNING
                    self.running.insert(0, output_seq)
                    self.rebatching_buffer_1.remove(output_seq.seq_id)
                
                # Move anything in reabtching_buffer_1 to rebatching_buffer_2
                while self.rebatching_buffer_1:
                    seq_id = self.rebatching_buffer_1.pop(0)
                    self.rebatching_buffer_2.append(seq_id)
                return
            else:
                raise ValueError(f"ee_from_layer: {ee_from_layer} is not valid")
        else:
            # No flush: Started from layer 0
            if ee_from_layer == 1:
                for input_seq_metadata in scheduled_seq_metadata_list:
                    if input_seq_metadata.seq.seq_id not in output_seq_ids:
                        input_seq_metadata.seq.set_status(SequenceStatus.IN_BUFFER)
                        self.rebatching_buffer_1.append(input_seq_metadata.seq.seq_id)
                        if input_seq_metadata.seq in self.running:
                            self.running.remove(input_seq_metadata.seq)
                return
            elif ee_from_layer == 2:
                for input_seq_metadata in scheduled_seq_metadata_list:
                    if input_seq_metadata.seq.seq_id not in output_seq_ids:
                        input_seq_metadata.seq.set_status(SequenceStatus.IN_BUFFER)
                        self.rebatching_buffer_2.append(input_seq_metadata.seq.seq_id)
                        if input_seq_metadata.seq in self.running:
                            self.running.remove(input_seq_metadata.seq)
                return


