import json
import logging
import os
import time

import ray
import wandb
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import numpy as np

from sarathi import LLMEngine, SamplingParams
from sarathi.benchmark.config import Config
from sarathi.benchmark.entities import Request
from sarathi.benchmark.request_generator import RequestGeneratorRegistry
from sarathi.benchmark.custom_types import ReplicaResourceMapping, ResourceMapping
from sarathi.benchmark.utils.random import set_seeds
from sarathi.config import MetricsConfig
from sarathi.metrics.metrics_store import MetricsStore
from sarathi.utils import get_ip
import bert_score
from rouge_score import rouge_scorer
from sarathi.benchmark.request_generator.real_request_generator import RealRequestGenerator

logger = logging.getLogger(__name__)


class BenchmarkRunner:

    def __init__(
        self,
        replica_id: int,
        config: Config,
        replica_resource_mapping: ResourceMapping = [],
    ) -> None:
        self._replica_id = replica_id
        self._config = config
        self._num_replicas = self._config.cluster_num_replicas

        self._time_limit = self._config.time_limit
        if not self._time_limit:
            self._time_limit = float("inf")

        output_dir = f"{self._config.output_dir}/replica_{replica_id}"
        os.makedirs(output_dir, exist_ok=True)

        # set_seeds(config.seed)
        set_seeds(42)
        print(f"[BenchmarkRunner]confi request generator: {self._config.request_generator_provider}")
        
        if self._config.request_generator_provider == "real":
            request_generator = RealRequestGenerator(self._config)
        else:
            request_generator = RequestGeneratorRegistry.get_from_str(
                self._config.request_generator_provider, self._config
            )
        self._requests = request_generator.generate()

        # select every nth request for this replica
        # e.g. if there are 4 replicas, and this is the 2nd replica, then
        # we will select the 2nd, 6th, 10th, ... requests
        # round robin scheduling
        self._requests = self._requests[self._replica_id :: self._num_replicas]

        if self._num_replicas == 1:
            wandb_project = self._config.metrics_store_wandb_project
            wandb_group = self._config.metrics_store_wandb_group
            wandb_run_name = self._config.metrics_store_wandb_run_name
        else:
            wandb_project = None
            wandb_group = None
            wandb_run_name = None

        chunk_size = None
        if self._config.replica_scheduler_provider == "sarathi":
            chunk_size = self._config.sarathi_scheduler_chunk_size
        elif self._config.replica_scheduler_provider == "simple_chunking":
            chunk_size = self._config.simple_chunking_scheduler_chunk_size
        
        self._config.model_load_format = "auto"
        self._config.download_dir = "/workspace/xutingl/downloaded_models/"

        self._llm_engine = LLMEngine.from_engine_args(
            # replica config
            replica_id=replica_id,
            replica_resource_mapping=replica_resource_mapping,
            output_dir=output_dir,
            # model config
            model=self._config.model_name,
            tokenizer=self._config.model_name,
            tensor_parallel_size=self._config.model_tensor_parallel_degree,
            pipeline_parallel_size=self._config.model_pipeline_parallel_degree,
            attention_backend=self._config.model_attention_backend,
            seed=42,
            dtype="float16", # "bfloat16" for llama3
            load_format=self._config.model_load_format,
            gpu_memory_utilization=self._config.gpu_memory_utilization,
            max_model_len=self._config.model_max_model_len,
            block_size=self._config.model_block_size,
            # scheduler config
            scheduler_type=self._config.replica_scheduler_provider,
            max_num_seqs=self._config.replica_scheduler_max_batch_size,
            # sarathi scheduler config
            chunk_size=chunk_size,
            enable_dynamic_chunking_schedule=self._config.sarathi_scheduler_enable_dynamic_chunking_schedule,
            low_chunk_size=self._config.sarathi_scheduler_low_chunk_size,
            high_chunk_size=self._config.sarathi_scheduler_high_chunk_size,
            chunk_schedule_max_tokens=self._config.sarathi_scheduler_chunk_schedule_max_tokens,
            chunk_schedule_stages=self._config.sarathi_scheduler_chunk_schedule_stages,
            # vllm scheduler config
            max_num_batched_tokens=self._config.vllm_scheduler_max_tokens_in_batch,
            # wandb config
            write_metrics=self._config.write_metrics,
            enable_chrome_trace=self._config.write_chrome_trace,
            wandb_project=wandb_project,
            wandb_group=wandb_group,
            wandb_run_name=wandb_run_name,
            wandb_sweep_id=self._config.metrics_store_wandb_sweep_id,
            wandb_run_id=self._config.metrics_store_wandb_run_id,
            # metrics config
            enable_op_level_metrics=self._config.metrics_store_enable_op_level_metrics,
            enable_cpu_op_level_metrics=self._config.metrics_store_enable_cpu_op_level_metrics,
            enable_request_outputs=self._config.metrics_store_enable_request_outputs,
            keep_individual_batch_metrics=self._config.metrics_store_keep_individual_batch_metrics,
            # engine config
            trust_remote_code=True,
            download_dir=self._config.download_dir,
            # EE config
            ee_policy=self._config.ee_policy,
            shallow_exit_layer=self._config.shallow_exit_layer,
            conf_threshold=self._config.conf_threshold,
            early_exit_head_path=self._config.early_exit_head_path,
            num_ee_threshold=self._config.num_ee_threshold,
            kv_method=self._config.kv_method,

            # Scheduler config for rebatching
            buffer_age_factor=self._config.buffer_age_factor,
        )

        self.ee_iter_count = [0, 0] # [# EE-iter, # non-EE-iter]
        
        self.normal_iter_times = []
        self.deep_iter_times = []
        self.ee_iter_times = []
        self.normal_iter_num_output_tokens = []
        self.ee_iter_num_output_tokens = []
        self.deep_iter_num_output_tokens = []

        self.prefill_times = []
        self.decode_times = []

    def _get_input_params(
        self, request: Request, first_request_time: float
    ) -> SamplingParams:
        sampling_params = SamplingParams(
            ignore_eos=False, #[TODO] How does True affect throughput and bert score?
            # max_tokens=request.num_decode_tokens,
            max_tokens=self._config.model_max_model_len // max(2, self._config.replica_scheduler_max_batch_size),
            #temperature=0.5,
            #top_p=0.5,
            #top_k=-1,
        )
        # prompt_token_ids = [1] * request.num_prefill_tokens

        return {
            "prompt": request.prompt,
            # "prompt_token_ids": prompt_token_ids,
            "sampling_params": sampling_params,
            "arrival_time": first_request_time + request.arrived_at,
            "seq_id": request._id,
        }

    def warmup(self) -> None:
        # warmup the engine
        self._llm_engine.add_request(
            **self._get_input_params(self._requests[0], time.monotonic())
        )

        is_completed = False
        while not is_completed:
            step_outputs = self._llm_engine.step()
            is_completed = step_outputs[0].finished

        self._llm_engine.reset_metrics()

    def _run(self) -> None:
        if self._config.enable_profiling:
            self._llm_engine.start_profiling()

        num_processed_requests = 0
        num_steps = 0
        pbar = tqdm(
            total=len(self._requests),
            desc=f"Replica {self._replica_id} processed requests",
        )
        num_output_tokens = 0
        start_time = time.monotonic()

        finished_seq_id_lst = []
        finished_output = []


        avg_conf_score_ee_lst = []
        avg_conf_score_non_ee_lst = []

        request_duration_lst = []

        # Run the engine.
        while num_processed_requests < len(self._requests):
            iter_start_time = time.perf_counter()
            elapsed_time = time.monotonic() - start_time
            if elapsed_time > self._time_limit:
                break
            
            #print(f"[BenchmarkRunner]step {num_steps} started")
            step_outputs, exited_rates, conf_score, is_ee, is_flush, is_prefill, latency_only_ee_iter_time = self._llm_engine.step()

            num_steps += 1


            for output in step_outputs:
                if output.finished:
                    num_processed_requests += 1
                    pbar.update(1)
                    num_output_tokens += len(output.token_ids)
                    print(f"[BenchmarkRunner._run] Output id {output.seq_id} Finished=====================================")
                    print(output.text)
                    print("=====================================")
                    raw_string = fr"{output.text}"
                    finished_seq_id_lst.append(output.seq_id)
                    finished_output.append(raw_string)
                    request_duration_lst.append(output.completion_time)




            iteration_time = time.perf_counter() - iter_start_time

            if is_prefill:
                self.prefill_times.append(iteration_time)
            else:
                # Only collect EE related metrics for decode iterations.
                self.decode_times.append(iteration_time)

                if latency_only_ee_iter_time is not None:
                    # To account for latency-only EE, we need to add the latency-only EE iter time to the decode time.
                    # Notice: this is "double counting" iteration time. So when policy is latency-only, `self.decode_times` can only be used to calculate TBT. It is inaccurate to use it to calculate throughput or TPOT.
                    self.decode_times.append(latency_only_ee_iter_time)


                if conf_score is not None:
                    if is_ee:
                        self.ee_iter_count[0] += 1
                        avg_conf_score_ee_lst.append(conf_score)
                    else:
                        self.ee_iter_count[1] += 1
                        avg_conf_score_non_ee_lst.append(conf_score)

                if is_flush:
                    self.deep_iter_times.append(iteration_time)
                    self.deep_iter_num_output_tokens.append(len(step_outputs))
                else:
                    if is_ee:
                        self.ee_iter_times.append(iteration_time)
                        self.ee_iter_num_output_tokens.append(len(step_outputs))
                    else:
                        self.normal_iter_times.append(iteration_time)
                        self.normal_iter_num_output_tokens.append(len(step_outputs))
        end_time = time.monotonic()
        pbar.close()

        if self._config.enable_profiling:
            self._llm_engine.stop_profiling()

        

        self._llm_engine.cleanup() # clean up the engine so we have gpu memory for bert_score

        req_spent_times = []
        req_num_output_tokens = []
        for finished_seq_id in finished_seq_id_lst:
            seq = self._llm_engine.get_seq(finished_seq_id)
            req_spent_times.append(seq.state.e2e_time)
            req_num_output_tokens.append(seq.state.num_output_tokens)
        
        prefill_time = self._llm_engine.prefill_spent_time
        decode_time = self._llm_engine.decode_spent_time
        tpot = decode_time / sum(req_num_output_tokens)



        # Get reference summaries for each request index
        self._config.num_requests = len(self._requests)
        reference_generator = RealRequestGenerator(self._config)
        reference_summaries = reference_generator.get_cnn_summaries()

        # Compute rougeL and bert_score for each request
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rougeL_scores = []
        bert_scores = []
        for idx, output in enumerate(finished_output):
            reference = reference_summaries[idx]
            # Compute rougeL fmeasure
            rougeL = scorer.score(reference, output)['rougeL'].fmeasure
            rougeL_scores.append(rougeL)
            # Compute bert_score F1
            P, R, F1 = bert_score.score([output], [reference], lang='en')
            bert_scores.append(F1[0].item())

        output_throughput = num_output_tokens / (end_time - start_time)

        tokens_per_iter = num_output_tokens / sum(self.ee_iter_count)

        if len(avg_conf_score_ee_lst) > 0:
            avg_conf_score_ee = sum(avg_conf_score_ee_lst) / len(avg_conf_score_ee_lst)
        else:
            avg_conf_score_ee = 0

        avg_conf_score_non_ee = sum(avg_conf_score_non_ee_lst) / len(avg_conf_score_non_ee_lst)
        avg_conf_score = (sum(avg_conf_score_ee_lst) + sum(avg_conf_score_non_ee_lst)) / (len(avg_conf_score_ee_lst) + len(avg_conf_score_non_ee_lst))

        # Iteration time stats
        normal_iter_count = len(self.normal_iter_times)
        ee_iter_count = len(self.ee_iter_times)
        deep_iter_count = len(self.deep_iter_times)
        total_iter_count = normal_iter_count + ee_iter_count + deep_iter_count
        avg_normal_iter_time = sum(self.normal_iter_times) / max(1,normal_iter_count)
        avg_ee_iter_time = sum(self.ee_iter_times) / max(1,ee_iter_count)
        avg_deep_iter_time = sum(self.deep_iter_times) / max(1,deep_iter_count)
        total_iter_time = sum(self.normal_iter_times) + sum(self.ee_iter_times) + sum(self.deep_iter_times)
        avg_normal_iter_num_output_tokens = sum(self.normal_iter_num_output_tokens) / max(1,normal_iter_count)
        avg_ee_iter_num_output_tokens = sum(self.ee_iter_num_output_tokens) / max(1,ee_iter_count)
        avg_deep_iter_num_output_tokens = sum(self.deep_iter_num_output_tokens) / max(1,deep_iter_count)
        total_iter_num_output_tokens = sum(self.normal_iter_num_output_tokens) + sum(self.ee_iter_num_output_tokens) + sum(self.deep_iter_num_output_tokens)

        ee_penalty_by_tokens = (1 - avg_conf_score_ee) / max(1, sum(self.ee_iter_num_output_tokens))
        ee_penalty_by_iter = (1 - avg_conf_score_ee) / max(1, ee_iter_count)

        # TBT: Time between tokens (inter token latency) = decoding iteration time
        tbt_avg = sum(self.decode_times) / max(1, len(self.decode_times))
        tbt_p95 = np.percentile(self.decode_times, 95)
        tbt_p99 = np.percentile(self.decode_times, 99)

        # Statictics of request duration
        request_duration_avg = sum(request_duration_lst) / max(1, len(request_duration_lst))
        request_duration_p95 = np.percentile(request_duration_lst, 95)
        request_duration_p99 = np.percentile(request_duration_lst, 99)

        baseline_bert_score = 0.8330790978670121 # For llama2-13b
        avg_bert_score = sum(bert_scores) / len(bert_scores)
        ee_bert_penalty_by_tokens = (baseline_bert_score - avg_bert_score) / max(1, sum(self.ee_iter_num_output_tokens))
        ee_bert_penalty_by_iter = (baseline_bert_score - avg_bert_score) / max(1, ee_iter_count)

        # Calculate overhead c and num_ee_threshold. This is not used in the model, jsut to check num_ee_threshold here is the same as what we get in llm_engine.
        overhead = 0
        rebatching_threshold_ratio = 0
        num_ee_threshold = 0
        if self._config.ee_policy == "rebatching":
            overhead = avg_ee_iter_time + avg_deep_iter_time - avg_normal_iter_time
            rebatching_threshold_ratio = overhead / avg_deep_iter_time
            num_ee_threshold = self._config.replica_scheduler_max_batch_size * rebatching_threshold_ratio

        df = pd.DataFrame({
            "seq_id": finished_seq_id_lst,
            "output": finished_output,
            "time": end_time - start_time,
            "throughput": output_throughput,
            "rougeL": rougeL_scores,
            "bert_score": bert_scores,
            "prefill_time": prefill_time,
            "decode_time": decode_time,
            "tpot": tpot,
            "num_ee_tokens": exited_rates[0],
            "num_no_ee_tokens": exited_rates[1],
            "avg_conf_score": avg_conf_score,
            "avg_conf_score_ee": avg_conf_score_ee,
            "avg_conf_score_non_ee": avg_conf_score_non_ee,
            "num_ee_iter": self.ee_iter_count[0],
            "num_no_ee_iter": self.ee_iter_count[1],
            "tokens_per_iter": tokens_per_iter,
            "avg_normal_iter_time": avg_normal_iter_time,
            "avg_ee_iter_time": avg_ee_iter_time,
            "avg_deep_iter_time": avg_deep_iter_time,
            "total_iter_time": total_iter_time,
            "avg_normal_iter_num_output_tokens": avg_normal_iter_num_output_tokens,
            "avg_ee_iter_num_output_tokens": avg_ee_iter_num_output_tokens,
            "avg_deep_iter_num_output_tokens": avg_deep_iter_num_output_tokens,
            "ee_penalty_by_tokens": ee_penalty_by_tokens,
            "ee_penalty_by_iter": ee_penalty_by_iter,
            "ee_bert_penalty_by_tokens": ee_bert_penalty_by_tokens,
            "ee_bert_penalty_by_iter": ee_bert_penalty_by_iter,
            "tbt_avg": tbt_avg,
            "tbt_p95": tbt_p95,
            "tbt_p99": tbt_p99,
            "request_duration": request_duration_lst,
            "request_duration_avg": request_duration_avg,
            "request_duration_p95": request_duration_p95,
            "request_duration_p99": request_duration_p99,
        })
        df = df.sort_values(by="seq_id")

        csv_path = Path(self._config.csv_path)
        csv_path.mkdir(parents=True, exist_ok=True)

        # numrequests_batchsize_layer_conf_policy_kvmethod.csv
        csv_file = f"{csv_path}/req_{len(self._requests)}_batch_{self._config.replica_scheduler_max_batch_size}_layer_{self._config.shallow_exit_layer}_conf_{self._config.conf_threshold}_{self._config.ee_policy}_{self._config.kv_method}_age_{self._config.buffer_age_factor}.csv"
        print(f"Saving results to {csv_file}")
        df.to_csv(csv_file, index=False, escapechar='\\')

        logger.info(
            f"Replica {self._replica_id} exiting after processing {len(self._requests)} ({num_steps} iterations), Total time taken: {end_time - start_time:.2f} seconds"
        )
        logger.info(f"Replica {self._replica_id} processed {num_output_tokens} output tokens. Time taken: {end_time - start_time:.2f} seconds. Throughput: {output_throughput:.2f} tokens/sec. Exited rates(#tokens generated via ee vs. non-ee): {exited_rates}.")
        logger.info(f"Num EE iter: {self.ee_iter_count[0]}, Num non-EE iter: {self.ee_iter_count[1]}. Total iter: {sum(self.ee_iter_count)}. Tokens per iter: {tokens_per_iter:.2f}")
        logger.info(f"Avg normal iter time: {avg_normal_iter_time}, Avg ee iter time: {avg_ee_iter_time}, Avg deep iter time: {avg_deep_iter_time}, Total iter time: {total_iter_time}")
        logger.info(f"Avg normal iter num output tokens: {avg_normal_iter_num_output_tokens}, Avg ee iter num output tokens: {avg_ee_iter_num_output_tokens}, Avg deep iter num output tokens: {avg_deep_iter_num_output_tokens}, Total iter num output tokens: {total_iter_num_output_tokens}")
        logger.info(f"Overhead: {overhead}, Rebaching threshold ratio: {rebatching_threshold_ratio}, Num EE threshold: {num_ee_threshold}. [Measured in benchmark_runner. The one measured in llm_engine is actually used.]")
        logger.info(f"Avg conf_score: {avg_conf_score}. Avg conf_score ee: {avg_conf_score_ee}. Avg conf_score non_ee: {avg_conf_score_non_ee}")
        logger.info(f"EE penalty by tokens: {ee_penalty_by_tokens}, EE penalty by iter: {ee_penalty_by_iter}")
        logger.info(f"EE BERT penalty by tokens: {ee_bert_penalty_by_tokens}, EE BERT penalty by iter: {ee_bert_penalty_by_iter}")
        logger.info(f"RougeL: {sum(rougeL_scores) / len(rougeL_scores)}, Bert_score: {sum(bert_scores) / len(bert_scores)}")
        logger.info(f"Prefill time: {prefill_time}, Decode time: {decode_time}, TPOT: {tpot}")
        logger.info(f"Prefill time (measured in benchmark_runner): {sum(self.prefill_times)}, Decode time (measured in benchmark_runner): {sum(self.decode_times)}")
        logger.info(f"TBT avg: {tbt_avg}, TBT p95: {tbt_p95}, TBT p99: {tbt_p99}")
        logger.info(f"Request duration avg: {request_duration_avg}, Request duration p95: {request_duration_p95}, Request duration p99: {request_duration_p99}")


    def _add_requests(self) -> None:
        index = 0
        # first_request_time = time.monotonic()
        first_request_time = 0.01
        while index < len(self._requests):
            request = self._requests[index]
            self._llm_engine.add_request(
                **self._get_input_params(request, first_request_time)
            )
            index += 1

    def run(self) -> None:
        self._llm_engine.reset_metrics()
        self._add_requests()
        self._run()

        # We don't need these metrics. We do cleanup in _run()
        # self._llm_engine.pull_worker_metrics()
        # metric_store = self._llm_engine.get_metric_store()
        # self._llm_engine.cleanup()
        # return metric_store


class BenchmarkRunnerLauncher:

    def __init__(self, config: Config) -> None:
        self._config = config
        self._is_multi_replica = self._config.cluster_num_replicas > 1

        ray.init(ignore_reinit_error=True)

        if self._is_multi_replica:
            self._validate_cluster_resources()
            self._runners = self._create_runners()
            self._aggregate_metric_store = self._create_aggregate_metric_store()
        else:
            replica_resource_mapping = self._get_replica_resource_mapping()
            assert len(replica_resource_mapping) == 1
            self._runner = BenchmarkRunner(
                0, self._config, replica_resource_mapping["0"]
            )

        if wandb.run is not None:
            wandb.config.update(self._config.__dict__)

    def _validate_cluster_resources(self):
        num_replicas = self._config.cluster_num_replicas
        tp_degree = self._config.model_tensor_parallel_degree
        pp_degree = self._config.model_pipeline_parallel_degree
        num_gpus_required = num_replicas * tp_degree * pp_degree

        available_resources = ray.available_resources()

        assert (
            available_resources["GPU"] >= num_gpus_required
        ), f"Insufficient GPUs. Required: {num_gpus_required}, Available: {available_resources['GPU']}"

    def _get_replica_resource_mapping(self) -> ReplicaResourceMapping:
        if self._config.replica_resource_mapping:
            replica_resource_mapping = json.loads(self._config.replica_resource_mapping)
            logger.info(f"Replica resource mapping: {replica_resource_mapping}")
            return replica_resource_mapping

        cluster_resources_keys = list(ray.available_resources().keys())
        num_gpus = ray.available_resources()["GPU"]
        ip_addresses = [
            x
            for x in cluster_resources_keys
            if x.startswith("node:") and x != "node:__internal_head__"
        ]

        runner_ip = f"node:{get_ip()}"
        # runner_ip = "node:158.130.4.64" # For Phastform machine

        ip_addresses.remove(runner_ip)
        ip_addresses.insert(0, runner_ip)

        num_nodes = len(ip_addresses)
        assert num_nodes > 0, "No nodes found in the cluster"
        assert num_gpus > 0, "No GPUs found in the cluster"
        assert (
            num_gpus % num_nodes == 0
        ), f"Number of GPUs ({num_gpus}) is not a multiple of number of nodes ({num_nodes})"
        num_gpus_per_node = int(num_gpus // num_nodes)
        num_replicas = self._config.cluster_num_replicas
        num_gpus_per_replica = (
            self._config.model_tensor_parallel_degree
            * self._config.model_pipeline_parallel_degree
        )

        assert (
            num_gpus >= num_replicas * num_gpus_per_replica
        ), f"Insufficient GPUs. Required: {num_replicas * num_gpus_per_replica}, Available: {num_gpus}"

        replica_resource_mapping = {}

        available_gpus = []
        for ip_address in ip_addresses:
            for gpu_id in reversed(range(num_gpus_per_node)):
                available_gpus.append((ip_address, gpu_id))

        for replica_id in range(num_replicas):
            replica_resource_mapping[str(replica_id)] = []
            for _ in range(num_gpus_per_replica):
                replica_resource_mapping[str(replica_id)].append(available_gpus.pop(0))

        logger.info(f"Replica resource mapping: {replica_resource_mapping}")

        return replica_resource_mapping

    def _create_runners(self):
        assert (
            self._config.model_tensor_parallel_degree > 1
            or self._config.model_pipeline_parallel_degree > 1
        )

        replica_resource_mapping = self._get_replica_resource_mapping()

        runner_class = ray.remote(num_cpus=1)(BenchmarkRunner)

        runners = []

        for replica_id in range(self._config.cluster_num_replicas):
            runners.append(
                runner_class.options(
                    resources={
                        replica_resource_mapping[str(replica_id)][0][0]: 0.01,
                    },
                ).remote(
                    replica_id, self._config, replica_resource_mapping[str(replica_id)]
                )
            )

        return runners

    def _create_aggregate_metric_store(self):
        metric_config = MetricsConfig(
            replica_id=0,  # dummy replica id
            write_metrics=self._config.write_metrics,
            output_dir=self._config.output_dir,
            wandb_project=self._config.metrics_store_wandb_project,
            wandb_group=self._config.metrics_store_wandb_group,
            wandb_run_name=self._config.metrics_store_wandb_run_name,
            enable_op_level_metrics=self._config.metrics_store_enable_op_level_metrics,
            enable_cpu_op_level_metrics=self._config.metrics_store_enable_cpu_op_level_metrics,
            enable_chrome_trace=self._config.write_chrome_trace,
            enable_request_outputs=self._config.metrics_store_enable_request_outputs,
            keep_individual_batch_metrics=self._config.metrics_store_keep_individual_batch_metrics,
        )
        metrics_store = MetricsStore(metric_config)
        metrics_store.mark_initial_memory_profiling_done()

        return metrics_store

    def run(self):
        if self._is_multi_replica:
            ray.get([runner.warmup.remote() for runner in self._runners])

            runner_metrics = ray.get([runner.run.remote() for runner in self._runners])

            for runner_metric in runner_metrics:
                self._aggregate_metric_store.merge(runner_metric)

            if wandb.run is not None:
                wandb.config.update(self._config.__dict__)

            self._aggregate_metric_store.plot()
        else:
            metric_store = self._runner.run()
            #metric_store.plot()

        wandb.finish()
