from typing import List

from sarathi.benchmark.entities import Request
from sarathi.benchmark.request_generator.base_request_generator import (
    BaseRequestGenerator,
)
from sarathi.benchmark.request_generator.request_interval_generator_registry import (
    RequestIntervalGeneratorRegistry,
)
from sarathi.benchmark.request_generator.request_length_generator_registry import (
    RequestLengthGeneratorRegistry,
)
from sarathi.benchmark.utils.random import set_seeds

from datasets import load_dataset



class SyntheticRequestGenerator(BaseRequestGenerator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._seed = self._config.seed

        self._request_length_generator = RequestLengthGeneratorRegistry.get_from_str(
            self._config.synthetic_request_generator_length_provider, self._config
        )
        self._request_interval_generator = (
            RequestIntervalGeneratorRegistry.get_from_str(
                self._config.synthetic_request_generator_interval_provider, self._config
            )
        )
        self.squad = load_dataset("rajpurkar/squad_v2", split="validation")
        self.cnn = load_dataset("abisee/cnn_dailymail", "3.0.0", split="validation")
    
    def _get_squad_prompt(self, idx: int) -> str:
        return self.squad[idx]["context"] + " " + self.squad[idx]["question"]
    
    def _get_cnn_prompt(self, idx: int) -> str:
        # https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00632/119276/Benchmarking-Large-Language-Models-for-News
        return "Article: " + self.cnn[idx]["article"][:500] + ". Summarize the article in three sentences. Summary:"

    def _generate_next_request(self, last_arrived_at: float, idx: int=0) -> Request:
        inter_request_time = (
            self._request_interval_generator.get_next_inter_request_time()
        )
        inter_request_time = 0.1
        if inter_request_time is None:
            return None
        arrived_at = last_arrived_at + inter_request_time

        # (
        #     prefill_tokens,
        #     decode_tokens,
        # ) = self._request_length_generator.get_next_num_tokens()

        # if prefill_tokens is None or decode_tokens is None:
        #     return None

        return Request(
            arrived_at=arrived_at,
            prompt=self._get_cnn_prompt(idx),
            # num_prefill_tokens=int(prefill_tokens),
            # num_decode_tokens=int(decode_tokens),
        )

    def _generate_requests(self) -> List[Request]:
        requests = []

        current_time = 0

        # first priority is duration
        if self._config.synthetic_request_generator_duration is not None:
            idx = 0
            while current_time < self._config.synthetic_request_generator_duration:
                request = self._generate_next_request(current_time, idx=idx)
                idx += 1
                current_time = request.arrived_at
                requests.append(request)
        elif self._config.synthetic_request_generator_num_requests is not None:
            for i in range(self._config.synthetic_request_generator_num_requests):
                request = self._generate_next_request(current_time, idx=i)
                current_time = request.arrived_at
                requests.append(request)
        else:
            assert self._config.synthetic_request_generator_interval_provider == "trace"
            idx = 0
            while True:
                request = self._generate_next_request(current_time, idx=idx)
                idx += 1
                if request is None:
                    break
                current_time = request.arrived_at
                requests.append(request)

        return requests

    def generate_requests(self) -> List[Request]:
        assert (
            self._config.synthetic_request_generator_num_requests
            or self._config.synthetic_request_generator_duration
            or self._config.synthetic_request_generator_interval_provider == "trace"
        )

        set_seeds(self._seed)

        requests = self._generate_requests()

        # sort requests by arrival time
        requests.sort(key=lambda x: x.arrived_at)
        # remove any requests that arrived after the time limit
        if self._config.synthetic_request_generator_duration is not None:
            requests = [
                request
                for request in requests
                if request.arrived_at
                < self._config.synthetic_request_generator_duration
            ]

        return requests
