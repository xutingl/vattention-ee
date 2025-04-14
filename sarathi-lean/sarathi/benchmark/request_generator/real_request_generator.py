from typing import List

from sarathi.benchmark.entities import Request
from sarathi.benchmark.request_generator.base_request_generator import (
    BaseRequestGenerator,
)

from datasets import load_dataset



class RealRequestGenerator(BaseRequestGenerator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.prompt_length = 1000
        self.squad = load_dataset("rajpurkar/squad_v2", split="validation")
        self.cnn = load_dataset("abisee/cnn_dailymail", "3.0.0", split="validation")
    
    def _get_squad_prompt(self, idx: int) -> str:
        return self.squad[idx]["context"] + " " + self.squad[idx]["question"]
    
    def _get_cnn_prompt(self, idx: int) -> str:
        # https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00632/119276/Benchmarking-Large-Language-Models-for-News
        return "Article: " + self.cnn[idx]["article"][:self.prompt_length] + ". Summarize the article in three sentences. Summary:"

    def _generate_next_request(self, last_arrived_at: float, idx: int=0) -> Request:
        
        inter_request_time = 0.01
        if inter_request_time is None:
            return None
        arrived_at = last_arrived_at + inter_request_time

        

        return Request(
            arrived_at=arrived_at,
            prompt=self._get_cnn_prompt(idx),
        )

    def _generate_requests(self) -> List[Request]:
        requests = []

        current_time = 0

        
        for i in range(self._config.num_requests):
            request = self._generate_next_request(current_time, idx=i)
            current_time = request.arrived_at
            requests.append(request)
        

        return requests

    def generate_requests(self) -> List[Request]:
       


        requests = self._generate_requests()

        # sort requests by arrival time
        requests.sort(key=lambda x: x.arrived_at)
        # remove any requests that arrived after the time limit
        

        return requests
    
    def get_cnn_prompts(self) -> List[str]:
        return [self._get_cnn_prompt(i)[:self.prompt_length] for i in range(self._config.num_requests)]
