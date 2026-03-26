from typing import List

from sarathi.benchmark.entities import Request
from sarathi.benchmark.request_generator.base_request_generator import (
    BaseRequestGenerator,
)

from datasets import load_dataset



class RealRequestGenerator(BaseRequestGenerator):

    def __init__(self, config, max_article_length: int = 2000):
        super().__init__(config)

        self.max_article_length = max_article_length
        # Get dataset name from config, default to "cnn"
        self.dataset_name = getattr(config, 'real_request_generator_dataset_name', 'cnn').lower()
        
        if self.dataset_name == "xsum":
            # Load XSUM directly from HuggingFace (latest version)
            self.dataset = load_dataset("EdinburghNLP/xsum", split="validation")
            print(f"Loaded XSUM dataset: {len(self.dataset)} samples")

            # Filter by document length (characters)
            self.filtered_indices = [
                i for i in range(len(self.dataset))
                if len(self.dataset[i]["document"]) <= self.max_article_length
            ]

            print(
                f"Filtered XSUM dataset: "
                f"{len(self.filtered_indices)} articles out of {len(self.dataset)} "
                f"are within {self.max_article_length} characters"
            )
        else:
            # Default to CNN/DailyMail
            self.dataset_name = "cnn"
            self.dataset = load_dataset("abisee/cnn_dailymail", "3.0.0", split="validation")
            # Filter articles to only include those within max_article_length
            self.filtered_indices = []
            for i in range(len(self.dataset)):
                if len(self.dataset[i]["article"]) <= self.max_article_length:
                    self.filtered_indices.append(i)
            print(f"Filtered CNN dataset: {len(self.filtered_indices)} articles out of {len(self.dataset)} are within {self.max_article_length} characters")
    
    def _get_squad_prompt(self, idx: int) -> str:
        return self.squad[idx]["context"] + " " + self.squad[idx]["question"]
    
    def _get_summary(self, idx: int) -> str:
        """Get the reference summary for the given index."""
        original_idx = self.filtered_indices[idx]
        if self.dataset_name == "xsum":
            return self.dataset[original_idx]["summary"]
        else:
            return self.dataset[original_idx]["highlights"]
    
    def _get_prompt(self, idx: int) -> str:
        """Get the prompt for the given index."""
        original_idx = self.filtered_indices[idx]
        if self.dataset_name == "xsum":
            # XSUM prompt format
            return "Article: " + self.dataset[original_idx]["document"] + ". Summarize the article in one sentence. Summary:"
        else:
            # CNN/DailyMail prompt format
            # https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00632/119276/Benchmarking-Large-Language-Models-for-News
            return "Article: " + self.dataset[original_idx]["article"] + ". Summarize the article in three sentences. Summary:"
    
    def _get_cnn_summary(self, idx: int) -> str:
        return self._get_summary(idx)
    
    def _get_cnn_prompt(self, idx: int) -> str:
        return self._get_prompt(idx)

    def _generate_next_request(self, last_arrived_at: float, idx: int=0) -> Request:
        
        inter_request_time = 0.01
        if inter_request_time is None:
            return None
        arrived_at = last_arrived_at + inter_request_time

        

        return Request(
            arrived_at=arrived_at,
            prompt=self._get_prompt(idx),
        )

    def _generate_requests(self) -> List[Request]:
        requests = []

        current_time = 0

        # Use the number of filtered articles as the limit
        num_available_articles = len(self.filtered_indices)
        num_requests = min(getattr(self._config, 'real_request_generator_num_requests', 500), num_available_articles)
        
        for i in range(num_requests):
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
    
    def get_prompts(self) -> List[str]:
        """Get all prompts for the configured dataset."""
        num_available_articles = len(self.filtered_indices)
        num_requests = min(getattr(self._config, 'real_request_generator_num_requests', 500), num_available_articles)
        return [self._get_prompt(i) for i in range(num_requests)]
    
    def get_summaries(self) -> List[str]:
        """Get all reference summaries for the configured dataset."""
        num_available_articles = len(self.filtered_indices)
        num_requests = min(getattr(self._config, 'real_request_generator_num_requests', 500), num_available_articles)
        return [self._get_summary(i) for i in range(num_requests)]
    
    # Keep backward compatibility methods
    def get_cnn_prompts(self) -> List[str]:
        return self.get_prompts()
    
    def get_cnn_summaries(self) -> List[str]:
        return self.get_summaries()
