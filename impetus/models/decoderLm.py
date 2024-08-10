
import time
import warnings
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)


from ..metrics import DecoderLmMetrics


class DecoderLm:
    def __init__(self, model_name, device) -> None:    
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
    def benchmark_inference(self, seq_len, num_runs=10) -> DecoderLmMetrics:
        dataset = load_dataset("wikimedia/wikipedia", '20231101.en', streaming=True)
        input_text = ""
        while len(input_text) < seq_len:
            input_text = next(iter(dataset['train'].shuffle()))['text'][:seq_len]
        
        input_ids = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        # Warm-up run
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.generate(**input_ids, max_new_tokens=1)
            
            # Measure Time to First Token (TTFT)
            ttft_times = []
            for _ in tqdm(range(num_runs), desc="Measuring time to first token", unit="run"):
                start_time = time.time()
                self.model.generate(**input_ids, max_new_tokens=1)
                ttft_times.append(time.time() - start_time)
            
            avg_ttft = np.mean(ttft_times)
            ttft_std = np.std(ttft_times)
            
            # Measure full generation latency
            latencies = []
            total_output_tokens = 0
            for _ in tqdm(range(num_runs), desc="Measuring latency", unit="run"):
                start_time = time.time()
                output = self.model.generate(
                    **input_ids,
                    max_new_tokens=100,
                    return_dict_in_generate=True,
                    output_scores=True
                )
                latencies.append(time.time() - start_time)
                total_output_tokens += len(output.sequences[0]) - len(input_ids[0])
            
        avg_latency = np.mean(latencies)
        
        # Calculate metrics
        avg_tokens_per_run = total_output_tokens / num_runs
        throughput = avg_tokens_per_run / avg_latency
        time_per_output_token = (avg_latency - avg_ttft) / avg_tokens_per_run
        
        return DecoderLmMetrics(
            latency=avg_latency,
            throughput=throughput,
            time_per_output_token=time_per_output_token,
            time_to_first_token=avg_ttft,
        )



