import time
import torch
import warnings
import itertools
import numpy as np
from tqdm import tqdm
import huggingface_hub
from datasets import load_dataset
from accelerate import cpu_offload
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from accelerate.utils import gather_object
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)


from ..metrics import DecoderOnlyLanguageModelMetrics


class DecoderOnlyLanguageModel:
    def __init__(self, model_name, device, precision) -> None:

        self.device = device
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.device.type == "cpu":
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        else:
            self.accelerator = Accelerator()
            print(f"Benhcmarking using {self.accelerator.num_processes} GPUs..")
            if precision == "4bit":
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, load_in_4bit=True, device_map="auto"
                )

            elif precision == "8bit":
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, load_in_8bit=True, device_map="auto"
                )

            elif precision == "full":
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if self.accelerator:
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            device_map={"": self.accelerator.process_index},
                            torch_dtype=torch.bfloat16,
                        )
                    else:
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_name
                        ).to(self.device)

            else:
                raise Exception(f"Invalid Precision : {precision}")

    def benchmark_inference(
        self, sequence_length, batch_size, num_runs, warmup_runs
    ) -> DecoderOnlyLanguageModelMetrics:
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dataset = load_dataset("wikimedia/wikipedia", "20231101.en", streaming=True)
        dataset_iter = iter(dataset["train"].shuffle())
        input_texts = [
            row["text"] for row in itertools.islice(dataset_iter, batch_size)
        ]

        input_ids = self.tokenizer(
            input_texts,
            return_tensors="pt",
            max_length=sequence_length,
            truncation=True,
            padding=True,
        ).to(self.device)

        # Warm-up run
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(warmup_runs):
                self.model.generate(**input_ids, max_new_tokens=1)

            # Measure Time to First Token (TTFT)
            ttft_times = []
            for _ in tqdm(
                range(num_runs), desc="Measuring time to first token", unit="run"
            ):
                start_time = time.time()
                with self.accelerator.split_between_processes(
                    input_ids
                ) as input_ids_per_process:
                    self.model.generate(**input_ids_per_process, max_new_tokens=1)
                ttft_times.append(time.time() - start_time)

            avg_ttft = np.mean(ttft_times)
            ttft_std = np.std(ttft_times)

            # Measure full generation latency
            latencies = []
            total_output_tokens = 0
            for _ in tqdm(range(num_runs), desc="Measuring latency", unit="run"):
                start_time = time.time()
                with self.accelerator.split_between_processes(
                    input_ids
                ) as input_ids_per_process:
                    output = self.model.generate(
                        **input_ids,
                        max_new_tokens=100,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )
                latencies.append(time.time() - start_time)
                total_output_tokens += len(output.sequences[0]) - len(input_ids[0])

        avg_latency = np.mean(latencies)

        # Calculate metrics
        avg_tokens_per_run = (total_output_tokens) / num_runs
        throughput = avg_tokens_per_run / avg_latency
        time_per_output_token = (avg_latency - avg_ttft) / avg_tokens_per_run

        return DecoderOnlyLanguageModelMetrics(
            seq_len=sequence_length,
            batch_size=batch_size,
            latency=avg_latency,
            tokens_per_batch=avg_tokens_per_run,
            throughput=throughput * batch_size,
            time_per_output_token=time_per_output_token,
            time_to_first_token=avg_ttft,
        )
