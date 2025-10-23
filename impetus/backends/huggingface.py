"""HuggingFace Transformers backend implementation."""

import time
import torch
import warnings
import itertools
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any, List, Optional

from .base import BaseBackend, BackendConfig, InferenceResult


class HuggingFaceBackend(BaseBackend):
    """HuggingFace Transformers inference backend."""
    
    def load_model(self):
        """Load model and tokenizer from HuggingFace."""
        if self.is_loaded:
            return
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with appropriate precision
        if self.config.device.type == "cpu":
            self.model = AutoModelForCausalLM.from_pretrained(self.config.model_name)
        else:
            # GPU loading with quantization support
            if self.config.precision == "4bit":
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    quantization_config=quantization_config,
                    device_map="auto"
                )
            elif self.config.precision == "8bit":
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    quantization_config=quantization_config,
                    device_map="auto"
                )
            elif self.config.precision in ["fp16", "float16"]:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            elif self.config.precision in ["bf16", "bfloat16"]:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    torch_dtype=torch.bfloat16,
                    device_map="auto"
                )
            else:  # fp32 or full precision
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name
                ).to(self.config.device)
        
        self.model.eval()
        self.is_loaded = True
    
    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        **kwargs
    ) -> List[InferenceResult]:
        """Generate text from prompts."""
        if not self.is_loaded:
            self.load_model()
        
        # Tokenize inputs
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_sequence_length,
        ).to(self.config.device)
        
        num_input_tokens = inputs.input_ids.shape[1]
        
        # Generate
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                **kwargs
            )
        
        latency_ms = (time.time() - start_time) * 1000
        
        num_output_tokens = outputs.shape[1] - num_input_tokens
        
        results = []
        for i, output in enumerate(outputs):
            results.append(InferenceResult(
                output_ids=output,
                num_input_tokens=num_input_tokens,
                num_output_tokens=num_output_tokens,
                latency_ms=latency_ms / len(prompts),  # Average per prompt
            ))
        
        return results
    
    def benchmark_inference(
        self,
        prompts: Optional[List[str]] = None,
        max_new_tokens: int = 100,
        num_runs: int = 10,
        warmup_runs: int = 2,
        sequence_length: int = 128,
        batch_size: int = 1,
    ) -> Dict[str, Any]:
        """Run comprehensive benchmark on the model."""
        if not self.is_loaded:
            self.load_model()
        
        # Use provided prompts or load from dataset
        if prompts is None:
            dataset = load_dataset("wikimedia/wikipedia", "20231101.en", streaming=True)
            dataset_iter = iter(dataset["train"].shuffle())
            prompts = [
                row["text"] for row in itertools.islice(dataset_iter, batch_size)
            ]
        
        # Ensure we have the right batch size
        prompts = prompts[:batch_size]
        if len(prompts) < batch_size:
            # Repeat if we don't have enough prompts
            prompts = prompts * (batch_size // len(prompts) + 1)
            prompts = prompts[:batch_size]
        
        # Tokenize inputs
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            max_length=sequence_length,
            truncation=True,
            padding=True,
        ).to(self.config.device)
        
        actual_seq_len = inputs.input_ids.shape[1]
        
        # Warm-up runs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with torch.no_grad():
                for _ in range(warmup_runs):
                    self.model.generate(**inputs, max_new_tokens=1)
        
        # Measure Time to First Token (TTFT)
        ttft_times = []
        for _ in tqdm(
            range(num_runs), desc="Measuring time to first token", unit="run"
        ):
            start_time = time.time()
            with torch.no_grad():
                self.model.generate(**inputs, max_new_tokens=1)
            ttft_times.append((time.time() - start_time) * 1000)  # Convert to ms
        
        avg_ttft = np.mean(ttft_times)
        
        # Measure full generation latency
        latencies = []
        total_output_tokens = 0
        
        for _ in tqdm(range(num_runs), desc="Measuring latency", unit="run"):
            start_time = time.time()
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            latency = (time.time() - start_time) * 1000  # Convert to ms
            latencies.append(latency)
            total_output_tokens += (output.sequences.shape[1] - inputs.input_ids.shape[1]) * batch_size
        
        avg_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        
        # Calculate metrics
        avg_tokens_per_run = total_output_tokens / num_runs
        throughput = avg_tokens_per_run / (avg_latency / 1000)  # tokens/sec
        time_per_output_token = (avg_latency - avg_ttft) / (avg_tokens_per_run / batch_size)
        
        return {
            "backend": "huggingface",
            "model_name": self.config.model_name,
            "precision": self.config.precision,
            "batch_size": batch_size,
            "sequence_length": actual_seq_len,
            "max_new_tokens": max_new_tokens,
            "num_runs": num_runs,
            
            # Latency metrics
            "avg_latency_ms": avg_latency,
            "std_latency_ms": std_latency,
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            
            # Token generation metrics
            "avg_time_to_first_token_ms": avg_ttft,
            "std_ttft_ms": np.std(ttft_times),
            "time_per_output_token_ms": time_per_output_token,
            "tokens_per_batch": avg_tokens_per_run / num_runs * batch_size,
            
            # Throughput
            "throughput_tokens_per_sec": throughput,
            "throughput_batches_per_sec": 1000 / avg_latency,
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self.is_loaded:
            self.load_model()
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Get config info
        config_dict = {}
        if hasattr(self.model, 'config'):
            config = self.model.config
            config_dict = {
                "model_type": getattr(config, "model_type", "unknown"),
                "num_layers": getattr(config, "num_hidden_layers", getattr(config, "n_layer", 0)),
                "hidden_size": getattr(config, "hidden_size", getattr(config, "n_embd", 0)),
                "num_attention_heads": getattr(config, "num_attention_heads", getattr(config, "n_head", 0)),
                "vocab_size": getattr(config, "vocab_size", 0),
                "max_position_embeddings": getattr(config, "max_position_embeddings", getattr(config, "n_positions", 0)),
            }
        
        return {
            "backend": "huggingface",
            "model_name": self.config.model_name,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "precision": self.config.precision,
            **config_dict,
        }
    
    def cleanup(self):
        """Clean up resources."""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.is_loaded = False
    
    def supports_distributed(self) -> bool:
        """HuggingFace supports distributed via device_map='auto'."""
        return True

