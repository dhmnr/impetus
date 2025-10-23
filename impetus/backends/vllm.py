"""vLLM backend implementation."""

import sys
import time
import warnings
from typing import Dict, Any, List, Optional

try:
    import vllm
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    VLLM_AVAILABLE = False
    if sys.platform == "win32":
        warnings.warn(
            "vLLM is not available on Windows (requires Triton which is Linux-only). "
            "Use WSL2 or the HuggingFace backend instead."
        )
    else:
        warnings.warn(f"vLLM not installed or import failed: {e}")

from .base import BaseBackend, BackendConfig, InferenceResult


class VLLMBackend(BaseBackend):
    """vLLM inference backend for high-throughput serving."""
    
    def __init__(self, config: BackendConfig):
        if not VLLM_AVAILABLE:
            if sys.platform == "win32":
                raise ImportError(
                    "vLLM is not supported on Windows.\n"
                    "vLLM requires Triton, which only has Linux wheels.\n"
                    "Options:\n"
                    "  1. Use WSL2 (Windows Subsystem for Linux)\n"
                    "  2. Use the HuggingFace backend (default)\n"
                    "  3. Run on a Linux system"
                )
            else:
                raise ImportError(
                    "vLLM is not installed. Install it with:\n"
                    "  pip install vllm\n"
                    "Or install impetus with vllm support:\n"
                    "  pip install impetus[vllm]"
                )
        
        super().__init__(config)
    
    def load_model(self):
        """Load model using vLLM."""
        if self.is_loaded:
            return
        
        # Configure vLLM
        tensor_parallel_size = self.config.backend_kwargs.get("tensor_parallel_size", 1)
        gpu_memory_utilization = self.config.backend_kwargs.get("gpu_memory_utilization", 0.9)
        
        # Determine dtype
        dtype_map = {
            "fp16": "float16",
            "float16": "float16",
            "bf16": "bfloat16",
            "bfloat16": "bfloat16",
            "fp32": "float32",
            "float32": "float32",
        }
        dtype = dtype_map.get(self.config.precision, "float16")
        
        # Initialize vLLM
        self.model = LLM(
            model=self.config.model_name,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
        )
        
        self.is_loaded = True
    
    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        **kwargs
    ) -> List[InferenceResult]:
        """Generate text using vLLM."""
        if not self.is_loaded:
            self.load_model()
        
        # Configure sampling params
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        
        # Generate
        start_time = time.time()
        outputs = self.model.generate(prompts, sampling_params)
        latency_ms = (time.time() - start_time) * 1000
        
        # Process results
        results = []
        for output in outputs:
            num_input_tokens = len(output.prompt_token_ids)
            num_output_tokens = len(output.outputs[0].token_ids)
            
            results.append(InferenceResult(
                output_ids=output.outputs[0].token_ids,
                num_input_tokens=num_input_tokens,
                num_output_tokens=num_output_tokens,
                latency_ms=latency_ms / len(prompts),
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
        """Run benchmark using vLLM."""
        if not self.is_loaded:
            self.load_model()
        
        # Use default prompts if none provided
        if prompts is None:
            prompts = ["This is a test prompt for benchmarking. " * 20] * batch_size
        
        sampling_params = SamplingParams(max_tokens=max_new_tokens, temperature=0.0)
        
        # Warmup
        for _ in range(warmup_runs):
            self.model.generate(prompts[:1], sampling_params)
        
        # Benchmark
        latencies = []
        total_input_tokens = 0
        total_output_tokens = 0
        
        for _ in range(num_runs):
            start_time = time.time()
            outputs = self.model.generate(prompts, sampling_params)
            latency_ms = (time.time() - start_time) * 1000
            latencies.append(latency_ms)
            
            for output in outputs:
                total_input_tokens += len(output.prompt_token_ids)
                total_output_tokens += len(output.outputs[0].token_ids)
        
        avg_latency = sum(latencies) / len(latencies)
        avg_output_tokens = total_output_tokens / num_runs
        throughput = (total_output_tokens / sum(latencies)) * 1000  # tokens/sec
        
        return {
            "backend": "vllm",
            "model_name": self.config.model_name,
            "precision": self.config.precision,
            "batch_size": batch_size,
            "sequence_length": total_input_tokens // (num_runs * batch_size),
            "max_new_tokens": max_new_tokens,
            "num_runs": num_runs,
            "avg_latency_ms": avg_latency,
            "std_latency_ms": (sum((l - avg_latency)**2 for l in latencies) / len(latencies)) ** 0.5,
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "avg_time_to_first_token_ms": 0,  # vLLM doesn't expose TTFT easily
            "time_per_output_token_ms": avg_latency / (avg_output_tokens / batch_size),
            "tokens_per_batch": avg_output_tokens,
            "throughput_tokens_per_sec": throughput,
            "throughput_batches_per_sec": 1000 / avg_latency,
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information from vLLM."""
        if not self.is_loaded:
            self.load_model()
        
        # vLLM doesn't expose model config easily, return basic info
        return {
            "backend": "vllm",
            "model_name": self.config.model_name,
            "precision": self.config.precision,
            "tensor_parallel_size": self.config.backend_kwargs.get("tensor_parallel_size", 1),
        }
    
    def cleanup(self):
        """Clean up vLLM resources."""
        if self.model is not None:
            del self.model
            self.model = None
        
        self.is_loaded = False
    
    def supports_distributed(self) -> bool:
        """vLLM supports tensor parallelism."""
        return True

