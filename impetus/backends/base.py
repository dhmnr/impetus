"""Base backend interface for inference engines."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import torch


@dataclass
class BackendConfig:
    """Configuration for inference backend."""
    model_name: str
    device: torch.device
    precision: str  # "4bit", "8bit", "fp16", "bf16", "fp32"
    max_batch_size: int = 1
    max_sequence_length: int = 2048
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    
    # Backend-specific options
    backend_kwargs: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.backend_kwargs is None:
            self.backend_kwargs = {}


class InferenceResult:
    """Result from an inference run."""
    def __init__(
        self,
        output_ids: torch.Tensor,
        num_input_tokens: int,
        num_output_tokens: int,
        latency_ms: float,
        time_to_first_token_ms: Optional[float] = None,
    ):
        self.output_ids = output_ids
        self.num_input_tokens = num_input_tokens
        self.num_output_tokens = num_output_tokens
        self.latency_ms = latency_ms
        self.time_to_first_token_ms = time_to_first_token_ms
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_input_tokens": self.num_input_tokens,
            "num_output_tokens": self.num_output_tokens,
            "latency_ms": self.latency_ms,
            "time_to_first_token_ms": self.time_to_first_token_ms,
        }


class BaseBackend(ABC):
    """Abstract base class for inference backends."""
    
    def __init__(self, config: BackendConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
    
    @abstractmethod
    def load_model(self):
        """Load the model and tokenizer."""
        pass
    
    @abstractmethod
    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        **kwargs
    ) -> List[InferenceResult]:
        """
        Generate text from prompts.
        
        Returns:
            List of InferenceResult objects containing generated tokens and metrics.
        """
        pass
    
    @abstractmethod
    def benchmark_inference(
        self,
        prompts: List[str],
        max_new_tokens: int = 100,
        num_runs: int = 10,
        warmup_runs: int = 2,
    ) -> Dict[str, Any]:
        """
        Run benchmark on the model.
        
        Returns:
            Dictionary containing benchmark metrics.
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model architecture info, parameter counts, etc.
        """
        pass
    
    @abstractmethod
    def cleanup(self):
        """Clean up resources (unload model, free memory, etc.)."""
        pass
    
    def supports_distributed(self) -> bool:
        """Check if backend supports distributed inference."""
        return False
    
    def get_backend_name(self) -> str:
        """Get the name of the backend."""
        return self.__class__.__name__
    
    def __enter__(self):
        """Context manager entry."""
        self.load_model()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()

