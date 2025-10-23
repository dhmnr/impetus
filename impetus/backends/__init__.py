"""Inference backend implementations for different engines."""

from .base import BaseBackend, BackendConfig
from .huggingface import HuggingFaceBackend

__all__ = [
    "BaseBackend",
    "BackendConfig",
    "HuggingFaceBackend",
]


def get_backend(backend_name: str, **kwargs) -> BaseBackend:
    """Factory function to get a backend by name."""
    backends = {
        "huggingface": HuggingFaceBackend,
        "hf": HuggingFaceBackend,
    }
    
    # Try to import optional backends
    try:
        from .vllm import VLLMBackend
        backends["vllm"] = VLLMBackend
    except ImportError:
        pass
    
    try:
        from .sglang import SGLangBackend
        backends["sglang"] = SGLangBackend
    except ImportError:
        pass
    
    try:
        from .tensorrt import TensorRTBackend
        backends["tensorrt"] = TensorRTBackend
        backends["trt"] = TensorRTBackend
    except ImportError:
        pass
    
    backend_class = backends.get(backend_name.lower())
    if backend_class is None:
        available = ", ".join(backends.keys())
        raise ValueError(
            f"Unknown backend: {backend_name}. Available backends: {available}"
        )
    
    return backend_class(**kwargs)

