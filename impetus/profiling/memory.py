"""Detailed memory profiling for GPU VRAM and model memory usage."""

import torch
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import gc


@dataclass
class MemorySnapshot:
    """Memory usage snapshot."""
    timestamp: float
    device: str
    
    # PyTorch memory stats
    allocated_bytes: int
    reserved_bytes: int
    active_bytes: int
    inactive_bytes: int
    
    # Detailed breakdown (if available)
    num_alloc_retries: int
    num_ooms: int
    max_split_size: int
    
    # Additional GPU memory info (from CUDA)
    total_memory: int
    free_memory: int
    used_memory: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelMemoryBreakdown:
    """Detailed breakdown of model memory usage."""
    
    # Weights
    total_params: int
    trainable_params: int
    param_memory_bytes: int
    
    # Estimated memory for different components
    estimated_activation_memory: int
    estimated_kv_cache_memory: int
    estimated_optimizer_memory: int
    
    # Per-layer breakdown
    layer_memory: Dict[str, int]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MemoryProfiler:
    """Profile GPU memory usage with detailed breakdown."""
    
    def __init__(self, device: Optional[torch.device] = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.is_cuda = self.device.type == "cuda"
        
        if self.is_cuda:
            # Reset peak memory stats
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.empty_cache()
    
    def get_current_snapshot(self) -> MemorySnapshot:
        """Get current memory usage snapshot."""
        import time
        
        if not self.is_cuda:
            return MemorySnapshot(
                timestamp=time.time(),
                device=str(self.device),
                allocated_bytes=0,
                reserved_bytes=0,
                active_bytes=0,
                inactive_bytes=0,
                num_alloc_retries=0,
                num_ooms=0,
                max_split_size=0,
                total_memory=0,
                free_memory=0,
                used_memory=0,
            )
        
        # Get PyTorch memory stats
        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        
        # Get detailed stats
        stats = torch.cuda.memory_stats(self.device)
        active = stats.get("active_bytes.all.current", 0)
        inactive = stats.get("inactive_split_bytes.all.current", 0)
        num_retries = stats.get("num_alloc_retries", 0)
        num_ooms = stats.get("num_ooms", 0)
        max_split = stats.get("max_split_size", 0)
        
        # Get GPU memory info
        total_mem = torch.cuda.get_device_properties(self.device).total_memory
        free_mem, total_mem_check = torch.cuda.mem_get_info(self.device)
        used_mem = total_mem - free_mem
        
        return MemorySnapshot(
            timestamp=time.time(),
            device=str(self.device),
            allocated_bytes=allocated,
            reserved_bytes=reserved,
            active_bytes=active,
            inactive_bytes=inactive,
            num_alloc_retries=num_retries,
            num_ooms=num_ooms,
            max_split_size=max_split,
            total_memory=total_mem,
            free_memory=free_mem,
            used_memory=used_mem,
        )
    
    def get_peak_memory_stats(self) -> Dict[str, int]:
        """Get peak memory usage statistics."""
        if not self.is_cuda:
            return {}
        
        return {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device),
        }
    
    def analyze_model_memory(self, model) -> ModelMemoryBreakdown:
        """Analyze memory usage of a model."""
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Calculate parameter memory (assuming fp16/bf16 or fp32)
        param_bytes = 0
        for p in model.parameters():
            param_bytes += p.numel() * p.element_size()
        
        # Estimate activation memory (rough estimate)
        # This is very approximate and depends on the actual forward pass
        estimated_activation = param_bytes  # Very rough estimate
        
        # Estimate KV cache memory for transformer models
        estimated_kv_cache = 0
        try:
            if hasattr(model.config, 'num_hidden_layers') and hasattr(model.config, 'hidden_size'):
                # Rough KV cache estimate for transformers
                # KV cache = 2 (K and V) * num_layers * hidden_size * sequence_length * batch_size
                # We'll estimate for a typical scenario
                num_layers = model.config.num_hidden_layers
                hidden_size = model.config.hidden_size
                # Assume seq_len=2048, batch=1, dtype=fp16
                estimated_kv_cache = 2 * num_layers * hidden_size * 2048 * 1 * 2
        except:
            pass
        
        # Estimate optimizer memory (if training)
        estimated_optimizer = param_bytes * 2  # Rough estimate for Adam (params + gradients)
        
        # Per-layer memory breakdown
        layer_memory = {}
        for name, module in model.named_modules():
            module_params = sum(p.numel() * p.element_size() for p in module.parameters(recurse=False))
            if module_params > 0:
                layer_memory[name] = module_params
        
        return ModelMemoryBreakdown(
            total_params=total_params,
            trainable_params=trainable_params,
            param_memory_bytes=param_bytes,
            estimated_activation_memory=estimated_activation,
            estimated_kv_cache_memory=estimated_kv_cache,
            estimated_optimizer_memory=estimated_optimizer,
            layer_memory=layer_memory,
        )
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get a comprehensive memory summary."""
        snapshot = self.get_current_snapshot()
        peak_stats = self.get_peak_memory_stats()
        
        summary = snapshot.to_dict()
        summary.update(peak_stats)
        
        # Add human-readable sizes
        for key in ['allocated_bytes', 'reserved_bytes', 'total_memory', 'used_memory', 'free_memory']:
            if key in summary:
                mb = summary[key] / (1024 * 1024)
                gb = mb / 1024
                summary[f"{key}_mb"] = mb
                summary[f"{key}_gb"] = gb
        
        # Calculate utilization
        if summary['total_memory'] > 0:
            summary['memory_utilization_percent'] = (summary['used_memory'] / summary['total_memory']) * 100
        
        return summary
    
    def estimate_kv_cache_size(
        self, 
        num_layers: int,
        hidden_size: int,
        num_attention_heads: int,
        sequence_length: int,
        batch_size: int = 1,
        dtype_bytes: int = 2,  # fp16/bf16
    ) -> int:
        """
        Estimate KV cache memory size for transformer models.
        
        KV cache stores keys and values for each attention head in each layer.
        Size = 2 (K and V) * num_layers * batch_size * num_heads * seq_len * head_dim * dtype_bytes
        """
        head_dim = hidden_size // num_attention_heads
        kv_cache_bytes = (
            2 *  # K and V
            num_layers *
            batch_size *
            num_attention_heads *
            sequence_length *
            head_dim *
            dtype_bytes
        )
        return kv_cache_bytes
    
    def reset_peak_stats(self):
        """Reset peak memory statistics."""
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
    
    def clear_cache(self):
        """Clear GPU memory cache."""
        if self.is_cuda:
            torch.cuda.empty_cache()
            gc.collect()

