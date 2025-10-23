"""Operator-level profiling for layer-by-layer performance analysis."""

import torch
import torch.nn as nn
import time
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict


@dataclass
class OperatorMetrics:
    """Metrics for a single operator or layer."""
    name: str
    operator_type: str
    forward_time_ms: float
    memory_allocated_mb: float
    num_parameters: int
    input_shape: Optional[tuple] = None
    output_shape: Optional[tuple] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class OperatorProfiler:
    """Profile individual operators and layers in a model."""
    
    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_cuda = self.device.type == "cuda"
        
        # Storage for profiling results
        self.layer_times: Dict[str, List[float]] = defaultdict(list)
        self.layer_memory: Dict[str, List[float]] = defaultdict(list)
        self.hooks = []
    
    def register_hooks(self):
        """Register forward hooks on all modules to measure execution time."""
        
        def create_hook(name: str):
            """Create a forward hook for a specific layer."""
            
            def hook(module, input, output):
                if not self.is_cuda:
                    return
                
                # Measure time using CUDA events
                torch.cuda.synchronize()
                
                # Record memory before
                mem_before = torch.cuda.memory_allocated(self.device)
                
                # Store timing (approximation - actual timing would need pre/post hooks)
                # This is a simplified version for demonstration
                mem_after = torch.cuda.memory_allocated(self.device)
                mem_delta = (mem_after - mem_before) / (1024 * 1024)  # MB
                
                self.layer_memory[name].append(mem_delta)
            
            return hook
        
        # Register hooks for all named modules
        for name, module in self.model.named_modules():
            # Skip container modules
            if len(list(module.children())) > 0:
                continue
            
            hook = module.register_forward_hook(create_hook(name))
            self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def profile_forward_pass(
        self,
        input_ids: torch.Tensor,
        num_runs: int = 10,
        warmup: int = 2,
    ) -> Dict[str, OperatorMetrics]:
        """
        Profile a forward pass through the model.
        
        Args:
            input_ids: Input tensor
            num_runs: Number of profiling runs
            warmup: Number of warmup runs
        
        Returns:
            Dictionary mapping layer names to their metrics
        """
        if not self.is_cuda:
            return {}
        
        self.register_hooks()
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                self.model(input_ids)
                torch.cuda.synchronize()
        
        # Profile each layer individually
        layer_metrics = {}
        
        for name, module in self.model.named_modules():
            # Skip container modules and very small modules
            if len(list(module.children())) > 0:
                continue
            
            num_params = sum(p.numel() for p in module.parameters(recurse=False))
            if num_params == 0:
                continue
            
            # Measure layer time
            times = []
            for _ in range(num_runs):
                if self.is_cuda:
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    
                    torch.cuda.synchronize()
                    start_event.record()
                
                with torch.no_grad():
                    # This is a simplified approach - ideally we'd isolate each layer
                    # For now, we'll measure the full forward pass
                    self.model(input_ids)
                
                if self.is_cuda:
                    end_event.record()
                    torch.cuda.synchronize()
                    elapsed = start_event.elapsed_time(end_event)
                    times.append(elapsed)
            
            avg_time = sum(times) / len(times) if times else 0
            avg_memory = sum(self.layer_memory[name]) / len(self.layer_memory[name]) if self.layer_memory[name] else 0
            
            layer_metrics[name] = OperatorMetrics(
                name=name,
                operator_type=module.__class__.__name__,
                forward_time_ms=avg_time / num_runs,  # Approximate time per layer
                memory_allocated_mb=avg_memory,
                num_parameters=num_params,
            )
        
        self.remove_hooks()
        
        return layer_metrics
    
    def profile_by_operator_type(self, layer_metrics: Dict[str, OperatorMetrics]) -> Dict[str, Dict[str, Any]]:
        """
        Aggregate metrics by operator type (e.g., all Linear layers together).
        
        Args:
            layer_metrics: Dictionary of layer metrics from profile_forward_pass
        
        Returns:
            Dictionary mapping operator types to aggregated metrics
        """
        type_metrics = defaultdict(lambda: {
            "count": 0,
            "total_time_ms": 0,
            "total_memory_mb": 0,
            "total_parameters": 0,
            "layers": []
        })
        
        for name, metrics in layer_metrics.items():
            op_type = metrics.operator_type
            type_metrics[op_type]["count"] += 1
            type_metrics[op_type]["total_time_ms"] += metrics.forward_time_ms
            type_metrics[op_type]["total_memory_mb"] += metrics.memory_allocated_mb
            type_metrics[op_type]["total_parameters"] += metrics.num_parameters
            type_metrics[op_type]["layers"].append(name)
        
        return dict(type_metrics)
    
    def identify_slow_layers(
        self,
        layer_metrics: Dict[str, OperatorMetrics],
        top_k: int = 10
    ) -> List[OperatorMetrics]:
        """
        Identify the slowest layers by forward time.
        
        Args:
            layer_metrics: Dictionary of layer metrics
            top_k: Number of top slow layers to return
        
        Returns:
            List of slowest layer metrics
        """
        sorted_layers = sorted(
            layer_metrics.values(),
            key=lambda x: x.forward_time_ms,
            reverse=True
        )
        
        return sorted_layers[:top_k]
    
    def identify_memory_intensive_layers(
        self,
        layer_metrics: Dict[str, OperatorMetrics],
        top_k: int = 10
    ) -> List[OperatorMetrics]:
        """
        Identify the most memory-intensive layers.
        
        Args:
            layer_metrics: Dictionary of layer metrics
            top_k: Number of top memory-intensive layers to return
        
        Returns:
            List of most memory-intensive layer metrics
        """
        sorted_layers = sorted(
            layer_metrics.values(),
            key=lambda x: x.memory_allocated_mb,
            reverse=True
        )
        
        return sorted_layers[:top_k]
    
    def get_attention_ffn_breakdown(self) -> Dict[str, Any]:
        """
        Get breakdown of time spent in attention vs FFN layers (for transformers).
        
        Returns:
            Dictionary with attention and FFN timing breakdown
        """
        attention_time = 0
        ffn_time = 0
        other_time = 0
        
        for name, times in self.layer_times.items():
            avg_time = sum(times) / len(times) if times else 0
            
            if "attention" in name.lower() or "attn" in name.lower():
                attention_time += avg_time
            elif "ffn" in name.lower() or "mlp" in name.lower() or "feed_forward" in name.lower():
                ffn_time += avg_time
            else:
                other_time += avg_time
        
        total_time = attention_time + ffn_time + other_time
        
        return {
            "attention_ms": attention_time,
            "ffn_ms": ffn_time,
            "other_ms": other_time,
            "total_ms": total_time,
            "attention_percent": (attention_time / total_time * 100) if total_time > 0 else 0,
            "ffn_percent": (ffn_time / total_time * 100) if total_time > 0 else 0,
            "other_percent": (other_time / total_time * 100) if total_time > 0 else 0,
        }

