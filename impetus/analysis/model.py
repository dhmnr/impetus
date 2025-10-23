"""Model architecture analysis and layer-level profiling."""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import torch
import torch.nn as nn


@dataclass
class LayerMetrics:
    """Metrics for a single layer."""
    name: str
    type: str
    params: int
    memory_bytes: int
    flops: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelArchitectureInfo:
    """Complete model architecture information."""
    model_name: str
    architecture_type: str  # "decoder-only", "encoder-decoder", etc.
    
    # Parameter counts
    total_params: int
    trainable_params: int
    non_trainable_params: int
    
    # Model dimensions
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    intermediate_size: int
    vocab_size: int
    max_position_embeddings: int
    
    # Memory breakdown
    total_memory_bytes: int
    embedding_memory_bytes: int
    layer_memory_bytes: int
    head_memory_bytes: int
    
    # Layer-wise breakdown
    layers: List[LayerMetrics]
    
    # Estimated FLOPs
    estimated_flops_per_token: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "layers": [layer.to_dict() for layer in self.layers],
        }


class ModelAnalyzer:
    """Analyze model architecture and provide detailed breakdown."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.config = getattr(model, 'config', None)
    
    def analyze(self) -> ModelArchitectureInfo:
        """Perform complete model architecture analysis."""
        
        # Get model name
        model_name = self.config.name_or_path if self.config and hasattr(self.config, 'name_or_path') else "Unknown"
        
        # Detect architecture type
        architecture_type = self._detect_architecture_type()
        
        # Count parameters
        total_params, trainable_params, non_trainable_params = self._count_parameters()
        
        # Get model dimensions from config
        dimensions = self._extract_dimensions()
        
        # Analyze layers
        layers, memory_breakdown = self._analyze_layers()
        
        # Estimate FLOPs
        estimated_flops = self._estimate_flops_per_token()
        
        return ModelArchitectureInfo(
            model_name=model_name,
            architecture_type=architecture_type,
            total_params=total_params,
            trainable_params=trainable_params,
            non_trainable_params=non_trainable_params,
            num_layers=dimensions["num_layers"],
            hidden_size=dimensions["hidden_size"],
            num_attention_heads=dimensions["num_attention_heads"],
            intermediate_size=dimensions["intermediate_size"],
            vocab_size=dimensions["vocab_size"],
            max_position_embeddings=dimensions["max_position_embeddings"],
            total_memory_bytes=memory_breakdown["total"],
            embedding_memory_bytes=memory_breakdown["embedding"],
            layer_memory_bytes=memory_breakdown["layers"],
            head_memory_bytes=memory_breakdown["head"],
            layers=layers,
            estimated_flops_per_token=estimated_flops,
        )
    
    def _detect_architecture_type(self) -> str:
        """Detect model architecture type."""
        model_class = self.model.__class__.__name__.lower()
        
        if "causallm" in model_class or "gpt" in model_class or "llama" in model_class:
            return "decoder-only"
        elif "seq2seq" in model_class or "t5" in model_class or "bart" in model_class:
            return "encoder-decoder"
        elif "bert" in model_class or "roberta" in model_class:
            return "encoder-only"
        else:
            return "unknown"
    
    def _count_parameters(self) -> Tuple[int, int, int]:
        """Count total, trainable, and non-trainable parameters."""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        non_trainable = total - trainable
        
        return total, trainable, non_trainable
    
    def _extract_dimensions(self) -> Dict[str, int]:
        """Extract model dimensions from config."""
        dimensions = {
            "num_layers": 0,
            "hidden_size": 0,
            "num_attention_heads": 0,
            "intermediate_size": 0,
            "vocab_size": 0,
            "max_position_embeddings": 0,
        }
        
        if not self.config:
            return dimensions
        
        # Try common config attribute names
        dimensions["num_layers"] = getattr(
            self.config,
            "num_hidden_layers",
            getattr(self.config, "n_layer", getattr(self.config, "num_layers", 0))
        )
        
        dimensions["hidden_size"] = getattr(
            self.config,
            "hidden_size",
            getattr(self.config, "n_embd", getattr(self.config, "d_model", 0))
        )
        
        dimensions["num_attention_heads"] = getattr(
            self.config,
            "num_attention_heads",
            getattr(self.config, "n_head", 0)
        )
        
        dimensions["intermediate_size"] = getattr(
            self.config,
            "intermediate_size",
            getattr(self.config, "n_inner", dimensions["hidden_size"] * 4)
        )
        
        dimensions["vocab_size"] = getattr(self.config, "vocab_size", 0)
        
        dimensions["max_position_embeddings"] = getattr(
            self.config,
            "max_position_embeddings",
            getattr(self.config, "n_positions", 0)
        )
        
        return dimensions
    
    def _analyze_layers(self) -> Tuple[List[LayerMetrics], Dict[str, int]]:
        """Analyze individual layers and calculate memory breakdown."""
        layers = []
        
        embedding_memory = 0
        layer_memory = 0
        head_memory = 0
        
        for name, module in self.model.named_modules():
            # Skip parent modules that contain other modules
            if list(module.children()):
                continue
            
            # Calculate parameters and memory for this module
            num_params = sum(p.numel() for p in module.parameters(recurse=False))
            memory_bytes = sum(p.numel() * p.element_size() for p in module.parameters(recurse=False))
            
            if num_params == 0:
                continue
            
            module_type = module.__class__.__name__
            
            # Categorize memory
            if "embed" in name.lower():
                embedding_memory += memory_bytes
            elif "lm_head" in name.lower() or "output" in name.lower():
                head_memory += memory_bytes
            else:
                layer_memory += memory_bytes
            
            layers.append(LayerMetrics(
                name=name,
                type=module_type,
                params=num_params,
                memory_bytes=memory_bytes,
            ))
        
        memory_breakdown = {
            "total": embedding_memory + layer_memory + head_memory,
            "embedding": embedding_memory,
            "layers": layer_memory,
            "head": head_memory,
        }
        
        return layers, memory_breakdown
    
    def _estimate_flops_per_token(self) -> Optional[int]:
        """
        Estimate FLOPs per token for transformer models.
        
        For a transformer layer:
        - Attention: 4 * hidden_size^2 + 2 * hidden_size * seq_len
        - FFN: 8 * hidden_size * intermediate_size
        - Total per layer ≈ 4 * hidden_size^2 + 8 * hidden_size * intermediate_size
        """
        if not self.config:
            return None
        
        try:
            hidden_size = self._extract_dimensions()["hidden_size"]
            intermediate_size = self._extract_dimensions()["intermediate_size"]
            num_layers = self._extract_dimensions()["num_layers"]
            
            if hidden_size == 0 or num_layers == 0:
                return None
            
            # FLOPs per layer (per token)
            # Q, K, V projections: 3 * 2 * hidden_size^2
            qkv_flops = 6 * hidden_size * hidden_size
            
            # Attention scores: 2 * seq_len * hidden_size (approximation)
            # For per-token, we use hidden_size as approximation
            attention_flops = 4 * hidden_size * hidden_size
            
            # Output projection: 2 * hidden_size^2
            output_proj_flops = 2 * hidden_size * hidden_size
            
            # FFN: 2 linear layers
            ffn_flops = 2 * (2 * hidden_size * intermediate_size)
            
            # Total per layer
            flops_per_layer = qkv_flops + attention_flops + output_proj_flops + ffn_flops
            
            # Total for all layers
            total_flops = num_layers * flops_per_layer
            
            return total_flops
        except:
            return None
    
    def get_layer_breakdown_by_type(self) -> Dict[str, Dict[str, Any]]:
        """Group layers by type and provide aggregate statistics."""
        layers, _ = self._analyze_layers()
        
        type_breakdown = {}
        
        for layer in layers:
            layer_type = layer.type
            
            if layer_type not in type_breakdown:
                type_breakdown[layer_type] = {
                    "count": 0,
                    "total_params": 0,
                    "total_memory_bytes": 0,
                    "layers": []
                }
            
            type_breakdown[layer_type]["count"] += 1
            type_breakdown[layer_type]["total_params"] += layer.params
            type_breakdown[layer_type]["total_memory_bytes"] += layer.memory_bytes
            type_breakdown[layer_type]["layers"].append(layer.name)
        
        return type_breakdown
    
    def identify_bottleneck_layers(self, top_k: int = 10) -> List[LayerMetrics]:
        """Identify the largest layers by parameter count (potential bottlenecks)."""
        layers, _ = self._analyze_layers()
        
        # Sort by parameter count (descending)
        sorted_layers = sorted(layers, key=lambda l: l.params, reverse=True)
        
        return sorted_layers[:top_k]

