"""Comprehensive metrics dataclasses for profiling results."""

from dataclasses import dataclass, asdict, field
from typing import Dict, Any, List, Optional


@dataclass
class DecoderOnlyLanguageModelMetrics:
    """Legacy class for keeping track of Decoder LM Metrics (backward compatibility)."""

    seq_len: int
    batch_size: int
    latency: float
    tokens_per_batch: float
    throughput: float
    time_per_output_token: float
    time_to_first_token: float

    def __str__(self):
        return (
            f"  Sequence length: {self.seq_len}\n"
            f"  Batch_size: {self.batch_size}\n"
            f"  Latency: {self.latency:.4f} seconds\n"
            f"  Tokens generated per batch: {self.tokens_per_batch:.4f}\n"
            f"  Throughput: {self.throughput:.2f} tokens/second\n"
            f"  Time per output token: {self.time_per_output_token:.4f} seconds\n"
            f"  Time to first token: {self.time_to_first_token:.4f} seconds\n"
        )


@dataclass
class HardwareMetrics:
    """Hardware-level profiling metrics."""
    
    # GPU metrics (per GPU)
    gpu_metrics: List[Dict[str, Any]] = field(default_factory=list)
    
    # Topology information
    topology: Dict[str, Any] = field(default_factory=dict)
    
    # Theoretical performance
    theoretical_performance: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SystemMetrics:
    """System-wide metrics."""
    
    # CPU and memory
    cpu_percent: float = 0.0
    ram_used_gb: float = 0.0
    ram_total_gb: float = 0.0
    ram_percent: float = 0.0
    
    # I/O
    disk_read_mbps: float = 0.0
    disk_write_mbps: float = 0.0
    net_sent_mbps: float = 0.0
    net_recv_mbps: float = 0.0
    
    # System info
    platform: str = ""
    python_version: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryMetrics:
    """Memory profiling metrics."""
    
    # GPU memory (bytes)
    allocated_bytes: int = 0
    reserved_bytes: int = 0
    peak_allocated_bytes: int = 0
    peak_reserved_bytes: int = 0
    total_memory_bytes: int = 0
    
    # Model breakdown
    param_memory_bytes: int = 0
    estimated_kv_cache_bytes: int = 0
    estimated_activation_bytes: int = 0
    
    # Utilization
    memory_utilization_percent: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CUDAMetrics:
    """CUDA kernel-level metrics."""
    
    # Overall timing
    total_cuda_time_ms: float = 0.0
    kernel_launch_overhead_ms: float = 0.0
    
    # Top kernels (by time)
    top_kernels: List[Dict[str, Any]] = field(default_factory=list)
    
    # Memory operations
    memory_copy_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelMetrics:
    """Model architecture metrics."""
    
    model_name: str = ""
    architecture_type: str = ""
    total_params: int = 0
    trainable_params: int = 0
    
    # Model dimensions
    num_layers: int = 0
    hidden_size: int = 0
    num_attention_heads: int = 0
    vocab_size: int = 0
    
    # Memory breakdown
    param_memory_mb: float = 0.0
    embedding_memory_mb: float = 0.0
    
    # FLOPs
    estimated_flops_per_token: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InferenceMetrics:
    """Inference performance metrics."""
    
    # Basic metrics
    batch_size: int = 1
    sequence_length: int = 0
    max_new_tokens: int = 0
    num_runs: int = 0
    
    # Latency (milliseconds)
    avg_latency_ms: float = 0.0
    std_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    
    # Token generation
    avg_time_to_first_token_ms: float = 0.0
    time_per_output_token_ms: float = 0.0
    tokens_per_batch: float = 0.0
    
    # Throughput
    throughput_tokens_per_sec: float = 0.0
    throughput_batches_per_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DistributedMetrics:
    """Distributed system metrics (TP, PP, DP)."""
    
    # Configuration
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    
    # Communication metrics
    all_reduce_latency_ms: float = 0.0
    all_gather_latency_ms: float = 0.0
    p2p_latency_ms: float = 0.0
    
    # Bandwidth
    nvlink_bandwidth_gbps: float = 0.0
    network_bandwidth_gbps: float = 0.0
    
    # Efficiency
    communication_overhead_percent: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ComprehensiveBenchmarkResults:
    """Complete benchmark results with all metrics."""
    
    # Metadata
    backend: str = ""
    model_name: str = ""
    precision: str = ""
    timestamp: str = ""
    
    # Core metrics
    inference: InferenceMetrics = field(default_factory=InferenceMetrics)
    model: ModelMetrics = field(default_factory=ModelMetrics)
    hardware: HardwareMetrics = field(default_factory=HardwareMetrics)
    system: SystemMetrics = field(default_factory=SystemMetrics)
    memory: MemoryMetrics = field(default_factory=MemoryMetrics)
    cuda: CUDAMetrics = field(default_factory=CUDAMetrics)
    distributed: Optional[DistributedMetrics] = None
    
    # Analysis
    roofline_analysis: Dict[str, Any] = field(default_factory=dict)
    bottlenecks: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "backend": self.backend,
            "model_name": self.model_name,
            "precision": self.precision,
            "timestamp": self.timestamp,
            "inference": self.inference.to_dict(),
            "model": self.model.to_dict(),
            "hardware": self.hardware.to_dict(),
            "system": self.system.to_dict(),
            "memory": self.memory.to_dict(),
            "cuda": self.cuda.to_dict(),
            "roofline_analysis": self.roofline_analysis,
            "bottlenecks": self.bottlenecks,
            "recommendations": self.recommendations,
        }
        
        if self.distributed is not None:
            result["distributed"] = self.distributed.to_dict()
        
        return result
    
    def summary_text(self) -> str:
        """Generate a text summary of the benchmark results."""
        lines = [
            "=" * 80,
            f"BENCHMARK RESULTS - {self.model_name}",
            "=" * 80,
            "",
            f"Backend: {self.backend}",
            f"Precision: {self.precision}",
            f"Timestamp: {self.timestamp}",
            "",
            "INFERENCE METRICS",
            "-" * 80,
            f"  Batch Size: {self.inference.batch_size}",
            f"  Sequence Length: {self.inference.sequence_length}",
            f"  Avg Latency: {self.inference.avg_latency_ms:.2f} ms",
            f"  Time to First Token: {self.inference.avg_time_to_first_token_ms:.2f} ms",
            f"  Time per Output Token: {self.inference.time_per_output_token_ms:.2f} ms",
            f"  Throughput: {self.inference.throughput_tokens_per_sec:.2f} tokens/sec",
            "",
            "MODEL METRICS",
            "-" * 80,
            f"  Architecture: {self.model.architecture_type}",
            f"  Total Parameters: {self.model.total_params:,}",
            f"  Layers: {self.model.num_layers}",
            f"  Hidden Size: {self.model.hidden_size}",
            f"  Parameter Memory: {self.model.param_memory_mb:.2f} MB",
            "",
            "MEMORY METRICS",
            "-" * 80,
            f"  Allocated: {self.memory.allocated_bytes / (1024**3):.2f} GB",
            f"  Peak Allocated: {self.memory.peak_allocated_bytes / (1024**3):.2f} GB",
            f"  Total GPU Memory: {self.memory.total_memory_bytes / (1024**3):.2f} GB",
            f"  Utilization: {self.memory.memory_utilization_percent:.2f}%",
            "",
        ]
        
        if self.hardware.gpu_metrics:
            lines.extend([
                "HARDWARE METRICS",
                "-" * 80,
            ])
            for i, gpu in enumerate(self.hardware.gpu_metrics):
                lines.append(f"  GPU {i}: {gpu.get('name', 'Unknown')}")
                lines.append(f"    Utilization: {gpu.get('utilization_gpu', 0)}%")
                lines.append(f"    Temperature: {gpu.get('temperature', 0)}°C")
                lines.append(f"    Power: {gpu.get('power_draw', 0):.1f} W")
            lines.append("")
        
        if self.bottlenecks:
            lines.extend([
                "IDENTIFIED BOTTLENECKS",
                "-" * 80,
            ])
            for bottleneck in self.bottlenecks:
                lines.append(f"  • {bottleneck}")
            lines.append("")
        
        if self.recommendations:
            lines.extend([
                "RECOMMENDATIONS",
                "-" * 80,
            ])
            for rec in self.recommendations:
                lines.append(f"  • {rec}")
            lines.append("")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)
