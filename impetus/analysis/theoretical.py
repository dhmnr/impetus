"""Theoretical performance calculations and roofline model analysis."""

from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import math


@dataclass
class TheoreticalPerformance:
    """Theoretical peak performance metrics."""
    
    # Required fields first (no defaults)
    # Compute (TFLOPS)
    peak_fp32_tflops: float
    peak_fp16_tflops: float
    peak_bf16_tflops: float
    
    # Memory bandwidth (GB/s)
    peak_memory_bandwidth_gbps: float
    
    # Architecture details
    compute_capability: str
    sm_count: int
    tensor_core_available: bool
    
    # Optional fields (with defaults)
    peak_fp8_tflops: Optional[float] = None
    peak_int8_tops: Optional[float] = None
    
    # Interconnect bandwidth (GB/s)
    nvlink_bandwidth_gbps: Optional[float] = None
    pcie_bandwidth_gbps: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RooflineAnalysis:
    """Roofline model analysis results."""
    
    # Operational intensity (FLOPs per byte)
    operational_intensity: float
    
    # Performance bounds
    memory_bandwidth_bound_tflops: float
    compute_bound_tflops: float
    
    # Actual performance
    actual_tflops: float
    
    # Bottleneck identification
    is_memory_bound: bool
    is_compute_bound: bool
    
    # Efficiency metrics
    compute_efficiency_percent: float
    memory_efficiency_percent: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TheoreticalAnalyzer:
    """Calculate theoretical peak performance and analyze roofline model."""
    
    # Known GPU specifications (approximate values)
    GPU_SPECS = {
        # NVIDIA Data Center GPUs
        "A100-SXM4-40GB": {
            "fp32_tflops": 19.5,
            "fp16_tflops": 312,
            "bf16_tflops": 312,
            "memory_bw": 1555,
            "nvlink_bw": 600,  # NVLink 3.0, 12 links
            "pcie_gen": 4,
            "sm_count": 108,
            "tensor_cores": True,
        },
        "A100-SXM4-80GB": {
            "fp32_tflops": 19.5,
            "fp16_tflops": 312,
            "bf16_tflops": 312,
            "memory_bw": 2039,
            "nvlink_bw": 600,
            "pcie_gen": 4,
            "sm_count": 108,
            "tensor_cores": True,
        },
        "H100-SXM5-80GB": {
            "fp32_tflops": 67,
            "fp16_tflops": 989,
            "bf16_tflops": 989,
            "fp8_tflops": 1979,
            "memory_bw": 3350,
            "nvlink_bw": 900,  # NVLink 4.0, 18 links
            "pcie_gen": 5,
            "sm_count": 132,
            "tensor_cores": True,
        },
        "V100-SXM2-16GB": {
            "fp32_tflops": 15.7,
            "fp16_tflops": 125,
            "memory_bw": 900,
            "nvlink_bw": 300,  # NVLink 2.0, 6 links
            "pcie_gen": 3,
            "sm_count": 80,
            "tensor_cores": True,
        },
        
        # NVIDIA Professional GPUs
        "L40": {
            "fp32_tflops": 90.5,
            "fp16_tflops": 181,
            "bf16_tflops": 181,
            "memory_bw": 864,
            "pcie_gen": 4,
            "sm_count": 142,
            "tensor_cores": True,
        },
        "A40": {
            "fp32_tflops": 37.4,
            "fp16_tflops": 149.7,
            "memory_bw": 696,
            "pcie_gen": 4,
            "sm_count": 84,
            "tensor_cores": True,
        },
        
        # NVIDIA Consumer GPUs
        "RTX 4090": {
            "fp32_tflops": 82.6,
            "fp16_tflops": 165.2,
            "memory_bw": 1008,
            "pcie_gen": 4,
            "sm_count": 128,
            "tensor_cores": True,
        },
        "RTX 3090": {
            "fp32_tflops": 35.6,
            "fp16_tflops": 142,
            "memory_bw": 936,
            "pcie_gen": 4,
            "sm_count": 82,
            "tensor_cores": True,
        },
    }
    
    def __init__(self, gpu_name: str, compute_capability: Tuple[int, int]):
        self.gpu_name = gpu_name
        self.compute_capability = compute_capability
        self.specs = self._get_gpu_specs(gpu_name)
    
    def _get_gpu_specs(self, gpu_name: str) -> Dict[str, Any]:
        """Get GPU specifications, matching against known GPUs."""
        # Try exact match first
        for known_gpu, specs in self.GPU_SPECS.items():
            if known_gpu in gpu_name:
                return specs
        
        # Try partial matches
        for known_gpu, specs in self.GPU_SPECS.items():
            gpu_model = known_gpu.split("-")[0]  # e.g., "A100" from "A100-SXM4-40GB"
            if gpu_model in gpu_name:
                return specs
        
        # Return empty dict if unknown
        return {}
    
    def get_theoretical_performance(self) -> TheoreticalPerformance:
        """Get theoretical peak performance for the GPU."""
        
        if not self.specs:
            # Return placeholder values if GPU is unknown
            return TheoreticalPerformance(
                peak_fp32_tflops=0.0,
                peak_fp16_tflops=0.0,
                peak_bf16_tflops=0.0,
                peak_memory_bandwidth_gbps=0.0,
                compute_capability=f"{self.compute_capability[0]}.{self.compute_capability[1]}",
                sm_count=0,
                tensor_core_available=self.compute_capability[0] >= 7,
            )
        
        # PCIe bandwidth calculation
        pcie_gen = self.specs.get("pcie_gen", 3)
        pcie_bandwidth = self._calculate_pcie_bandwidth(pcie_gen)
        
        return TheoreticalPerformance(
            peak_fp32_tflops=self.specs.get("fp32_tflops", 0.0),
            peak_fp16_tflops=self.specs.get("fp16_tflops", 0.0),
            peak_bf16_tflops=self.specs.get("bf16_tflops", 0.0),
            peak_fp8_tflops=self.specs.get("fp8_tflops"),
            peak_int8_tops=self.specs.get("int8_tops"),
            peak_memory_bandwidth_gbps=self.specs.get("memory_bw", 0.0),
            nvlink_bandwidth_gbps=self.specs.get("nvlink_bw"),
            pcie_bandwidth_gbps=pcie_bandwidth,
            compute_capability=f"{self.compute_capability[0]}.{self.compute_capability[1]}",
            sm_count=self.specs.get("sm_count", 0),
            tensor_core_available=self.specs.get("tensor_cores", False),
        )
    
    def _calculate_pcie_bandwidth(self, gen: int, lanes: int = 16) -> float:
        """Calculate PCIe bandwidth in GB/s."""
        # PCIe bandwidth per lane per generation (GB/s)
        bandwidth_per_lane = {
            3: 0.985,   # PCIe 3.0: ~1 GB/s per lane
            4: 1.969,   # PCIe 4.0: ~2 GB/s per lane
            5: 3.938,   # PCIe 5.0: ~4 GB/s per lane
        }
        
        bw_per_lane = bandwidth_per_lane.get(gen, 0.985)
        return bw_per_lane * lanes
    
    def analyze_roofline(
        self,
        actual_tflops: float,
        flops: int,
        memory_bytes: int,
        dtype: str = "fp16"
    ) -> RooflineAnalysis:
        """
        Perform roofline model analysis.
        
        Args:
            actual_tflops: Measured performance in TFLOPS
            flops: Number of floating point operations
            memory_bytes: Amount of memory transferred in bytes
            dtype: Data type used (fp16, bf16, fp32)
        """
        theoretical = self.get_theoretical_performance()
        
        # Get peak compute for the dtype
        if dtype == "fp32":
            peak_compute = theoretical.peak_fp32_tflops
        elif dtype in ["fp16", "bf16"]:
            peak_compute = max(theoretical.peak_fp16_tflops, theoretical.peak_bf16_tflops)
        else:
            peak_compute = theoretical.peak_fp16_tflops
        
        # Calculate operational intensity (FLOPs per byte)
        operational_intensity = flops / memory_bytes if memory_bytes > 0 else 0
        
        # Memory bandwidth bound (TFLOPS)
        # Performance = Memory Bandwidth (GB/s) * Operational Intensity (FLOPs/byte) / 1000
        memory_bound = (theoretical.peak_memory_bandwidth_gbps * operational_intensity) / 1000.0
        
        # Compute bound (TFLOPS)
        compute_bound = peak_compute
        
        # Determine bottleneck
        is_memory_bound = memory_bound < compute_bound
        is_compute_bound = not is_memory_bound
        
        # Calculate efficiency
        if peak_compute > 0:
            compute_efficiency = (actual_tflops / peak_compute) * 100
        else:
            compute_efficiency = 0.0
        
        if memory_bound > 0:
            memory_efficiency = (actual_tflops / memory_bound) * 100 if is_memory_bound else 100.0
        else:
            memory_efficiency = 0.0
        
        return RooflineAnalysis(
            operational_intensity=operational_intensity,
            memory_bandwidth_bound_tflops=memory_bound,
            compute_bound_tflops=compute_bound,
            actual_tflops=actual_tflops,
            is_memory_bound=is_memory_bound,
            is_compute_bound=is_compute_bound,
            compute_efficiency_percent=min(compute_efficiency, 100.0),
            memory_efficiency_percent=min(memory_efficiency, 100.0),
        )
    
    def estimate_communication_time(
        self,
        data_size_bytes: int,
        communication_type: str,
        num_gpus: int = 1
    ) -> float:
        """
        Estimate communication time for distributed operations.
        
        Args:
            data_size_bytes: Size of data to transfer
            communication_type: "nvlink", "pcie", "network"
            num_gpus: Number of GPUs involved
        
        Returns:
            Estimated time in seconds
        """
        theoretical = self.get_theoretical_performance()
        
        if communication_type == "nvlink" and theoretical.nvlink_bandwidth_gbps:
            # NVLink bandwidth (full duplex)
            bandwidth_gbps = theoretical.nvlink_bandwidth_gbps
        elif communication_type == "pcie" and theoretical.pcie_bandwidth_gbps:
            # PCIe bandwidth
            bandwidth_gbps = theoretical.pcie_bandwidth_gbps
        elif communication_type == "network":
            # Assume InfiniBand or high-speed Ethernet (100-200 Gbps)
            bandwidth_gbps = 100.0
        else:
            # Default to PCIe
            bandwidth_gbps = 16.0
        
        # Account for collective operation overhead (e.g., all-reduce)
        # For ring all-reduce: time = 2 * (N-1) / N * data_size / bandwidth
        if num_gpus > 1:
            collective_overhead = 2 * (num_gpus - 1) / num_gpus
        else:
            collective_overhead = 1.0
        
        data_size_gb = data_size_bytes / (1024**3)
        time_seconds = (data_size_gb / bandwidth_gbps) * collective_overhead
        
        return time_seconds
    
    def calculate_model_flops(
        self,
        num_params: int,
        sequence_length: int,
        batch_size: int = 1,
        is_training: bool = False
    ) -> int:
        """
        Estimate FLOPs for transformer model inference/training.
        
        For transformers: FLOPs ≈ 2 * num_params * sequence_length * batch_size
        (factor of 2 for forward pass, multiply by 3 for training with backward pass)
        """
        forward_flops = 2 * num_params * sequence_length * batch_size
        
        if is_training:
            # Training includes forward + backward (2x forward) + optimizer step
            total_flops = forward_flops * 3
        else:
            total_flops = forward_flops
        
        return total_flops

