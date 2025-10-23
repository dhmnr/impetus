"""CUDA kernel-level profiling using PyTorch profiler."""

import torch
import torch.cuda.profiler as profiler
from torch.profiler import profile, ProfilerActivity, record_function
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
import time


@dataclass
class KernelMetrics:
    """Metrics for a single CUDA kernel."""
    name: str
    count: int
    total_time_us: float
    avg_time_us: float
    min_time_us: float
    max_time_us: float
    cuda_time_us: float
    self_cuda_time_us: float
    occupancy_percent: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CUDAProfilingResults:
    """Results from CUDA profiling session."""
    total_time_ms: float
    cuda_time_ms: float
    cpu_time_ms: float
    
    # Kernel-level metrics
    kernels: List[KernelMetrics]
    
    # Memory operations
    memory_copy_time_ms: float
    memory_alloc_count: int
    memory_free_count: int
    
    # Aggregate statistics
    total_kernel_time_ms: float
    kernel_launch_overhead_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_time_ms": self.total_time_ms,
            "cuda_time_ms": self.cuda_time_ms,
            "cpu_time_ms": self.cpu_time_ms,
            "kernels": [k.to_dict() for k in self.kernels],
            "memory_copy_time_ms": self.memory_copy_time_ms,
            "memory_alloc_count": self.memory_alloc_count,
            "memory_free_count": self.memory_free_count,
            "total_kernel_time_ms": self.total_kernel_time_ms,
            "kernel_launch_overhead_ms": self.kernel_launch_overhead_ms,
        }


class CUDAProfiler:
    """Profile CUDA kernel execution and performance."""
    
    def __init__(self, device: Optional[torch.device] = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.is_cuda = self.device.type == "cuda"
        self.profiling_enabled = self.is_cuda
    
    def profile_function(
        self,
        func: Callable,
        *args,
        record_shapes: bool = True,
        with_stack: bool = False,
        **kwargs
    ) -> tuple[Any, CUDAProfilingResults]:
        """
        Profile a function's CUDA kernel execution.
        
        Returns: (function_result, profiling_results)
        """
        if not self.is_cuda:
            result = func(*args, **kwargs)
            return result, None
        
        # Warm-up and synchronization
        torch.cuda.synchronize(self.device)
        
        # Profile the function
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=record_shapes,
            with_stack=with_stack,
            profile_memory=True,
        ) as prof:
            result = func(*args, **kwargs)
            torch.cuda.synchronize(self.device)
        
        # Parse profiling results
        profiling_results = self._parse_profile(prof)
        
        return result, profiling_results
    
    def _parse_profile(self, prof) -> CUDAProfilingResults:
        """Parse PyTorch profiler results into structured metrics."""
        
        # Get key averages
        key_averages = prof.key_averages()
        
        # Aggregate metrics
        total_cpu_time = 0
        total_cuda_time = 0
        total_self_cuda_time = 0
        memory_copy_time = 0
        memory_alloc_count = 0
        memory_free_count = 0
        
        # Kernel-level metrics
        kernel_metrics = {}
        
        for event in key_averages:
            # Aggregate times (convert from microseconds to milliseconds)
            total_cpu_time += event.cpu_time_total / 1000.0
            total_cuda_time += event.cuda_time_total / 1000.0
            total_self_cuda_time += event.self_cuda_time_total / 1000.0
            
            # Track memory operations
            if "Memcpy" in event.key or "MemCpy" in event.key:
                memory_copy_time += event.cuda_time_total / 1000.0
            if "cudaMalloc" in event.key or "Malloc" in event.key:
                memory_alloc_count += event.count
            if "cudaFree" in event.key or "Free" in event.key:
                memory_free_count += event.count
            
            # Track kernel metrics
            if event.device_type == torch.profiler.DeviceType.CUDA and event.is_legacy:
                kernel_name = event.key
                
                if kernel_name not in kernel_metrics:
                    kernel_metrics[kernel_name] = {
                        "count": 0,
                        "total_time": 0,
                        "min_time": float('inf'),
                        "max_time": 0,
                        "cuda_time": 0,
                        "self_cuda_time": 0,
                    }
                
                km = kernel_metrics[kernel_name]
                km["count"] += event.count
                km["total_time"] += event.cuda_time_total
                km["cuda_time"] += event.cuda_time_total
                km["self_cuda_time"] += event.self_cuda_time_total
                
                # Note: min/max per-call time is approximate here
                avg_time = event.cuda_time_total / event.count if event.count > 0 else 0
                km["min_time"] = min(km["min_time"], avg_time)
                km["max_time"] = max(km["max_time"], avg_time)
        
        # Convert kernel metrics to structured format
        kernels = []
        for name, metrics in kernel_metrics.items():
            avg_time = metrics["total_time"] / metrics["count"] if metrics["count"] > 0 else 0
            min_time = metrics["min_time"] if metrics["min_time"] != float('inf') else 0
            
            kernels.append(KernelMetrics(
                name=name,
                count=metrics["count"],
                total_time_us=metrics["total_time"],
                avg_time_us=avg_time,
                min_time_us=min_time,
                max_time_us=metrics["max_time"],
                cuda_time_us=metrics["cuda_time"],
                self_cuda_time_us=metrics["self_cuda_time"],
            ))
        
        # Sort kernels by total time (descending)
        kernels.sort(key=lambda k: k.total_time_us, reverse=True)
        
        # Calculate kernel launch overhead (approximate)
        total_kernel_time = sum(k.total_time_us for k in kernels) / 1000.0  # to ms
        kernel_launch_overhead = max(0, total_cuda_time - total_kernel_time)
        
        return CUDAProfilingResults(
            total_time_ms=(total_cpu_time + total_cuda_time),
            cuda_time_ms=total_cuda_time,
            cpu_time_ms=total_cpu_time,
            kernels=kernels,
            memory_copy_time_ms=memory_copy_time,
            memory_alloc_count=memory_alloc_count,
            memory_free_count=memory_free_count,
            total_kernel_time_ms=total_kernel_time,
            kernel_launch_overhead_ms=kernel_launch_overhead,
        )
    
    def measure_kernel_time(self, func: Callable, *args, num_runs: int = 10, warmup: int = 2, **kwargs) -> Dict[str, float]:
        """
        Measure kernel execution time using CUDA events.
        More accurate than Python time.time() for GPU operations.
        """
        if not self.is_cuda:
            return {}
        
        times = []
        
        # Warm-up runs
        for _ in range(warmup):
            func(*args, **kwargs)
            torch.cuda.synchronize(self.device)
        
        # Timed runs
        for _ in range(num_runs):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            func(*args, **kwargs)
            end_event.record()
            
            torch.cuda.synchronize(self.device)
            elapsed_time = start_event.elapsed_time(end_event)  # milliseconds
            times.append(elapsed_time)
        
        return {
            "mean_ms": sum(times) / len(times),
            "min_ms": min(times),
            "max_ms": max(times),
            "std_ms": (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5,
            "median_ms": sorted(times)[len(times)//2],
        }
    
    def get_cuda_properties(self) -> Dict[str, Any]:
        """Get CUDA device properties."""
        if not self.is_cuda:
            return {}
        
        props = torch.cuda.get_device_properties(self.device)
        
        return {
            "name": props.name,
            "major": props.major,
            "minor": props.minor,
            "compute_capability": f"{props.major}.{props.minor}",
            "total_memory_gb": props.total_memory / (1024**3),
            "multi_processor_count": props.multi_processor_count,
            "max_threads_per_multi_processor": props.max_threads_per_multi_processor,
            "max_threads_per_block": props.max_threads_per_block,
            "max_block_dim": props.max_block_dim,
            "max_grid_dim": props.max_grid_dim,
            "warp_size": 32,  # Always 32 on NVIDIA GPUs
            "is_multi_gpu_board": props.is_multi_gpu_board,
        }
    
    def estimate_occupancy(
        self,
        threads_per_block: int,
        registers_per_thread: int,
        shared_memory_per_block: int
    ) -> float:
        """
        Estimate theoretical occupancy for a kernel configuration.
        
        Note: This is a simplified estimate. For accurate occupancy,
        use NVIDIA Nsight Compute.
        """
        if not self.is_cuda:
            return 0.0
        
        props = torch.cuda.get_device_properties(self.device)
        
        max_threads_per_sm = props.max_threads_per_multi_processor
        max_blocks_per_sm = max_threads_per_sm // threads_per_block
        
        # Simplified occupancy calculation
        # In reality, this depends on register usage, shared memory, and other factors
        theoretical_occupancy = (threads_per_block * max_blocks_per_sm) / max_threads_per_sm
        
        return min(theoretical_occupancy * 100, 100.0)  # Return as percentage
    
    def benchmark_memory_bandwidth(self, size_mb: int = 100) -> Dict[str, float]:
        """Benchmark memory bandwidth (HtoD, DtoH, DtoD)."""
        if not self.is_cuda:
            return {}
        
        size_bytes = size_mb * 1024 * 1024
        num_elements = size_bytes // 4  # float32
        
        # Host to Device
        h2d_times = []
        for _ in range(5):
            data_cpu = torch.randn(num_elements)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            data_gpu = data_cpu.to(self.device)
            end.record()
            torch.cuda.synchronize()
            
            h2d_times.append(start.elapsed_time(end) / 1000.0)  # seconds
        
        # Device to Host
        d2h_times = []
        data_gpu = torch.randn(num_elements, device=self.device)
        for _ in range(5):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            data_cpu = data_gpu.cpu()
            end.record()
            torch.cuda.synchronize()
            
            d2h_times.append(start.elapsed_time(end) / 1000.0)  # seconds
        
        # Device to Device
        d2d_times = []
        src = torch.randn(num_elements, device=self.device)
        for _ in range(5):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            dst = src.clone()
            end.record()
            torch.cuda.synchronize()
            
            d2d_times.append(start.elapsed_time(end) / 1000.0)  # seconds
        
        size_gb = size_bytes / (1024**3)
        
        return {
            "h2d_bandwidth_gbps": size_gb / (sum(h2d_times) / len(h2d_times)),
            "d2h_bandwidth_gbps": size_gb / (sum(d2h_times) / len(d2h_times)),
            "d2d_bandwidth_gbps": size_gb / (sum(d2d_times) / len(d2d_times)),
        }

