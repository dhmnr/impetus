"""Distributed system profiling for TP, PP, DP communication patterns."""

import torch
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class CommunicationMetrics:
    """Metrics for distributed communication operations."""
    
    operation_type: str  # "all_reduce", "all_gather", "send", "recv", etc.
    data_size_bytes: int
    latency_ms: float
    bandwidth_gbps: float
    participants: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DistributedTopology:
    """Distributed system topology information."""
    
    world_size: int
    local_world_size: int
    num_nodes: int
    rank: int
    local_rank: int
    
    # Parallelism configuration
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    
    # Interconnect info
    has_nvlink: bool = False
    has_infiniband: bool = False
    network_topology: str = "unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DistributedProfiler:
    """Profile distributed training/inference communication patterns."""
    
    def __init__(self):
        self.is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        
        if self.is_distributed:
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        else:
            self.world_size = 1
            self.rank = 0
            self.local_rank = 0
    
    def get_topology(self) -> DistributedTopology:
        """Get distributed system topology information."""
        
        if not self.is_distributed:
            return DistributedTopology(
                world_size=1,
                local_world_size=1,
                num_nodes=1,
                rank=0,
                local_rank=0,
            )
        
        # Detect local world size (GPUs per node)
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", torch.cuda.device_count()))
        num_nodes = self.world_size // local_world_size
        
        # Detect interconnect (simplified detection)
        has_nvlink = self._detect_nvlink()
        has_infiniband = self._detect_infiniband()
        
        if has_infiniband:
            network_topology = "infiniband"
        elif self.world_size > local_world_size:
            network_topology = "ethernet"
        else:
            network_topology = "single_node"
        
        return DistributedTopology(
            world_size=self.world_size,
            local_world_size=local_world_size,
            num_nodes=num_nodes,
            rank=self.rank,
            local_rank=self.local_rank,
            has_nvlink=has_nvlink,
            has_infiniband=has_infiniband,
            network_topology=network_topology,
        )
    
    def _detect_nvlink(self) -> bool:
        """Detect if NVLink is available."""
        if not torch.cuda.is_available():
            return False
        
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            # Check if any NVLink is active
            for link in range(6):  # NVLink typically has up to 6 links
                try:
                    state = pynvml.nvmlDeviceGetNvLinkState(handle, link)
                    if state == pynvml.NVML_FEATURE_ENABLED:
                        return True
                except:
                    break
        except:
            pass
        
        return False
    
    def _detect_infiniband(self) -> bool:
        """Detect if InfiniBand is available."""
        import os
        import subprocess
        
        try:
            # Check for IB devices
            result = subprocess.run(
                ["ibstat"],
                capture_output=True,
                text=True,
                timeout=1
            )
            return result.returncode == 0 and "State: Active" in result.stdout
        except:
            pass
        
        # Check for environment variables
        return "NCCL_IB_DISABLE" not in os.environ or os.environ.get("NCCL_IB_DISABLE") == "0"
    
    def profile_all_reduce(
        self,
        tensor_size: int = 1024 * 1024,  # 1MB default
        num_iterations: int = 10,
        warmup: int = 2,
    ) -> CommunicationMetrics:
        """
        Profile all-reduce operation performance.
        
        Args:
            tensor_size: Size of tensor in elements (fp32)
            num_iterations: Number of iterations to measure
            warmup: Number of warmup iterations
        
        Returns:
            Communication metrics
        """
        if not self.is_distributed:
            return CommunicationMetrics(
                operation_type="all_reduce",
                data_size_bytes=0,
                latency_ms=0,
                bandwidth_gbps=0,
                participants=1,
            )
        
        # Create tensor
        tensor = torch.randn(tensor_size, device='cuda')
        data_size_bytes = tensor.numel() * tensor.element_size()
        
        # Warmup
        for _ in range(warmup):
            torch.distributed.all_reduce(tensor)
            torch.cuda.synchronize()
        
        # Measure
        timings = []
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start = time.time()
            
            torch.distributed.all_reduce(tensor)
            
            torch.cuda.synchronize()
            elapsed = time.time() - start
            timings.append(elapsed * 1000)  # Convert to ms
        
        avg_latency_ms = sum(timings) / len(timings)
        
        # Calculate bandwidth
        # For ring all-reduce: 2 * (N-1) / N * data_size
        alg_bandwidth_factor = 2 * (self.world_size - 1) / self.world_size
        data_transferred_gb = (data_size_bytes * alg_bandwidth_factor) / (1024**3)
        bandwidth_gbps = data_transferred_gb / (avg_latency_ms / 1000)
        
        return CommunicationMetrics(
            operation_type="all_reduce",
            data_size_bytes=data_size_bytes,
            latency_ms=avg_latency_ms,
            bandwidth_gbps=bandwidth_gbps,
            participants=self.world_size,
        )
    
    def profile_all_gather(
        self,
        tensor_size: int = 1024 * 1024,
        num_iterations: int = 10,
        warmup: int = 2,
    ) -> CommunicationMetrics:
        """Profile all-gather operation performance."""
        if not self.is_distributed:
            return CommunicationMetrics(
                operation_type="all_gather",
                data_size_bytes=0,
                latency_ms=0,
                bandwidth_gbps=0,
                participants=1,
            )
        
        # Create tensors
        tensor = torch.randn(tensor_size, device='cuda')
        output_tensors = [torch.zeros_like(tensor) for _ in range(self.world_size)]
        data_size_bytes = tensor.numel() * tensor.element_size()
        
        # Warmup
        for _ in range(warmup):
            torch.distributed.all_gather(output_tensors, tensor)
            torch.cuda.synchronize()
        
        # Measure
        timings = []
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start = time.time()
            
            torch.distributed.all_gather(output_tensors, tensor)
            
            torch.cuda.synchronize()
            elapsed = time.time() - start
            timings.append(elapsed * 1000)
        
        avg_latency_ms = sum(timings) / len(timings)
        
        # Bandwidth calculation
        data_transferred_gb = (data_size_bytes * (self.world_size - 1)) / (1024**3)
        bandwidth_gbps = data_transferred_gb / (avg_latency_ms / 1000)
        
        return CommunicationMetrics(
            operation_type="all_gather",
            data_size_bytes=data_size_bytes,
            latency_ms=avg_latency_ms,
            bandwidth_gbps=bandwidth_gbps,
            participants=self.world_size,
        )
    
    def profile_p2p(
        self,
        tensor_size: int = 1024 * 1024,
        num_iterations: int = 10,
        warmup: int = 2,
    ) -> Optional[CommunicationMetrics]:
        """Profile point-to-point send/recv operations."""
        if not self.is_distributed or self.world_size < 2:
            return None
        
        tensor = torch.randn(tensor_size, device='cuda')
        data_size_bytes = tensor.numel() * tensor.element_size()
        
        # Only measure between rank 0 and 1
        if self.rank > 1:
            return None
        
        # Warmup
        for _ in range(warmup):
            if self.rank == 0:
                torch.distributed.send(tensor, dst=1)
            else:
                torch.distributed.recv(tensor, src=0)
            torch.cuda.synchronize()
        
        # Measure
        timings = []
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start = time.time()
            
            if self.rank == 0:
                torch.distributed.send(tensor, dst=1)
            else:
                torch.distributed.recv(tensor, src=0)
            
            torch.cuda.synchronize()
            elapsed = time.time() - start
            timings.append(elapsed * 1000)
        
        avg_latency_ms = sum(timings) / len(timings)
        
        # Bandwidth
        data_transferred_gb = data_size_bytes / (1024**3)
        bandwidth_gbps = data_transferred_gb / (avg_latency_ms / 1000)
        
        return CommunicationMetrics(
            operation_type="p2p",
            data_size_bytes=data_size_bytes,
            latency_ms=avg_latency_ms,
            bandwidth_gbps=bandwidth_gbps,
            participants=2,
        )
    
    def comprehensive_profile(self) -> Dict[str, Any]:
        """Run comprehensive distributed profiling."""
        results = {
            "topology": self.get_topology().to_dict(),
            "communication_ops": {},
        }
        
        if not self.is_distributed:
            return results
        
        # Profile different sizes
        sizes = [1024, 1024*1024, 16*1024*1024]  # 4KB, 4MB, 64MB
        
        for size in sizes:
            size_label = f"{size * 4 // 1024}KB" if size < 1024*1024 else f"{size * 4 // (1024*1024)}MB"
            
            # All-reduce
            metrics = self.profile_all_reduce(tensor_size=size, num_iterations=10)
            results["communication_ops"][f"all_reduce_{size_label}"] = metrics.to_dict()
            
            # All-gather
            metrics = self.profile_all_gather(tensor_size=size, num_iterations=10)
            results["communication_ops"][f"all_gather_{size_label}"] = metrics.to_dict()
        
        # P2P (only one size)
        if self.rank <= 1:
            metrics = self.profile_p2p(tensor_size=1024*1024, num_iterations=10)
            if metrics:
                results["communication_ops"]["p2p_4MB"] = metrics.to_dict()
        
        return results


# Add missing import
import os

