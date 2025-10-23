"""Hardware profiling using NVIDIA Management Library (NVML)."""

import torch
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import logging

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    logging.warning("pynvml not available. Hardware profiling will be limited.")


@dataclass
class GPUInfo:
    """Information about a single GPU."""
    index: int
    name: str
    architecture: str
    compute_capability: tuple
    total_memory: int  # bytes
    pci_bus_id: str
    driver_version: str
    cuda_version: str
    multi_processor_count: int
    max_threads_per_multiprocessor: int
    max_threads_per_block: int
    clock_rate: int  # MHz
    memory_clock_rate: int  # MHz
    memory_bus_width: int  # bits
    l2_cache_size: int  # bytes
    
    # Theoretical capabilities
    theoretical_fp16_tflops: Optional[float] = None
    theoretical_fp32_tflops: Optional[float] = None
    theoretical_memory_bandwidth_gbps: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GPUMetrics:
    """Real-time metrics for a single GPU."""
    index: int
    temperature: int  # Celsius
    power_draw: float  # Watts
    power_limit: float  # Watts
    utilization_gpu: int  # Percent
    utilization_memory: int  # Percent
    memory_used: int  # bytes
    memory_free: int  # bytes
    memory_total: int  # bytes
    sm_clock: int  # MHz
    memory_clock: int  # MHz
    pcie_link_gen: int
    pcie_link_width: int
    pcie_tx_throughput: int  # KB/s
    pcie_rx_throughput: int  # KB/s
    
    # NVLink info (if available)
    nvlink_count: int = 0
    nvlink_total_tx_bytes: int = 0
    nvlink_total_rx_bytes: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HardwareProfiler:
    """Profile NVIDIA GPU hardware using NVML."""
    
    def __init__(self):
        self.nvml_initialized = False
        self.gpu_count = 0
        self.gpu_handles = []
        self.gpu_info: List[GPUInfo] = []
        
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.nvml_initialized = True
                self.gpu_count = pynvml.nvmlDeviceGetCount()
                
                # Get handles for all GPUs
                for i in range(self.gpu_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    self.gpu_handles.append(handle)
                    self.gpu_info.append(self._get_gpu_info(handle, i))
                    
                logging.info(f"NVML initialized. Found {self.gpu_count} GPU(s)")
            except Exception as e:
                logging.error(f"Failed to initialize NVML: {e}")
                self.nvml_initialized = False
        else:
            logging.warning("NVML not available. Using fallback methods.")
            # Fallback to PyTorch
            if torch.cuda.is_available():
                self.gpu_count = torch.cuda.device_count()
                self.gpu_info = [self._get_gpu_info_fallback(i) for i in range(self.gpu_count)]
    
    def _get_gpu_info(self, handle, index: int) -> GPUInfo:
        """Get static information about a GPU using NVML."""
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode('utf-8')
            
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pci_info = pynvml.nvmlDeviceGetPciInfo(handle)
        
        # Get compute capability
        try:
            major = pynvml.nvmlDeviceGetCudaComputeCapability(handle)[0]
            minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)[1]
            compute_capability = (major, minor)
        except:
            compute_capability = (0, 0)
        
        # Get architecture name from compute capability
        arch_map = {
            (7, 0): "Volta", (7, 5): "Turing",
            (8, 0): "Ampere", (8, 6): "Ampere", (8, 9): "Ada Lovelace",
            (9, 0): "Hopper",
        }
        architecture = arch_map.get(compute_capability, "Unknown")
        
        # Get various properties
        try:
            multi_processor_count = pynvml.nvmlDeviceGetNumGpuCores(handle)
        except:
            try:
                multi_processor_count = pynvml.nvmlDeviceGetMultiProcessorCount(handle)
            except:
                multi_processor_count = 0
        
        try:
            max_threads_per_mp = pynvml.nvmlDeviceGetMaxThreadsPerMultiProcessor(handle)
        except:
            max_threads_per_mp = 0
            
        try:
            max_threads_per_block = pynvml.nvmlDeviceGetMaxThreadsPerBlock(handle)
        except:
            max_threads_per_block = 0
        
        try:
            clock_rate = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_SM)
        except:
            clock_rate = 0
            
        try:
            memory_clock = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_MEM)
        except:
            memory_clock = 0
        
        try:
            memory_bus_width = pynvml.nvmlDeviceGetMemoryBusWidth(handle)
        except:
            memory_bus_width = 0
            
        try:
            l2_cache_size = pynvml.nvmlDeviceGetL2CacheSize(handle)
        except:
            l2_cache_size = 0
        
        driver_version = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(driver_version, bytes):
            driver_version = driver_version.decode('utf-8')
            
        cuda_version = pynvml.nvmlSystemGetCudaDriverVersion()
        cuda_version_str = f"{cuda_version // 1000}.{(cuda_version % 1000) // 10}"
        
        pci_bus_id = pci_info.busId
        if isinstance(pci_bus_id, bytes):
            pci_bus_id = pci_bus_id.decode('utf-8')
        
        # Calculate theoretical performance
        theoretical_fp16, theoretical_fp32, theoretical_bw = self._calculate_theoretical_performance(
            name, architecture, compute_capability, multi_processor_count, 
            clock_rate, memory_clock, memory_bus_width
        )
        
        return GPUInfo(
            index=index,
            name=name,
            architecture=architecture,
            compute_capability=compute_capability,
            total_memory=memory_info.total,
            pci_bus_id=pci_bus_id,
            driver_version=driver_version,
            cuda_version=cuda_version_str,
            multi_processor_count=multi_processor_count,
            max_threads_per_multiprocessor=max_threads_per_mp,
            max_threads_per_block=max_threads_per_block,
            clock_rate=clock_rate,
            memory_clock_rate=memory_clock,
            memory_bus_width=memory_bus_width,
            l2_cache_size=l2_cache_size,
            theoretical_fp16_tflops=theoretical_fp16,
            theoretical_fp32_tflops=theoretical_fp32,
            theoretical_memory_bandwidth_gbps=theoretical_bw,
        )
    
    def _get_gpu_info_fallback(self, index: int) -> GPUInfo:
        """Get GPU info using PyTorch when NVML is not available."""
        props = torch.cuda.get_device_properties(index)
        
        arch_map = {
            (7, 0): "Volta", (7, 5): "Turing",
            (8, 0): "Ampere", (8, 6): "Ampere", (8, 9): "Ada Lovelace",
            (9, 0): "Hopper",
        }
        architecture = arch_map.get((props.major, props.minor), "Unknown")
        
        return GPUInfo(
            index=index,
            name=props.name,
            architecture=architecture,
            compute_capability=(props.major, props.minor),
            total_memory=props.total_memory,
            pci_bus_id=f"0000:{index:02d}:00.0",
            driver_version="Unknown",
            cuda_version=torch.version.cuda or "Unknown",
            multi_processor_count=props.multi_processor_count,
            max_threads_per_multiprocessor=props.max_threads_per_multi_processor,
            max_threads_per_block=props.max_threads_per_block,
            clock_rate=0,
            memory_clock_rate=0,
            memory_bus_width=0,
            l2_cache_size=0,
        )
    
    def _calculate_theoretical_performance(
        self, name: str, arch: str, compute_cap: tuple,
        sm_count: int, clock_mhz: int, mem_clock_mhz: int, mem_bus_width: int
    ) -> tuple[Optional[float], Optional[float], Optional[float]]:
        """Calculate theoretical peak performance based on GPU specs."""
        
        # Known specs for common GPUs (approximate values)
        known_specs = {
            # Format: (fp16_tflops, fp32_tflops, memory_bandwidth_gbps)
            "A100": (312, 19.5, 1935),
            "H100": (989, 67, 3350),
            "V100": (125, 15.7, 900),
            "RTX 4090": (165, 82.6, 1008),
            "RTX 3090": (142, 35.6, 936),
            "L40": (181, 90.5, 864),
            "A40": (75, 37.4, 696),
        }
        
        # Try to match known GPU
        for gpu_name, specs in known_specs.items():
            if gpu_name in name:
                return specs
        
        # Estimate if we have enough info
        fp16_tflops = None
        fp32_tflops = None
        mem_bandwidth_gbps = None
        
        if sm_count > 0 and clock_mhz > 0:
            # Very rough estimation based on architecture
            if arch == "Ampere":
                # Ampere has 128 FP32 cores per SM
                fp32_tflops = (sm_count * 128 * clock_mhz * 2) / 1e6  # 2 for FMA
                fp16_tflops = fp32_tflops * 2  # Tensor cores can be much higher
            elif arch == "Hopper":
                fp32_tflops = (sm_count * 128 * clock_mhz * 2) / 1e6
                fp16_tflops = fp32_tflops * 2
        
        if mem_clock_mhz > 0 and mem_bus_width > 0:
            # Memory bandwidth = clock * bus_width * 2 (DDR) / 8 (bits to bytes)
            mem_bandwidth_gbps = (mem_clock_mhz * 1e6 * mem_bus_width * 2) / (8 * 1e9)
        
        return fp16_tflops, fp32_tflops, mem_bandwidth_gbps
    
    def get_current_metrics(self, gpu_index: Optional[int] = None) -> List[GPUMetrics]:
        """Get current real-time metrics for GPU(s)."""
        if not self.nvml_initialized:
            return []
        
        indices = [gpu_index] if gpu_index is not None else range(self.gpu_count)
        metrics = []
        
        for idx in indices:
            handle = self.gpu_handles[idx]
            
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except:
                temp = 0
            
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
            except:
                power = 0.0
                
            try:
                power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
            except:
                power_limit = 0.0
            
            try:
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                util_gpu = utilization.gpu
                util_mem = utilization.memory
            except:
                util_gpu = 0
                util_mem = 0
            
            try:
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                mem_used = memory.used
                mem_free = memory.free
                mem_total = memory.total
            except:
                mem_used = mem_free = mem_total = 0
            
            try:
                sm_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
            except:
                sm_clock = 0
                
            try:
                mem_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
            except:
                mem_clock = 0
            
            try:
                pcie_gen = pynvml.nvmlDeviceGetCurrPcieLinkGeneration(handle)
                pcie_width = pynvml.nvmlDeviceGetCurrPcieLinkWidth(handle)
            except:
                pcie_gen = 0
                pcie_width = 0
            
            try:
                pcie_tx = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_TX_BYTES)
                pcie_rx = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_RX_BYTES)
            except:
                pcie_tx = pcie_rx = 0
            
            # NVLink info
            nvlink_count = 0
            nvlink_tx = 0
            nvlink_rx = 0
            try:
                for link in range(pynvml.NVML_NVLINK_MAX_LINKS):
                    try:
                        state = pynvml.nvmlDeviceGetNvLinkState(handle, link)
                        if state == pynvml.NVML_FEATURE_ENABLED:
                            nvlink_count += 1
                            tx_bytes = pynvml.nvmlDeviceGetNvLinkUtilizationCounter(
                                handle, link, 0  # TX counter
                            )[0]
                            rx_bytes = pynvml.nvmlDeviceGetNvLinkUtilizationCounter(
                                handle, link, 1  # RX counter
                            )[0]
                            nvlink_tx += tx_bytes
                            nvlink_rx += rx_bytes
                    except:
                        break
            except:
                pass
            
            metrics.append(GPUMetrics(
                index=idx,
                temperature=temp,
                power_draw=power,
                power_limit=power_limit,
                utilization_gpu=util_gpu,
                utilization_memory=util_mem,
                memory_used=mem_used,
                memory_free=mem_free,
                memory_total=mem_total,
                sm_clock=sm_clock,
                memory_clock=mem_clock,
                pcie_link_gen=pcie_gen,
                pcie_link_width=pcie_width,
                pcie_tx_throughput=pcie_tx,
                pcie_rx_throughput=pcie_rx,
                nvlink_count=nvlink_count,
                nvlink_total_tx_bytes=nvlink_tx,
                nvlink_total_rx_bytes=nvlink_rx,
            ))
        
        return metrics
    
    def get_topology_info(self) -> Dict[str, Any]:
        """Get GPU topology information (NVLink connections, PCIe switches, etc.)."""
        if not self.nvml_initialized:
            return {}
        
        topology = {
            "gpu_count": self.gpu_count,
            "nvlink_topology": {},
            "pcie_switches": {},
        }
        
        # NVLink topology
        for i in range(self.gpu_count):
            handle_i = self.gpu_handles[i]
            connections = []
            
            for j in range(self.gpu_count):
                if i == j:
                    continue
                handle_j = self.gpu_handles[j]
                
                try:
                    # Check if there's an NVLink between GPUs
                    path = pynvml.nvmlDeviceGetTopologyCommonAncestor(handle_i, handle_j)
                    if path == pynvml.NVML_TOPOLOGY_NVLINK:
                        connections.append(j)
                except:
                    pass
            
            topology["nvlink_topology"][i] = connections
        
        return topology
    
    def __del__(self):
        """Cleanup NVML."""
        if self.nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except:
                pass

