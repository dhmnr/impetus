"""System-wide profiling (CPU, memory, power, thermal, I/O)."""

import psutil
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import platform


@dataclass
class SystemMetrics:
    """System-wide metrics snapshot."""
    timestamp: float
    
    # CPU
    cpu_percent: float
    cpu_count_physical: int
    cpu_count_logical: int
    cpu_freq_current: float  # MHz
    cpu_freq_max: float  # MHz
    
    # Memory
    ram_total: int  # bytes
    ram_used: int
    ram_available: int
    ram_percent: float
    swap_total: int
    swap_used: int
    swap_percent: float
    
    # Disk I/O (cumulative)
    disk_read_bytes: int
    disk_write_bytes: int
    disk_read_count: int
    disk_write_count: int
    
    # Network I/O (cumulative)
    net_sent_bytes: int
    net_recv_bytes: int
    net_sent_packets: int
    net_recv_packets: int
    
    # System info
    platform: str
    python_version: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SystemProfiler:
    """Profile system-wide resources and performance."""
    
    def __init__(self):
        self.initial_disk_io = psutil.disk_io_counters()
        self.initial_net_io = psutil.net_io_counters()
        self.start_time = time.time()
    
    def get_current_metrics(self) -> SystemMetrics:
        """Get current system metrics snapshot."""
        
        # CPU info
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_count_physical = psutil.cpu_count(logical=False) or 0
        cpu_count_logical = psutil.cpu_count(logical=True) or 0
        
        cpu_freq = psutil.cpu_freq()
        cpu_freq_current = cpu_freq.current if cpu_freq else 0
        cpu_freq_max = cpu_freq.max if cpu_freq else 0
        
        # Memory info
        ram = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        disk_read_bytes = disk_io.read_bytes if disk_io else 0
        disk_write_bytes = disk_io.write_bytes if disk_io else 0
        disk_read_count = disk_io.read_count if disk_io else 0
        disk_write_count = disk_io.write_count if disk_io else 0
        
        # Network I/O
        net_io = psutil.net_io_counters()
        net_sent_bytes = net_io.bytes_sent if net_io else 0
        net_recv_bytes = net_io.bytes_recv if net_io else 0
        net_sent_packets = net_io.packets_sent if net_io else 0
        net_recv_packets = net_io.packets_recv if net_io else 0
        
        return SystemMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            cpu_count_physical=cpu_count_physical,
            cpu_count_logical=cpu_count_logical,
            cpu_freq_current=cpu_freq_current,
            cpu_freq_max=cpu_freq_max,
            ram_total=ram.total,
            ram_used=ram.used,
            ram_available=ram.available,
            ram_percent=ram.percent,
            swap_total=swap.total,
            swap_used=swap.used,
            swap_percent=swap.percent,
            disk_read_bytes=disk_read_bytes,
            disk_write_bytes=disk_write_bytes,
            disk_read_count=disk_read_count,
            disk_write_count=disk_write_count,
            net_sent_bytes=net_sent_bytes,
            net_recv_bytes=net_recv_bytes,
            net_sent_packets=net_sent_packets,
            net_recv_packets=net_recv_packets,
            platform=platform.platform(),
            python_version=platform.python_version(),
        )
    
    def get_delta_metrics(self, current: SystemMetrics) -> Dict[str, Any]:
        """Calculate delta metrics since profiler initialization."""
        
        delta = {
            "disk_read_delta_mb": (current.disk_read_bytes - self.initial_disk_io.read_bytes) / (1024 * 1024)
                if self.initial_disk_io else 0,
            "disk_write_delta_mb": (current.disk_write_bytes - self.initial_disk_io.write_bytes) / (1024 * 1024)
                if self.initial_disk_io else 0,
            "net_sent_delta_mb": (current.net_sent_bytes - self.initial_net_io.bytes_sent) / (1024 * 1024)
                if self.initial_net_io else 0,
            "net_recv_delta_mb": (current.net_recv_bytes - self.initial_net_io.bytes_recv) / (1024 * 1024)
                if self.initial_net_io else 0,
            "elapsed_time_seconds": current.timestamp - self.start_time,
        }
        
        # Calculate rates
        elapsed = delta["elapsed_time_seconds"]
        if elapsed > 0:
            delta["disk_read_mbps"] = delta["disk_read_delta_mb"] / elapsed
            delta["disk_write_mbps"] = delta["disk_write_delta_mb"] / elapsed
            delta["net_sent_mbps"] = delta["net_sent_delta_mb"] / elapsed
            delta["net_recv_mbps"] = delta["net_recv_delta_mb"] / elapsed
        
        return delta
    
    def get_process_metrics(self, pid: Optional[int] = None) -> Dict[str, Any]:
        """Get metrics for a specific process (default: current process)."""
        try:
            process = psutil.Process(pid) if pid else psutil.Process()
            
            with process.oneshot():
                cpu_percent = process.cpu_percent()
                memory_info = process.memory_info()
                io_counters = process.io_counters() if hasattr(process, 'io_counters') else None
                num_threads = process.num_threads()
                num_fds = process.num_fds() if hasattr(process, 'num_fds') else None
                
                return {
                    "pid": process.pid,
                    "name": process.name(),
                    "cpu_percent": cpu_percent,
                    "memory_rss_mb": memory_info.rss / (1024 * 1024),
                    "memory_vms_mb": memory_info.vms / (1024 * 1024),
                    "num_threads": num_threads,
                    "num_fds": num_fds,
                    "io_read_bytes": io_counters.read_bytes if io_counters else 0,
                    "io_write_bytes": io_counters.write_bytes if io_counters else 0,
                }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return {}

