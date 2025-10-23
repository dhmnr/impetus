"""Rich console output for beautiful terminal display."""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.tree import Tree
from rich.layout import Layout
from typing import Dict, Any, List
import time

from ..metrics import ComprehensiveBenchmarkResults


class ConsoleReporter:
    """Rich console reporter for beautiful terminal output."""
    
    def __init__(self, verbose: bool = False):
        self.console = Console()
        self.verbose = verbose
    
    def print_header(self, title: str):
        """Print a formatted header."""
        self.console.print(
            Panel(
                f"[bold cyan]{title}[/bold cyan]",
                style="bold white on blue"
            )
        )
    
    def print_section(self, title: str):
        """Print a section header."""
        self.console.print(f"\n[bold yellow]{title}[/bold yellow]")
        self.console.print("-" * 80)
    
    def print_gpu_info(self, gpu_info: List[Dict[str, Any]]):
        """Print GPU information in a table."""
        if not gpu_info:
            return
        
        self.print_section("GPU Information")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("GPU", style="cyan", width=5)
        table.add_column("Name", style="green")
        table.add_column("Memory", justify="right")
        table.add_column("Compute Cap", justify="center")
        table.add_column("SMs", justify="right")
        table.add_column("FP16 Peak", justify="right")
        table.add_column("Memory BW", justify="right")
        
        for gpu in gpu_info:
            table.add_row(
                str(gpu.get("index", "?")),
                gpu.get("name", "Unknown"),
                f"{gpu.get('total_memory', 0) / (1024**3):.1f} GB",
                f"{gpu.get('compute_capability', ('?', '?'))[0]}.{gpu.get('compute_capability', ('?', '?'))[1]}",
                str(gpu.get("multi_processor_count", 0)),
                f"{gpu.get('theoretical_fp16_tflops', 0):.1f} TF" if gpu.get('theoretical_fp16_tflops') else "N/A",
                f"{gpu.get('theoretical_memory_bandwidth_gbps', 0):.0f} GB/s" if gpu.get('theoretical_memory_bandwidth_gbps') else "N/A",
            )
        
        self.console.print(table)
    
    def print_system_metrics(self, system_metrics: Dict[str, Any]):
        """Print system metrics."""
        self.print_section("System Metrics")
        
        table = Table(show_header=False, box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Platform", system_metrics.get("platform", "Unknown"))
        table.add_row("Python Version", system_metrics.get("python_version", "Unknown"))
        table.add_row("CPU Usage", f"{system_metrics.get('cpu_percent', 0):.1f}%")
        table.add_row("RAM Used", f"{system_metrics.get('ram_used_gb', 0):.2f} / {system_metrics.get('ram_total_gb', 0):.2f} GB")
        table.add_row("RAM Usage", f"{system_metrics.get('ram_percent', 0):.1f}%")
        
        self.console.print(table)
    
    def print_inference_metrics(self, metrics: Dict[str, Any]):
        """Print inference performance metrics."""
        self.print_section("Inference Performance")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="green")
        table.add_column("Unit", style="dim")
        
        table.add_row("Batch Size", str(metrics.get("batch_size", 1)), "")
        table.add_row("Sequence Length", str(metrics.get("sequence_length", 0)), "tokens")
        table.add_row("Avg Latency", f"{metrics.get('avg_latency_ms', 0):.2f}", "ms")
        table.add_row("Std Latency", f"{metrics.get('std_latency_ms', 0):.2f}", "ms")
        table.add_row("Time to First Token", f"{metrics.get('avg_time_to_first_token_ms', 0):.2f}", "ms")
        table.add_row("Time per Token", f"{metrics.get('time_per_output_token_ms', 0):.2f}", "ms")
        table.add_row("Throughput", f"{metrics.get('throughput_tokens_per_sec', 0):.2f}", "tokens/sec")
        
        self.console.print(table)
    
    def print_memory_metrics(self, metrics: Dict[str, Any]):
        """Print memory usage metrics."""
        self.print_section("Memory Usage")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="green")
        
        table.add_row("Allocated", f"{metrics.get('allocated_bytes', 0) / (1024**3):.2f} GB")
        table.add_row("Reserved", f"{metrics.get('reserved_bytes', 0) / (1024**3):.2f} GB")
        table.add_row("Peak Allocated", f"{metrics.get('peak_allocated_bytes', 0) / (1024**3):.2f} GB")
        table.add_row("Total GPU Memory", f"{metrics.get('total_memory_bytes', 0) / (1024**3):.2f} GB")
        table.add_row("Utilization", f"{metrics.get('memory_utilization_percent', 0):.1f}%")
        
        self.console.print(table)
    
    def print_model_info(self, model_info: Dict[str, Any]):
        """Print model architecture information."""
        self.print_section("Model Architecture")
        
        table = Table(show_header=False, box=None)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Model Name", model_info.get("model_name", "Unknown"))
        table.add_row("Architecture", model_info.get("architecture_type", "Unknown"))
        table.add_row("Total Parameters", f"{model_info.get('total_params', 0):,}")
        table.add_row("Layers", str(model_info.get("num_layers", 0)))
        table.add_row("Hidden Size", str(model_info.get("hidden_size", 0)))
        table.add_row("Attention Heads", str(model_info.get("num_attention_heads", 0)))
        table.add_row("Vocab Size", str(model_info.get("vocab_size", 0)))
        table.add_row("Parameter Memory", f"{model_info.get('param_memory_mb', 0):.2f} MB")
        
        self.console.print(table)
    
    def print_comprehensive_results(self, results: ComprehensiveBenchmarkResults):
        """Print comprehensive benchmark results."""
        self.print_header(f"Benchmark Results - {results.model_name}")
        
        # Model info
        self.print_model_info(results.model.to_dict())
        
        # Hardware info
        if results.hardware.gpu_metrics:
            self.print_gpu_info(results.hardware.gpu_metrics)
        
        # System metrics
        self.print_system_metrics(results.system.to_dict())
        
        # Inference performance
        self.print_inference_metrics(results.inference.to_dict())
        
        # Memory usage
        self.print_memory_metrics(results.memory.to_dict())
        
        # CUDA metrics (if verbose)
        if self.verbose and results.cuda.top_kernels:
            self.print_section("Top CUDA Kernels")
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Kernel", style="cyan")
            table.add_column("Count", justify="right")
            table.add_column("Total Time", justify="right", style="yellow")
            table.add_column("Avg Time", justify="right", style="green")
            
            for kernel in results.cuda.top_kernels[:10]:
                table.add_row(
                    kernel.get("name", "")[:50],  # Truncate long names
                    str(kernel.get("count", 0)),
                    f"{kernel.get('total_time_us', 0) / 1000:.2f} ms",
                    f"{kernel.get('avg_time_us', 0) / 1000:.3f} ms",
                )
            
            self.console.print(table)
        
        # Bottlenecks
        if results.bottlenecks:
            self.print_section("Identified Bottlenecks")
            for bottleneck in results.bottlenecks:
                self.console.print(f"  [red]•[/red] {bottleneck}")
        
        # Recommendations
        if results.recommendations:
            self.print_section("Recommendations")
            for rec in results.recommendations:
                self.console.print(f"  [green]•[/green] {rec}")
        
        self.console.print("\n" + "═" * 80 + "\n")
    
    def create_progress_bar(self, description: str, total: int):
        """Create a progress bar for long-running operations."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        )
    
    def print_info(self, message: str):
        """Print info message."""
        self.console.print(f"[blue][i][/i][/blue] {message}")
    
    def print_success(self, message: str):
        """Print success message."""
        self.console.print(f"[green][OK][/green] {message}")
    
    def print_warning(self, message: str):
        """Print warning message."""
        self.console.print(f"[yellow][!][/yellow] {message}")
    
    def print_error(self, message: str):
        """Print error message."""
        self.console.print(f"[red][X][/red] {message}")

