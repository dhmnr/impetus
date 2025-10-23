"""Main CLI entry point for Impetus."""

import click
import torch
import sys

from .benchmark import ComprehensiveBenchmark
from .profiling import HardwareProfiler, SystemProfiler
from .reporting import ConsoleReporter


@click.group()
@click.version_option(version="0.2.0")
def cli():
    """
    Impetus - Comprehensive GPU profiling and characterization tool for LLM workloads.
    
    Profile your hardware from single GPUs to multi-node datacenters.
    Measure everything from application metrics (TPS, TTFT) down to CUDA kernel performance.
    """
    pass


@cli.command()
@click.option(
    "--model",
    type=click.STRING,
    required=True,
    help="LLM model to benchmark (HuggingFace model name or path).",
)
@click.option(
    "--backend",
    type=click.Choice(["huggingface", "hf", "vllm", "sglang", "tensorrt", "trt"]),
    default="huggingface",
    help="Inference backend to use.",
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda", "auto"]),
    default="auto",
    help="Device to run the benchmark on.",
)
@click.option(
    "--batch-size",
    type=int,
    default=1,
    help="Batch size for inference.",
)
@click.option(
    "--sequence-length",
    type=int,
    default=128,
    help="Input sequence length.",
)
@click.option(
    "--max-new-tokens",
    type=int,
    default=100,
    help="Maximum new tokens to generate.",
)
@click.option(
    "--precision",
    type=click.Choice(["4bit", "8bit", "fp16", "bf16", "fp32"]),
    default="fp16",
    help="Precision to use for the model.",
)
@click.option(
    "--num-runs",
    type=int,
    default=10,
    help="Number of inference iterations.",
)
@click.option(
    "--warmup-runs",
    type=int,
    default=2,
    help="Number of warmup iterations.",
)
@click.option(
    "--profile-level",
    type=click.Choice(["basic", "detailed", "comprehensive"]),
    default="detailed",
    help="Level of profiling detail.",
)
@click.option(
    "--output-format",
    type=click.Choice(["text", "json", "csv", "html"]),
    default="text",
    help="Output format for results.",
)
@click.option(
    "--output-path",
    type=click.Path(),
    help="Path to save output file (for json/csv/html formats).",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose output with detailed profiling info.",
)
def benchmark(
    model,
    backend,
    device,
    batch_size,
    sequence_length,
    max_new_tokens,
    precision,
    num_runs,
    warmup_runs,
    profile_level,
    output_format,
    output_path,
    verbose,
):
    """
    Run comprehensive LLM benchmark with hardware profiling.
    
    This command profiles your model across all levels:
    - Application: TPS, TTFT, latency, throughput
    - Hardware: GPU utilization, memory, power, temperature
    - System: CPU, RAM, disk I/O, network
    - CUDA: Kernel execution, memory transfers
    - Model: Architecture analysis, parameter breakdown
    
    Example:
        impetus benchmark --model microsoft/phi-2 --precision fp16
        impetus benchmark --model meta-llama/Llama-2-7b-hf --backend vllm --batch-size 8
    """
    try:
        # Create comprehensive benchmark
        benchmark_runner = ComprehensiveBenchmark(
            model_name=model,
            backend=backend,
            device=device,
            precision=precision,
            profile_level=profile_level,
            verbose=verbose,
        )
        
        # Determine output path if format specified but no path given
        if output_format != "text" and not output_path:
            model_safe_name = model.replace("/", "_")
            output_path = f"impetus_results_{model_safe_name}.{output_format}"
        
        # Run benchmark
        results = benchmark_runner.run(
            batch_size=batch_size,
            sequence_length=sequence_length,
            max_new_tokens=max_new_tokens,
            num_runs=num_runs,
            warmup_runs=warmup_runs,
            output_format=output_format if output_format != "text" else None,
            output_path=output_path,
        )
        
    except Exception as e:
        click.echo(f"Error running benchmark: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.option(
    "--verbose",
    is_flag=True,
    help="Show detailed system information.",
)
def system_info(verbose):
    """
    Display detailed hardware and system information.
    
    Shows GPU capabilities, memory, compute capacity, theoretical performance,
    system configuration, and topology information.
    
    Use this to understand your hardware before running benchmarks.
    """
    reporter = ConsoleReporter(verbose=verbose)
    
    reporter.print_header("System Information")
    
    # Hardware profiling
    reporter.print_section("Hardware Detection")
    hw_profiler = HardwareProfiler()
    
    if hw_profiler.gpu_count == 0:
        reporter.print_warning("No GPUs detected")
    else:
        reporter.print_success(f"Detected {hw_profiler.gpu_count} GPU(s)")
        
        # Display GPU info
        gpu_info_list = [gpu.to_dict() for gpu in hw_profiler.gpu_info]
        reporter.print_gpu_info(gpu_info_list)
        
        # Display current metrics
        if verbose:
            reporter.print_section("Current GPU Metrics")
            current_metrics = hw_profiler.get_current_metrics()
            for metrics in current_metrics:
                metrics_dict = metrics.to_dict()
                reporter.print_info(f"GPU {metrics.index}:")
                reporter.console.print(f"  Utilization: {metrics.utilization_gpu}%")
                reporter.console.print(f"  Memory: {metrics.memory_used / (1024**3):.2f} / {metrics.memory_total / (1024**3):.2f} GB")
                reporter.console.print(f"  Temperature: {metrics.temperature}°C")
                reporter.console.print(f"  Power: {metrics.power_draw:.1f} / {metrics.power_limit:.1f} W")
                if metrics.nvlink_count > 0:
                    reporter.console.print(f"  NVLink: {metrics.nvlink_count} links")
        
        # Display topology
        topology = hw_profiler.get_topology_info()
        if topology.get("nvlink_topology") and verbose:
            reporter.print_section("GPU Topology")
            nvlink_topo = topology["nvlink_topology"]
            for gpu_id, connections in nvlink_topo.items():
                if connections:
                    reporter.console.print(f"  GPU {gpu_id} → NVLink connections: {connections}")
    
    # System profiling
    reporter.print_section("System Information")
    sys_profiler = SystemProfiler()
    system_metrics = sys_profiler.get_current_metrics()
    reporter.print_system_metrics(system_metrics.to_dict())
    
    # CUDA info
    if torch.cuda.is_available():
        reporter.print_section("CUDA Information")
        reporter.console.print(f"  CUDA Available: Yes")
        reporter.console.print(f"  CUDA Version: {torch.version.cuda}")
        reporter.console.print(f"  cuDNN Version: {torch.backends.cudnn.version()}")
        reporter.console.print(f"  Number of GPUs: {torch.cuda.device_count()}")
    else:
        reporter.print_warning("CUDA not available")
    
    # PyTorch info
    reporter.print_section("PyTorch Information")
    reporter.console.print(f"  PyTorch Version: {torch.__version__}")
    reporter.console.print(f"  CUDA Enabled: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        reporter.console.print(f"  CUDA Device: {torch.cuda.get_device_name(0)}")


@cli.command()
@click.argument("result_files", nargs=-1, type=click.Path(exists=True))
@click.option(
    "--output",
    type=click.Path(),
    help="Output file for comparison results.",
)
def compare(result_files, output):
    """
    Compare multiple benchmark results.
    
    Load and compare results from previous benchmark runs.
    
    Example:
        impetus compare results1.json results2.json results3.json
    """
    if len(result_files) < 2:
        click.echo("Error: Need at least 2 result files to compare", err=True)
        sys.exit(1)
    
    reporter = ConsoleReporter()
    reporter.print_header("Benchmark Comparison")
    
    # TODO: Implement comparison logic
    reporter.print_warning("Comparison feature coming soon!")
    
    for i, file_path in enumerate(result_files):
        reporter.console.print(f"  [{i+1}] {file_path}")


if __name__ == "__main__":
    cli()
