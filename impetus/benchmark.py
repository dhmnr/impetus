"""Comprehensive benchmark orchestrator."""

import torch
import time
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

from .backends import get_backend, BackendConfig
from .profiling import HardwareProfiler, SystemProfiler, MemoryProfiler, CUDAProfiler
from .analysis import TheoreticalAnalyzer, ModelAnalyzer
from .metrics import (
    ComprehensiveBenchmarkResults,
    InferenceMetrics,
    ModelMetrics,
    HardwareMetrics,
    SystemMetrics,
    MemoryMetrics,
    CUDAMetrics,
)
from .reporting import ConsoleReporter, JSONExporter, CSVExporter, HTMLReporter


class ComprehensiveBenchmark:
    """Orchestrate comprehensive benchmarking with all profiling modules."""
    
    def __init__(
        self,
        model_name: str,
        backend: str = "huggingface",
        device: str = "auto",
        precision: str = "fp16",
        profile_level: str = "detailed",
        verbose: bool = False,
    ):
        """
        Initialize comprehensive benchmark.
        
        Args:
            model_name: Model name or path
            backend: Backend to use (huggingface, vllm, sglang, tensorrt)
            device: Device to run on (cpu, cuda, auto)
            precision: Precision (4bit, 8bit, fp16, bf16, fp32)
            profile_level: Profiling level (basic, detailed, comprehensive)
            verbose: Enable verbose output
        """
        self.model_name = model_name
        self.backend_name = backend
        self.precision = precision
        self.profile_level = profile_level
        self.verbose = verbose
        
        # Determine device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize profilers
        self.hardware_profiler = HardwareProfiler()
        self.system_profiler = SystemProfiler()
        self.memory_profiler = MemoryProfiler(self.device)
        self.cuda_profiler = CUDAProfiler(self.device) if self.device.type == "cuda" else None
        
        # Initialize reporters
        self.console_reporter = ConsoleReporter(verbose=verbose)
        
        # Backend will be initialized later
        self.backend = None
        self.backend_config = None
    
    def run(
        self,
        batch_size: int = 1,
        sequence_length: int = 128,
        max_new_tokens: int = 100,
        num_runs: int = 10,
        warmup_runs: int = 2,
        output_format: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> ComprehensiveBenchmarkResults:
        """
        Run comprehensive benchmark.
        
        Args:
            batch_size: Batch size for inference
            sequence_length: Input sequence length
            max_new_tokens: Maximum new tokens to generate
            num_runs: Number of benchmark runs
            warmup_runs: Number of warmup runs
            output_format: Output format (json, csv, html)
            output_path: Path to save output
        
        Returns:
            Comprehensive benchmark results
        """
        self.console_reporter.print_header("Impetus - Comprehensive LLM Profiling")
        
        # Initialize backend
        self.console_reporter.print_info(f"Initializing {self.backend_name} backend...")
        self._initialize_backend(batch_size, sequence_length)
        
        # Collect hardware info
        self.console_reporter.print_info("Collecting hardware information...")
        hardware_metrics = self._collect_hardware_metrics()
        
        # Get system baseline
        self.console_reporter.print_info("Collecting system metrics...")
        system_metrics_start = self.system_profiler.get_current_metrics()
        
        # Load model
        self.console_reporter.print_info(f"Loading model: {self.model_name}...")
        self.backend.load_model()
        
        # Analyze model architecture
        self.console_reporter.print_info("Analyzing model architecture...")
        model_metrics = self._analyze_model()
        
        # Get memory baseline after model load
        memory_baseline = self.memory_profiler.get_current_snapshot()
        
        # Run benchmark
        self.console_reporter.print_info(f"Running benchmark ({num_runs} runs)...")
        inference_results = self.backend.benchmark_inference(
            prompts=None,  # Will use default dataset
            max_new_tokens=max_new_tokens,
            num_runs=num_runs,
            warmup_runs=warmup_runs,
            sequence_length=sequence_length,
            batch_size=batch_size,
        )
        
        # Collect final metrics
        self.console_reporter.print_info("Collecting final metrics...")
        memory_final = self.memory_profiler.get_memory_summary()
        system_metrics_end = self.system_profiler.get_current_metrics()
        
        # Create comprehensive results
        results = self._create_results(
            inference_results,
            model_metrics,
            hardware_metrics,
            system_metrics_end,
            memory_final,
        )
        
        # Generate recommendations
        results.recommendations = self._generate_recommendations(results)
        results.bottlenecks = self._identify_bottlenecks(results)
        
        # Display results
        self.console_reporter.print_comprehensive_results(results)
        
        # Export results if requested
        if output_format and output_path:
            self._export_results(results, output_format, output_path)
        
        # Cleanup
        self.backend.cleanup()
        
        return results
    
    def _initialize_backend(self, batch_size: int, sequence_length: int):
        """Initialize inference backend."""
        self.backend_config = BackendConfig(
            model_name=self.model_name,
            device=self.device,
            precision=self.precision,
            max_batch_size=batch_size,
            max_sequence_length=sequence_length,
        )
        
        self.backend = get_backend(self.backend_name, config=self.backend_config)
    
    def _collect_hardware_metrics(self) -> HardwareMetrics:
        """Collect hardware metrics."""
        gpu_info_list = []
        
        for gpu_info in self.hardware_profiler.gpu_info:
            gpu_info_list.append(gpu_info.to_dict())
        
        # Get current GPU metrics
        current_metrics = self.hardware_profiler.get_current_metrics()
        for i, metrics in enumerate(current_metrics):
            if i < len(gpu_info_list):
                gpu_info_list[i].update(metrics.to_dict())
        
        # Get topology
        topology = self.hardware_profiler.get_topology_info()
        
        # Get theoretical performance
        theoretical = {}
        if gpu_info_list:
            first_gpu = self.hardware_profiler.gpu_info[0]
            analyzer = TheoreticalAnalyzer(
                first_gpu.name,
                first_gpu.compute_capability
            )
            theoretical_perf = analyzer.get_theoretical_performance()
            theoretical = theoretical_perf.to_dict()
        
        return HardwareMetrics(
            gpu_metrics=gpu_info_list,
            topology=topology,
            theoretical_performance=theoretical,
        )
    
    def _analyze_model(self) -> ModelMetrics:
        """Analyze model architecture."""
        model_analyzer = ModelAnalyzer(self.backend.model)
        arch_info = model_analyzer.analyze()
        
        return ModelMetrics(
            model_name=arch_info.model_name,
            architecture_type=arch_info.architecture_type,
            total_params=arch_info.total_params,
            trainable_params=arch_info.trainable_params,
            num_layers=arch_info.num_layers,
            hidden_size=arch_info.hidden_size,
            num_attention_heads=arch_info.num_attention_heads,
            vocab_size=arch_info.vocab_size,
            param_memory_mb=arch_info.total_memory_bytes / (1024**2),
            embedding_memory_mb=arch_info.embedding_memory_bytes / (1024**2),
            estimated_flops_per_token=arch_info.estimated_flops_per_token,
        )
    
    def _create_results(
        self,
        inference_results: Dict[str, Any],
        model_metrics: ModelMetrics,
        hardware_metrics: HardwareMetrics,
        system_metrics: Any,
        memory_metrics: Dict[str, Any],
    ) -> ComprehensiveBenchmarkResults:
        """Create comprehensive benchmark results."""
        
        # Convert inference results
        inference = InferenceMetrics(
            batch_size=inference_results.get("batch_size", 1),
            sequence_length=inference_results.get("sequence_length", 0),
            max_new_tokens=inference_results.get("max_new_tokens", 0),
            num_runs=inference_results.get("num_runs", 0),
            avg_latency_ms=inference_results.get("avg_latency_ms", 0),
            std_latency_ms=inference_results.get("std_latency_ms", 0),
            min_latency_ms=inference_results.get("min_latency_ms", 0),
            max_latency_ms=inference_results.get("max_latency_ms", 0),
            avg_time_to_first_token_ms=inference_results.get("avg_time_to_first_token_ms", 0),
            time_per_output_token_ms=inference_results.get("time_per_output_token_ms", 0),
            tokens_per_batch=inference_results.get("tokens_per_batch", 0),
            throughput_tokens_per_sec=inference_results.get("throughput_tokens_per_sec", 0),
            throughput_batches_per_sec=inference_results.get("throughput_batches_per_sec", 0),
        )
        
        # Convert system metrics
        system = SystemMetrics(
            cpu_percent=system_metrics.cpu_percent,
            ram_used_gb=system_metrics.ram_used / (1024**3),
            ram_total_gb=system_metrics.ram_total / (1024**3),
            ram_percent=system_metrics.ram_percent,
            platform=system_metrics.platform,
            python_version=system_metrics.python_version,
        )
        
        # Convert memory metrics
        memory = MemoryMetrics(
            allocated_bytes=memory_metrics.get("allocated_bytes", 0),
            reserved_bytes=memory_metrics.get("reserved_bytes", 0),
            peak_allocated_bytes=memory_metrics.get("peak_allocated_bytes", 0),
            peak_reserved_bytes=memory_metrics.get("peak_reserved_bytes", 0),
            total_memory_bytes=memory_metrics.get("total_memory", 0),
            memory_utilization_percent=memory_metrics.get("memory_utilization_percent", 0),
        )
        
        # CUDA metrics (placeholder for now)
        cuda = CUDAMetrics()
        
        return ComprehensiveBenchmarkResults(
            backend=self.backend_name,
            model_name=self.model_name,
            precision=self.precision,
            timestamp=datetime.now().isoformat(),
            inference=inference,
            model=model_metrics,
            hardware=hardware_metrics,
            system=system,
            memory=memory,
            cuda=cuda,
        )
    
    def _generate_recommendations(self, results: ComprehensiveBenchmarkResults) -> list[str]:
        """Generate optimization recommendations based on results."""
        recommendations = []
        
        # Memory recommendations
        if results.memory.memory_utilization_percent > 90:
            recommendations.append(
                "High memory utilization detected. Consider using quantization (4bit/8bit) or reducing batch size."
            )
        
        # Batch size recommendations
        if results.inference.batch_size == 1 and results.memory.memory_utilization_percent < 50:
            recommendations.append(
                "Low memory utilization with batch size 1. Consider increasing batch size for better throughput."
            )
        
        # Precision recommendations
        if results.precision == "fp32" and results.hardware.gpu_metrics:
            if any(gpu.get("tensor_core_available") for gpu in [results.hardware.theoretical_performance]):
                recommendations.append(
                    "GPU supports Tensor Cores. Consider using fp16/bf16 for significant speedup."
                )
        
        # Throughput recommendations
        if results.inference.throughput_tokens_per_sec < 10:
            recommendations.append(
                "Low throughput detected. Consider optimizing model loading, using a faster backend (vLLM, TensorRT-LLM), or upgrading hardware."
            )
        
        return recommendations
    
    def _identify_bottlenecks(self, results: ComprehensiveBenchmarkResults) -> list[str]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        
        # Memory bottleneck
        if results.memory.memory_utilization_percent > 95:
            bottlenecks.append("Memory: GPU memory is nearly full, limiting batch size and performance")
        
        # TTFT bottleneck
        ttft_ratio = results.inference.avg_time_to_first_token_ms / results.inference.avg_latency_ms
        if ttft_ratio > 0.5:
            bottlenecks.append("Latency: Time to first token is high relative to total latency")
        
        # GPU utilization
        if results.hardware.gpu_metrics:
            avg_util = sum(gpu.get("utilization_gpu", 0) for gpu in results.hardware.gpu_metrics) / len(results.hardware.gpu_metrics)
            if avg_util < 50:
                bottlenecks.append(f"GPU Utilization: Average GPU utilization is low ({avg_util:.1f}%), indicating potential CPU or I/O bottleneck")
        
        return bottlenecks
    
    def _export_results(self, results: ComprehensiveBenchmarkResults, format: str, path: str):
        """Export results to file."""
        output_path = Path(path)
        
        if format == "json":
            JSONExporter.export(results, output_path)
            self.console_reporter.print_success(f"Results exported to {output_path}")
        elif format == "csv":
            CSVExporter.export(results, output_path)
            self.console_reporter.print_success(f"Results exported to {output_path}")
        elif format == "html":
            HTMLReporter.generate(results, output_path)
            self.console_reporter.print_success(f"HTML report generated: {output_path}")
        else:
            self.console_reporter.print_error(f"Unknown export format: {format}")

