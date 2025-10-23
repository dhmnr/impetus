"""CSV export functionality for benchmark results."""

import csv
from pathlib import Path
from typing import Union, List

from ..metrics import ComprehensiveBenchmarkResults


class CSVExporter:
    """Export benchmark results to CSV format."""
    
    @staticmethod
    def export(results: ComprehensiveBenchmarkResults, output_path: Union[str, Path]) -> None:
        """
        Export benchmark results to a CSV file.
        
        Creates a flattened CSV with key metrics.
        
        Args:
            results: Comprehensive benchmark results
            output_path: Path to output CSV file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Define columns
        columns = [
            "timestamp",
            "backend",
            "model_name",
            "precision",
            "batch_size",
            "sequence_length",
            "avg_latency_ms",
            "std_latency_ms",
            "ttft_ms",
            "time_per_token_ms",
            "throughput_tokens_per_sec",
            "total_params",
            "num_layers",
            "hidden_size",
            "allocated_memory_gb",
            "peak_memory_gb",
            "memory_utilization_percent",
            "gpu_utilization_percent",
            "temperature_c",
            "power_draw_w",
        ]
        
        # Extract data
        row = {
            "timestamp": results.timestamp,
            "backend": results.backend,
            "model_name": results.model_name,
            "precision": results.precision,
            "batch_size": results.inference.batch_size,
            "sequence_length": results.inference.sequence_length,
            "avg_latency_ms": results.inference.avg_latency_ms,
            "std_latency_ms": results.inference.std_latency_ms,
            "ttft_ms": results.inference.avg_time_to_first_token_ms,
            "time_per_token_ms": results.inference.time_per_output_token_ms,
            "throughput_tokens_per_sec": results.inference.throughput_tokens_per_sec,
            "total_params": results.model.total_params,
            "num_layers": results.model.num_layers,
            "hidden_size": results.model.hidden_size,
            "allocated_memory_gb": results.memory.allocated_bytes / (1024**3),
            "peak_memory_gb": results.memory.peak_allocated_bytes / (1024**3),
            "memory_utilization_percent": results.memory.memory_utilization_percent,
        }
        
        # Add GPU metrics if available
        if results.hardware.gpu_metrics:
            first_gpu = results.hardware.gpu_metrics[0]
            row["gpu_utilization_percent"] = first_gpu.get("utilization_gpu", 0)
            row["temperature_c"] = first_gpu.get("temperature", 0)
            row["power_draw_w"] = first_gpu.get("power_draw", 0)
        else:
            row["gpu_utilization_percent"] = 0
            row["temperature_c"] = 0
            row["power_draw_w"] = 0
        
        # Write CSV
        file_exists = output_path.exists()
        
        with open(output_path, 'a' if file_exists else 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(row)
    
    @staticmethod
    def export_detailed(results: ComprehensiveBenchmarkResults, output_dir: Union[str, Path]) -> None:
        """
        Export detailed benchmark results to multiple CSV files.
        
        Creates separate CSV files for different metric categories.
        
        Args:
            results: Comprehensive benchmark results
            output_dir: Directory to output CSV files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export inference metrics
        CSVExporter._export_dict_to_csv(
            results.inference.to_dict(),
            output_dir / "inference_metrics.csv"
        )
        
        # Export model metrics
        CSVExporter._export_dict_to_csv(
            results.model.to_dict(),
            output_dir / "model_metrics.csv"
        )
        
        # Export memory metrics
        CSVExporter._export_dict_to_csv(
            results.memory.to_dict(),
            output_dir / "memory_metrics.csv"
        )
        
        # Export system metrics
        CSVExporter._export_dict_to_csv(
            results.system.to_dict(),
            output_dir / "system_metrics.csv"
        )
        
        # Export GPU metrics
        if results.hardware.gpu_metrics:
            with open(output_dir / "gpu_metrics.csv", 'w', newline='', encoding='utf-8') as f:
                if results.hardware.gpu_metrics:
                    writer = csv.DictWriter(f, fieldnames=results.hardware.gpu_metrics[0].keys())
                    writer.writeheader()
                    writer.writerows(results.hardware.gpu_metrics)
    
    @staticmethod
    def _export_dict_to_csv(data: dict, output_path: Path) -> None:
        """Helper to export a dictionary to CSV."""
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            writer.writeheader()
            writer.writerow(data)
    
    @staticmethod
    def export_comparison(results_list: List[ComprehensiveBenchmarkResults], output_path: Union[str, Path]) -> None:
        """
        Export multiple benchmark results for comparison.
        
        Args:
            results_list: List of benchmark results
            output_path: Path to output CSV file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        columns = [
            "run_id",
            "timestamp",
            "backend",
            "model_name",
            "precision",
            "batch_size",
            "sequence_length",
            "avg_latency_ms",
            "ttft_ms",
            "throughput_tokens_per_sec",
            "memory_utilization_percent",
        ]
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            
            for i, results in enumerate(results_list):
                writer.writerow({
                    "run_id": i,
                    "timestamp": results.timestamp,
                    "backend": results.backend,
                    "model_name": results.model_name,
                    "precision": results.precision,
                    "batch_size": results.inference.batch_size,
                    "sequence_length": results.inference.sequence_length,
                    "avg_latency_ms": results.inference.avg_latency_ms,
                    "ttft_ms": results.inference.avg_time_to_first_token_ms,
                    "throughput_tokens_per_sec": results.inference.throughput_tokens_per_sec,
                    "memory_utilization_percent": results.memory.memory_utilization_percent,
                })

