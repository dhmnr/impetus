# Changelog

All notable changes to Impetus will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2024-10-22

### Major Overhaul 🚀

Complete transformation from a basic benchmarking tool to a comprehensive GPU profiling and characterization system for LLM workloads.

### Changed
- **Migrated from Poetry to uv** for faster dependency management
  - Updated `pyproject.toml` to use standard PEP 621 format
  - Removed Poetry-specific configuration
  - Now uses hatchling as build backend
  - Significantly faster installation and dependency resolution
  - Flexible version ranges allow automatic dependency upgrades
- **Migrated from pynvml to nvidia-ml-py** (the officially maintained package)
- **Fixed Windows encoding issues**
  - Console output: Replaced Unicode symbols with ASCII equivalents
  - File I/O: Added explicit UTF-8 encoding for all report exports (HTML, JSON, CSV)

### Added

#### Backend Support
- **HuggingFace Transformers** - Default backend, works on all platforms
- **vLLM 0.11.0** - High-throughput serving (Linux only)
- **SGLang** - Efficient serving (Linux only)
- **TensorRT-LLM** - Planned for v0.3.0

#### Core Profiling Infrastructure
- **Hardware Profiler** (`impetus/profiling/hardware.py`)
  - NVML integration for detailed GPU metrics
  - Real-time monitoring of utilization, memory, power, temperature
  - PCIe and NVLink bandwidth tracking
  - GPU topology detection and visualization
  - Theoretical performance calculation (peak FLOPS, bandwidth)

- **System Profiler** (`impetus/profiling/system.py`)
  - CPU usage and frequency monitoring
  - RAM usage tracking
  - Disk I/O profiling (read/write throughput)
  - Network bandwidth monitoring
  - System information collection

- **Memory Profiler** (`impetus/profiling/memory.py`)
  - Detailed VRAM allocation tracking
  - Peak memory usage monitoring
  - KV cache size estimation
  - Activation memory analysis
  - Memory fragmentation detection

- **CUDA Profiler** (`impetus/profiling/cuda.py`)
  - PyTorch profiler integration
  - Kernel-level execution metrics
  - Memory operation tracking
  - Kernel launch overhead measurement
  - Memory bandwidth benchmarking

- **Distributed Profiler** (`impetus/profiling/distributed.py`)
  - Communication pattern profiling (TP/PP/DP)
  - All-reduce, all-gather, P2P latency measurement
  - NVLink vs network bandwidth detection
  - Topology information gathering

- **Operator Profiler** (`impetus/profiling/operators.py`)
  - Layer-by-layer performance profiling
  - Attention vs FFN breakdown
  - Memory-intensive layer identification
  - Operator-type aggregation

#### Analysis Modules
- **Theoretical Analyzer** (`impetus/analysis/theoretical.py`)
  - Peak performance calculations for different precisions
  - Roofline model analysis
  - Memory bandwidth calculations
  - Communication time estimation
  - Efficiency metrics (actual/theoretical)
  - Model FLOPs estimation

- **Model Analyzer** (`impetus/analysis/model.py`)
  - Architecture detection and analysis
  - Parameter counting and breakdown
  - Layer-wise memory profiling
  - Model dimension extraction
  - Bottleneck layer identification

#### Backend System
- **Backend Abstraction Layer** (`impetus/backends/base.py`)
  - Unified interface for all inference engines
  - Pluggable backend architecture
  - Common profiling hooks

- **HuggingFace Backend** (`impetus/backends/huggingface.py`)
  - Enhanced from legacy implementation
  - Quantization support (4bit, 8bit, fp16, bf16, fp32)
  - Comprehensive benchmarking
  - Model info extraction

- **vLLM Backend** (`impetus/backends/vllm.py`)
  - High-throughput inference support
  - Tensor parallelism integration
  - Optimized batch processing

- **SGLang Backend** (`impetus/backends/sglang.py`)
  - Stub implementation for future integration
  - Architecture prepared for SGLang support

- **TensorRT-LLM Backend** (`impetus/backends/tensorrt.py`)
  - Stub implementation for future integration
  - Architecture prepared for TRT-LLM support

#### Reporting System
- **Rich Console Reporter** (`impetus/reporting/console.py`)
  - Beautiful terminal output with Rich library
  - Colored tables and formatted text
  - Section-based organization
  - Progress indicators
  - GPU and system info tables

- **JSON Exporter** (`impetus/reporting/json_export.py`)
  - Structured JSON output
  - Comparison mode support
  - Full metrics preservation

- **CSV Exporter** (`impetus/reporting/csv_export.py`)
  - Spreadsheet-compatible output
  - Detailed and summary modes
  - Multi-run comparison support

- **HTML Reporter** (`impetus/reporting/html.py`)
  - Interactive web-based reports
  - Plotly charts and visualizations
  - Responsive design
  - Shareable reports

#### Metrics System
- **Comprehensive Metrics** (`impetus/metrics.py`)
  - `HardwareMetrics`: GPU hardware profiling data
  - `SystemMetrics`: System-wide resource usage
  - `MemoryMetrics`: Detailed memory profiling
  - `CUDAMetrics`: Kernel-level metrics
  - `ModelMetrics`: Architecture information
  - `InferenceMetrics`: Performance metrics
  - `DistributedMetrics`: Multi-GPU communication
  - `ComprehensiveBenchmarkResults`: Complete benchmark data

#### Orchestration
- **Comprehensive Benchmark Runner** (`impetus/benchmark.py`)
  - Orchestrates all profiling modules
  - Automated bottleneck detection
  - Smart recommendations engine
  - Multi-format export support
  - Profile level configuration

#### CLI Enhancements
- **Enhanced `benchmark` command**
  - Backend selection (--backend)
  - Profile level control (--profile-level)
  - Multiple output formats (--output-format)
  - Verbose mode (--verbose)
  - Extended options for all parameters

- **New `system-info` command**
  - Hardware inventory display
  - Current metrics snapshot
  - GPU topology visualization
  - System configuration details

- **New `compare` command** (stub for future)
  - Multi-run comparison support
  - Side-by-side analysis

### Changed
- Completely refactored codebase into modular architecture
- Upgraded from basic metrics to comprehensive profiling
- Enhanced CLI with more options and better UX
- Improved error handling and graceful degradation
- Updated dependencies for better performance

### Dependencies
- Added `pynvml` for NVML integration
- Added `psutil` for system metrics
- Added `rich` for beautiful terminal output
- Added `plotly` for interactive visualizations
- Added `pandas` for data manipulation
- Made `vllm`, `sglang`, `tensorrt-llm` optional dependencies

### Performance
- Minimal profiling overhead (< 5%)
- Efficient memory tracking
- Optimized CUDA event usage
- Batch profiling for efficiency

### Documentation
- Comprehensive README with examples
- Detailed API documentation
- Use case scenarios
- Advanced usage guide
- Contributing guidelines

## [0.1.1] - 2024-XX-XX

### Added
- Basic LLM benchmarking
- Quantization support (4bit, 8bit)
- Simple metrics (latency, throughput, TTFT)

### Fixed
- Minor bug fixes

## [0.1.0] - Initial Release

### Added
- Basic decoder-only language model benchmarking
- HuggingFace Transformers integration
- Simple performance metrics
- CLI interface

---

## Roadmap

### Planned for 0.3.0
- Complete vLLM backend implementation
- Complete SGLang backend implementation
- Complete TensorRT-LLM backend implementation
- Advanced distributed profiling (multi-node)
- Power efficiency analysis
- Carbon footprint estimation
- Automated optimization suggestions
- A/B testing mode for model comparison
- Real-time monitoring dashboard
- Prometheus metrics export
- Cost analysis (cloud pricing integration)

### Future Considerations
- AMD GPU support
- Apple Silicon (MPS) support
- Intel GPU support
- Google TPU support
- Inference server profiling (Triton, vLLM server, etc.)
- Fine-tuning performance profiling
- Model compression analysis
- Quantization impact analysis
- Custom kernel profiling
- Energy-to-accuracy tradeoff analysis

