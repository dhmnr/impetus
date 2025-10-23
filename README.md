# ⚡ Impetus

<div align="center">

**Comprehensive GPU Profiling & Characterization Tool for LLM Workloads**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

*Profile your hardware from single GPUs to multi-node datacenters. Measure everything from application metrics (TPS, TTFT) down to CUDA kernel performance.*

</div>

---

## 🎯 Overview

Impetus is a comprehensive profiling tool designed to characterize and benchmark LLM workloads on NVIDIA GPU hardware. Whether you're a researcher testing a new GPU, an enterprise validating a multi-node cluster, or a datacenter operator characterizing thousands of GPUs, Impetus provides deep insights into your hardware's LLM capabilities.

### Key Features

- **🔍 Multi-Level Profiling**: From application metrics down to CUDA kernels and warp occupancy
- **🖥️ Hardware Characterization**: GPU utilization, memory bandwidth, power consumption, thermal profiling
- **💾 Memory Analysis**: Detailed VRAM tracking, KV cache estimation, memory fragmentation analysis
- **⚙️ System Metrics**: CPU, RAM, disk I/O, network bandwidth monitoring
- **🧠 Model Analysis**: Architecture breakdown, parameter counting, layer-by-layer profiling
- **📊 Theoretical Analysis**: Peak FLOPS, bandwidth calculations, roofline model, efficiency metrics
- **🔗 Distributed Profiling**: TP/PP/DP communication overhead, NVLink/PCIe bandwidth
- **🚀 Multiple Backends**: HuggingFace Transformers, vLLM, SGLang, TensorRT-LLM support
- **📈 Rich Reporting**: Beautiful console output, JSON/CSV export, interactive HTML reports
- **💡 Smart Recommendations**: Automatic bottleneck identification and optimization suggestions

---

## 📦 Installation

### Basic Installation

Using pip:
```bash
pip install impetus

# For CUDA support (Linux), install from PyTorch index
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Using uv with **automatic CUDA detection** (recommended):
```bash
# Automatically detects your GPU and installs the right CUDA version
uv pip install impetus --torch-backend=auto
```

Or manually specify CUDA version with uv:
```bash
# For CUDA 12.6-12.9 (PyTorch 2.8.0+ recommended)
uv pip install impetus --index-url https://download.pytorch.org/whl/cu126

# For CUDA 12.4
uv pip install impetus --index-url https://download.pytorch.org/whl/cu124

# For CUDA 12.1
uv pip install impetus --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
uv pip install impetus --index-url https://download.pytorch.org/whl/cu118

# CPU-only
uv pip install impetus --torch-backend=cpu
```

See [uv's PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/) for more options.

### With Optional Backend Support

**⚠️ Linux Only**: vLLM and SGLang require Linux. On Windows, use WSL2 or stick with the HuggingFace backend.

Using pip (Linux):
```bash
# Install with vLLM support
pip install "impetus[vllm]"

# Install with SGLang support
pip install "impetus[sglang]"

# Install with all backends (vLLM + SGLang)
pip install "impetus[all-backends]"
```

Using uv (Linux):
```bash
# Install with vLLM support
uv sync --extra vllm

# Install with SGLang support  
uv sync --extra sglang

# Install with all backends (vLLM + SGLang)
uv sync --extra all-backends
```

**Note**: Installing backends will automatically upgrade dependencies (like `transformers`) to compatible versions.

### TensorRT-LLM Installation

🚧 **Coming Soon**: TensorRT-LLM backend support is planned for v0.3.0.

For now, TensorRT-LLM can be installed separately but is not yet integrated:

```bash
pip install --extra-index-url https://pypi.nvidia.com/ tensorrt-llm
```

### Development Installation

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
   git clone https://github.com/dhmnr/impetus.git
cd impetus

# Sync with automatic CUDA detection
UV_TORCH_BACKEND=auto uv sync

# Or manually specify CUDA version
uv sync  # Uses CUDA 12.4 on Linux, CPU on macOS/Windows
```

---

## 🎯 Supported Backends

Impetus supports multiple inference backends for different use cases:

| Backend | Platforms | Status | Installation |
|---------|-----------|--------|--------------|
| **HuggingFace** | Windows, Linux, macOS | ✅ Default | Included in base install |
| **vLLM** | Linux only | ✅ Supported | `uv sync --extra vllm` |
| **SGLang** | Linux only | ✅ Supported | `uv sync --extra sglang` |
| **TensorRT-LLM** | Linux, Windows | 🚧 Coming Soon | Manual installation required |

**Default Backend:** HuggingFace Transformers (works on all platforms)

---

## 🚀 Quick Start

### Basic Benchmark

```bash
# Benchmark a model with default settings
impetus benchmark --model microsoft/phi-2

# Specify precision and batch size
impetus benchmark --model meta-llama/Llama-2-7b-hf --precision fp16 --batch-size 4

# Use a different backend
impetus benchmark --model mistralai/Mistral-7B-v0.1 --backend vllm --precision fp16
```

### System Information

```bash
# Display hardware and system information
impetus system-info

# Show detailed GPU topology and metrics
impetus system-info --verbose
```

### Export Results

```bash
# Export to JSON
impetus benchmark --model microsoft/phi-2 --output-format json --output-path results.json

# Export to CSV
impetus benchmark --model microsoft/phi-2 --output-format csv --output-path results.csv

# Generate interactive HTML report
impetus benchmark --model microsoft/phi-2 --output-format html --output-path report.html
```

---

## 📊 What Does Impetus Measure?

### Application-Level Metrics
- **Latency**: Average, min, max, standard deviation
- **Time to First Token (TTFT)**: Critical for interactive applications
- **Throughput**: Tokens per second, batches per second
- **Time per Output Token**: Generation speed per token

### Hardware Metrics (per GPU)
- **Utilization**: GPU compute and memory utilization
- **Memory**: Allocated, reserved, peak usage, total capacity
- **Power**: Current draw, power limit, efficiency
- **Temperature**: Operating temperature monitoring
- **Clock Speeds**: SM clock, memory clock
- **Interconnect**: PCIe bandwidth, NVLink topology and bandwidth
- **Theoretical Performance**: Peak FLOPS (FP16/BF16/FP32), memory bandwidth

### System Metrics
- **CPU**: Usage percentage, core count, frequency
- **RAM**: Total, used, available, utilization percentage
- **Disk I/O**: Read/write throughput and IOPS
- **Network**: Send/receive bandwidth and packet rates

### Memory Profiling
- **VRAM Allocation**: Real-time and peak GPU memory usage
- **KV Cache**: Estimated size based on model architecture
- **Parameter Memory**: Weight memory breakdown by layer
- **Activation Memory**: Estimated activation memory requirements
- **Memory Fragmentation**: Detection and analysis

### CUDA-Level Metrics
- **Kernel Execution**: Time per kernel, execution count
- **Kernel Launch Overhead**: Overhead of launching CUDA kernels
- **Memory Operations**: Memory copy time, allocation/deallocation counts
- **Occupancy**: Theoretical and actual warp occupancy (when available)

### Model Architecture Analysis
- **Parameter Counting**: Total, trainable, non-trainable parameters
- **Layer Breakdown**: Memory and compute per layer
- **Architecture Type**: Decoder-only, encoder-decoder detection
- **Model Dimensions**: Hidden size, attention heads, vocabulary size
- **FLOPs Estimation**: Estimated FLOPs per token

### Distributed System Profiling
- **Topology Detection**: NVLink connections, multi-node setup
- **Communication Latency**: All-reduce, all-gather, P2P operations
- **Bandwidth Measurement**: Actual vs theoretical bandwidth
- **Parallelism Overhead**: TP/PP/DP communication overhead
- **Network Type Detection**: NVLink, InfiniBand, Ethernet

### Theoretical Analysis
- **Peak Performance**: Theoretical FLOPS and memory bandwidth
- **Roofline Model**: Compute vs memory bound analysis
- **Efficiency Metrics**: Actual/theoretical performance ratios
- **Communication Time**: Estimated for different topologies

---

### Local Development

For local development without installing:

```bash
# Run directly with uv
uv run impetus benchmark --model microsoft/phi-2

# Or activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
impetus benchmark --model microsoft/phi-2
```

## 🏗️ Architecture

Impetus is built with a modular architecture:

```
impetus/
├── backends/           # Inference engine backends
│   ├── huggingface.py  # HuggingFace Transformers (default)
│   ├── vllm.py         # vLLM high-throughput backend
│   ├── sglang.py       # SGLang backend
│   └── tensorrt.py     # TensorRT-LLM backend
├── profiling/          # Profiling modules
│   ├── hardware.py     # GPU metrics via NVML
│   ├── system.py       # System-wide metrics
│   ├── memory.py       # Memory profiling
│   ├── cuda.py         # CUDA kernel profiling
│   ├── distributed.py  # Distributed communication profiling
│   └── operators.py    # Layer-level profiling
├── analysis/           # Analysis modules
│   ├── theoretical.py  # Theoretical performance calculations
│   └── model.py        # Model architecture analysis
├── reporting/          # Output formats
│   ├── console.py      # Rich terminal output
│   ├── json_export.py  # JSON export
│   ├── csv_export.py   # CSV export
│   └── html.py         # Interactive HTML reports
├── benchmark.py        # Main orchestrator
└── __main__.py         # CLI entry point
```

---

## 💻 Command-Line Interface

### `impetus benchmark`

Run comprehensive LLM benchmark with hardware profiling.

#### Options:
- `--model TEXT`: Model name or path (required)
- `--backend [huggingface|vllm|sglang|tensorrt]`: Inference backend (default: huggingface)
- `--device [cpu|cuda|auto]`: Device to run on (default: auto)
- `--batch-size INT`: Batch size for inference (default: 1)
- `--sequence-length INT`: Input sequence length (default: 128)
- `--max-new-tokens INT`: Maximum tokens to generate (default: 100)
- `--precision [4bit|8bit|fp16|bf16|fp32]`: Model precision (default: fp16)
- `--num-runs INT`: Number of benchmark iterations (default: 10)
- `--warmup-runs INT`: Number of warmup iterations (default: 2)
- `--profile-level [basic|detailed|comprehensive]`: Profiling detail level (default: detailed)
- `--output-format [text|json|csv|html]`: Output format (default: text)
- `--output-path PATH`: File path for output
- `--verbose`: Enable verbose output with detailed profiling

#### Examples:

```bash
# Basic benchmark
impetus benchmark --model microsoft/phi-2

# Production-like settings
impetus benchmark \
  --model meta-llama/Llama-2-7b-hf \
  --backend vllm \
  --precision fp16 \
  --batch-size 8 \
  --sequence-length 512 \
  --max-new-tokens 200 \
  --num-runs 50

# Comprehensive profiling with all outputs
impetus benchmark \
  --model mistralai/Mistral-7B-v0.1 \
  --precision bf16 \
  --profile-level comprehensive \
  --output-format html \
  --output-path mistral_report.html \
  --verbose
```

### `impetus system-info`

Display detailed hardware and system information.

#### Options:
- `--verbose`: Show detailed system information and metrics

#### Examples:

```bash
# Basic system info
impetus system-info

# Detailed info with current metrics
impetus system-info --verbose
```

### `impetus compare` (Coming Soon)

Compare multiple benchmark results side-by-side.

---

## 📈 Use Cases

### 1. Research Lab: Testing New Hardware

```bash
# Characterize your new 8xA100 node
impetus benchmark --model meta-llama/Llama-2-7b-hf --precision fp16 --batch-size 8 --verbose
```

**What you get:**
- GPU utilization and memory patterns
- NVLink topology and bandwidth
- Bottleneck identification
- Optimization recommendations

### 2. Enterprise: Pre-Production Validation

```bash
# Validate cluster performance before deployment
impetus benchmark \
  --model your-org/custom-model \
  --backend vllm \
  --precision fp16 \
  --batch-size 32 \
  --output-format json \
  --output-path validation_results.json
```

**What you get:**
- Throughput at production batch sizes
- Memory requirements validation
- Cost-per-token estimates
- Exportable results for analysis

### 3. Datacenter: Infrastructure Characterization

```bash
# Profile different GPU configurations
for gpu in A100 H100 L40; do
  impetus benchmark \
    --model meta-llama/Llama-2-70b-hf \
    --backend vllm \
    --precision fp16 \
    --output-format csv \
    --output-path ${gpu}_benchmark.csv
done
```

**What you get:**
- Comparative performance data
- TCO analysis inputs
- Infrastructure planning metrics
- Capacity planning data

### 4. Individual: Local Development

```bash
# Test model on your RTX 4090
impetus benchmark --model microsoft/phi-2 --precision fp16 --batch-size 1
```

**What you get:**
- Real-world inference speeds
- Memory requirements
- Thermal and power characteristics
- Optimization tips for your hardware

---

## 🎨 Output Formats

### Console (Terminal)

Beautiful, colorful terminal output with tables and formatting:

```
═══════════════════════════════════════════════════════════════════════════════
BENCHMARK RESULTS - microsoft/phi-2
═══════════════════════════════════════════════════════════════════════════════

Backend: huggingface
Precision: fp16
Timestamp: 2024-10-22 15:30:45

INFERENCE METRICS
───────────────────────────────────────────────────────────────────────────────
  Batch Size: 1
  Sequence Length: 128
  Avg Latency: 45.23 ms
  Time to First Token: 12.34 ms
  Time per Output Token: 0.33 ms
  Throughput: 3030.30 tokens/sec
  ...
```

### JSON Export

Structured data for programmatic analysis:

```json
{
  "backend": "huggingface",
  "model_name": "microsoft/phi-2",
  "precision": "fp16",
  "inference": {
    "avg_latency_ms": 45.23,
    "throughput_tokens_per_sec": 3030.30,
    ...
  },
  "hardware": {
    "gpu_metrics": [...],
    ...
  }
}
```

### CSV Export

Easy spreadsheet analysis:

```csv
timestamp,backend,model_name,precision,avg_latency_ms,throughput_tokens_per_sec,...
2024-10-22T15:30:45,huggingface,microsoft/phi-2,fp16,45.23,3030.30,...
```

### HTML Report

Interactive web-based reports with charts and visualizations:
- Responsive design
- Interactive Plotly charts
- Detailed metrics tables
- Color-coded recommendations
- Exportable/shareable

---

## 🔧 Advanced Usage

### Using as a Python Library

```python
from impetus import ComprehensiveBenchmark

# Create benchmark instance
benchmark = ComprehensiveBenchmark(
    model_name="microsoft/phi-2",
    backend="huggingface",
    precision="fp16",
    verbose=True
)

# Run benchmark
results = benchmark.run(
    batch_size=1,
    sequence_length=128,
    max_new_tokens=100,
    num_runs=10
)

# Access results
print(f"Throughput: {results.inference.throughput_tokens_per_sec:.2f} tokens/sec")
print(f"Memory Usage: {results.memory.allocated_bytes / (1024**3):.2f} GB")

# Export results
from impetus.reporting import JSONExporter
JSONExporter.export(results, "results.json")
```

### Custom Backend Configuration

```python
from impetus.backends import get_backend, BackendConfig
import torch

# Configure vLLM with tensor parallelism
config = BackendConfig(
    model_name="meta-llama/Llama-2-70b-hf",
    device=torch.device("cuda"),
    precision="fp16",
    backend_kwargs={
        "tensor_parallel_size": 4,
        "gpu_memory_utilization": 0.9
    }
)

backend = get_backend("vllm", config=config)
```

### Profiling Individual Components

```python
from impetus.profiling import HardwareProfiler, MemoryProfiler, CUDAProfiler

# Hardware profiling
hw_profiler = HardwareProfiler()
gpu_info = hw_profiler.gpu_info
current_metrics = hw_profiler.get_current_metrics()

# Memory profiling
mem_profiler = MemoryProfiler()
snapshot = mem_profiler.get_current_snapshot()
peak_stats = mem_profiler.get_peak_memory_stats()

# CUDA profiling
cuda_profiler = CUDAProfiler()
result, metrics = cuda_profiler.profile_function(my_function, *args)
```

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Bug Reports**: Open an issue with detailed information
2. **Feature Requests**: Describe the feature and its use case
3. **Pull Requests**: Fork, create a feature branch, and submit a PR
4. **Documentation**: Help improve docs and examples
5. **Testing**: Test on different hardware configurations

### Development Setup

```bash
git clone https://github.com/dhmnr/impetus.git
cd impetus
uv sync --extra dev
uv run pytest  # Run tests (when available)
```

---

## 📄 License

Impetus is released under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- Built with [PyTorch](https://pytorch.org/)
- Hardware profiling via [NVML](https://developer.nvidia.com/nvidia-management-library-nvml)
- Beautiful terminal output with [Rich](https://rich.readthedocs.io/)
- Visualizations with [Plotly](https://plotly.com/)
- Backend integrations: [HuggingFace](https://huggingface.co/), [vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang), [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)

---

## 📚 Citation

If you use Impetus in your research, please cite:

```bibtex
@software{impetus2024,
  title = {Impetus: Comprehensive GPU Profiling for LLM Workloads},
  author = {Manur, Dheemanth},
  year = {2024},
  url = {https://github.com/dhmnr/impetus}
}
```

---

## 📞 Contact

- **Issues**: [GitHub Issues](https://github.com/dhmnr/impetus/issues)
- **Discussions**: [GitHub Discussions](https://github.com/dhmnr/impetus/discussions)

---

<div align="center">

**⚡ Impetus** - *Know Your Hardware, Optimize Your LLMs*

Made with ❤️ by [Dheemanth Manur](https://github.com/dhmnr)

</div>
