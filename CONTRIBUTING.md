# Contributing to Impetus

Thank you for your interest in contributing to Impetus! This document provides guidelines and instructions for contributing.

## 🌟 Ways to Contribute

- **Bug Reports**: Help us identify and fix issues
- **Feature Requests**: Suggest new features or improvements
- **Code Contributions**: Submit bug fixes or new features
- **Documentation**: Improve docs, add examples, fix typos
- **Testing**: Test on different hardware configurations
- **Review**: Review pull requests from other contributors

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- uv for dependency management (fast, modern Python package installer)
- Git for version control
- NVIDIA GPU (for GPU-related testing)

### Development Setup

1. **Fork the repository**
   ```bash
   # Click "Fork" on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/impetus.git
   cd impetus
   ```

2. **Install dependencies**
   ```bash
   # Install uv if you haven't already
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Or on Windows (PowerShell)
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

   # Install project dependencies
   uv sync

   # Activate virtual environment
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install optional dependencies for testing**
   ```bash
   # For vLLM testing
   uv sync --extra vllm

   # For all backends (vLLM + SGLang)
   uv sync --extra all-backends
   
   # For development tools
   uv sync --extra dev
   
   # For TensorRT-LLM (requires manual installation from NVIDIA)
   uv pip install --extra-index-url https://pypi.nvidia.com/ tensorrt-llm
   ```

4. **Verify installation**
   ```bash
   # Run system info to check setup
   uv run impetus system-info

   # Run a quick benchmark
   uv run impetus benchmark --model gpt2 --num-runs 2
   
   # Or if venv is activated
   impetus system-info
   impetus benchmark --model gpt2 --num-runs 2
   ```

## 📋 Development Workflow

### 1. Create a Branch

```bash
# Create a new branch for your feature/fix
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/bug-description
```

Branch naming conventions:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Test additions or modifications

### 2. Make Your Changes

- Write clean, readable code
- Follow existing code style
- Add type hints to all functions
- Add docstrings to public APIs
- Keep functions focused and testable

### 3. Test Your Changes

```bash
# Run linters
uv run ruff check impetus/

# Run type checker
uv run mypy impetus/

# Test your changes manually
uv run impetus benchmark --model microsoft/phi-2

# Test different configurations
uv run impetus benchmark --model gpt2 --precision fp16
uv run impetus system-info --verbose
```

### 4. Commit Your Changes

```bash
# Stage your changes
git add .

# Commit with a descriptive message
git commit -m "Add feature: descriptive commit message"
```

Commit message guidelines:
- Use present tense ("Add feature" not "Added feature")
- Be descriptive but concise
- Reference issues when applicable (`Fixes #123`)
- Examples:
  - `Add CUDA profiling for kernel execution times`
  - `Fix memory leak in hardware profiler`
  - `Update README with new CLI options`
  - `Refactor backend abstraction layer`

### 5. Push and Create Pull Request

```bash
# Push to your fork
git push origin feature/your-feature-name

# Go to GitHub and create a Pull Request
```

## 📝 Code Style Guidelines

### Python Style

- Follow PEP 8 style guide
- Use 4 spaces for indentation (not tabs)
- Maximum line length: 100 characters
- Use meaningful variable names

### Type Hints

Always include type hints:

```python
def process_metrics(
    data: Dict[str, Any],
    threshold: float = 0.5
) -> List[MetricResult]:
    """Process metrics and return filtered results."""
    pass
```

### Docstrings

Use Google-style docstrings:

```python
def benchmark_inference(
    model: str,
    batch_size: int = 1,
    num_runs: int = 10
) -> BenchmarkResults:
    """
    Run inference benchmark on a model.
    
    Args:
        model: Model name or path
        batch_size: Batch size for inference
        num_runs: Number of benchmark iterations
    
    Returns:
        BenchmarkResults containing all metrics
    
    Raises:
        ValueError: If model not found
        RuntimeError: If CUDA not available
    """
    pass
```

### Error Handling

- Use specific exception types
- Provide helpful error messages
- Gracefully degrade when features unavailable

```python
try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    logging.warning("NVML not available. GPU metrics will be limited.")
```

## 🧪 Testing Guidelines

### Manual Testing

Before submitting a PR, test:

1. **Basic functionality**
   ```bash
   impetus benchmark --model gpt2
   impetus system-info
   ```

2. **Different configurations**
   ```bash
   # Different precisions
   impetus benchmark --model gpt2 --precision fp16
   impetus benchmark --model gpt2 --precision fp32

   # Different backends (if available)
   impetus benchmark --model gpt2 --backend vllm
   ```

3. **Edge cases**
   - CPU-only systems
   - Single GPU
   - Multi-GPU
   - Different output formats

### What to Test

- [ ] Code runs without errors
- [ ] Output looks correct
- [ ] Memory usage is reasonable
- [ ] No performance regression
- [ ] Documentation is updated
- [ ] New features have examples

## 📚 Documentation

### When to Update Documentation

- Adding new features
- Changing CLI interface
- Modifying public APIs
- Adding new configuration options

### What to Update

- **README.md**: User-facing features, examples
- **CHANGELOG.md**: All changes, following format
- **TODO.md**: Completed items, new planned features
- **Docstrings**: All public functions and classes
- **Examples**: Add usage examples for new features

## 🎯 Contribution Areas

### High-Impact Contributions

These contributions have the most value:

1. **Complete Backend Implementations**
   - Full vLLM backend with all features
   - SGLang backend implementation
   - TensorRT-LLM backend implementation

2. **Hardware Support**
   - AMD GPU profiling (ROCm)
   - Apple Silicon support (MPS)
   - Intel GPU support

3. **Advanced Features**
   - Power efficiency analysis
   - Cost analysis and TCO calculations
   - Real-time monitoring dashboard

4. **Testing & Validation**
   - Test on different GPU models (V100, A100, H100, RTX series)
   - Multi-node testing
   - Edge case handling

### Good First Issues

Looking for something to start with?

- Fix documentation typos
- Add code examples
- Improve error messages
- Add unit tests
- Test on different hardware

## 🔍 Code Review Process

### What We Look For

- **Correctness**: Does it work as intended?
- **Style**: Follows project conventions
- **Tests**: Adequate testing coverage
- **Documentation**: Well documented
- **Performance**: No significant regression

### Review Timeline

- Initial review: Within 1-2 weeks
- Follow-up: Based on complexity
- Merge: After approval from maintainers

## 🐛 Reporting Bugs

### Before Reporting

1. Check existing issues
2. Try latest version
3. Reproduce the bug

### Bug Report Template

```markdown
## Description
Clear description of the bug

## To Reproduce
Steps to reproduce:
1. Run command: `impetus benchmark --model X`
2. Observe error: Y

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- OS: [e.g., Ubuntu 22.04]
- Python: [e.g., 3.10.12]
- CUDA: [e.g., 12.1]
- GPU: [e.g., RTX 4090]
- Impetus: [e.g., 0.2.0]

## Additional Context
Error messages, logs, screenshots
```

## 💡 Feature Requests

### Feature Request Template

```markdown
## Problem
Describe the problem this feature would solve

## Proposed Solution
Describe your proposed solution

## Alternatives
Alternative solutions you've considered

## Use Case
Who would use this and how?

## Additional Context
Any other relevant information
```

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Recognition

Contributors will be:
- Listed in the GitHub contributors page
- Mentioned in release notes for significant contributions
- Added to CONTRIBUTORS.md (if it exists)

## 📧 Questions?

- **GitHub Issues**: For bugs and features
- **GitHub Discussions**: For questions and ideas
- **Email**: For private matters

---

Thank you for contributing to Impetus! 🚀

