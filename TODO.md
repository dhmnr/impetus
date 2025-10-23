# TODO - Impetus Development

## ✅ Completed (v0.2.0)

### Core Infrastructure
- [x] Refactor existing code into modular architecture
- [x] Create backend abstraction layer
- [x] Implement comprehensive metrics dataclasses
- [x] Build orchestration system

### Profiling Modules
- [x] Hardware profiler with NVML integration
- [x] System-wide profiling (CPU, memory, I/O)
- [x] Memory profiler with detailed breakdown
- [x] CUDA kernel profiling
- [x] Distributed communication profiling
- [x] Operator-level profiling

### Analysis Modules
- [x] Theoretical performance calculator
- [x] Model architecture analyzer
- [x] Roofline model implementation
- [x] Bottleneck identification
- [x] Recommendations engine

### Backend Support
- [x] HuggingFace backend (enhanced)
- [x] vLLM backend (basic implementation)
- [x] SGLang backend (stub)
- [x] TensorRT-LLM backend (stub)

### Reporting
- [x] Rich console output
- [x] JSON export
- [x] CSV export
- [x] HTML report generator

### CLI
- [x] Enhanced benchmark command
- [x] system-info command
- [x] compare command (stub)

### Documentation
- [x] Comprehensive README
- [x] CHANGELOG
- [x] Updated TODO

---

## 🚧 In Progress (v0.3.0)

### Backend Implementations
- [ ] Complete vLLM backend with all features
  - [ ] Streaming support
  - [ ] Advanced sampling parameters
  - [ ] Full metrics extraction
- [ ] Complete SGLang backend
  - [ ] Model loading
  - [ ] Generation implementation
  - [ ] Profiling integration
- [ ] Complete TensorRT-LLM backend
  - [ ] Engine loading
  - [ ] Inference implementation
  - [ ] Optimization profiling

### Distributed Profiling
- [ ] Multi-node distributed profiling
- [ ] NCCL operation interception
- [ ] Network topology advanced detection
- [ ] Inter-node vs intra-node breakdown
- [ ] Pipeline parallelism bubble time measurement

### Advanced Features
- [ ] Real-time monitoring mode
- [ ] Live dashboard (web interface)
- [ ] Prometheus metrics export
- [ ] Grafana dashboard templates

---

## 📋 Planned Features

### High Priority

#### Power & Efficiency
- [ ] Power efficiency analysis (tokens/joule)
- [ ] Carbon footprint estimation
- [ ] Energy consumption tracking
- [ ] Thermal efficiency analysis
- [ ] Power cap optimization recommendations

#### Cost Analysis
- [ ] Cloud provider pricing integration (AWS, Azure, GCP)
- [ ] Cost per token calculation
- [ ] TCO analysis for different configurations
- [ ] Cost-performance tradeoff visualization

#### Comparison & Optimization
- [ ] A/B testing mode for model comparison
- [ ] Multi-model comparison reports
- [ ] Historical trend analysis
- [ ] Regression detection
- [ ] Automated optimization workflow

#### Advanced Profiling
- [ ] Custom CUDA kernel profiling
- [ ] Operator fusion opportunity detection
- [ ] Memory access pattern analysis
- [ ] Cache hit/miss rate profiling
- [ ] Quantization impact analysis

### Medium Priority

#### Backend Integrations
- [ ] Inference server profiling (Triton Inference Server)
- [ ] vLLM server endpoint profiling
- [ ] SGLang runtime profiling
- [ ] TGI (Text Generation Inference) support
- [ ] DeepSpeed integration

#### Hardware Support
- [ ] AMD GPU support (ROCm)
- [ ] Apple Silicon (MPS) support
- [ ] Intel GPU support (oneAPI)
- [ ] Google TPU support
- [ ] AWS Inferentia/Trainium support

#### Advanced Analysis
- [ ] Roofline model visualization
- [ ] Flame graphs for execution profiling
- [ ] Memory access patterns visualization
- [ ] Communication pattern visualization
- [ ] Kernel fusion recommendations

#### Automation
- [ ] Auto-tuning for optimal batch size
- [ ] Auto-tuning for optimal precision
- [ ] Auto-scaling recommendations
- [ ] CI/CD integration examples
- [ ] Automated regression testing

### Low Priority

#### Extended Features
- [ ] Fine-tuning performance profiling
- [ ] Data loading profiling
- [ ] Preprocessing pipeline profiling
- [ ] Multi-modal model support (vision-language)
- [ ] Mixture-of-Experts (MoE) specific profiling

#### Reporting Enhancements
- [ ] PDF report generation
- [ ] Markdown report generation
- [ ] LaTeX table generation for papers
- [ ] Jupyter notebook integration
- [ ] Google Sheets export

#### Developer Tools
- [ ] Python API documentation (Sphinx)
- [ ] Tutorial notebooks
- [ ] Video tutorials
- [ ] Best practices guide
- [ ] Troubleshooting guide

#### Integration
- [ ] MLflow integration
- [ ] Weights & Biases integration
- [ ] TensorBoard integration
- [ ] Kubeflow integration
- [ ] Ray integration

---

## 🐛 Known Issues

### Current Limitations
- SGLang backend not fully implemented (stub only)
- TensorRT-LLM backend not fully implemented (stub only)
- Compare command not implemented (stub only)
- Multi-node profiling needs testing
- Some metrics may require elevated permissions (power on some systems)
- Windows support for certain profiling features limited

### Bug Fixes Needed
- [ ] Improve error messages for missing dependencies
- [ ] Add fallback for systems without NVML
- [ ] Handle edge cases in distributed detection
- [ ] Validate all metrics on different GPU architectures
- [ ] Test on Windows with CUDA

---

## 🎯 Goals by Version

### v0.3.0 (Next Release)
- Complete vLLM, SGLang, TensorRT-LLM backends
- Advanced distributed profiling
- Power and efficiency analysis
- Cost analysis features
- Real-time monitoring mode

### v0.4.0
- AMD, Apple Silicon, Intel GPU support
- Inference server profiling
- Advanced visualization features
- Automated optimization tools

### v0.5.0
- Multi-modal model support
- CI/CD integration
- Advanced reporting features
- MLOps platform integrations

### v1.0.0
- Production-ready on all major platforms
- Complete documentation
- Comprehensive test coverage
- Enterprise features (SSO, RBAC, audit logs)
- Professional support options

---

## 🤝 Contributing

Want to help? Pick any item from the TODO list and:
1. Open an issue to discuss the implementation
2. Fork the repository
3. Create a feature branch
4. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## 📝 Notes

### Development Principles
- **Modularity**: Keep components loosely coupled
- **Extensibility**: Easy to add new backends and profilers
- **Performance**: Minimal overhead from profiling
- **User Experience**: Clear outputs and helpful messages
- **Reliability**: Graceful degradation when features unavailable

### Code Quality
- Maintain type hints throughout
- Keep functions focused and testable
- Document all public APIs
- Add tests for new features
- Update documentation with changes

---

Last Updated: 2024-10-22
