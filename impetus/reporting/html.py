"""HTML report generator with interactive visualizations."""

from pathlib import Path
from typing import Union
import json

from ..metrics import ComprehensiveBenchmarkResults


class HTMLReporter:
    """Generate interactive HTML reports with charts."""
    
    @staticmethod
    def generate(results: ComprehensiveBenchmarkResults, output_path: Union[str, Path]) -> None:
        """
        Generate an interactive HTML report.
        
        Args:
            results: Comprehensive benchmark results
            output_path: Path to output HTML file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert results to dict for JavaScript
        results_dict = results.to_dict()
        results_json = json.dumps(results_dict, indent=2)
        
        # Generate HTML
        html_content = HTMLReporter._generate_html(results, results_json)
        
        # Write to file with UTF-8 encoding (important for Windows)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    @staticmethod
    def _generate_html(results: ComprehensiveBenchmarkResults, results_json: str) -> str:
        """Generate HTML content."""
        
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Impetus Benchmark Report - {results.model_name}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        .header {{
            background: white;
            border-radius: 16px;
            padding: 32px;
            margin-bottom: 24px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        }}
        
        .header h1 {{
            font-size: 2.5em;
            color: #667eea;
            margin-bottom: 8px;
        }}
        
        .header .subtitle {{
            color: #666;
            font-size: 1.1em;
        }}
        
        .meta-info {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-top: 24px;
        }}
        
        .meta-item {{
            background: #f8f9fa;
            padding: 16px;
            border-radius: 8px;
        }}
        
        .meta-item .label {{
            font-size: 0.85em;
            color: #666;
            margin-bottom: 4px;
        }}
        
        .meta-item .value {{
            font-size: 1.2em;
            font-weight: 600;
            color: #333;
        }}
        
        .section {{
            background: white;
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }}
        
        .section h2 {{
            font-size: 1.8em;
            color: #667eea;
            margin-bottom: 20px;
            padding-bottom: 12px;
            border-bottom: 3px solid #667eea;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 24px;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }}
        
        .metric-card .metric-label {{
            font-size: 0.9em;
            opacity: 0.9;
            margin-bottom: 8px;
        }}
        
        .metric-card .metric-value {{
            font-size: 2em;
            font-weight: 700;
        }}
        
        .metric-card .metric-unit {{
            font-size: 0.8em;
            opacity: 0.8;
        }}
        
        .table-container {{
            overflow-x: auto;
            margin-top: 20px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        
        th {{
            background: #f8f9fa;
            padding: 12px;
            text-align: left;
            font-weight: 600;
            border-bottom: 2px solid #dee2e6;
        }}
        
        td {{
            padding: 12px;
            border-bottom: 1px solid #dee2e6;
        }}
        
        .chart-container {{
            margin-top: 24px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
        }}
        
        .alert {{
            padding: 16px;
            border-radius: 8px;
            margin-bottom: 16px;
        }}
        
        .alert-warning {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            color: #856404;
        }}
        
        .alert-info {{
            background: #d1ecf1;
            border-left: 4px solid #17a2b8;
            color: #0c5460;
        }}
        
        .alert h3 {{
            font-size: 1.1em;
            margin-bottom: 8px;
        }}
        
        .alert ul {{
            margin-left: 20px;
        }}
        
        .footer {{
            text-align: center;
            color: white;
            padding: 24px;
            margin-top: 40px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>⚡ Impetus Benchmark Report</h1>
            <p class="subtitle">{results.model_name}</p>
            
            <div class="meta-info">
                <div class="meta-item">
                    <div class="label">Backend</div>
                    <div class="value">{results.backend}</div>
                </div>
                <div class="meta-item">
                    <div class="label">Precision</div>
                    <div class="value">{results.precision}</div>
                </div>
                <div class="meta-item">
                    <div class="label">Timestamp</div>
                    <div class="value">{results.timestamp[:19]}</div>
                </div>
                <div class="meta-item">
                    <div class="label">Batch Size</div>
                    <div class="value">{results.inference.batch_size}</div>
                </div>
            </div>
        </div>
        
        <!-- Inference Metrics -->
        <div class="section">
            <h2>📊 Inference Performance</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Average Latency</div>
                    <div class="metric-value">{results.inference.avg_latency_ms:.2f}</div>
                    <div class="metric-unit">milliseconds</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Time to First Token</div>
                    <div class="metric-value">{results.inference.avg_time_to_first_token_ms:.2f}</div>
                    <div class="metric-unit">milliseconds</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Throughput</div>
                    <div class="metric-value">{results.inference.throughput_tokens_per_sec:.1f}</div>
                    <div class="metric-unit">tokens/second</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Time per Token</div>
                    <div class="metric-value">{results.inference.time_per_output_token_ms:.2f}</div>
                    <div class="metric-unit">milliseconds</div>
                </div>
            </div>
        </div>
        
        <!-- Model Information -->
        <div class="section">
            <h2>🧠 Model Architecture</h2>
            <div class="table-container">
                <table>
                    <tr>
                        <th>Property</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Architecture</td>
                        <td>{results.model.architecture_type}</td>
                    </tr>
                    <tr>
                        <td>Total Parameters</td>
                        <td>{results.model.total_params:,}</td>
                    </tr>
                    <tr>
                        <td>Number of Layers</td>
                        <td>{results.model.num_layers}</td>
                    </tr>
                    <tr>
                        <td>Hidden Size</td>
                        <td>{results.model.hidden_size}</td>
                    </tr>
                    <tr>
                        <td>Attention Heads</td>
                        <td>{results.model.num_attention_heads}</td>
                    </tr>
                    <tr>
                        <td>Parameter Memory</td>
                        <td>{results.model.param_memory_mb:.2f} MB</td>
                    </tr>
                </table>
            </div>
        </div>
        
        <!-- Memory Metrics -->
        <div class="section">
            <h2>💾 Memory Usage</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Allocated</div>
                    <div class="metric-value">{results.memory.allocated_bytes / (1024**3):.2f}</div>
                    <div class="metric-unit">GB</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Peak Allocated</div>
                    <div class="metric-value">{results.memory.peak_allocated_bytes / (1024**3):.2f}</div>
                    <div class="metric-unit">GB</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Total GPU Memory</div>
                    <div class="metric-value">{results.memory.total_memory_bytes / (1024**3):.2f}</div>
                    <div class="metric-unit">GB</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Utilization</div>
                    <div class="metric-value">{results.memory.memory_utilization_percent:.1f}</div>
                    <div class="metric-unit">%</div>
                </div>
            </div>
            
            <div class="chart-container">
                <div id="memoryChart"></div>
            </div>
        </div>
        
        <!-- Hardware Metrics -->
        {"".join([f'''
        <div class="section">
            <h2>🖥️ GPU {i} - {gpu.get("name", "Unknown")}</h2>
            <div class="table-container">
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Utilization</td>
                        <td>{gpu.get("utilization_gpu", 0)}%</td>
                    </tr>
                    <tr>
                        <td>Temperature</td>
                        <td>{gpu.get("temperature", 0)}°C</td>
                    </tr>
                    <tr>
                        <td>Power Draw</td>
                        <td>{gpu.get("power_draw", 0):.1f} W</td>
                    </tr>
                    <tr>
                        <td>Memory Used</td>
                        <td>{gpu.get("memory_used", 0) / (1024**3):.2f} GB / {gpu.get("memory_total", 0) / (1024**3):.2f} GB</td>
                    </tr>
                </table>
            </div>
        </div>
        ''' for i, gpu in enumerate(results.hardware.gpu_metrics)])}
        
        <!-- Bottlenecks and Recommendations -->
        {"" if not results.bottlenecks else f'''
        <div class="section">
            <div class="alert alert-warning">
                <h3>⚠️ Identified Bottlenecks</h3>
                <ul>
                    {"".join([f"<li>{b}</li>" for b in results.bottlenecks])}
                </ul>
            </div>
        </div>
        '''}
        
        {"" if not results.recommendations else f'''
        <div class="section">
            <div class="alert alert-info">
                <h3>💡 Recommendations</h3>
                <ul>
                    {"".join([f"<li>{r}</li>" for r in results.recommendations])}
                </ul>
            </div>
        </div>
        '''}
        
        <div class="footer">
            <p>Generated by <strong>Impetus v0.2.0</strong> - Comprehensive GPU Profiling for LLMs</p>
        </div>
    </div>
    
    <script>
        // Store results data
        const resultsData = {results_json};
        
        // Create memory chart
        const memoryData = [{{
            type: 'bar',
            x: ['Allocated', 'Peak Allocated', 'Total'],
            y: [
                resultsData.memory.allocated_bytes / (1024**3),
                resultsData.memory.peak_allocated_bytes / (1024**3),
                resultsData.memory.total_memory_bytes / (1024**3)
            ],
            marker: {{
                color: ['#667eea', '#764ba2', '#48bb78']
            }}
        }}];
        
        const memoryLayout = {{
            title: 'GPU Memory Usage (GB)',
            xaxis: {{ title: 'Memory Type' }},
            yaxis: {{ title: 'Memory (GB)' }},
            paper_bgcolor: '#f8f9fa',
            plot_bgcolor: '#f8f9fa'
        }};
        
        Plotly.newPlot('memoryChart', memoryData, memoryLayout, {{responsive: true}});
    </script>
</body>
</html>
"""

