import click
import torch

from .models.decoderLm import DecoderLm

@click.group()
@click.version_option(version='0.1.0')
def cli():
    """CLI for benchmarking LLM runtime performance on local GPU or CPU."""
    pass

@cli.command()
@click.option('--model', type=click.STRING, required=True, help='LLM model to benchmark.')
@click.option('--device', type=click.Choice(['cpu', 'cuda','auto']), default='auto', help='Device to run the benchmark on.')
@click.option('--batch-size', type=int, default=1, help='Batch size for inference.')
@click.option('--seq_len', type=int, default=128, help='Input sequence length.')
@click.option('--precision', type=click.Choice(['4bit', '8bit', 'full']), default='full', help='Precison to use for the models')
@click.option('--num-runs', type=int, default=10, help='Number of inference iterations.')
@click.option('--warmup-runs', type=int, default=1, help='Number of warmup iterations.')
@click.option('--custom-model-path', type=click.Path(exists=True), help='Path to custom model weights.')
@click.option('--output-format', type=click.Choice(['text', 'json', 'csv']), default='text', help='Output format for results.')
@click.option('--save-trace', is_flag=True, help='Save execution trace for profiling.')
@click.option('--verbose', is_flag=True, help='Enable verbose output.')
def benchmark(model, device, batch_size, seq_len, precision, num_runs, warmup_runs, custom_model_path, output_format, save_trace, verbose):
    """Run LLM benchmark with specified options."""
    if device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    click.echo(f"Benchmarking {model} on {device}")
    click.echo(f"Batch size: {batch_size}, Sequence length: {seq_len}")
    click.echo(f"Running {num_runs} iterations with {warmup_runs} warmup iterations")

    lm = DecoderLm(model, device=device, precision=precision)
    metrics = lm.benchmark_inference(seq_len=seq_len, num_runs=num_runs, warmup_runs=warmup_runs)

    
    click.echo("Benchmark completed.\n")
    click.echo("****** Performance Metrics ******\n")
    click.echo(metrics)

    # click.echo("Average inference time: 10ms")
    # click.echo("Throughput: 100 sequences/second")

# @cli.command()
# @click.option('--device', type=click.Choice(['cpu', 'gpu']), required=True, help='Device to profile.')
# def profile_device(device):
#     """Profile the specified device (CPU or GPU)."""
#     click.echo(f"Profiling {device.upper()} capabilities")
#     # Placeholder for device profiling logic
#     click.echo(f"{device.upper()} profile completed")

if __name__ == '__main__':
    cli()