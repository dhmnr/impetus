from dataclasses import dataclass


@dataclass
class DecoderLmMetrics():
    """Class for keeping track of Decoder LM Metrics."""
    latency: float
    throughput: float
    time_per_output_token: float
    time_to_first_token: float 

    def __str__(self):
        return (
            f"Decoder LM Metrics:\n"
            f"  Latency: {self.latency:.4f} seconds\n"
            f"  Throughput: {self.throughput:.2f} tokens/second\n"
            f"  Time per output token: {self.time_per_output_token:.4f} seconds\n"
            f"  Time to first token: {self.time_to_first_token:.4f} seconds\n"
        )