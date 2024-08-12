from dataclasses import dataclass


@dataclass
class DecoderOnlyLanguageModelMetrics:
    """Class for keeping track of Decoder LM Metrics."""

    seq_len: int
    batch_size: int
    latency: float
    tokens_per_batch: float
    throughput: float
    time_per_output_token: float
    time_to_first_token: float

    def __str__(self):
        return (
            f"  Sequence length: {self.seq_len}\n"
            f"  Batch_size: {self.batch_size}\n"
            f"  Latency: {self.latency:.4f} seconds\n"
            f"  Tokens generated per batch: {self.tokens_per_batch:.4f}\n"
            f"  Throughput: {self.throughput:.2f} tokens/second\n"
            f"  Time per output token: {self.time_per_output_token:.4f} seconds\n"
            f"  Time to first token: {self.time_to_first_token:.4f} seconds\n"
        )
