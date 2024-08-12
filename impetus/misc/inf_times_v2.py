import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from transformers import BitsAndBytesConfig


def load_quantized_model(model_name, quantization):
    if quantization == "fp32":
        return AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
    elif quantization == "int8":
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map="auto",
        )
    elif quantization == "int4":
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype="fp4"
            ),
            device_map="auto",
        )
    else:
        raise ValueError(f"Unsupported quantization: {quantization}")


def benchmark_inference(model, tokenizer, input_text, num_runs=10):
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to("cuda")

    # Warm-up run
    model.generate(input_ids, max_new_tokens=100)

    total_time = 0
    for _ in range(num_runs):
        start_time = time.time()
        model.generate(input_ids, max_new_tokens=100)
        end_time = time.time()
        total_time += end_time - start_time

    return total_time / num_runs


def main():
    model_name = "microsoft/phi-2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load a small subset of a dataset for benchmarking
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    input_text = dataset[0][
        "instruction"
    ]  # Use the first 500 characters of the first example

    quantization_levels = ["fp32", "int8", "int4"]
    results = {}

    for quant in quantization_levels:
        print(f"Benchmarking {quant} quantization...")
        model = load_quantized_model(model_name, quant)
        model.eval()

        avg_time = benchmark_inference(model, tokenizer, input_text)
        results[quant] = avg_time
        print(f"{quant} average inference time: {avg_time:.4f} seconds")

        del model  # Free up memory
        torch.cuda.empty_cache()

    print("\nFinal Results:")
    for quant, time in results.items():
        print(f"{quant}: {time:.4f} seconds")


if __name__ == "__main__":
    main()
