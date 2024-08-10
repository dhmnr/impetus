import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

model_name = "microsoft/phi-2"  # or any other model you want to benchmark
model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True,  device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prepare input
text = "Hello, how are you? " * 10  # Repeat to make the input longer
inputs = tokenizer(text, return_tensors="pt").to('cuda')

# Dictionary to store timing results
layer_start = {}
layer_end = {}
layer_times = {}
# CUDA events for accurate GPU timing
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

# Hook function to measure layer time
def pre_hook_fn(module, input):
    torch.cuda.synchronize()
    torch.cuda.current_stream().record_event(start_event)
    torch.cuda.synchronize()

        
    
def post_hook_callable(layer_name):
    def post_hook_fn(module, input, output):
        torch.cuda.synchronize()
        print(torch.cuda.current_stream())
        torch.cuda.current_stream().record_event(end_event)
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) 
        if layer_name not in layer_times:
            layer_times[layer_name] = [elapsed_time]
        else:
            layer_times[layer_name].append(elapsed_time)
    return post_hook_fn

# Register hooks for all decoder layers
for name, module in model.named_modules():
    
    if "PhiDecoderLayer" in str(module.__class__.__name__):
        module.register_forward_pre_hook(pre_hook_fn)
        module.register_forward_hook(post_hook_callable(name))

# Warmup run
with torch.no_grad():
    _ = model(**inputs)

# Benchmark run
torch.cuda.synchronize()
with torch.no_grad():
    outputs = model(**inputs)
torch.cuda.synchronize()

# Print results
print("Average time per layer type:")
for layer_name, times in layer_times.items():
    avg_time = sum(times[1:]) / len(times[1:])
    print(f"{layer_name}: {avg_time:.6f} ms")

# Detailed breakdown
# print("\nDetailed breakdown:")
# for layer_name, times in layer_times.items():
#     print(f"\n{layer_name}:")
#     for i, t in enumerate(times):
#         print(f"  Layer {i}: {t:.6f} ms")

# Calculate total inference time
total_time = sum([sum(times) for times[1:] in layer_times.values()])
print(f"\nTotal inference time: {total_time:.6f} ms")