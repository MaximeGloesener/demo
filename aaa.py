import subprocess
import time
import torch


def get_gpu_memory():
    """Fetch GPU memory usage using nvidia-smi."""
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.free,memory.total', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE)
    memory_info = result.stdout.decode().strip().split(',')
    used_memory = int(memory_info[0])
    free_memory = int(memory_info[1])
    total_memory = int(memory_info[2])
    return used_memory, free_memory, total_memory

def track_max_gpu_memory(model, inference_fn, num_iterations=10):
    """
    Track the maximum GPU memory usage during inference.

    Args:
    - inference_fn: The inference function to run.
    - num_iterations: Number of times to run the inference to monitor memory usage.

    Returns:
    - max_memory_used: The maximum memory used during inference.
    """
    max_memory_used = 0

    for _ in range(num_iterations):
        # Before inference: check GPU memory
        used, _, _ = get_gpu_memory()

        # Run inference (simulating with the provided function)
        inference_fn(model)

        # After inference: check GPU memory
        post_used, _, _ = get_gpu_memory()

        # Track the maximum memory usage
        max_memory_used = max(max_memory_used, post_used)

        # Optional: Add a short sleep to avoid overloading the system with checks
        time.sleep(0.1)

    return max_memory_used

from pytorch_bench import benchmark

# Example inference function (replace with actual model inference)
def dummy_inference(model):
    """Simulate model inference by adding a sleep to mimic processing time."""
    out = model(torch.randn(1, 3, 224, 224, device='cuda'))

model = torch.load("model_full2.pth", map_location=torch.device('cpu' if not torch.cuda.is_available() else 'cuda'))

# Run the memory tracking
max_memory = track_max_gpu_memory(model, dummy_inference, num_iterations=20)
print(f"Maximum GPU Memory Used: {max_memory} MB")

results = benchmark(model, torch.randn(1, 3, 224, 224), gpu_only=True)
