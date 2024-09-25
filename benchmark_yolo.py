# implement the benchmark for YOLOv8

import torch
from contextlib import contextmanager
import torch
import numpy as np
from codecarbon import EmissionsTracker
import time
from tqdm import tqdm

Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB


def format_number(num):
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return f"{num:.2f}"


@contextmanager
def track_gpu_memory():
    """
    Context manager to track GPU memory usage during inference
    """
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    try:
        yield
    finally:
        max_memory = torch.cuda.max_memory_allocated() / 1e6
        current_memory = torch.cuda.memory_allocated() / 1e6

    # Store the memory usage in the context manager object
    track_gpu_memory.max_memory = max_memory
    track_gpu_memory.current_memory = current_memory



def benchmark_yolo(model, imgsz):
    # input data
    input_data = np.random.rand(imgsz, imgsz, 3).astype(np.float32)

    # Warmup runs for gpu
    for _ in range(3):
        for _ in range(20):
            model(input_data, imgsz=imgsz, verbose=False)

    # Compute number of runs as higher of min_time or num_timed_runs
    num_runs = 5000

    # Timed runs
    with EmissionsTracker(log_level='critical') as tracker:
        run_times = []
        for _ in tqdm(range(num_runs)):
            results = model(input_data, imgsz=imgsz, verbose=False)
            run_times.append(results[0].speed["inference"])  # Convert to milliseconds

    # Compute statistics
    run_times = np.array(run_times)
    mean_time = np.mean(run_times)
    std_time = np.std(run_times)
    fps = 1000 / mean_time
    print(f"Mean inference time: {mean_time:.2f} ms")
    print(f'FPS GPU: {1000/mean_time:.2f}')
    # get emissions
    total_emissions = tracker.final_emissions
    total_energy = tracker.final_emissions_data.energy_consumed

    # average energy per inference
    avg_emissions = total_emissions / num_runs
    avg_energy = total_energy / num_runs

    return mean_time, std_time, fps, total_emissions, total_energy, avg_emissions, avg_energy

@torch.no_grad()
def benchmark(model, imgsz, type='pt'):
    input_data = np.random.rand(imgsz, imgsz, 3).astype(np.float32)

    with track_gpu_memory():
        mean_syn_gpu, std_syn_gpu, fps_gpu, total_emissions, total_energy, avg_emissions, avg_energy = benchmark_yolo(model, imgsz)
    max_memory_used = track_gpu_memory.max_memory
    current_memory_used = track_gpu_memory.current_memory

    if type == "pt": # si .engine on ne peut pas get les infos comme ça
        model.fuse()
        layers, params, gradients, flops = model.info()
        print("benchmark results base model")
        print('----------------------------')
        print('Number of layers: ', layers)
        print(f"Number of parameters: {format_number(params)}")
        print(f"Number of FLOPs: {format_number(flops)}")

    print("\nMemory Usage:")
    print(f"  Max memory used: {format_number(max_memory_used)} MB")
    print(f"  Current memory used: {format_number(current_memory_used)} MB")

    print("\nPerformance:")
    print(f"  GPU: {fps_gpu:.2f} FPS (±{std_syn_gpu:.2f} ms)")

    print("\nEnergy Consumption:")
    print(f"  Total emissions: {total_emissions*1e3:.8f} gCO2")
    print(f"  Total energy: {total_energy*1e3:.8f} Wh")

    print("\nAverage Energy Consumption Per Inference:")
    print(f"  Average emissions: {avg_emissions*1e3:.8f} gCO2")
    print(f"  Average energy: {avg_energy*1e3:.8f} Wh")

    # Eval model
    metrics = model.val(
        data="data.yaml",
        batch=1,
        imgsz=640,
        verbose=False,
        device="cuda")

    map5095 = metrics.box.map
    map50 = metrics.box.map50


    # return a dict with all the results
    return {
        "max_memory_used": max_memory_used,
        "current_memory_used": current_memory_used,
        "fps_gpu": fps_gpu,
        "total_emissions": total_emissions,
        "total_energy": total_energy,
        "avg_emissions": avg_emissions,
        "avg_energy": avg_energy,
        "map50": map50,
        "map5095": map5095
    }