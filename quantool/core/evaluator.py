import torch
import time
import os
import tracemalloc
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from typing import Dict, Any, Tuple

def get_model_size(model: torch.nn.Module) -> float:
    """
    Calculates the model's size in megabytes by saving its state dictionary to disk.
    This is the most reliable way to measure the size of a quantized model.
    """
    torch.save(model.state_dict(), "temp_model.p")
    size_mb = os.path.getsize("temp_model.p") / (1024 * 1024)
    os.remove("temp_model.p")
    return size_mb

def evaluate_accuracy(model: torch.nn.Module, data_loader: DataLoader, device: torch.device) -> float:
    """
    Evaluates the model's accuracy on a given dataset.

    Args:
        model (torch.nn.Module): The model to evaluate.
        data_loader (DataLoader): DataLoader for the evaluation data.
        device (torch.device): The device to run evaluation on (e.g., 'cpu', 'cuda').

    Returns:
        float: The accuracy score.
    """
    model.eval()
    model.to(device)
    
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating Accuracy"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return accuracy_score(all_labels, all_preds)

def measure_performance(model: torch.nn.Module, data_loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    """
    Measures latency, peak memory usage, and model size.

    Args:
        model (torch.nn.Module): The model to profile.
        data_loader (DataLoader): DataLoader to get a sample batch for profiling.
        device (torch.device): The device to run profiling on.

    Returns:
        Dict[str, Any]: A dictionary with performance metrics.
    """
    model.eval()
    model.to(device)

    # --- Latency Measurement ---
    latencies = []
    sample_batch = next(iter(data_loader))
    input_ids = sample_batch['input_ids'].to(device)
    attention_mask = sample_batch['attention_mask'].to(device)
    
    # Warm-up run to load model onto GPU, etc.
    with torch.no_grad():
        _ = model(input_ids, attention_mask=attention_mask)

    # Timed runs for stable measurement
    num_runs = 50
    for _ in tqdm(range(num_runs), desc="Measuring Latency"):
        start_time = time.perf_counter()
        with torch.no_grad():
            _ = model(input_ids, attention_mask=attention_mask)
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000) # Convert to milliseconds

    avg_latency_ms = np.mean(latencies)
    latency_std_dev = np.std(latencies)

    # --- Memory Measurement ---
    tracemalloc.start()
    with torch.no_grad():
        _ = model(input_ids, attention_mask=attention_mask)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_memory_mb = peak / (1024 * 1024)

    # --- Model Size ---
    model_size_mb = get_model_size(model)

    return {
        "avg_latency_ms": avg_latency_ms,
        "latency_std_dev": latency_std_dev,
        "peak_memory_mb": peak_memory_mb,
        "model_size_mb": model_size_mb,
    }

def run_evaluation_pipeline(model: torch.nn.Module, tokenizer, dataset_info: Dict) -> Dict[str, Any]:
    """
    Runs the full evaluation pipeline: accuracy and performance metrics.
    
    Args:
        model (torch.nn.Module): The model to evaluate.
        tokenizer: The model's tokenizer.
        dataset_info (Dict): Dictionary with dataset name and subset.
        
    Returns:
        Dict[str, Any]: A dictionary containing all evaluation results.
    """
    # Dynamically select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on device: {device}")

    # Lazily import to avoid circular dependency if this file grows
    from quantool.benchmarks.datasets import load_and_prepare_dataset
    
    data_loader = load_and_prepare_dataset(
        dataset_name=dataset_info['name'],
        subset=dataset_info['subset'],
        tokenizer=tokenizer
    )

    accuracy = evaluate_accuracy(model, data_loader, device)
    performance_metrics = measure_performance(model, data_loader, device)

    return {
        "accuracy": accuracy,
        **performance_metrics
    }

