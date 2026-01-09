import time
import torch
import psutil
import os
import numpy as np
from typing import Dict, Any
from sklearn.metrics import accuracy_score
from tqdm import tqdm


class PyTorchEvaluator:
    """
    Standardized PyTorch evaluator for QuanTool.
    Measures accuracy, latency, memory, and model size.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    # --------------------------------------------------
    # Inference
    # --------------------------------------------------
    def infer(self, inputs: Dict[str, torch.Tensor]):
        """
        Run a single forward pass and measure latency.
        """
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            start = time.perf_counter()
            outputs = self.model(**inputs)
            latency_ms = (time.perf_counter() - start) * 1000

        return outputs, latency_ms

    # --------------------------------------------------
    # Accuracy
    # --------------------------------------------------
    def evaluate_accuracy(self, dataloader) -> float:
        all_preds, all_labels = [], []

        for batch in tqdm(dataloader, desc="Evaluating accuracy"):
            inputs = {
                "input_ids": batch["input_ids"].to(self.device),
                "attention_mask": batch["attention_mask"].to(self.device),
            }
            labels = batch["labels"].to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                preds = outputs.logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        return accuracy_score(all_labels, all_preds)

    # --------------------------------------------------
    # Latency (batched, per-sample)
    # --------------------------------------------------
    def benchmark_latency(self, dataloader, num_batches: int = 50) -> float:
        latencies = []

        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break

            inputs = {
                "input_ids": batch["input_ids"].to(self.device),
                "attention_mask": batch["attention_mask"].to(self.device),
            }

            batch_size = inputs["input_ids"].shape[0]
            _, latency = self.infer(inputs)
            latencies.append(latency / batch_size)

        return float(np.mean(latencies))

    # --------------------------------------------------
    # Model Size
    # --------------------------------------------------
    def get_model_size_mb(self) -> float:
        param_size = sum(
            p.nelement() * p.element_size() for p in self.model.parameters()
        )
        buffer_size = sum(
            b.nelement() * b.element_size() for b in self.model.buffers()
        )
        return (param_size + buffer_size) / (1024 ** 2)

    # --------------------------------------------------
    # Peak Memory
    # --------------------------------------------------
    def get_peak_memory_mb(self) -> float:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 ** 2)
