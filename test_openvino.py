import os
import numpy as np
from transformers import AutoTokenizer

from quantool.core.openvino_engine import (
    export_to_openvino,
    quantize_openvino_int8,
    compress_openvino_int4,
    OpenVINOEvaluator,
)

# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "textattack/bert-base-uncased-SST-2"

FP32_DIR = "./openvino_models/bert_sst2_fp32"
INT8_DIR = "./openvino_models/bert_sst2_int8"
INT4_DIR = "./openvino_models/bert_sst2_int4"

TEXT_SAMPLE = [
    "This movie was absolutely fantastic!",
    "The plot was boring and predictable."
]

WARMUP_RUNS = 5
TIMED_RUNS = 10

# ============================================================
# UTILS
# ============================================================

def model_exists(model_dir: str) -> bool:
    return os.path.exists(os.path.join(model_dir, "openvino_model.xml"))

def get_model_size_mb(model_dir: str) -> float:
    bin_path = os.path.join(model_dir, "openvino_model.bin")
    return os.path.getsize(bin_path) / (1024 * 1024)

def benchmark_model(evaluator, inputs, warmup=WARMUP_RUNS, runs=TIMED_RUNS):
    """
    Warm up model, then return average inference latency.
    """
    for _ in range(warmup):
        evaluator.infer(inputs)

    latencies = []
    for _ in range(runs):
        _, latency = evaluator.infer(inputs)
        latencies.append(latency)

    return sum(latencies) / len(latencies)

# ============================================================
# 1. EXPORT FP32 MODEL
# ============================================================

print("\n=== STEP 1: Export FP32 OpenVINO Model ===")
if not model_exists(FP32_DIR):
    export_to_openvino(MODEL_NAME, FP32_DIR)
else:
    print("FP32 model already exists.")

# ============================================================
# 2. QUANTIZE TO INT8
# ============================================================

print("\n=== STEP 2: Quantize OpenVINO Model to INT8 ===")
if not model_exists(INT8_DIR):
    quantize_openvino_int8(FP32_DIR, INT8_DIR)
else:
    print("INT8 model already exists.")

# ============================================================
# 3. COMPRESS TO INT4 (WEIGHT-ONLY)
# ============================================================

print("\n=== STEP 3: Compress OpenVINO Model to INT4 (Weight-Only) ===")
if not model_exists(INT4_DIR):
    compress_openvino_int4(FP32_DIR, INT4_DIR)
else:
    print("INT4 model already exists.")

# ============================================================
# 4. TOKENIZATION
# ============================================================

print("\n=== STEP 4: Tokenization ===")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

encoded = tokenizer(
    TEXT_SAMPLE,
    padding=True,
    truncation=True,
    return_tensors="np",
)

inputs = {
    "input_ids": encoded["input_ids"].astype(np.int64),
    "attention_mask": encoded["attention_mask"].astype(np.int64),
}

if "token_type_ids" in encoded:
    inputs["token_type_ids"] = encoded["token_type_ids"].astype(np.int64)

# ============================================================
# 5. BENCHMARK FP32
# ============================================================

print("\n=== STEP 5: FP32 OpenVINO Benchmark ===")
fp32_eval = OpenVINOEvaluator(FP32_DIR)
fp32_latency = benchmark_model(fp32_eval, inputs)
print(f"FP32 Avg Latency: {fp32_latency:.2f} ms")

# ============================================================
# 6. BENCHMARK INT8
# ============================================================

print("\n=== STEP 6: INT8 OpenVINO Benchmark ===")
int8_eval = OpenVINOEvaluator(INT8_DIR)
int8_latency = benchmark_model(int8_eval, inputs)
print(f"INT8 Avg Latency: {int8_latency:.2f} ms")

# ============================================================
# 7. BENCHMARK INT4
# ============================================================

print("\n=== STEP 7: INT4 OpenVINO Benchmark (Weight-Only) ===")
int4_eval = OpenVINOEvaluator(INT4_DIR)
int4_latency = benchmark_model(int4_eval, inputs)
print(f"INT4 Avg Latency: {int4_latency:.2f} ms")

# ============================================================
# 8. SANITY CHECK
# ============================================================

print("\n=== STEP 8: Output Sanity Check ===")
fp32_out, _ = fp32_eval.infer(inputs)
int8_out, _ = int8_eval.infer(inputs)
int4_out, _ = int4_eval.infer(inputs)

print("FP32 logits shape:", list(fp32_out.values())[0].shape)
print("INT8 logits shape:", list(int8_out.values())[0].shape)
print("INT4 logits shape:", list(int4_out.values())[0].shape)

# ============================================================
# 9. MODEL SIZE COMPARISON
# ============================================================

print("\n=== STEP 9: Model Size Comparison ===")

fp32_size = get_model_size_mb(FP32_DIR)
int8_size = get_model_size_mb(INT8_DIR)
int4_size = get_model_size_mb(INT4_DIR)

print(f"FP32 Model Size: {fp32_size:.2f} MB")
print(f"INT8 Model Size: {int8_size:.2f} MB")
print(f"INT4 Model Size: {int4_size:.2f} MB")

print(f"\nINT8 Size Reduction: {fp32_size / int8_size:.2f}x")
print(f"INT4 Size Reduction: {fp32_size / int4_size:.2f}x")

# ============================================================
# 10. FINAL SUMMARY
# ============================================================

print("\n=== FINAL SUMMARY ===")
print(f"INT8 Speedup over FP32: {fp32_latency / int8_latency:.2f}x")
print(f"INT4 Speedup over FP32: {fp32_latency / int4_latency:.2f}x")

print("\n✅ Clean, reproducible OpenVINO evaluation completed.")
