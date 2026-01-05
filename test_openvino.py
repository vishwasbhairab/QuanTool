import torch
import numpy as np
from transformers import AutoTokenizer
from quantool.core.openvino_engine import (
    export_to_openvino,
    quantize_openvino_int8,
    OpenVINOEvaluator,
)

# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "textattack/bert-base-uncased-SST-2"
FP32_DIR = "./openvino_models/bert_sst2_fp32"
INT8_DIR = "./openvino_models/bert_sst2_int8"

TEXT_SAMPLE = [
    "This movie was absolutely fantastic!",
    "The plot was boring and predictable."
]

# ============================================================
# 1. EXPORT FP32 MODEL (if not exists)
# ============================================================

print("\n=== STEP 1: Export FP32 OpenVINO Model ===")
export_to_openvino(MODEL_NAME, FP32_DIR)

# ============================================================
# 2. QUANTIZE TO INT8 (if not exists)
# ============================================================

print("\n=== STEP 2: Quantize OpenVINO Model to INT8 ===")
quantize_openvino_int8(FP32_DIR, INT8_DIR)

# ============================================================
# 3. TOKENIZATION
# ============================================================

print("\n=== STEP 3: Tokenizing Input ===")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

encoded = tokenizer(
    TEXT_SAMPLE,
    padding=True,
    truncation=True,
    return_tensors="np",   # IMPORTANT for OpenVINO
)

inputs = {
    "input_ids": encoded["input_ids"].astype(np.int64),
    "attention_mask": encoded["attention_mask"].astype(np.int64),
    "token_type_ids": np.zeros_like(encoded["input_ids"], dtype=np.int64),
}

# ============================================================
# 4. RUN FP32 INFERENCE
# ============================================================

print("\n=== STEP 4: FP32 OpenVINO Inference ===")
fp32_evaluator = OpenVINOEvaluator(FP32_DIR)
fp32_outputs, fp32_latency = fp32_evaluator.infer(inputs)

print(f"FP32 Latency: {fp32_latency:.2f} ms")

# ============================================================
# 5. RUN INT8 INFERENCE
# ============================================================

print("\n=== STEP 5: INT8 OpenVINO Inference ===")
int8_evaluator = OpenVINOEvaluator(INT8_DIR)
int8_outputs, int8_latency = int8_evaluator.infer(inputs)

print(f"INT8 Latency: {int8_latency:.2f} ms")

# ============================================================
# 6. QUICK SANITY CHECK
# ============================================================

print("\n=== STEP 6: Output Sanity Check ===")
print("FP32 logits shape:", fp32_outputs[list(fp32_outputs.keys())[0]].shape)
print("INT8 logits shape:", int8_outputs[list(int8_outputs.keys())[0]].shape)

speedup = fp32_latency / int8_latency
print(f"\nINT8 Speedup over FP32: {speedup:.2f}x")

print("\n✅ OpenVINO test completed successfully.")
