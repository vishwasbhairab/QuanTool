import argparse
import os
import numpy as np
import pandas as pd
import torch
from quantool.models import model_loader
from quantool.core import quantizer, evaluator
from quantool.benchmarks.datasets import load_and_prepare_dataset


# --------------------------------------------------
# Validation
# --------------------------------------------------
def validate_backend_precision(backend, precision):
    if backend == "pytorch" and precision == "int4":
        raise ValueError("INT4 precision is only supported with OpenVINO backend.")


# --------------------------------------------------
# Main
# --------------------------------------------------
def main(args):
    print("\n--- Starting QuanTool Benchmark ---")

    validate_backend_precision(args.backend, args.precision)

    MODEL_NAME = args.model_name
    TASK_NAME = args.task

    print(f"\nModel    : {MODEL_NAME}")
    print(f"Task     : {TASK_NAME}")
    print(f"Backend  : {args.backend}")
    print(f"Precision: {args.precision}")

    # --------------------------------------------------
    # 1. Load HF model ONCE (source of truth)
    # --------------------------------------------------
    hf_model, tokenizer = model_loader.load_model_and_tokenizer(
        MODEL_NAME, TASK_NAME
    )

    id2label = hf_model.config.id2label
    label2id = hf_model.config.label2id

    print("Label mapping:", id2label)

    # --------------------------------------------------
    # 2. Dataset
    # --------------------------------------------------
    eval_loader = load_and_prepare_dataset(
        dataset_name="glue",
        subset=TASK_NAME,
        tokenizer=tokenizer,
        split="validation",
        batch_size=8,
    )

    # --------------------------------------------------
    # 3. Backend-specific setup
    # --------------------------------------------------
    if args.backend == "pytorch":
        model = hf_model

        if args.precision == "int8":
            print("\nApplying PyTorch INT8 dynamic quantization...")
            model = quantizer.quantize_int8_dynamic(model)

        evaluator_instance = evaluator.PyTorchEvaluator(model)

    elif args.backend == "openvino":
        from quantool.core.openvino_engine import (
            export_to_openvino,
            quantize_openvino_int8,
            compress_openvino_int4,
            OpenVINOEvaluator,
        )

        base = MODEL_NAME.replace("/", "_")
        fp32_dir = f"./openvino_models/{base}_{TASK_NAME}_fp32"
        int8_dir = f"./openvino_models/{base}_{TASK_NAME}_int8"
        int4_dir = f"./openvino_models/{base}_{TASK_NAME}_int4"

        # 🔑 ALWAYS load HF fine-tuned model ONCE
        hf_model, _ = model_loader.load_model_and_tokenizer(
            MODEL_NAME, TASK_NAME
        )

        # 🔑 Export fine-tuned model ONLY if needed
        if not os.path.exists(fp32_dir):
            export_to_openvino(
                hf_model.name_or_path,   # ✅ fine-tuned checkpoint
                fp32_dir
            )

        # Select precision
        if args.precision == "fp32":
            model_dir = fp32_dir

        elif args.precision == "int8":
            if not os.path.exists(int8_dir):
                quantize_openvino_int8(fp32_dir, int8_dir)
            model_dir = int8_dir

        elif args.precision == "int4":
            if not os.path.exists(int4_dir):
                compress_openvino_int4(fp32_dir, int4_dir)
            model_dir = int4_dir

        # ✅ Pass model_config for correct labels
        evaluator_instance = OpenVINOEvaluator(
            model_dir=model_dir,
            device="CPU",
            model_config=hf_model.config
        )


    else:
        raise ValueError("Unsupported backend")

# --------------------------------------------------
# Warm-up (important for fair latency)
# --------------------------------------------------
    print("\n--- Warm-up ---")

    warmup_batch = next(iter(eval_loader))

        # PyTorch warm-up
    if args.backend == "pytorch":
        warmup_inputs = {
            "input_ids": warmup_batch["input_ids"],
            "attention_mask": warmup_batch["attention_mask"],
        }

        # OpenVINO warm-up
    elif args.backend == "openvino":
        warmup_inputs = {
            "input_ids": warmup_batch["input_ids"].cpu().numpy(),
            "attention_mask": warmup_batch["attention_mask"].cpu().numpy(),
        }

        if "token_type_ids" in evaluator_instance.input_names:
            warmup_inputs["token_type_ids"] = np.zeros_like(
                warmup_inputs["input_ids"]
            )

        # Run warm-up iterations
    for _ in range(5):
        evaluator_instance.infer(warmup_inputs)


    # --------------------------------------------------
    # 5. Evaluation
    # --------------------------------------------------
    print("\n--- Running Evaluation ---")

    latencies = []
    correct = 0
    total = 0

    for batch in eval_loader:

        # PyTorch backend
        if args.backend == "pytorch":
            inputs = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
            }


        # OpenVINO backend
        elif args.backend == "openvino":
            inputs = {
                "input_ids": batch["input_ids"].cpu().numpy(),
                "attention_mask": batch["attention_mask"].cpu().numpy(),
            }

            # Add token_type_ids only if model expects it
            if "token_type_ids" in evaluator_instance.input_names:
                inputs["token_type_ids"] = np.zeros_like(inputs["input_ids"])



        outputs, latency = evaluator_instance.infer(inputs)
        latencies.append(latency / inputs["input_ids"].shape[0])
        logits = list(outputs.values())[0]

        # Extract logits
        raw_preds = logits.argmax(axis=-1)
        raw_preds = np.atleast_1d(raw_preds)

        # --------------------------------------------------
        # Universal GLUE label alignment (SST2, QNLI, RTE, MRPC)
        # --------------------------------------------------
        preds = raw_preds

        if hasattr(evaluator_instance, "label2id") and evaluator_instance.label2id is not None:
            label2id = evaluator_instance.label2id

            # Case 1: SST-2 → POSITIVE label mapping
            if "POSITIVE" in label2id:
                if label2id["POSITIVE"] == 0:
                    preds = 1 - raw_preds

            # Case 2: QNLI / RTE → entailment mapping
            elif "entailment" in label2id:
                # GLUE expects entailment = 1
                if label2id["entailment"] == 0:
                    preds = 1 - raw_preds

            # Case 3: MRPC → equivalent mapping
            elif "equivalent" in label2id:
                # GLUE expects equivalent = 1
                if label2id["equivalent"] == 0:
                    preds = 1 - raw_preds

        # otherwise default ordering is correct

        labels = batch.get("labels", batch.get("label"))

        if hasattr(labels, "cpu"):
            labels = labels.cpu().numpy()

        labels = np.atleast_1d(labels)

        correct += int((preds == labels).sum())
        total += labels.shape[0]


    accuracy = correct / total
    avg_latency_ms = float(np.mean(latencies))
    model_size_mb = evaluator_instance.get_model_size_mb()

    # --------------------------------------------------
    # 6. Results
    # --------------------------------------------------
    result = {
        "model": MODEL_NAME,
        "task": TASK_NAME,
        "backend": args.backend,
        "precision": args.precision,
        "accuracy": round(accuracy, 4),
        "avg_latency_ms": round(avg_latency_ms, 2),
        "model_size_mb": round(model_size_mb, 2),
    }

    df = pd.DataFrame([result])
    print("\n--- Benchmark Result ---")
    print(df)

    os.makedirs("results", exist_ok=True)
    out_path = f"results/benchmark_{args.backend}_{args.precision}_{MODEL_NAME.replace('/', '_')}_{TASK_NAME}.csv"
    df.to_csv(out_path, index=False)

    print(f"\nResults saved to: {out_path}")
    print("\n--- Benchmark Complete ---")



# --------------------------------------------------
# CLI
# --------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="QuanTool: Backend-aware Quantization Benchmark"
    )

    parser.add_argument("--model-name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--task", type=str, default="sst2")
    parser.add_argument("--backend", choices=["pytorch", "openvino"], default="pytorch")
    parser.add_argument("--precision", choices=["fp32", "int8", "int4"], default="fp32")

    args = parser.parse_args()
    main(args)
