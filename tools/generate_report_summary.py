import pandas as pd
import os

RESULTS_FILE = "results/master_results.csv"
REPORT_FILE = "results/report_summary.txt"

def generate_summary():
    if not os.path.exists(RESULTS_FILE):
        print("❌ master_results.csv not found. Run aggregation first.")
        return

    df = pd.read_csv(RESULTS_FILE)

    lines = []
    lines.append("QUANTOOL – AUTOMATED BENCHMARK REPORT SUMMARY\n")
    lines.append("="*60 + "\n")

    # Overall table snapshot
    lines.append("Overall Benchmark Results:\n")
    lines.append(df.to_string(index=False))
    lines.append("\n\n")

    # Group by model-task-backend
    grouped = df.groupby(["model", "task", "backend"])

    for (model, task, backend), group in grouped:
        lines.append(f"\nModel: {model}\n")
        lines.append(f"Task: {task.upper()}\n")
        lines.append(f"Backend: {backend.upper()}\n")
        lines.append("-"*50 + "\n")

        # FP32 baseline
        fp32 = group[group["precision"]=="fp32"].iloc[0]

        lines.append(f"FP32 Baseline Accuracy: {fp32['accuracy']:.4f}\n")
        lines.append(f"FP32 Latency: {fp32['avg_latency_ms']:.2f} ms\n")
        lines.append(f"FP32 Model Size: {fp32['model_size_mb']:.2f} MB\n\n")

        # Best quantized choice (highest accuracy under quantization)
        quantized = group[group["precision"]!="fp32"]
        best = quantized.sort_values("accuracy", ascending=False).iloc[0]

        lines.append(f"Best Quantized Precision: {best['precision'].upper()}\n")
        lines.append(f"Accuracy: {best['accuracy']:.4f}\n")
        lines.append(f"Latency: {best['avg_latency_ms']:.2f} ms\n")
        lines.append(f"Model Size: {best['model_size_mb']:.2f} MB\n")

        # Gains
        acc_drop = fp32['accuracy'] - best['accuracy']
        speedup = fp32['avg_latency_ms'] / best['avg_latency_ms']
        size_reduction = fp32['model_size_mb'] / best['model_size_mb']

        lines.append(f"Accuracy Drop: {acc_drop:.4f}\n")
        lines.append(f"Latency Speedup: {speedup:.2f}×\n")
        lines.append(f"Size Reduction: {size_reduction:.2f}×\n")

        # Interpretation sentence
        lines.append("\nInterpretation:\n")
        lines.append(
            f"For {model} on {task.upper()} using {backend.upper()}, "
            f"{best['precision'].upper()} quantization achieves "
            f"{speedup:.2f}× faster inference and "
            f"{size_reduction:.2f}× smaller model size, "
            f"with only {acc_drop:.4f} accuracy degradation.\n"
        )

        lines.append("\n")

    # Save report
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n✅ Report summary generated at: {REPORT_FILE}")


if __name__ == "__main__":
    generate_summary()
