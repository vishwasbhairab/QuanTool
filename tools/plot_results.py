import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_FILE = "results/master_results.csv"
PLOTS_DIR = "results/plots"

def plot_all():
    if not os.path.exists(RESULTS_FILE):
        print("❌ master_results.csv not found. Run aggregate_results.py first.")
        return

    os.makedirs(PLOTS_DIR, exist_ok=True)

    df = pd.read_csv(RESULTS_FILE)

    # Styling
    sns.set(style="whitegrid")
    palette = {"fp32": "#1f77b4", "int8": "#ff7f0e", "int4": "#2ca02c"}

    # Group by model-task-backend
    grouped = df.groupby(["model", "task", "backend"])

    for (model, task, backend), group in grouped:
        title_prefix = f"{model} | {task.upper()} | {backend.upper()}"

        # ---------------- Accuracy ----------------
        plt.figure(figsize=(6,4))
        sns.barplot(data=group, x="precision", y="accuracy", palette=palette)
        plt.title(f"Accuracy\n{title_prefix}")
        plt.ylim(0,1)
        plt.ylabel("Accuracy")
        plt.xlabel("Precision")
        plt.tight_layout()
        plt.savefig(f"{PLOTS_DIR}/accuracy_{model}_{task}_{backend}.png")
        plt.close()

        # ---------------- Latency ----------------
        plt.figure(figsize=(6,4))
        sns.barplot(data=group, x="precision", y="avg_latency_ms", palette=palette)
        plt.title(f"Average Latency (ms)\n{title_prefix}")
        plt.ylabel("Latency (ms)")
        plt.xlabel("Precision")
        plt.tight_layout()
        plt.savefig(f"{PLOTS_DIR}/latency_{model}_{task}_{backend}.png")
        plt.close()

        # ---------------- Model Size ----------------
        plt.figure(figsize=(6,4))
        sns.barplot(data=group, x="precision", y="model_size_mb", palette=palette)
        plt.title(f"Model Size (MB)\n{title_prefix}")
        plt.ylabel("Size (MB)")
        plt.xlabel("Precision")
        plt.tight_layout()
        plt.savefig(f"{PLOTS_DIR}/size_{model}_{task}_{backend}.png")
        plt.close()

        # ---------------- Speedup ----------------
        plt.figure(figsize=(6,4))
        sns.barplot(data=group, x="precision", y="latency_speedup_x", palette=palette)
        plt.title(f"Latency Speedup vs FP32\n{title_prefix}")
        plt.ylabel("Speedup (x)")
        plt.xlabel("Precision")
        plt.tight_layout()
        plt.savefig(f"{PLOTS_DIR}/speedup_{model}_{task}_{backend}.png")
        plt.close()

        # ---------------- Size Reduction ----------------
        plt.figure(figsize=(6,4))
        sns.barplot(data=group, x="precision", y="size_reduction_x", palette=palette)
        plt.title(f"Size Reduction vs FP32\n{title_prefix}")
        plt.ylabel("Reduction (x)")
        plt.xlabel("Precision")
        plt.tight_layout()
        plt.savefig(f"{PLOTS_DIR}/size_reduction_{model}_{task}_{backend}.png")
        plt.close()

    print(f"\n✅ All plots saved in: {PLOTS_DIR}")


if __name__ == "__main__":
    plot_all()
