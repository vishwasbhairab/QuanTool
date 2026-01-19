import pandas as pd
import matplotlib.pyplot as plt
import os

RESULTS_FILE = "results/master_results.csv"
PLOT_FILE = "results/overall_quantool_dashboard.png"

def plot_dashboard():
    if not os.path.exists(RESULTS_FILE):
        print("❌ master_results.csv not found. Run aggregation first.")
        return

    df = pd.read_csv(RESULTS_FILE)

    # Create a combined identifier
    df["experiment"] = (
        df["model"] + " | " +
        df["task"].str.upper() + " | " +
        df["backend"].str.upper()
    )

    experiments = df["experiment"].unique()
    precisions = ["fp32", "int8", "int4"]

    # Prepare subplots
    fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)

    metrics = ["accuracy", "avg_latency_ms", "model_size_mb"]
    titles = ["Accuracy (Higher is Better)",
              "Latency (ms) (Lower is Better)",
              "Model Size (MB) (Lower is Better)"]

    for ax, metric, title in zip(axes, metrics, titles):
        for exp in experiments:
            subset = df[df["experiment"] == exp]
            subset = subset.set_index("precision").reindex(precisions)

            ax.plot(
                precisions,
                subset[metric],
                marker="o",
                linewidth=2,
                label=exp
            )

        ax.set_title(title, fontsize=13)
        ax.grid(True, linestyle="--", alpha=0.5)

    axes[-1].set_xlabel("Precision")

    # Put legend outside
    axes[0].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)

    plt.suptitle("QuanTool – Full Benchmark Bird’s-Eye View", fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.78, 0.96])

    plt.savefig(PLOT_FILE, dpi=300)
    print(f"\n✅ Bird’s-eye dashboard saved to: {PLOT_FILE}")
    plt.close()

if __name__ == "__main__":
    plot_dashboard()
