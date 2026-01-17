import os
import glob
import pandas as pd

RESULTS_DIR = "results"
OUTPUT_FILE = os.path.join(RESULTS_DIR, "master_results.csv")

def aggregate_results():
    csv_files = glob.glob(os.path.join(RESULTS_DIR, "benchmark_*.csv"))

    if not csv_files:
        print("No benchmark CSV files found in results/")
        return

    print(f"Found {len(csv_files)} benchmark files")

    dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        dfs.append(df)

    master_df = pd.concat(dfs, ignore_index=True)

    # Sort nicely
    master_df = master_df.sort_values(
        by=["model", "task", "backend", "precision"]
    ).reset_index(drop=True)

    # ------------------------------------------------------------
    # Derived Metrics
    # ------------------------------------------------------------
    master_df["accuracy_drop_%"] = 0.0
    master_df["latency_speedup_x"] = 1.0
    master_df["size_reduction_x"] = 1.0

    # Compute relative to FP32 baseline per (model, task, backend)
    grouped = master_df.groupby(["model", "task", "backend"])

    for (model, task, backend), group in grouped:
        # Find FP32 baseline row
        baseline = group[group["precision"] == "fp32"]

        if baseline.empty:
            continue

        base_acc = float(baseline["accuracy"].values[0])
        base_lat = float(baseline["avg_latency_ms"].values[0])
        base_size = float(baseline["model_size_mb"].values[0])

        for idx in group.index:
            acc = master_df.loc[idx, "accuracy"]
            lat = master_df.loc[idx, "avg_latency_ms"]
            size = master_df.loc[idx, "model_size_mb"]

            master_df.loc[idx, "accuracy_drop_%"] = round((acc - base_acc) * 100, 3)
            master_df.loc[idx, "latency_speedup_x"] = round(base_lat / lat, 3)
            master_df.loc[idx, "size_reduction_x"] = round(base_size / size, 3)

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    master_df.to_csv(OUTPUT_FILE, index=False)

    print("\nMaster results saved to:", OUTPUT_FILE)
    print("\nPreview:\n")
    print(master_df)


if __name__ == "__main__":
    aggregate_results()
