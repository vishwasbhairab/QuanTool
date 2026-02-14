import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

# ============================================================
# QuanTool Benchmark Dashboard
# Week 12 Deliverable
# ============================================================

st.set_page_config(
    page_title="QuanTool Dashboard",
    layout="wide"
)

st.title("🚀 QuanTool Benchmark Dashboard")
st.markdown(
    """
    QuanTool is a benchmarking framework for evaluating  
    **Post-Training Quantization (FP32 / INT8 / INT4)**  
    across **PyTorch and OpenVINO backends**.
    """
)

# ============================================================
# Load Master Results CSV
# ============================================================

RESULTS_PATH = "results/master_results.csv"

if not os.path.exists(RESULTS_PATH):
    st.error("❌ master_results.csv not found!")
    st.stop()

df = pd.read_csv(RESULTS_PATH)

st.success("✅ Loaded benchmark results successfully!")

# ============================================================
# Sidebar Filters
# ============================================================

st.sidebar.header("🔍 Filter Experiments")

model = st.sidebar.selectbox(
    "Select Model",
    df["model"].unique()
)

task = st.sidebar.selectbox(
    "Select Task",
    df["task"].unique()
)

backend = st.sidebar.selectbox(
    "Select Backend",
    df["backend"].unique()
)

filtered = df[
    (df["model"] == model) &
    (df["task"] == task) &
    (df["backend"] == backend)
]

# ============================================================
# Display Filtered Results Table
# ============================================================

st.subheader("📌 Filtered Benchmark Results")

if filtered.empty:
    st.warning("No results found for selected filters.")
    st.stop()

st.dataframe(filtered, use_container_width=True)

# ============================================================
# Show Key Metrics (FP32 Baseline)
# ============================================================

st.subheader("📊 Key Metrics Summary")

fp32_row = filtered[filtered["precision"] == "fp32"].iloc[0]

col1, col2, col3 = st.columns(3)

col1.metric(
    "FP32 Accuracy",
    f"{fp32_row['accuracy']:.4f}"
)

col2.metric(
    "FP32 Latency (ms)",
    f"{fp32_row['avg_latency_ms']:.2f}"
)

col3.metric(
    "FP32 Model Size (MB)",
    f"{fp32_row['model_size_mb']:.2f}"
)

# ============================================================
# Latency Bar Chart
# ============================================================

st.subheader("⏱ Latency Comparison Across Precisions")

fig1, ax1 = plt.subplots()

ax1.bar(
    filtered["precision"],
    filtered["avg_latency_ms"]
)

ax1.set_ylabel("Latency (ms)")
ax1.set_xlabel("Precision")
ax1.set_title(f"Latency Comparison ({model}, {task}, {backend})")

st.pyplot(fig1)

# ============================================================
# Model Size Comparison Chart
# ============================================================

st.subheader("📦 Model Size Comparison Across Precisions")

fig2, ax2 = plt.subplots()

ax2.bar(
    filtered["precision"],
    filtered["model_size_mb"]
)

ax2.set_ylabel("Model Size (MB)")
ax2.set_xlabel("Precision")
ax2.set_title(f"Model Size Reduction ({model}, {task}, {backend})")

st.pyplot(fig2)

# ============================================================
# Accuracy vs Compression Scatter Plot
# ============================================================

st.subheader("🎯 Accuracy vs Compression Trade-off")

fig3, ax3 = plt.subplots()

ax3.scatter(
    filtered["size_reduction_x"],
    filtered["accuracy"],
    s=200
)

for i, row in filtered.iterrows():
    ax3.text(
        row["size_reduction_x"] + 0.05,
        row["accuracy"],
        row["precision"]
    )

ax3.set_xlabel("Size Reduction (×)")
ax3.set_ylabel("Accuracy")
ax3.set_title("Accuracy vs Compression Trade-off")

st.pyplot(fig3)

# ============================================================
# Download Filtered CSV
# ============================================================

st.subheader("⬇ Export Filtered Results")

csv_data = filtered.to_csv(index=False)

st.download_button(
    label="Download Filtered Results CSV",
    data=csv_data,
    file_name=f"{model}_{task}_{backend}_results.csv",
    mime="text/csv"
)

# ============================================================
# Footer
# ============================================================

st.markdown("---")
st.caption("QuanTool Dashboard • Week 12 Deliverable • Built with Streamlit")
