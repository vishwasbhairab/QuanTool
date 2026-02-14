# QuanTool 🚀

**A Framework for Benchmarking Post-Training Quantization Techniques for Transformer Models**

QuanTool is a specialized benchmarking framework developed to evaluate the efficiency and impact of Post-Training Quantization (PTQ) on Transformer models. By focusing on reproducible best practices, it provides a clear picture of the trade-offs between model size, inference speed, and accuracy.

---

## 📌 Motivation

Quantization can significantly reduce model size and inference latency, but improper evaluation (e.g., using non-fine-tuned models) leads to misleading results.

QuanTool provides:
- **Verified fine-tuned model loading**: Ensuring evaluation is done on high-quality weights.
- **Backend-aware benchmarking**: Comparative analysis between PyTorch and OpenVINO.
- **Automated experiment execution**: Streamlined testing across multiple precisions (FP32, INT8, INT4).
- **Clean Analysis**: Direct accuracy–latency–compression reporting for optimized deployment.

---

## 🧠 Supported Models & Tasks

The framework currently supports the following configurations for benchmarking:

| Model | Task | Dataset/Metric |
|-------|------|----------------|
| `distilbert-base-uncased` | SST-2 | Sentiment Analysis (Accuracy) |
| `bert-base-uncased` | QNLI | Question Answering (Accuracy) |

---

## ⚙️ Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/vishwasbhairab/QuanTool.git
cd QuanTool
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Run a Specific Benchmark

To evaluate a model across specific parameters:

```bash
python main.py --model "distilbert-base-uncased" --task "sst2" --backends "pytorch" --precision int8
```

### 2. Run Full Benchmark Suite (One Command)

To execute the complete testing matrix (FP32, INT8, and INT4) across all supported backends:

```bash
python scripts/run_all_benchmarks.py
```

Results are stored in seperate csv files wrt model, task, backend and precision.

To aggregate results into one csv file:
```bash
python tools/aggregate_results.py
```

### 3. Generate Visual Analytics

To visualize the performance trade-offs:

```bash
python scripts/plot_benchmarks.py
                 OR
python tools/plot_overall_dashboard.py
```

This will generate Latency vs. Accuracy and Size Comparison charts in the `/plots` directory as PNG and PDF files.

---

## 📈 Metrics Reported

- **Accuracy**: Maintains task-specific performance scores (F1/Accuracy).
- **Average Latency (ms)**: Measured per inference pass.
- **Model Size (MB)**: Total disk footprint after quantization.
- **Speedup (×)**: Latency gain relative to the FP32 baseline.
- **Size Reduction (×)**: Storage gain relative to the FP32 baseline.

---

## 🧪 Experimental Best Practices

- **Verified Models**: Only uses fine-tuned models to ensure valid accuracy metrics.
- **Baseline Validation**: Every test is compared against a verified FP32 baseline.
- **Warm-up Runs**: Performs inference "warm-ups" before recording timing to avoid cold-start bias.
- **Backend Handling**: Specialized input handling for both PyTorch and OpenVINO engines.

---

## 🧩 Known Limitations

- Static INT8 quantization for Transformers can be unstable depending on the calibration set.
- Current support is optimized for CPU-based inference.
- **INT4 Precision**: Utilizes weight-only compression techniques.

---

## 🛡️ License

This project is licensed under the MIT License - see the LICENSE file for details.

---
