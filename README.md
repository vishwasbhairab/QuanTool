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
python main.py --model "distilbert-base-uncased" --task "sst2" --backends "pytorch" --precision "int8"
```

### 2. Run Full Benchmark Suite (One Command)

To execute the complete testing matrix (FP32, INT8, and INT4) across all supported backends:

```bash
python run_all_benchmarks.py
```

Results are automatically consolidated in `results/master_results.csv`.

### 3. Generate Visual Analytics

To visualize the performance trade-offs:

```bash
python plot_benchmarks.py
```

This will generate Latency vs. Accuracy and Size Comparison charts in the `/plots` directory as PNG and PDF files.

### 4. Interactive Streamlit Dashboard

Explore results interactively through our web-based dashboard:

```bash
streamlit run dashboard.py
```

**Live Demo**: [Streamlit Dashboard](https://quantool.streamlit.app/)

Features:
- **Model Selection**: Choose from available models via dropdown
- **Task Selection**: Filter by specific tasks
- **Backend Comparison**: Compare PyTorch and OpenVINO performance
- **Interactive Tables**: View detailed benchmark results
- **Visualization**: Dynamic plots for latency vs. accuracy trade-offs
- **Export Functionality**: Download filtered results as CSV

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

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Vishwas Kumar Pandey**  
B.Tech (Computer Science Engineering), Final Year

---

## 🤝 Contributors

We welcome contributions from the community! If you'd like to contribute to QuanTool, please follow these steps:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### Current Contributors

<!-- Add contributors here -->
- **[Vishwas Kumar Pandey]** - [vishwasbhairab](https://github.com/vishwasbhairab)
- **[Naina Jain]** - [Naina2308](https://github.com/Naina2308)
- **[Vidhi Soni]** - [Vvidhuu](https://github.com/Vvidhuu)

### How to Contribute

Contributions are welcome in the following areas:
- Adding support for new models (e.g., RoBERTa, ALBERT, T5)
- Implementing additional quantization techniques
- Extending backend support (TensorRT, ONNX Runtime)
- Improving documentation and examples
- Bug fixes and performance optimizations
- Adding new benchmarking tasks and datasets

### Acknowledgments

Special thanks to:
- The Hugging Face team for their transformers library
- OpenVINO team for optimization tools
- The open-source community for continuous support
