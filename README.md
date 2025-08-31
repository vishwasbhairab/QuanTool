# 🚀 QuanTool Core: A Post-Training Quantization Toolkit

## 📌 1. Project Description

**QuanTool** is a **Python-based benchmarking framework** for evaluating the performance impact of **post-training quantization** on Hugging Face Transformer models.  

It provides a clean, automated pipeline to compare a model's **original full-precision (Float32)** version against its **8-bit integer (INT8)** counterparts.  

The goal is to deliver **data-driven insights** into the trade-offs between:  
- ✅ Model accuracy  
- ⚡ Inference speed (latency)  
- 💾 Memory usage  
- 📦 Model size  

This enables developers to make **informed deployment decisions** for optimized models.

---

## ⚙️ 2. Installation

Ensure you have **Python 3.8+** and `pip` installed.  

1. **Clone the repository or download the source files.**  
   ```bash
   git clone https://github.com/your-username/quantool-core.git
   cd quantool-core
   ```
2. **Install the required dependencies:**  
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ 3. Running the Benchmark

The main entry point is **`run_benchmark.py`**, which orchestrates the entire process.  
You can specify which model to test using the `--model-name` argument.  

### Example commands:
```bash
# Run benchmark on DistilBERT
python run_benchmark.py --model-name "distilbert-base-uncased"

# Run benchmark on BERT
python run_benchmark.py --model-name "bert-base-uncased"
```

---

## 🔄 4. What Happens When You Run It?

The script will:  
1. 📥 Download the specified model and the **GLUE/SST-2 dataset**.  
2. 🧪 Benchmark the **original FP32 model**.  
3. ⚡ Apply and benchmark **INT8 Dynamic Quantization**.  
4. 🛠️ Attempt **INT8 Static Quantization** (gracefully handle failures).  
5. 📊 Print a **summary table** in the console.  
6. 💾 Save results as:  
   - `.csv` → benchmark results  
   - `.png` → summary plot  

---

## 📈 5. Example Output

After a successful run, you will see a table like this in your terminal:  

*(Note: Numbers may vary depending on your hardware.)*  

| Model        | Accuracy | Accuracy Drop | Avg Latency (ms) | Latency Speedup (x) | Peak Memory (MB) | Memory Reduction (x) | Model Size (MB) | Size Reduction (x) |
|--------------|----------|---------------|------------------|----------------------|------------------|-----------------------|-----------------|---------------------|
| Float32      | 0.8933   | 0.0000        | 2753.4512        | 1.0000               | 834.1134         | 1.0000                | 256.3398        | 1.0000              |
| INT8-Dynamic | 0.8853   | 0.0080        | 1305.1221        | 2.1100               | 835.4321         | 0.9984                | 66.7219         | 3.8418              |

> ✅ Dynamic quantization achieves **2.1x latency speedup** and **3.8x model size reduction** with only a **0.8% accuracy drop**.  
> ⚠️ Static quantization may fail for certain Transformer models (expected behavior).  

---

## 🙌 Acknowledgements
- [Hugging Face Transformers](https://github.com/huggingface/transformers)  
- [PyTorch Quantization](https://pytorch.org/docs/stable/quantization.html)  
