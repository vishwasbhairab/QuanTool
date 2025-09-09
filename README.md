# 🚀 QuanTool Core: A Post-Training Quantization Toolkit

## 📌 1. Project Description
QuanTool is a **Python-based benchmarking framework** for evaluating the performance impact of **post-training quantization** on Hugging Face Transformer models.

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

1. Clone the repository or download the source files.  
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ 3. Running the Benchmark
The main entry point is **`run_benchmark.py`**.  
You can specify the model and **GLUE task** using command-line arguments.

### Example Commands:
```bash
# Run on DistilBERT with the default task (sst2)
python run_benchmark.py --model-name "distilbert-base-uncased"

# Run on BERT with the default task (sst2)
python run_benchmark.py --model-name "bert-base-uncased"

# Run on DistilBERT with a different task (qnli)
python run_benchmark.py --model-name "distilbert-base-uncased" --task "qnli"
```

---

## 🔄 4. What Happens When You Run It?
The script will:

1. 📥 Download the specified model and **GLUE task dataset** (e.g., `sst2`, `qnli`).  
2. 🧪 Benchmark the **original FP32 model**.  
3. ⚡ Apply and benchmark **INT8 Dynamic Quantization**.  
4. 🛠️ Attempt **INT8 Static Quantization** (and gracefully handle failures).  
5. 📊 Print a **summary table** in the console.  
6. 💾 Save results to:  
   - `.csv` file  
   - `.png` summary plot  

---

## 📈 5. Example Output
After a successful run, you will see a table like this in your terminal.  
This example is for `distilbert-base-uncased` on the **sst2** task.  

*(Note: Numbers may vary depending on your hardware.)*

| Model        | Accuracy | Accuracy Drop | Avg Latency (ms) | Latency Speedup (x) |
|--------------|----------|---------------|------------------|----------------------|
| Float32      | 0.8933   | 0.0000        | 2753.45          | 1.00                 |
| INT8-Dynamic | 0.8853   | 0.0080        | 1305.12          | 2.11                 |

> ✅ Dynamic quantization achieves a **2.1x latency speedup** and **3.8x model size reduction** with only a **0.8% accuracy drop**.  
> ⚠️ Static quantization is expected to fail for some Transformer models, which the tool handles automatically.  
---

## Collaborators

<table>
  <tr>
    <td align="center"><a href="https://github.com/Naina2308"><img src="https://avatars.githubusercontent.com/Naina2308" width="100px;" alt=""/><br /><sub><b>Naina Jain</b></sub></a></td>
    <td align="center"><a href="https://github.com/Vvidhuu"><img src="https://avatars.githubusercontent.com/Vvidhuu" width="100px;" alt=""/><br /><sub><b>Vidhi Soni</b></sub></a></td>
    <td align="center"><a href="https://github.com/vishwasbhairab"><img src="https://avatars.githubusercontent.com/vishwasbhairab" width="100px;" alt=""/><br /><sub><b>Vishwas Kumar Pandey</b></sub></a></td>
  </tr>
</table>

