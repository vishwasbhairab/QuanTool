# **QuanTool Core: A Post-Training Quantization Toolkit**

## **1\. Project Description**

QuanTool is a Python-based framework for benchmarking and evaluating the performance impact of post-training quantization on Hugging Face Transformer models. It provides a clean, automated pipeline to compare a model's original full-precision (Float32) version against its 8-bit integer (INT8) counterparts.

The primary goal of this toolkit is to provide a clear, data-driven analysis of the trade-offs between model accuracy, inference speed (latency), memory usage, and model size, enabling developers to make informed decisions about deploying optimized models.

## **2\. Installation**

Ensure you have Python 3.8+ and pip installed.

1. **Clone the repository or download the source files.**  
2. **Install the required dependencies:**  
   pip install \-r requiremnts.txt

## **3\. How to Run the Benchmark**

The main script run\_benchmark.py orchestrates the entire process. You can specify which model to test using the \--model-name argument.

\# Example: Run the benchmark on "distilbert-base-uncased"  
python run\_benchmark.py \--model-name "distilbert-base-uncased"

\# Example: Run the benchmark on "bert-base-uncased"  
python run\_benchmark.py \--model-name "bert-base-uncased"

The script will:

1. Download the specified model and the glue/sst2 dataset.  
2. Benchmark the original FP32 model.  
3. Apply and benchmark INT8 Dynamic Quantization.  
4. Attempt to apply and benchmark INT8 Static Quantization (and handle failures gracefully).  
5. Print a summary table to the console.  
6. Save the results to a .csv file and a summary plot to a .png file.

## **4\. Example Output**

After a successful run, you will see a summary table in your terminal like the one below. This example shows the results for distilbert-base-uncased, where static quantization failed as expected.

*(Note: Your numbers may vary slightly based on your hardware.)*

\--- Final Benchmark Results \---  
              accuracy  Accuracy Drop  avg\_latency\_ms  Latency Speedup (x)  peak\_memory\_mb  Memory Reduction (x)  model\_size\_mb  Size Reduction (x)  
Float32         0.8933         0.0000       2753.4512               1.0000        834.1134                1.0000       256.3398              1.0000  
INT8-Dynamic    0.8853         0.0080       1305.1221               2.1100        835.4321                0.9984        66.7219              3.8418  
