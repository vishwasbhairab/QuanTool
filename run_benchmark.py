import argparse
import torch
import pandas as pd
import matplotlib.pyplot as plt
from quantool.models import model_loader
from quantool.core import quantizer, evaluator
from quantool.benchmarks.datasets import load_and_prepare_dataset

def main(args):
    """Main function to run the full quantization and evaluation benchmark."""
    print("--- Starting QuanTool Benchmark ---")
    
    # --- Configuration ---
    MODEL_NAME = args.model_name
    TASK_NAME = args.task
    DATASET_INFO = {'name': 'glue', 'subset': TASK_NAME}
    
    all_results = []
    index_labels = []
    
    # 1. Load FINE-TUNED Model (THIS IS THE KEY CHANGE)
    fp32_model, tokenizer = model_loader.load_model_and_tokenizer(MODEL_NAME, TASK_NAME)
    
    # Prepare calibration data
    print("\n--- Preparing Calibration Data ---")
    calibration_loader = load_and_prepare_dataset(
        dataset_name=DATASET_INFO['name'],
        subset=DATASET_INFO['subset'],
        tokenizer=tokenizer,
        split='train',
        batch_size=8
    )
    
    # --- Baseline Evaluation ---
    print(f"\n--- Evaluating Float32 Model ({MODEL_NAME} on {TASK_NAME}) ---")
    fp32_results = evaluator.run_evaluation_pipeline(fp32_model, tokenizer, DATASET_INFO)
    all_results.append(fp32_results)
    index_labels.append('Float32')
    
    # Verify baseline accuracy is reasonable
    if fp32_results['accuracy'] < 0.70:
        print(f"\n⚠️  WARNING: Baseline accuracy ({fp32_results['accuracy']:.2%}) is suspiciously low!")
        print("This suggests the model may not be properly fine-tuned.")
        print("Expected accuracy for fine-tuned models: 85-93%")
    
    # --- Dynamic Quantization ---
    print("\n--- Quantizing Model to INT8 (Dynamic) ---")
    model_dynamic, _ = model_loader.load_model_and_tokenizer(MODEL_NAME, TASK_NAME)
    int8_dynamic_model = quantizer.quantize_int8_dynamic(model_dynamic)
    
    print("\n--- Evaluating INT8 Dynamic Quantized Model ---")
    int8_dynamic_results = evaluator.run_evaluation_pipeline(int8_dynamic_model, tokenizer, DATASET_INFO)
    all_results.append(int8_dynamic_results)
    index_labels.append('INT8-Dynamic')

    # --- Static Quantization with Error Handling ---
    try:
        print("\n--- Quantizing Model to INT8 (Static) ---")
        model_static, _ = model_loader.load_model_and_tokenizer(MODEL_NAME, TASK_NAME)
        int8_static_model = quantizer.quantize_int8_static(model_static, calibration_loader)
        
        print("\n--- Evaluating INT8 Static Quantized Model ---")
        int8_static_results = evaluator.run_evaluation_pipeline(int8_static_model, tokenizer, DATASET_INFO)
        all_results.append(int8_static_results)
        index_labels.append('INT8-Static')
    except Exception as e:
        print("\n--- Static quantization failed for this model. Skipping. ---")
        print(f"    Underlying Error: {e}")
    
    # --- INT4 Dynamic Quantization (Weight-Only) ---
    try:
        print("\n--- Quantizing Model to INT4 (Dynamic, Weight-Only) ---")
        model_int4, _ = model_loader.load_model_and_tokenizer(MODEL_NAME, TASK_NAME)

        int4_dynamic_model = quantizer.quantize_int4_dynamic(model_int4)

        print("\n--- Evaluating INT4 Dynamic Quantized Model ---")
        int4_results = evaluator.run_evaluation_pipeline(
            int4_dynamic_model, tokenizer, DATASET_INFO
        )

        all_results.append(int4_results)
        index_labels.append('INT4-Dynamic')

    except Exception as e:
        print("\n--- INT4 quantization failed. Skipping. ---")
        print(f"    Underlying Error: {e}")


    # --- Reporting with Multiple Runs ---
    print("\n--- Final Benchmark Results ---")
    results_df = pd.DataFrame(all_results, index=index_labels)
    
    # Calculate comparative metrics
    if not results_df.empty and len(results_df) > 1:
        baseline_acc = results_df['accuracy'].iloc[0]
        baseline_lat = results_df['avg_latency_ms'].iloc[0]
        baseline_mem = results_df['peak_memory_mb'].iloc[0]
        baseline_size = results_df['model_size_mb'].iloc[0]
        
        results_df['Accuracy Change (%)'] = (results_df['accuracy'] - baseline_acc) * 100
        results_df['Latency Speedup (x)'] = baseline_lat / results_df['avg_latency_ms']
        results_df['Memory Reduction (x)'] = baseline_mem / results_df['peak_memory_mb']
        results_df['Size Reduction (x)'] = baseline_size / results_df['model_size_mb']
    
        display_columns = [
            'accuracy', 'Accuracy Change (%)', 'avg_latency_ms', 'Latency Speedup (x)',
            'peak_memory_mb', 'Memory Reduction (x)', 'model_size_mb', 'Size Reduction (x)'
        ]
        pd.set_option('display.precision', 4)
        print(results_df[display_columns])
        
    # Rest of the code including saving CSV and plots remains the same, but let's ensure it's correct.
    # We will replace it just to be safe.
    
    # --- Save Results to Files ---
    safe_model_name = MODEL_NAME.replace('/', '_')
    safe_task_name = TASK_NAME.replace('/', '_')
    
    # Save table to CSV
    csv_filename = f"benchmark_results_{safe_model_name}_{safe_task_name}.csv"
    results_df.to_csv(csv_filename)
    print(f"\nResults saved to {csv_filename}")
    
    # Save plot
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f'Quantization Benchmark: {MODEL_NAME} on {TASK_NAME}', fontsize=16)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    num_results = len(results_df)

    results_df['avg_latency_ms'].plot(kind='bar', ax=axes[0], title='Average Latency (ms)', color=colors[:num_results])
    axes[0].set_ylabel('Milliseconds (Lower is Better)')
    axes[0].tick_params(axis='x', rotation=0)

    results_df['model_size_mb'].plot(kind='bar', ax=axes[1], title='Model Size (MB)', color=colors[:num_results])
    axes[1].set_ylabel('Megabytes (Lower is Better)')
    axes[1].tick_params(axis='x', rotation=0)
    
    results_df['accuracy'].plot(kind='bar', ax=axes[2], title='Accuracy', color=colors[:num_results])
    axes[2].set_ylabel('Accuracy Score (Higher is Better)')
    if not results_df.empty:
        axes[2].set_ylim(bottom=max(0, results_df['accuracy'].min() - 0.05), top=min(1, results_df['accuracy'].max() + 0.05))
    axes[2].tick_params(axis='x', rotation=0)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_filename = f"benchmark_plot_{safe_model_name}_{safe_task_name}.png"
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")

    print("\n--- Benchmark Complete ---")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="QuanTool: A Framework for Evaluating Post-Training Quantization.")
    parser.add_argument(
        '--model-name', 
        type=str, 
        default='distilbert-base-uncased', 
        help='The name of the HuggingFace model to benchmark (e.g., "distilbert-base-uncased").'
    )
    parser.add_argument(
        '--task',
        type=str,
        default='sst2',
        help='The GLUE task to run the benchmark on (e.g., "sst2", "qnli").'
    )
    
    args = parser.parse_args()
    main(args)



'''

**To Run the Full Benchmark:**

```bash
python run_benchmark.py --model-name "distilbert-base-uncased" --task "qnli"

This will produce a new set of `.csv` and `.png` files with the task name included. 
Once this is done, your benchmarking phase will be complete, 
and you'll have all the data needed for your final report.

'''