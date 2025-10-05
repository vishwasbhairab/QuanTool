import subprocess
import sys

# Define experiments to run
EXPERIMENTS = [
    ('bert-base-uncased', 'sst2'),
    ('bert-base-uncased', 'qnli'),
    ('bert-base-uncased', 'mrpc'),
    ('bert-base-uncased', 'rte'),
    ('distilbert-base-uncased', 'sst2'),
    ('distilbert-base-uncased', 'qnli'),
    ('distilbert-base-uncased', 'mrpc'),
    ('distilbert-base-uncased', 'rte'),
]

def run_all_experiments():
    """Run all experiments sequentially."""
    print("="*80)
    print("RUNNING COMPLETE BENCHMARK SUITE")
    print("="*80)
    
    failed_experiments = []
    
    for model, task in EXPERIMENTS:
        print(f"\n{'='*80}")
        print(f"Running: {model} on {task}")
        print(f"{'='*80}\n")
        
        try:
            result = subprocess.run(
                [sys.executable, 'run_benchmark.py',
                 '--model-name', model,
                 '--task', task],
                check=True,
                capture_output=False
            )
        except subprocess.CalledProcessError as e:
            print(f"\n❌ FAILED: {model} on {task}")
            failed_experiments.append((model, task))
            continue

    print("\n" + "="*80)
    print("BENCHMARK SUITE COMPLETE")
    print("="*80)
    
    if failed_experiments:
        print("\n⚠️  Some experiments failed:")
        for model, task in failed_experiments:
            print(f"  - {model} on {task}")
    else:
        print("\n✅ All experiments completed successfully!")

if __name__ == '__main__':
    run_all_experiments()