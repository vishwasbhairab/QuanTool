import subprocess
import itertools
import numpy as np
MODELS = [
    "distilbert-base-uncased",
    "bert-base-uncased"
]

TASKS = [
    "sst2",
    "qnli"
]

BACKENDS = {
    "pytorch": ["fp32", "int8"],
    "openvino": ["fp32", "int8", "int4"]
}

PYTHON  = r"venv311\Scripts\python.exe"
   # or full venv python path if needed

def run():
    commands = []

    for model, task in itertools.product(MODELS, TASKS):
        for backend, precisions in BACKENDS.items():
            for precision in precisions:
                cmd = [
                    PYTHON,
                    "run_benchmark.py",
                    "--model-name", model,
                    "--task", task,
                    "--backend", backend,
                    "--precision", precision
                ]
                commands.append(cmd)

    print(f"Total experiments to run: {len(commands)}\n")

    for i, cmd in enumerate(commands, 1):
        print("=" * 80)
        print(f"[{i}/{len(commands)}] Running:")
        print(" ".join(cmd))
        print("=" * 80)

        subprocess.run(cmd, check=True)

    print("\n✅ ALL BENCHMARKS COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    run()
