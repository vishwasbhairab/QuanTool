import logging
import json
from datetime import datetime
from pathlib import Path

class BenchmarkLogger:
    """Structured logging for benchmark experiments."""

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.log_dir = Path('logs')
        self.log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = self.log_dir / f'{experiment_name}_{timestamp}.log'

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(experiment_name)

        self.results = {
            'experiment_name': experiment_name,
            'timestamp': timestamp,
            'runs': []
        }

    def log_run(self, model_type: str, results: dict):
        """Log results from a single run."""
        self.logger.info(f"{model_type} Results: {results}")
        self.results['runs'].append({
            'model_type': model_type,
            'results': results
        })

    def save_summary(self):
        """Save summary JSON."""
        summary_file = self.log_dir / f'{self.experiment_name}_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        self.logger.info(f"Summary saved to {summary_file}")