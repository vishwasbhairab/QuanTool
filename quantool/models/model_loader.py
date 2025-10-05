import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Tuple

# Mapping of tasks to fine-tuned models available on Hugging Face
FINETUNED_MODELS = {
    'sst2': {
        'bert-base-uncased': 'textattack/bert-base-uncased-SST-2',
        'distilbert-base-uncased': 'distilbert-base-uncased-finetuned-sst-2-english'
    },
    'qnli': {
        'bert-base-uncased': 'textattack/bert-base-uncased-QNLI',
        'distilbert-base-uncased': 'textattack/distilbert-base-uncased-QNLI'
    },
    'mrpc': {
        'bert-base-uncased': 'textattack/bert-base-uncased-MRPC',
        'distilbert-base-uncased': 'textattack/distilbert-base-uncased-MRPC'
    },
    'rte': {
        'bert-base-uncased': 'textattack/bert-base-uncased-RTE',
        'distilbert-base-uncased': 'textattack/distilbert-base-uncased-RTE'
    }
}

def load_model_and_tokenizer(model_name: str, task: str) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """
    Loads a fine-tuned transformer model and tokenizer for a specific task.
    
    Args:
        model_name (str): Base model architecture (e.g., 'bert-base-uncased')
        task (str): GLUE task name (e.g., 'sst2', 'qnli')
        
    Returns:
        Tuple[torch.nn.Module, AutoTokenizer]: Fine-tuned model and tokenizer
    """
    try:
        # Get fine-tuned model name
        if task in FINETUNED_MODELS and model_name in FINETUNED_MODELS[task]:
            finetuned_model_name = FINETUNED_MODELS[task][model_name]
            print(f"Loading fine-tuned model: {finetuned_model_name}")
        else:
            raise ValueError(
                f"No fine-tuned model found for {model_name} on task {task}. "
                f"Available combinations: {list(FINETUNED_MODELS.keys())}"
            )
        
        model = AutoModelForSequenceClassification.from_pretrained(finetuned_model_name)
        tokenizer = AutoTokenizer.from_pretrained(finetuned_model_name)
        
        print(f"Successfully loaded fine-tuned model for {task}")
        return model, tokenizer
            
    except Exception as e:
        print(f"Error loading model: {e}")
        raise