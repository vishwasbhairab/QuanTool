import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Tuple

def load_model_and_tokenizer(model_name: str, num_labels: int) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """
    Loads a pre-trained transformer model and its tokenizer from HuggingFace Hub.

    Args:
        model_name (str): The name of the model to load (e.g., 'bert-base-uncased').
        num_labels (int): The number of labels for the classification head.

    Returns:
        Tuple[torch.nn.Module, AutoTokenizer]: A tuple containing the model and the tokenizer.
    """
    try:
        print(f"Loading model: {model_name}...")
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("Model and tokenizer loaded successfully.")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        raise

