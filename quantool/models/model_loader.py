import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
)
from typing import Tuple, Dict
# --------------------------------------------------
# Mapping of tasks → fine-tuned models on Hugging Face
# --------------------------------------------------

FINETUNED_MODELS: Dict[str, Dict[str, str]] = {
    "sst2": {
        "bert-base-uncased": "textattack/bert-base-uncased-SST-2",
        "distilbert-base-uncased": "distilbert-base-uncased-finetuned-sst-2-english",
    },
    "qnli": {
        "bert-base-uncased": "textattack/bert-base-uncased-QNLI",
        "distilbert-base-uncased": "textattack/distilbert-base-uncased-QNLI",
    },
    "mrpc": {
        "bert-base-uncased": "textattack/bert-base-uncased-MRPC",
        "distilbert-base-uncased": "textattack/distilbert-base-uncased-MRPC",
    },
    "rte": {
        "bert-base-uncased": "textattack/bert-base-uncased-RTE",
        "distilbert-base-uncased": "textattack/distilbert-base-uncased-RTE",
    },
}

# --------------------------------------------------
# Loader API
# --------------------------------------------------

def load_model_and_tokenizer(
    model_name: str,
    task: str,
    device: str = "cpu",
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """
    Load a fine-tuned Hugging Face classification model and tokenizer.

    Args:
        model_name (str): Base architecture name
            (e.g. "bert-base-uncased", "distilbert-base-uncased")
        task (str): GLUE task name
            (e.g. "sst2", "qnli", "mrpc", "rte")
        device (str): Torch device ("cpu" or "cuda")

    Returns:
        model (torch.nn.Module): Fine-tuned classification model
        tokenizer (AutoTokenizer): Corresponding tokenizer
    """
    if task not in FINETUNED_MODELS:
        raise ValueError(
            f"Unsupported task '{task}'. "
            f"Supported tasks: {list(FINETUNED_MODELS.keys())}"
        )

    if model_name not in FINETUNED_MODELS[task]:
        raise ValueError(
            f"No fine-tuned model for '{model_name}' on task '{task}'. "
            f"Available models: {list(FINETUNED_MODELS[task].keys())}"
        )

    finetuned_model_name = FINETUNED_MODELS[task][model_name]

    print(f"Loading fine-tuned model: {finetuned_model_name}")

    # Load config explicitly (important for label mapping)
    config = AutoConfig.from_pretrained(finetuned_model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        finetuned_model_name,
        config=config,
    )
    tokenizer = AutoTokenizer.from_pretrained(finetuned_model_name)

    model.to(device)
    model.eval()

    print(f"Successfully loaded fine-tuned model for task '{task}'")

    return model, tokenizer
