from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from typing import Dict

def load_and_prepare_dataset(dataset_name: str, subset: str, tokenizer: AutoTokenizer, split: str = 'validation', batch_size: int = 32) -> DataLoader:
    """
    Loads a dataset, tokenizes it, and prepares it for evaluation in a DataLoader.

    Args:
        dataset_name (str): The name of the dataset on HuggingFace Hub (e.g., 'glue').
        subset (str): The specific subset of the dataset (e.g., 'sst2').
        tokenizer (AutoTokenizer): The tokenizer to use for processing the text.
        split (str): The dataset split to use (e.g., 'validation').
        batch_size (int): The batch size for the DataLoader.

    Returns:
        DataLoader: A DataLoader containing the processed dataset.
    """
    print(f"Loading and preparing dataset: {dataset_name}/{subset} [{split}]...")
    try:
        # Load the dataset from HuggingFace
        dataset = load_dataset(dataset_name, subset, split=split)

        def tokenize_function(examples: Dict) -> Dict:
            # Handle different key names for the text columns in GLUE tasks
            if 'sentence1' in examples and 'sentence2' in examples:
                # For sentence-pair tasks like MRPC, RTE
                return tokenizer(examples['sentence1'], examples['sentence2'], padding='max_length', truncation=True, max_length=128)
            elif 'question' in examples and 'sentence' in examples:
                # For question-answering tasks like QNLI
                return tokenizer(examples['question'], examples['sentence'], padding='max_length', truncation=True, max_length=128)
            elif 'sentence' in examples:
                # For single-sentence tasks like SST-2
                return tokenizer(examples['sentence'], padding='max_length', truncation=True, max_length=128)
            else:
                raise ValueError(f"Could not find standard sentence/question columns in dataset features: {list(examples.keys())}")

        # Apply the tokenization across the entire dataset
        tokenized_dataset = dataset.map(tokenize_function, batched=True)

        # Remove original text columns and set format for PyTorch
        columns_to_remove = [col for col in dataset.column_names if col not in ['input_ids', 'attention_mask', 'label']]
        tokenized_dataset = tokenized_dataset.remove_columns(columns_to_remove)
        tokenized_dataset.set_format('torch')

        # Create and return the DataLoader
        data_loader = DataLoader(tokenized_dataset, batch_size=batch_size)
        print("Dataset preparation complete.")
        return data_loader
    except Exception as e:
        print(f"Error preparing dataset {dataset_name}/{subset}: {e}")
        raise

