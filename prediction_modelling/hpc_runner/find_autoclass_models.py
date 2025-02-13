from transformers import AutoModelForSequenceClassification
import torch
from tqdm import tqdm

from huggingface_hub import hf_cache_home

# 1. Get the cache directory path:
cache_dir = hf_cache_home()
print(f"Hugging Face cache directory: {cache_dir}")


def get_model_size(model_name):
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        total_params = sum(p.numel() for p in model.parameters())
        size_mb = (total_params * 4) / (1024 * 1024)  # assuming float32
        return total_params, size_mb
    except Exception as e:
        return None, None

# Dictionary mapping model types to their common pretrained versions
model_mappings = {
    "Albert": ["albert-base-v2", "albert-large-v2", "albert-xlarge-v2"],
    "Bart": ["facebook/bart-base", "facebook/bart-large"],
    "Bert": ["bert-base-uncased", "bert-large-uncased"],
    "BigBird": ["google/bigbird-roberta-base", "google/bigbird-roberta-large"],
    "Deberta": ["microsoft/deberta-base", "microsoft/deberta-large", "microsoft/deberta-v3-large"],
    "DistilBert": ["distilbert-base-uncased"],
    "Electra": ["google/electra-small-discriminator", "google/electra-base-discriminator"],
    "GPT2": ["gpt2", "gpt2-medium", "gpt2-large"],
    "Llama": ["meta-llama/Llama-2-7b", "meta-llama/Llama-2-13b"],
    "Roberta": ["roberta-base", "roberta-large"],
    "XLMRoberta": ["xlm-roberta-base", "xlm-roberta-large"],
    # Add more mappings as needed
}

def print_model_sizes():
    print("Model Size Analysis:")
    print("===================")
    
    for model_type, versions in model_mappings.items():
        print(f"\n{model_type} Models:")
        print("-" * 50)
        for version in versions:
            params, size = get_model_size(version)
            if params is not None:
                print(f"{version}:")
                print(f"  Parameters: {params:,}")
                print(f"  Approximate size: {size:.2f} MB")
            else:
                print(f"{version}: Unable to load model")

if __name__ == "__main__":
    print_model_sizes()
