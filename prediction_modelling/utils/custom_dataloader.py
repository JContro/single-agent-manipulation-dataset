import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd
import numpy as np

class ManipulationDataset(Dataset):
    def __init__(self, X, y, text_column, model, target_columns):
        self.encodings = create_encodings(X, text_column, model)
        self.labels = create_labels(y, target_columns)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

def create_encodings(X, text_column, model):
    tokenizer = AutoTokenizer.from_pretrained(model)
    return tokenizer(X[text_column].values.tolist(), truncation=True, padding=True)

def create_labels(y, target_columns):
    # Align labels with X indices and convert to numpy array
    return y[target_columns].values


