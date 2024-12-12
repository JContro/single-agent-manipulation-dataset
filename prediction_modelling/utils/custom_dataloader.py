import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd
import numpy as np

class ManipulationDataset(Dataset):
    def __init__(self, X, y, text_column, model, target_columns, max_length=6000):
        self.X = X
        self.text_column = text_column
        self.tokenizer = AutoTokenizer.from_pretrained(
            model,
            model_max_length=max_length
        )
        self.max_length = max_length
        self.labels = create_labels(y, target_columns)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        text = self.X[self.text_column].iloc[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        label = torch.tensor(self.labels[idx]).float()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': label
        }

def create_labels(y, target_columns):
    return y[target_columns].values