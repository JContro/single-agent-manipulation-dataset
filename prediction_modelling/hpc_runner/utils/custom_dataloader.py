import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd
import numpy as np

class ManipulationDataset(Dataset):
    def __init__(self, X, y, text_column, model, target_columns, max_length=6000):
        self.X = X
        self.text_column = text_column
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        
        # Store UUIDs
        self.uuids = X.index.tolist()  # Assuming UUID is the index
        # Alternative if UUID is a column:
        # self.uuids = X['uuid'].tolist()
        
        # Set padding token and strategy
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.max_length = max_length
        self.labels = create_labels(y, target_columns)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.X[self.text_column].iloc[idx]
        uuid = self.uuids[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors=None
        )
        
        return {
            'input_ids': torch.tensor(encoding['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float),
            'uuid': uuid
        }


def create_labels(y, target_columns):
    return y[target_columns].values