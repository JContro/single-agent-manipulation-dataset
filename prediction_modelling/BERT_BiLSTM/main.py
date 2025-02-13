import os
import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
import pandas as pd
import wandb
from dotenv import load_dotenv
import argparse
from datetime import datetime

# At the top of the file, keep your original import
from utils.load_data import process_conversation_data

# Load environment variables
load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser(description='Train conversation classifier')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory containing the data')
    parser.add_argument('--num-epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--disable-wandb', action='store_true',
                        help='Disable W&B logging')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode with small dataset')
    return parser.parse_args()

class ConversationDataset(Dataset):
    def __init__(self, conversations, labels, tokenizer, max_length=512):
        self.conversations = conversations
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        conversation = self.conversations[idx]
        labels = self.labels[idx]
        
        # Split conversation into utterances
        utterances = conversation.split('\n')  # Adjust split pattern based on your data format
        
        # Encode each utterance separately
        utterance_embeddings = []
        for utterance in utterances:
            if utterance.strip():  # Skip empty utterances
                encoding = self.tokenizer(
                    utterance,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                utterance_embeddings.append({
                    'input_ids': encoding['input_ids'].squeeze(),
                    'attention_mask': encoding['attention_mask'].squeeze(),
                })
        
        # Pad sequence of utterances if necessary
        max_utterances = 50  # Adjust based on your needs
        while len(utterance_embeddings) < max_utterances:
            # Add padding utterance
            padding_encoding = self.tokenizer(
                '',
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            utterance_embeddings.append({
                'input_ids': padding_encoding['input_ids'].squeeze(),
                'attention_mask': padding_encoding['attention_mask'].squeeze(),
            })
        
        # Truncate if too many utterances
        utterance_embeddings = utterance_embeddings[:max_utterances]
        
        # Stack tensors
        input_ids = torch.stack([u['input_ids'] for u in utterance_embeddings])
        attention_mask = torch.stack([u['attention_mask'] for u in utterance_embeddings])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.FloatTensor(labels)
        }

class BERTBiLSTMClassifier(nn.Module):
    def __init__(self, bert_model, num_classes, hidden_size=256, num_layers=2, dropout=0.5):
        super().__init__()
        self.bert = bert_model
        self.hidden_size = hidden_size
        
        # BiLSTM layer
        self.lstm = nn.LSTM(
            input_size=bert_model.config.hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.size(0)
        num_utterances = input_ids.size(1)
        
        # Reshape for BERT processing
        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        
        # Get BERT embeddings for each utterance
        with torch.no_grad():
            bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            # Use [CLS] token embedding for each utterance
            utterance_embeddings = bert_outputs.last_hidden_state[:, 0, :]
        
        # Reshape back to (batch_size, num_utterances, bert_hidden_size)
        utterance_embeddings = utterance_embeddings.view(batch_size, num_utterances, -1)
        
        # Pass through BiLSTM
        lstm_output, (hidden, cell) = self.lstm(utterance_embeddings)
        
        # Get final hidden states from both directions
        hidden_forward = hidden[-2, :, :]
        hidden_backward = hidden[-1, :, :]
        final_hidden = torch.cat((hidden_forward, hidden_backward), dim=1)
        
        # Classification layers
        dropped = self.dropout(final_hidden)
        logits = self.classifier(dropped)
        outputs = self.sigmoid(logits)
        
        return outputs

def setup_wandb(args, device):
    """Initialize Weights & Biases"""
    if not args.disable_wandb:
        wandb_api_key = os.getenv('WANDB_API_KEY')
        if not wandb_api_key:
            raise ValueError("WANDB_API_KEY not found in .env file")
        
        wandb.login(key=wandb_api_key)
        config = {
            "model_type": "bert-bilstm",
            "bert_model": "bert-base-uncased",
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "device": str(device)
        }
        wandb.init(
            project="conversation-classifier",
            config=config
        )

def train_epoch(model, data_loader, optimizer, criterion, device, args):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(data_loader):
        input_ids = batch['input_ids'].to(device)      # Shape: [batch_size, num_utterances, seq_len]
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if not args.disable_wandb and batch_idx % 10 == 0:
            wandb.log({
                "batch_loss": loss.item(),
                "batch": batch_idx
            })
            
    return total_loss / len(data_loader)
    
def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return total_loss / len(data_loader), np.array(all_preds), np.array(all_labels)


def get_debug_dataset(df, sample_size=16):
    return df.sample(n=sample_size, random_state=42)

def main():
    args = parse_args()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup W&B
    setup_wandb(args, device)
    
    # Load and process data
    df = process_conversation_data(args.data_dir)
    
    if args.debug:
        df = get_debug_dataset(df)
        args.num_epochs = 2
    
    # Prepare data
    conversations = df['chat_completion'].tolist()
    binary_cols = [col for col in df.columns if col.endswith('_binary')]
    labels = df[binary_cols].values
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    bert_model = AutoModel.from_pretrained('bert-base-uncased')
    
    # Setup cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(conversations, np.argmax(labels, axis=1)), 1):
        print(f'\nFold {fold}')
        if not args.disable_wandb:
            wandb.log({"fold": fold})
        
        # Create datasets
        train_dataset = ConversationDataset(
            [conversations[i] for i in train_idx],
            labels[train_idx],
            tokenizer
        )
        val_dataset = ConversationDataset(
            [conversations[i] for i in val_idx],
            labels[val_idx],
            tokenizer
        )
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        
        # Initialize model and optimizer
        model = BERTBiLSTMClassifier(bert_model, num_classes=len(binary_cols))
        model.to(device)
        
        criterion = nn.BCELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        
        # Training loop
        for epoch in range(args.num_epochs):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device, args)
            val_loss, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
            
            if not args.disable_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss
                })
            
            print(f'Epoch {epoch+1}/{args.num_epochs}:')
            print(f'Train Loss: {train_loss:.4f}')
            print(f'Val Loss: {val_loss:.4f}')
            
            # Log metrics for each binary column
            for i, col in enumerate(binary_cols):
                report = classification_report(val_labels[:, i], val_preds[:, i], output_dict=True)
                if not args.disable_wandb:
                    wandb.log({
                        f"{col}_f1": report['weighted avg']['f1-score'],
                        f"{col}_precision": report['weighted avg']['precision'],
                        f"{col}_recall": report['weighted avg']['recall']
                    })
                
                print(f'\nMetrics for {col}:')
                print(classification_report(val_labels[:, i], val_preds[:, i]))
        
        fold_metrics = {
            'fold': fold,
            'final_val_loss': val_loss,
            'val_predictions': val_preds,
            'val_labels': val_labels
        }
        all_metrics.append(fold_metrics)
    
    if not args.disable_wandb:
        wandb.finish()
    
    return all_metrics

if __name__ == "__main__":
    main()
