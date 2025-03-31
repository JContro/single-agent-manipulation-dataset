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
from tqdm import tqdm
import re
import json
from pathlib import Path
from utils.load_data import process_conversation_data

# Load environment variables
load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser(description='Train conversation classifier')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory containing the data')
    parser.add_argument('--num-epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--disable-wandb', action='store_true',
                        help='Disable W&B logging')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode with small dataset')
    return parser.parse_args()

class PrecomputedConversationDataset(Dataset):
    def __init__(self, conversations, labels, tokenizer, bert_model, device, max_length=512):
        self.labels = labels
        self.max_utterances = 50
        
        # Pre-compute all BERT embeddings
        self.precomputed_embeddings = []
        self.valid_utterance_masks = []
        
        print("Pre-computing BERT embeddings...")
        bert_model.eval()
        bert_model.to(device)
        
        for conversation in tqdm(conversations, desc="Processing conversations"):
            # Split conversation into utterances by AGENT and USER
            utterances = conversation.split("@@@")
            
            utterances = utterances[:self.max_utterances]
            
            # Create mask for valid utterances
            valid_mask = torch.zeros(self.max_utterances)
            valid_mask[:len(utterances)] = 1
            self.valid_utterance_masks.append(valid_mask)
            
            # Encode and get embeddings for each utterance
            utterance_embeddings = []
            for utterance in utterances:
                if utterance.strip():
                    encoding = tokenizer(
                        utterance,
                        max_length=max_length,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt'
                    ).to(device)
                    
                    with torch.no_grad():
                        outputs = bert_model(**encoding)
                        embedding = outputs.last_hidden_state[:, 0, :].cpu()
                    utterance_embeddings.append(embedding)

            # Pad if necessary
            while len(utterance_embeddings) < self.max_utterances:
                utterance_embeddings.append(torch.zeros_like(utterance_embeddings[0]))
            
            # Stack all utterance embeddings for this conversation
            conversation_tensor = torch.cat(utterance_embeddings, dim=0)
            self.precomputed_embeddings.append(conversation_tensor)
        
        # Convert to tensor
        self.precomputed_embeddings = torch.stack(self.precomputed_embeddings)
        self.valid_utterance_masks = torch.stack(self.valid_utterance_masks)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'embeddings': self.precomputed_embeddings[idx],
            'valid_mask': self.valid_utterance_masks[idx],
            'labels': torch.FloatTensor(self.labels[idx])
        }

class BERTBiLSTMClassifier(nn.Module):
    def __init__(self, bert_hidden_size, num_classes, hidden_size=128, num_layers=2, dropout=0.5):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=bert_hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, embeddings, valid_mask=None):
        lstm_output, (hidden, cell) = self.lstm(embeddings)
        hidden_forward = hidden[-2, :, :]
        hidden_backward = hidden[-1, :, :]
        final_hidden = torch.cat((hidden_forward, hidden_backward), dim=1)
        dropped = self.dropout(final_hidden)
        logits = self.classifier(dropped)
        outputs = self.sigmoid(logits)
        return outputs

def train_epoch(model, data_loader, optimizer, criterion, device, args):
    model.train()
    total_loss = 0
    progress_bar = tqdm(data_loader, desc="Training")
    
    for batch_idx, batch in enumerate(progress_bar):
        embeddings = batch['embeddings'].to(device)
        valid_mask = batch['valid_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(embeddings, valid_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
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
    all_probs = []
    all_labels = []
    
    progress_bar = tqdm(data_loader, desc="Evaluating")
    with torch.no_grad():
        for batch in progress_bar:
            embeddings = batch['embeddings'].to(device)
            valid_mask = batch['valid_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(embeddings, valid_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return total_loss / len(data_loader), np.array(all_preds), np.array(all_probs), np.array(all_labels)

def save_fold_predictions(fold_metrics, binary_cols, save_dir, conversations, val_idx, df):
    fold = fold_metrics['fold']
    predictions = fold_metrics['val_predictions']
    probabilities = fold_metrics['val_probabilities']
    labels = fold_metrics['val_labels']
    
    # Create temporary DataFrame with conversations and their predictions
    temp_df = pd.DataFrame({'conversation': [conversations[i] for i in val_idx]})
    
    # Match conversations with the original DataFrame to get UUIDs
    # Create a temporary conversation column in the original df for matching
    df_temp = df.copy()
    df_temp['conversation'] = df['chat_completion']
    
    # Merge to get the UUIDs
    results_df = temp_df.merge(df_temp[['conversation', 'uuid']], 
                             on='conversation', 
                             how='left',
                             validate='1:1')  # Ensure one-to-one matching
    
    # Verify the merge worked correctly
    if len(results_df) != len(temp_df):
        raise ValueError(f"Mismatch in merge: {len(results_df)} results vs {len(temp_df)} predictions")
    
    if results_df['uuid'].isna().any():
        raise ValueError("Some conversations could not be matched to UUIDs")
    
    # Drop the conversation column as it's no longer needed
    results_df = results_df.drop('conversation', axis=1)
    
    # Add predicted labels (binary)
    for i, col in enumerate(binary_cols):
        results_df[col] = predictions[:, i]
    
    # Add probabilities
    for i, col in enumerate(binary_cols):
        results_df[f'{col}_prob'] = probabilities[:, i]
    
    # Add true labels
    for i, col in enumerate(binary_cols):
        results_df[f'{col}_true'] = labels[:, i]
    
    # Add fold number
    results_df['fold'] = fold
    
    # Save to CSV
    csv_path = save_dir / f'fold_{fold}_predictions.csv'
    results_df.to_csv(csv_path, index=False)
    
    return csv_path

def combine_fold_predictions(save_dir):
    all_predictions = []
    for file in save_dir.glob('fold_*_predictions.csv'):
        df = pd.read_csv(file)
        all_predictions.append(df)
    
    combined_df = pd.concat(all_predictions, axis=0, ignore_index=True)
    combined_path = save_dir / 'all_fold_predictions.csv'
    combined_df.to_csv(combined_path, index=False)
    return combined_path

def get_debug_dataset(df, sample_size=10):
    return df.sample(n=sample_size, random_state=42)

def get_folds(conversations, df, fold_pred_path):
    conv_df = pd.DataFrame({"chat_completion" : conversations})
    conv_df = conv_df.reset_index()
    merged_df = pd.merge(df, conv_df, on='chat_completion', how='inner')  
    
    
    fold_df = pd.read_csv(fold_pred_path)
    final_df = pd.merge(fold_df, merged_df, on='uuid', how='inner')
    final_df[['uuid', 'fold', 'index', 'chat_completion']]
    all_idx = set(final_df['index'])
    fold_info = []
    for fold in final_df['fold'].unique():
        val_idx = set(final_df[final_df['fold'] == fold]['index'])
        train_idx = all_idx - val_idx
        fold_info.append((fold, list(train_idx), list(val_idx)))
    return fold_info


def main():
    args = parse_args()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create directory for saving predictions
    save_dir = Path(f'results/run_{timestamp}')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup W&B with group
    run_group = f"exp_{timestamp}"
    if not args.disable_wandb:
        wandb_api_key = os.getenv('WANDB_API_KEY')
        if not wandb_api_key:
            raise ValueError("WANDB_API_KEY not found in .env file")
        wandb.login(key=wandb_api_key)
    
    # Load and process data
    df = process_conversation_data(args.data_dir)
    df = df.reset_index()
    
    
    if args.debug:
        print("Running in debug mode with reduced dataset")
        df = get_debug_dataset(df)
        args.num_epochs = 2
    
    conversations = df['chat_completion'].tolist()
    binary_cols = [col for col in df.columns if col.endswith('_binary')]
    labels = df[binary_cols].values
    
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    bert_model = AutoModel.from_pretrained('bert-base-uncased')
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_metrics = []
    
    config = {
        "timestamp": timestamp,
        "model_type": "bert-bilstm",
        "bert_model": "bert-base-uncased",
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "device": str(device),
        "binary_columns": binary_cols,
        "debug_mode": args.debug
    }
    
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    fold_info = get_folds(conversations, df, fold_pred_path='data/longformer_all_fold_predictions.csv')

    for (fold, train_idx, val_idx) in fold_info:
        
        if not args.disable_wandb:
            wandb.init(
                project="conversation-classifier",
                group=run_group,
                name=f"fold_{fold}",
                config=config,
                reinit=True
            )
        
        print(f'\nFold {fold}')
        
        train_dataset = PrecomputedConversationDataset(
            [conversations[i] for i in train_idx],
            labels[train_idx],
            tokenizer,
            bert_model,
            device
        )
        val_dataset = PrecomputedConversationDataset(
            [conversations[i] for i in val_idx],
            labels[val_idx],
            tokenizer,
            bert_model,
            device
        )
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        
        model = BERTBiLSTMClassifier(
            bert_hidden_size=bert_model.config.hidden_size,
            num_classes=len(binary_cols)
        )
        model.to(device)
        
        criterion = nn.BCELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        
        for epoch in range(args.num_epochs):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device, args)
            val_loss, val_preds, val_probs, val_labels = evaluate(model, val_loader, criterion, device)
            
            if not args.disable_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss
                })
            
            print(f'Epoch {epoch+1}/{args.num_epochs}:')
            print(f'Train Loss: {train_loss:.4f}')
            print(f'Val Loss: {val_loss:.4f}')
            
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
            'val_probabilities': val_probs,
            'val_labels': val_labels
        }
        
        save_path = save_fold_predictions(fold_metrics, binary_cols, save_dir, conversations, val_idx, df)
        print(f"Saved fold {fold} predictions to {save_path}")
        
        if not args.disable_wandb:
            wandb.save(str(save_path))
            wandb.finish()
        
        all_metrics.append(fold_metrics)
    
    # Combine all fold predictions
    combined_path = combine_fold_predictions(save_dir)
    print(f"Combined predictions saved to: {combined_path}")
    
    return all_metrics, save_dir

if __name__ == "__main__":
    metrics, save_dir = main()
    print(f"\nResults saved in: {save_dir}")
