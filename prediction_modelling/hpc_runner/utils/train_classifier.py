import torch
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import logging
import os
from datetime import datetime
import pandas as pd


from dataclasses import dataclass
from typing import Dict, List
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import TrainerCallback
import wandb

class CustomWandbCallback(TrainerCallback):
    """Custom callback for logging training metrics to W&B"""
    def __init__(self, trainer, target_columns):
        self.trainer = trainer
        self.target_columns = target_columns
        self.best_f1 = 0.0
        
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Log evaluation metrics"""
        if state.is_world_process_zero:
            # Log main metrics
            eval_metrics = {
                'epoch': state.epoch,
                'train_loss': metrics.get('train_loss', 0),
                'eval_loss': metrics.get('eval_loss', 0),
                'eval_f1': metrics.get('eval_f1', 0),
                'eval_accuracy': metrics.get('eval_accuracy', 0),
                'learning_rate': metrics.get('learning_rate', 0)
            }
            
            # Track best model
            current_f1 = metrics.get('eval_f1', 0)
            if current_f1 > self.best_f1:
                self.best_f1 = current_f1
                eval_metrics['best_f1'] = self.best_f1
            
            wandb.log(eval_metrics)
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log training metrics"""
        if state.is_world_process_zero and logs:
            # Only log training loss and learning rate during training steps
            metrics_to_log = {
                'train/step': state.global_step,
                'train/loss': logs.get('loss', 0),
                'train/learning_rate': logs.get('learning_rate', 0)
            }
            wandb.log(metrics_to_log)


@dataclass
class DataCollatorWithPadding:
    tokenizer: AutoTokenizer
    padding: bool = True
    max_length: int = 512
    pad_to_multiple_of: int = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Separate the input features
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = torch.stack([f["labels"] for f in features])

        # Pad the sequences
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    

def compute_metrics(eval_pred):
    """
    Compute metrics for multi-label classification
    """
    predictions, labels = eval_pred
    # Move computations to CPU since metrics computation typically expects numpy arrays
    predictions = (torch.sigmoid(torch.Tensor(predictions)) > 0.5).numpy()
    accuracies = [accuracy_score(labels[:, i], predictions[:, i]) for i in range(labels.shape[1])]
    f1_scores = [f1_score(labels[:, i], predictions[:, i], average='binary') for i in range(labels.shape[1])]
    
    return {
        'accuracy': np.mean(accuracies),
        'f1': np.mean(f1_scores),
    }

def setup_trainer(
    train_dataset,
    test_dataset,
    model_name: str,
    output_dir: str,
    num_labels: int,
    model_cache_dir: str,
    batch_size: int = 16,
    num_epochs: int = 10,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    seed: int = 42,
    device: str = None,
    target_columns: List[str] = None
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        cache_dir=model_cache_dir
    )
    
    model.config.pad_token_id = train_dataset.tokenizer.pad_token_id
    model.config.use_cache = False
    model = model.to(device)

    # Create data collator
    data_collator = DataCollatorWithPadding(
        tokenizer=train_dataset.tokenizer,
        max_length=train_dataset.max_length
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",     # Changed to save each epoch
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        metric_for_best_model="eval_f1",
        save_total_limit=2,        # Keep only the 2 best models
        seed=seed,
        logging_dir=f'logs/runs_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        fp16=True,
        gradient_accumulation_steps=4,
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
        optim="adamw_torch",
        # Add more frequent logging
        logging_steps=10,          # Log every 10 steps
        logging_first_step=True,   # Log the first training step
        report_to=["wandb"],      # Enable wandb reporting
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Add custom W&B callback
    trainer.add_callback(CustomWandbCallback(trainer, target_columns))

    return trainer

def save_predictions(trainer, test_dataset, output_dir, target_columns):
    """
    Generate and save predictions
    """
    predictions = trainer.predict(test_dataset)
    # Move predictions to CPU for post-processing
    predicted_labels = (torch.sigmoid(torch.Tensor(predictions.predictions).cpu()) > 0.5).numpy()
    results_df = pd.DataFrame(predicted_labels, columns=target_columns)
    predictions_path = os.path.join(output_dir, 'predictions.csv')
    results_df.to_csv(predictions_path, index=False)
    return predictions, results_df