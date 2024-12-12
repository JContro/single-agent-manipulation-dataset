import torch
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import logging
import os
from datetime import datetime
import pandas as pd 

def compute_metrics(eval_pred):
    """
    Compute metrics for multi-label classification
    """
    predictions, labels = eval_pred
    predictions = (torch.sigmoid(torch.Tensor(predictions)) > 0.5).numpy()
    
    accuracies = [accuracy_score(labels[:, i], predictions[:, i]) for i in range(labels.shape[1])]
    f1_scores = [f1_score(labels[:, i], predictions[:, i], average='binary') for i in range(labels.shape[1])]
    
    metrics = {
        'accuracy': np.mean(accuracies),
        'f1': np.mean(f1_scores),
    }
    
    return metrics

def setup_trainer(
    train_dataset,
    test_dataset,
    model_name: str,
    output_dir: str,
    num_labels: int,
    batch_size: int = 16,
    num_epochs: int = 10,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    seed: int = 42,
    device: str = None
):
    """
    Setup and return a trainer for multi-label classification
    """
    # Set device
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    torch.device(device)

    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification"
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        save_total_limit=2,
        seed=seed,
        logging_dir=f'logs/runs_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    return trainer

def save_predictions(trainer, test_dataset, output_dir, target_columns):
    """
    Generate and save predictions
    """
    predictions = trainer.predict(test_dataset)
    predicted_labels = (torch.sigmoid(torch.Tensor(predictions.predictions)) > 0.5).numpy()
    
    results_df = pd.DataFrame(predicted_labels, columns=target_columns)
    predictions_path = os.path.join(output_dir, 'predictions.csv')
    results_df.to_csv(predictions_path, index=False)
    
    return predictions, results_df