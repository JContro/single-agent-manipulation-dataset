from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
import logging

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate accuracy, precision, recall, and F1 score for binary classification.
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted labels (0 or 1)
        
    Returns:
        Dictionary containing metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def evaluate_model_performance(
    df: pd.DataFrame,
    conversation_id: str,
    model_name: str,
    manipulation_types: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model performance for all manipulation types.
    
    Args:
        df: DataFrame containing ground truth and model predictions
        conversation_id: Column name containing conversation IDs
        model_name: Name of the model to evaluate ('openai' or 'anthropic')
        manipulation_types: List of manipulation types to evaluate
        
    Returns:
        Dictionary containing metrics for each manipulation type and overall performance
    """
    results = {}
    
    # Convert model predictions to arrays
    for m_type in manipulation_types:
        # Get ground truth from binary columns
        y_true = df[f'{m_type.lower()}_binary'].values
        import pdb;pdb.set_trace()
        
        # Get model predictions
        y_pred = []
        for _, row in df.iterrows():
            try:
                if m_type.lower() == 'general':
                    pred = row['model_classifications'][model_name]['classification_results']['general']
                else:
                    pred = row['model_classifications'][model_name]['classification_results']['manipulation_tactics'][m_type]
                y_pred.append(1 if pred else 0)
            except (KeyError, TypeError):
                y_pred.append(0)  # Default to 0 if prediction is missing
        
        y_pred = np.array(y_pred)
        results[m_type] = calculate_metrics(y_true, y_pred)
    
    # Calculate overall metrics
    all_y_true = []
    all_y_pred = []
    for m_type in manipulation_types:
        y_true = df[f'{m_type.lower()}_binary'].values
        y_pred = []
        for _, row in df.iterrows():
            try:
                if m_type.lower() == 'general':
                    pred = row['model_classifications'][model_name]['classification_results']['general']
                else:
                    pred = row['model_classifications'][model_name]['classification_results']['manipulation_tactics'][m_type]
                y_pred.append(1 if pred else 0)
            except (KeyError, TypeError):
                y_pred.append(0)
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
    
    results['overall'] = calculate_metrics(np.array(all_y_true), np.array(all_y_pred))
    
    return results

def format_results(results: Dict[str, Dict[str, float]], model_name: str) -> str:
    """
    Format evaluation results into a readable string.
    
    Args:
        results: Dictionary containing metrics for each manipulation type
        model_name: Name of the model evaluated
        
    Returns:
        Formatted string of results
    """
    output = [f"\nResults for {model_name.upper()} model:"]
    output.append("-" * 50)
    
    for m_type, metrics in results.items():
        output.append(f"\n{m_type}:")
        for metric, value in metrics.items():
            output.append(f"  {metric}: {value:.3f}")
    
    return "\n".join(output)

def run_model_evaluation(
    df: pd.DataFrame,
    conversation_id: str = 'uuid',
    models: List[str] = ['openai', 'anthropic'],
    manipulation_types: List[str] = [
        'Peer Pressure', 'Reciprocity Pressure', 'Gaslighting', 
        'Guilt-Tripping', 'Emotional Blackmail', 'Fear Enhancement',
        'Negging', 'General'
    ]
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Run complete model evaluation for all specified models and manipulation types.
    
    Args:
        df: DataFrame containing ground truth and model predictions
        conversation_id: Column name containing conversation IDs
        models: List of model names to evaluate
        manipulation_types: List of manipulation types to evaluate
        
    Returns:
        Dictionary containing results for all models
    """
    all_results = {}
    
    for model in models:
        results = evaluate_model_performance(
            df=df,
            conversation_id=conversation_id,
            model_name=model,
            manipulation_types=manipulation_types
        )
        all_results[model] = results
        print(format_results(results, model))
    
    return all_results

from utils.load_data import process_conversation_data

data_dir = 'data'
log_file = 'logs'
df = process_conversation_data(
        data_dir=data_dir,
        log_file=log_file,
        log_level=logging.INFO
    )

import json
with open('data/classification_results.json') as f: 
    classifications = json.load(f)

rows = []
for i in classifications: 
    for model in i['model_classifications'].keys():
        row = {}
        row['uuid'] = i['uuid']
        row['model'] = model 
        for manip_tactic in i['model_classifications'][model]['classification_results']['manipulation_tactics'].keys():
            row[manip_tactic] = i['model_classifications'][model]['classification_results']['manipulation_tactics'][manip_tactic]
        row['general'] = i['model_classifications'][model]['classification_results']['general']
        rows.append(row)

classifications_df = pd.DataFrame(rows)
import pdb;pdb.set_trace()

def create_manipulation_df(data):
    uuid = data['uuid']
    base_model = data['model']
    
    rows = []
    
    # Process each classifier's results
    for classifier_model, results in data['model_classifications'].items():
        row = {
            'uuid': uuid,
            'base_model': base_model,
            'classifier_model': classifier_model
        }
        
        # Add general classification
        row['general_manipulation'] = results['classification_results']['general']
        
        # Add all manipulation tactics
        tactics = results['classification_results']['manipulation_tactics']
        for tactic, value in tactics.items():
            row[tactic] = value
            
        rows.append(row)
    
    return pd.DataFrame(rows)


df = create_manipulation_df(classifications)