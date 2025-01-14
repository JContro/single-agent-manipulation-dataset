from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
import logging
import json
from utils.validate_classification_data import validate_keys

# Import the data processing function
try:
    from utils.load_data import process_conversation_data
except ImportError:
    # If utils is not in the Python path, try relative import
    from ..utils.load_data import process_conversation_data

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate accuracy, precision, recall, and F1 score for binary classification.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def prepare_evaluation_data(
    ground_truth_df: pd.DataFrame,
    classifications: List[Dict]
) -> pd.DataFrame:
    """
    Prepare joined dataset of ground truth and model predictions.
    """
    print(f"Ground truth DataFrame shape: {ground_truth_df.shape}")
    print("Ground truth columns:", ground_truth_df.columns.tolist())
    
    # Create classifications dataframe
    rows = []
    for entry in classifications:
        uuid = entry['uuid']
        # Debug print for the first entry
        if len(rows) == 0:
            print("\nFirst classification entry structure:")
            print(json.dumps(entry, indent=2)[:500] + "...")
        
        for model_name, model_results in entry['model_classifications'].items():
            row = {
                'uuid': uuid,
                'model_name': model_name, 
                'general': model_results['classification_results']['general']
            }
            # Add manipulation tactics
            for tactic, value in model_results['classification_results']['manipulation_tactics'].items():
                row[f'{tactic.lower()}_pred'] = value
            rows.append(row)
    
    classifications_df = pd.DataFrame(rows)
    
    print(f"\nClassifications DataFrame shape: {classifications_df.shape}")
    print("Classifications columns:", classifications_df.columns.tolist())
    
    # Join with ground truth
    ground_truth_df = ground_truth_df.reset_index()
    merged_df = ground_truth_df.merge(classifications_df, on='uuid', how='inner')
    print(f"\nMerged DataFrame shape: {merged_df.shape}")
    print("Merged columns:", merged_df.columns.tolist())
    
    return merged_df

def evaluate_model_performance(
    df: pd.DataFrame,
    model_name: str,
    manipulation_types: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model performance for all manipulation types.
    """
    print(f"\nEvaluating model: {model_name}")
    print(f"Available models in data: {df['model_name'].unique()}")
    
    # Filter for specific model
    model_df = df[df['model_name'] == model_name]  # Changed from 'model' to 'model_name'
    print(f"Filtered model DataFrame shape: {model_df.shape}")
    
    results = {}
    all_y_true = []
    all_y_pred = []
    
    for m_type in manipulation_types:
        m_type_lower = m_type.lower()
        try:
            if m_type_lower == 'general':
                y_true = model_df[f'{m_type_lower}_binary'].values
                y_pred = model_df['general_y'].values
            else:
                y_true = model_df[f'{m_type_lower}_binary'].values
                y_pred = model_df[f'{m_type_lower}_pred'].values
            
            results[m_type] = calculate_metrics(y_true, y_pred)
            all_y_true.extend(y_true)
            all_y_pred.extend(y_pred)
        except KeyError as e:
            print(f"Missing column for {m_type}: {e}")
            print("Available columns:", model_df.columns.tolist())
            raise
    
    results['overall'] = calculate_metrics(np.array(all_y_true), np.array(all_y_pred))
    return results

def format_results(results: Dict[str, Dict[str, float]], model_name: str) -> str:
    """
    Format evaluation results into a readable string.
    """
    output = [f"\nResults for {model_name.upper()} model:"]
    output.append("-" * 50)
    
    for m_type, metrics in results.items():
        output.append(f"\n{m_type}:")
        for metric, value in metrics.items():
            output.append(f"  {metric}: {value:.3f}")
    
    return "\n".join(output)

def run_model_evaluation(
    ground_truth_df: pd.DataFrame,
    classifications: List[Dict],
    models: List[str] = ['openai', 'anthropic'],
    manipulation_types: List[str] = [
        'Peer Pressure', 'Reciprocity Pressure', 'Gaslighting', 
        'Guilt-Tripping', 'Emotional Blackmail', 'Fear Enhancement',
        'Negging', 'General'
    ]
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Run complete model evaluation for all specified models and manipulation types.
    """
    # Prepare joined dataset
    evaluation_df = prepare_evaluation_data(ground_truth_df, classifications)
    
    
    all_results = {}
    for model in models:
        try:
            results = evaluate_model_performance(
                df=evaluation_df,
                model_name=model,
                manipulation_types=manipulation_types
            )
            all_results[model] = results
            print(format_results(results, model))
        except Exception as e:
            print(f"Error evaluating model {model}: {str(e)}")
            continue
    
    return all_results

# Example usage:
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load ground truth data
    data_dir = 'data'
    log_file = 'logs'
    ground_truth_df = process_conversation_data(
        data_dir=data_dir,
        log_file=log_file,
        log_level=logging.INFO
    )
    
    # Load classifications
    with open('data/classification_results.json') as f:
        classifications = json.load(f)
        print(f"Loaded {len(classifications)} classification entries")
    
    for t in classifications:
        validate_keys(t['model_classifications'])

       

    # Run evaluation
    results = run_model_evaluation(
        ground_truth_df=ground_truth_df,
        classifications=classifications
    )