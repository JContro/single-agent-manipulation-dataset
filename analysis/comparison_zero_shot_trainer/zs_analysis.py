from typing import Dict, List
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
import json
from scipy import stats
import logging
import argparse
from pathlib import Path

def parse_zs_output(classification):
    classification = classification.strip()
    if classification.startswith('```json\n'):
        classification = classification[len('```json\n'):]
        if classification.endswith('```'):
            classification = classification[:-3]
    elif classification.startswith('```\n'):  # Handle the case you provided
        classification = classification[len('```\n'):]
        if classification.endswith('```'):
            classification = classification[:-3]

    try:
        data = json.loads(classification)
        return data
    except json.JSONDecodeError as e:
        import pdb; pdb.set_trace()
        print(f"Error decoding JSON: {e}")
        return None

def get_zs_classification_data(path):
    with open(path, 'r') as f:
        _data = json.load(f)

    new_rows = []
    for row in _data:
        parsed_data = parse_zs_output(row['classification'])
        if parsed_data:
            
            new_row = parsed_data['manipulation_tactics']
            try:
                new_row['general'] = parsed_data['general']
            except:
                try:
                    new_row['general'] = parsed_data['manipulation_tactics']['general']
                except:
                    import pdb; pdb.set_trace()
                    
            new_row['uuid'] = row['conversation_id']
            new_row['model'] = row['model']
            new_rows.append(new_row)
            

    dataset = pd.DataFrame(new_rows)
    return dataset

def load_and_prepare_data(model_df: pd.DataFrame, csv_path: str) -> pd.DataFrame:
    """
    Load and merge the model DataFrame and CSV data.
    """
    # Prepare model DataFrame
    model_df = model_df.set_index('uuid')
    
    # Get all manipulation tactics from the data
    manipulation_columns = [col for col in model_df.columns 
                          if col not in ['uuid', 'model']]
    
    # Create column mapping (standardize column names)
    column_mapping = {col: col.lower() for col in manipulation_columns}
    
    model_df = model_df.rename(columns=column_mapping)
    model_df = model_df.add_suffix('_zs')
    
    # Load CSV data
    csv_df = pd.read_csv(csv_path, index_col=0)
    
    # Merge the dataframes
    merged_df = csv_df.join(model_df)
    merged_df = merged_df.dropna()
    
    return merged_df

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate accuracy, precision, recall, and F1 score for multi-label classification.
    """
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    
    hamming_accuracy = np.mean(y_true == y_pred)
    subset_accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, 
        y_pred, 
        average='macro',
        zero_division=0
    )
    
    return {
        'hamming_accuracy': hamming_accuracy,
        'subset_accuracy': subset_accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def calculate_statistics(values: List[float]) -> Dict:
    """
    Calculate statistical measures for a list of values.
    """
    values = np.array(values)
    mean = np.mean(values)
    std = np.std(values)
    confidence_level = 0.95
    degrees_of_freedom = len(values) - 1
    t_value = stats.t.ppf((1 + confidence_level) / 2, degrees_of_freedom)
    margin_of_error = t_value * (std / np.sqrt(len(values)))
    
    return {
        'mean': mean,
        'std': std,
        'var': np.var(values),
        'confidence_interval_95': (mean - margin_of_error, mean + margin_of_error)
    }

def evaluate_model(data: pd.DataFrame) -> Dict:
    """
    Evaluate the model against Longformer results.
    """
    # Detect manipulation types from the data
    manipulation_types = [col.replace('_binary_true', '') 
                        for col in data.columns 
                        if col.endswith('_binary_true')]
    
    folds = data['fold'].unique()
    results_per_fold = {}
    fold_metrics = {
        'overall': {metric: [] for metric in ['hamming_accuracy', 'subset_accuracy', 'precision', 'recall', 'f1']},
        'per_manipulation': {manip: {'precision': [], 'recall': [], 'f1': []} for manip in manipulation_types}
    }
    
    for fold in folds:
        fold_data = data[data['fold'] == fold]
        
        # Prepare true and predicted values
        y_true = np.column_stack([fold_data[f'{manip}_binary_true'] for manip in manipulation_types])
        y_pred = np.column_stack([fold_data[f'{manip.lower()}_zs'] for manip in manipulation_types])
        
        # Calculate metrics
        overall_metrics = calculate_metrics(y_true, y_pred)
        
        # Calculate per-manipulation metrics
        manip_metrics = {}
        y_true = y_true.astype(bool)
        y_pred = y_pred.astype(bool)
        for i, manip in enumerate(manipulation_types):
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true[:, i],
                y_pred[:, i],
                average='binary',
                zero_division=0
            )
            manip_metrics[manip] = {'precision': precision, 'recall': recall, 'f1': f1}
        
        results_per_fold[f'fold_{fold}'] = {
            'overall': overall_metrics,
            'per_manipulation': manip_metrics
        }
        
        # Store metrics for statistical analysis
        for metric, value in overall_metrics.items():
            fold_metrics['overall'][metric].append(value)
        for manip, metrics in manip_metrics.items():
            for metric, value in metrics.items():
                fold_metrics['per_manipulation'][manip][metric].append(value)
    
    statistical_analysis = {
        'overall': {metric: calculate_statistics(values) 
                   for metric, values in fold_metrics['overall'].items()},
        'per_manipulation': {manip: {metric: calculate_statistics(values) 
                                   for metric, values in metrics.items()}
                           for manip, metrics in fold_metrics['per_manipulation'].items()}
    }
    
    return {
        'per_fold': results_per_fold,
        'statistical_analysis': statistical_analysis
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate model classifications')
    parser.add_argument('--model_predictions', type=str, required=True,
                      help='Path to the model predictions JSON file')
    parser.add_argument('--ground_truth', type=str, required=True,
                      help='Path to the ground truth CSV file')
    parser.add_argument('--output', type=str, required=True,
                      help='Path to save the evaluation results')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load model classifications
    logger.info(f"Loading model classifications from {args.model_predictions}...")
    cls_df = get_zs_classification_data(path=args.model_predictions)
    
    # Load and prepare merged data
    logger.info("Loading and preparing merged data...")
    merged_data = load_and_prepare_data(cls_df, args.ground_truth)
    
    # Evaluate model
    logger.info("Evaluating model...")
    results = evaluate_model(merged_data)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Evaluation complete. Results saved to {args.output}")

if __name__ == "__main__":
    main()