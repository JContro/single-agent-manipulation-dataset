from typing import Dict, List
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
import json
from scipy import stats
import logging

def load_and_prepare_data(json_path: str, csv_path: str) -> pd.DataFrame:
    """
    Load and merge the JSON and CSV data.
    """
    # Load JSON data
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    # Convert JSON to DataFrame
    json_rows = []
    for entry in json_data:
        row = {
            'uuid': entry['uuid'],
        }
        
        # Extract OpenAI classifications
        if 'openai' in entry.get('model_classifications', {}):
                if entry['model_classifications']['openai'].get('classification_results') is None:
                    continue
                openai_results = entry['model_classifications']['openai']['classification_results']
                for tactic, value in openai_results['manipulation_tactics'].items():
                    
                    row[f'openai_{tactic.lower()}'] = value
                
                row['openai_general'] = openai_results['general']
                
             
        
        # Extract Anthropic classifications
        if 'anthropic' in entry.get('model_classifications', {}):
            if entry['model_classifications']['anthropic'].get('classification_results') is None:
                    continue
            anthropic_results = entry['model_classifications']['anthropic']['classification_results']
            for tactic, value in anthropic_results['manipulation_tactics'].items():
                row[f'anthropic_{tactic.lower()}'] = value
            row['anthropic_general'] = anthropic_results['general']
              

        json_rows.append(row)
        
    
    json_df = pd.DataFrame(json_rows)
    json_df = json_df.set_index('uuid')
    json_df = json_df.add_suffix('_zs')
    
    # Load CSV data
    csv_df = pd.read_csv(csv_path, index_col=0)


    
    # Merge the dataframes
    merged_df = csv_df.join(json_df)

    
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

def evaluate_model(data: pd.DataFrame, model: str) -> Dict:
    """
    Evaluate the model (OpenAI or Anthropic) against Longformer results.
    """
    manipulation_types = ['peer pressure', 'reciprocity pressure', 'gaslighting', 
                         'guilt-tripping', 'emotional blackmail', 'fear enhancement', 
                         'negging', 'general']
    
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
        y_pred = np.column_stack([fold_data[f'{model}_{manip.lower()}_zs'] for manip in manipulation_types])
        
        

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
    
    # Calculate statistical analysis
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

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load and prepare data
    logger.info("Loading and preparing data...")
    data = load_and_prepare_data('data/classification_results.json', 
                                'data/longformer_all_fold_predictions.csv')
    import pdb; pdb.set_trace()
    
    # Evaluate OpenAI model
    logger.info("Evaluating OpenAI model...")
    openai_results = evaluate_model(data, 'openai')
    
    # Evaluate Anthropic model
    logger.info("Evaluating Anthropic model...")
    anthropic_results = evaluate_model(data, 'anthropic')
    
    # Save results
    results = {
        'openai': openai_results,
        'anthropic': anthropic_results
    }
    
    with open('model_comparison_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info("Evaluation complete. Results saved to model_comparison_results.json")
