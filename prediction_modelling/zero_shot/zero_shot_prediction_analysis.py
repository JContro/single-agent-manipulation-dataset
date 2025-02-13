from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import f1_score
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
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    
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
    c= 0
    for entry in classifications:
        try:
            
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
        except Exception as e:

            c+=1
    print(f"error entries: {c}")
            
    
    classifications_df = pd.DataFrame(rows)
    
    print(f"\nClassifications DataFrame shape: {classifications_df.shape}")
    print("Classifications columns:", classifications_df.columns.tolist())
    
    # Join with ground truth
    ground_truth_df = ground_truth_df.reset_index()
    merged_df = ground_truth_df.merge(classifications_df, on='uuid', how='inner')
    print(f"\nMerged DataFrame shape: {merged_df.shape}")
    print("Merged columns:", merged_df.columns.tolist())
    
    return merged_df

# def evaluate_model_performance(
#     df: pd.DataFrame,
#     model_name: str,
#     manipulation_types: List[str]
# ) -> Dict[str, Dict[str, float]]:
#     """
#     Evaluate model performance for all manipulation types.
#     """
#     print(f"\nEvaluating model: {model_name}")
#     print(f"Available models in data: {df['model_name'].unique()}")
    
#     # Filter for specific model
#     model_df = df[df['model_name'] == model_name]  # Changed from 'model' to 'model_name'
#     print(f"Filtered model DataFrame shape: {model_df.shape}")
    
#     results = {}
#     all_y_true = []
#     all_y_pred = []
    
#     for m_type in manipulation_types:
#         m_type_lower = m_type.lower()
#         try:
#             if m_type_lower == 'general':
#                 y_true = model_df[f'{m_type_lower}_binary'].values
#                 y_pred = model_df['general_y'].values
#             else:
#                 y_true = model_df[f'{m_type_lower}_binary'].values
#                 y_pred = model_df[f'{m_type_lower}_pred'].values
            
#             results[m_type] = calculate_metrics(y_true, y_pred)
#             all_y_true.append(y_true)
#             all_y_pred.append(y_pred)
#         except KeyError as e:
#             print(f"Missing column for {m_type}: {e}")
#             print("Available columns:", model_df.columns.tolist())
#             raise
    
#     rows = []
#     for i in range(len(all_y_true[0])): 
#         row = []
#         for j in range(len(all_y_true)):
#             row.append(all_y_true[j][i])
#         rows.append(row)
        
#     rows = np.array(rows)


#     results['overall'] = calculate_metrics(np.array(all_y_true), np.array(all_y_pred))
#     return results


def evaluate_model_performance(
    df: pd.DataFrame,
    model_name: str,
    manipulation_types: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model performance using multilabel classification metrics across all manipulation types.
    Returns metrics in a dictionary format compatible with multilabel classification.
    """
    print(f"\nEvaluating model: {model_name}")
    print(f"Available models in data: {df['model_name'].unique()}")
    
    # Filter for specific model
    model_df = df[df['model_name'] == model_name].copy()
    print(f"Filtered model DataFrame shape: {model_df.shape}")
    
    # Validate required columns exist
    true_cols = [f"{mt.lower()}_binary" for mt in manipulation_types]
    pred_cols = []
    
    errors = []
    for mt in manipulation_types:
        mtl = mt.lower()
        if mtl == 'general':
            pred_col = 'general_y'
        else:
            pred_col = f"{mtl}_pred"
            
        if f"{mtl}_binary" not in model_df.columns:
            errors.append(f"{mtl}_binary")
        if pred_col not in model_df.columns:
            errors.append(pred_col)
        pred_cols.append(pred_col)
    
    if errors:
        print(f"Missing required columns: {errors}")
        print("Available columns:", model_df.columns.tolist())
        raise KeyError("Missing columns required for evaluation")
    
    # Construct multilabel formats
    try:
        y_true = model_df[true_cols].values.astype(int)
        y_pred = model_df[pred_cols].values
        
        # Ensure predictions are proper format (convert probabilities to binary if needed)
        # Example thresholding (comment out if predictions are already binary):
        # y_pred = (y_pred >= 0.5).astype(int)
        
        # Check dimension match
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch between y_true {y_true.shape} and y_pred {y_pred.shape}")
    except Exception as e:
        print("Error constructing multilabel arrays:", str(e))
        raise

    # Calculate multilabel metrics
    results = {'overall': calculate_multilabel_metrics(y_true, y_pred)}
    
    return results

def calculate_multilabel_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate multilabel classification metrics"""
    from sklearn.metrics import (
        accuracy_score,
        hamming_loss,
        precision_score,
        recall_score,
        f1_score,
        classification_report
    )
    
    metrics = {
        # Subset accuracy (exact match)
        'exact_accuracy': accuracy_score(y_true, y_pred),
        # First convert to 1D array for samples-average metrics
        'hamming_loss': hamming_loss(y_true, y_pred),
        # Sample-average metrics
        'precision': precision_score(y_true, y_pred, average='samples', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='samples', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='samples', zero_division=0)
    }
    
    # Uncomment for per-class metrics
    # print("\nClassification Report:")
    # print(classification_report(y_true, y_pred, target_names=manipulation_types))
    
    return metrics

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
    

    # Run evaluation
    results = run_model_evaluation(
        ground_truth_df=ground_truth_df,
        classifications=classifications
    )