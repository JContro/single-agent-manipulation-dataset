from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import f1_score, roc_curve, auc
import pandas as pd
import logging
import json
from scipy import stats
import matplotlib.pyplot as plt

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


def plot_roc_curves(y_true: np.ndarray, y_pred_prob: np.ndarray, 
                   manipulation_types: List[str], save_path: str = None):
    """
    Plot ROC curves for each manipulation type with larger font sizes.
    """
    plt.figure(figsize=(10, 8))
    
    # Set font sizes
    SMALL_SIZE = 14
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 18
    
    # Set font sizes for different elements
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    for i, manip_type in enumerate(manipulation_types):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred_prob[:, i])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f'{manip_type} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves by Manipulation Type')
    plt.legend(loc="lower right", bbox_to_anchor=(1.0, 0.0))
    
    # Add tight layout to prevent label cutoff
    plt.tight_layout()
    
    if save_path:
        # Save as PNG
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Save as PDF
        pdf_path = save_path.rsplit('.', 1)[0] + '.pdf'
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.show()

def calculate_manipulation_specific_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                         manipulation_types: List[str]) -> Dict:
    """
    Calculate metrics for each manipulation type separately.
    """
    metrics_per_type = {}
    
    for i, manip_type in enumerate(manipulation_types):
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true[:, i],
            y_pred[:, i],
            average='binary',
            zero_division=0
        )
        
        metrics_per_type[manip_type] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    return metrics_per_type

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

def run_model_evaluation(classifications: pd.DataFrame) -> Dict:
    """
    Enhanced evaluation including manipulation-specific metrics and ROC curves.
    """
    folds = classifications['fold'].unique()
    
    pred_cols = [col for col in classifications.columns if col.endswith('_binary') 
                 and not col.endswith('_binary_true') and not col.endswith('_binary_prob')]
    true_cols = [col + '_true' for col in pred_cols]
    prob_cols = [col + '_prob' for col in pred_cols]
    
    manipulation_types = [col.replace('_binary', '') for col in pred_cols]
    
    fold_metrics = {
        'overall': {
            'hamming_accuracy': [],
            'subset_accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        },
        'per_manipulation': {manip: {'precision': [], 'recall': [], 'f1': []} 
                           for manip in manipulation_types}
    }
    
    results_per_fold = {}
    for fold in folds:
        fold_mask = classifications['fold'] == fold
        y_true = classifications.loc[fold_mask, true_cols].values
        y_pred = classifications.loc[fold_mask, pred_cols].values
        y_pred_prob = classifications.loc[fold_mask, prob_cols].values
        
        overall_metrics = calculate_metrics(y_true, y_pred)
        manip_metrics = calculate_manipulation_specific_metrics(
            y_true, y_pred, manipulation_types
        )
        
        results_per_fold[f'fold_{fold}'] = {
            'overall': overall_metrics,
            'per_manipulation': manip_metrics
        }
        
        for metric, value in overall_metrics.items():
            fold_metrics['overall'][metric].append(value)
            
        for manip, metrics in manip_metrics.items():
            for metric, value in metrics.items():
                fold_metrics['per_manipulation'][manip][metric].append(value)
    
    statistical_analysis = {
        'overall': {},
        'per_manipulation': {}
    }
    
    for metric, values in fold_metrics['overall'].items():
        statistical_analysis['overall'][metric] = calculate_statistics(values)
    
    for manip in manipulation_types:
        statistical_analysis['per_manipulation'][manip] = {}
        for metric in ['precision', 'recall', 'f1']:
            values = fold_metrics['per_manipulation'][manip][metric]
            statistical_analysis['per_manipulation'][manip][metric] = calculate_statistics(values)
    
    plot_roc_curves(y_true, y_pred_prob, manipulation_types, 'roc_curves.png')
    
    return {
        'per_fold': results_per_fold,
        'statistical_analysis': statistical_analysis
    }

def format_results(results: Dict) -> str:
    """
    Format the results into a readable string.
    """
    output = []
    
    # Overall results per fold
    output.append("\nOverall Results per fold:")
    output.append("=" * 50)
    for fold, fold_results in results['per_fold'].items():
        output.append(f"\n{fold}:")
        output.append("Overall metrics:")
        for metric, value in fold_results['overall'].items():
            output.append(f"{metric}: {value:.4f}")
        
        output.append("\nPer-manipulation metrics:")
        for manip, metrics in fold_results['per_manipulation'].items():
            output.append(f"\n{manip}:")
            for metric, value in metrics.items():
                output.append(f"{metric}: {value:.4f}")
    
    # Statistical analysis
    output.append("\n\nStatistical Analysis:")
    output.append("=" * 50)
    
    output.append("\nOverall Metrics:")
    for metric, stats in results['statistical_analysis']['overall'].items():
        output.append(f"\n{metric.upper()}:")
        output.append(f"Mean: {stats['mean']:.4f}")
        output.append(f"Standard Deviation: {stats['std']:.4f}")
        output.append(f"Variance: {stats['var']:.4f}")
        output.append(f"95% CI: ({stats['confidence_interval_95'][0]:.4f}, {stats['confidence_interval_95'][1]:.4f})")
    
    output.append("\nPer-Manipulation Metrics:")
    for manip, metrics in results['statistical_analysis']['per_manipulation'].items():
        output.append(f"\n{manip.upper()}:")
        for metric, stats in metrics.items():
            output.append(f"\n{metric}:")
            output.append(f"Mean: {stats['mean']:.4f}")
            output.append(f"Standard Deviation: {stats['std']:.4f}")
            output.append(f"Variance: {stats['var']:.4f}")
            output.append(f"95% CI: ({stats['confidence_interval_95'][0]:.4f}, {stats['confidence_interval_95'][1]:.4f})")
    
    return "\n".join(output)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load classifications
    logger.info("Loading classification results...")
    classifications = pd.read_csv('data/all_fold_predictions.csv')
    
    # Run evaluation
    logger.info("Running evaluation...")
    results = run_model_evaluation(classifications)
    
    # Print formatted results
    print(format_results(results))
    
    # Save results to file
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info("Evaluation complete. Results saved to evaluation_results.json")
