import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
import os
import json
import logging

def plot_kfold_distributions(fold_splits, stratify_columns, target_columns=None, output_dir=None):
    """
    Plot and save the distribution of classes across k folds.
    
    Parameters:
    -----------
    fold_splits : dict
        Dictionary containing k-fold splits
    stratify_columns : list
        List of columns used for stratification
    target_columns : list, optional
        Additional target columns to plot distributions for
    output_dir : str, optional
        Directory to save plots to. If None, plots will be displayed instead
    """
    all_columns = stratify_columns.copy()
    if target_columns:
        all_columns.extend([col for col in target_columns if col not in stratify_columns])
    
    n_folds = len(fold_splits)
    n_targets = len(all_columns)
    
    # Distribution plots
    fig, axes = plt.subplots(n_targets, 2, figsize=(15, 6 * n_targets))
    if n_targets == 1:
        axes = axes.reshape(1, -1)
    
    for idx, col in enumerate(all_columns):
        # Calculate distributions for each fold
        fold_distributions = []
        for fold_idx, split in fold_splits.items():
            y_train, y_test = split['y_train'], split['y_test']
            train_dist = y_train[col].value_counts(normalize=True)
            test_dist = y_test[col].value_counts(normalize=True)
            fold_distributions.append((train_dist, test_dist))
        
        # Calculate average distribution across folds
        avg_train_dist = pd.concat([d[0] for d in fold_distributions]).groupby(level=0).mean()
        avg_test_dist = pd.concat([d[1] for d in fold_distributions]).groupby(level=0).mean()
        
        # Bar plot
        x = np.arange(len(avg_train_dist))
        width = 0.35
        axes[idx, 0].bar(x - width/2, avg_train_dist, width, label='Train (avg)', alpha=0.8)
        axes[idx, 0].bar(x + width/2, avg_test_dist, width, label='Test (avg)', alpha=0.5)
        axes[idx, 0].set_title(f'{col} Average Distribution Across {n_folds} Folds')
        axes[idx, 0].set_xticks(x)
        axes[idx, 0].set_xticklabels(avg_train_dist.index)
        axes[idx, 0].legend()
        
        # Add percentage labels on bars
        for i, v in enumerate(avg_train_dist):
            axes[idx, 0].text(i - width/2, v, f'{v:.1%}', ha='center', va='bottom')
        for i, v in enumerate(avg_test_dist):
            axes[idx, 0].text(i + width/2, v, f'{v:.1%}', ha='center', va='bottom')
        
        # Box plot showing distribution variance across folds
        fold_data = []
        for fold_idx in range(n_folds):
            for class_label in avg_train_dist.index:
                fold_data.extend([
                    {
                        'Fold': fold_idx,
                        'Class': class_label,
                        'Proportion': fold_distributions[fold_idx][0].get(class_label, 0),
                        'Set': 'Train'
                    },
                    {
                        'Fold': fold_idx,
                        'Class': class_label,
                        'Proportion': fold_distributions[fold_idx][1].get(class_label, 0),
                        'Set': 'Test'
                    }
                ])
        
        fold_df = pd.DataFrame(fold_data)
        sns.boxplot(data=fold_df, x='Class', y='Proportion', hue='Set', ax=axes[idx, 1])
        axes[idx, 1].set_title(f'{col} Distribution Variance Across Folds')
        axes[idx, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    if output_dir:
        try:
            plt.savefig(os.path.join(output_dir, 'distribution_plots.png'))
        except Exception as e:
            logging.error(f"Failed to save distribution plots: {e}")
        plt.close()
    else:
        plt.show()
    
    # Box plots for distribution variance
    plt.figure(figsize=(15, 5 * n_targets))
    for idx, col in enumerate(all_columns):
        plt.subplot(n_targets, 1, idx + 1)
        
        fold_data = []
        for fold_idx, split in fold_splits.items():
            train_dist = split['y_train'][col].value_counts(normalize=True)
            test_dist = split['y_test'][col].value_counts(normalize=True)
            
            for class_label in train_dist.index:
                fold_data.extend([
                    {
                        'Fold': fold_idx,
                        'Class': class_label,
                        'Proportion': train_dist[class_label],
                        'Set': 'Train'
                    },
                    {
                        'Fold': fold_idx,
                        'Class': class_label,
                        'Proportion': test_dist.get(class_label, 0),
                        'Set': 'Test'
                    }
                ])
        
        fold_df = pd.DataFrame(fold_data)
        sns.boxplot(data=fold_df, x='Class', y='Proportion', hue='Set')
        plt.title(f'{col} Distribution Variance Across Folds')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    if output_dir:
        try:
            plt.savefig(os.path.join(output_dir, 'distribution_variance.png'))
        except Exception as e:
            logging.error(f"Failed to save distribution variance plots: {e}")
        plt.close()
    else:
        plt.show()
    
    # Save distribution statistics if output directory is provided
    if output_dir:
        stats = []
        for fold_idx, split in fold_splits.items():
            for col in all_columns:
                train_dist = split['y_train'][col].value_counts(normalize=True)
                test_dist = split['y_test'][col].value_counts(normalize=True)
                stats.append({
                    'fold': fold_idx,
                    'column': col,
                    'set': 'train',
                    'distribution': train_dist.to_dict()
                })
                stats.append({
                    'fold': fold_idx,
                    'column': col,
                    'set': 'test',
                    'distribution': test_dist.to_dict()
                })
        
        try:
            with open(os.path.join(output_dir, 'distribution_stats.json'), 'w') as f:
                json.dump(stats, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save distribution statistics: {e}")

def perform_kfold_stratified_split(df, stratify_columns, target_columns=None, n_splits=5, 
                                 random_state=42, plot=True, plot_output_dir=None):
    """
    Performs k-fold stratified split on a pandas DataFrame, stratifying by specified columns
    while preserving additional target columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame
    stratify_columns : list
        Columns to use for stratification
    target_columns : list, optional
        Additional target columns to preserve (not used for stratification)
    n_splits : int, default=5
        Number of folds for k-fold cross-validation
    random_state : int, default=42
        Random seed for reproducibility
    plot : bool, default=True
        Whether to plot the distributions of the splits
    plot_output_dir : str, optional
        Directory to save plots to. If None, plots will be displayed
        
    Returns:
    --------
    fold_splits : dict
        Dictionary containing the splits for each fold:
        {fold_index: {'X_train': X_train, 'X_test': X_test, 
                     'y_train': y_train, 'y_test': y_test}}
    """
    # Verify all columns exist in the DataFrame
    all_target_cols = stratify_columns.copy()
    if target_columns:
        all_target_cols.extend([col for col in target_columns if col not in stratify_columns])
    
    missing_cols = [col for col in all_target_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
    
    # Check for empty DataFrame
    if len(df) == 0:
        raise ValueError("Input DataFrame is empty")
    
    # Separate features and targets
    X = df.drop(all_target_cols, axis=1)
    y = df[all_target_cols]
    
    # Create a combined category for stratification
    combined_strat = y[stratify_columns].apply(lambda row: '_'.join(row.astype(str)), axis=1)
    
    # Initialize KFold cross-validator
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Store splits
    fold_splits = {}
    
    try:
        # Perform k-fold split
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, combined_strat)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            fold_splits[fold_idx] = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            }
            
            # Print distribution for current fold
            print(f"\nFold {fold_idx + 1}/{n_splits}")
            for col in stratify_columns:
                print(f"\nColumn: {col}")
                print("Training set distribution:")
                print(y_train[col].value_counts(normalize=True))
                print("\nTest set distribution:")
                print(y_test[col].value_counts(normalize=True))
    
    except ValueError as e:
        if "The least populated class in y has only 1 member" in str(e):
            print("\nWarning: Some classes have too few samples for stratification.")
            raise e
        else:
            raise e
    
    # Plot distributions if requested
    if plot:
        plot_kfold_distributions(
            fold_splits, 
            stratify_columns, 
            target_columns,
            output_dir=plot_output_dir
        )
    
    return fold_splits