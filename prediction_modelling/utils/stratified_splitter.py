import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# TODO: make use of the logger for this
def plot_distributions(y_train, y_test, target_columns):
    """
    Plot the distribution of classes in train and test sets.
    
    Parameters:
    -----------
    y_train : pandas.DataFrame or Series
        Training set targets
    y_test : pandas.DataFrame or Series
        Test set targets
    target_columns : list
        List of target column names
    """
    n_targets = len(target_columns)
    fig, axes = plt.subplots(n_targets, 2, figsize=(15, 5 * n_targets))
    
    # Handle single target case
    if n_targets == 1:
        axes = axes.reshape(1, -1)
    
    for idx, col in enumerate(target_columns):
        # Training set distribution
        train_dist = y_train[col].value_counts(normalize=True)
        test_dist = y_test[col].value_counts(normalize=True)
        
        # Ensure both distributions have the same indices
        all_classes = sorted(set(train_dist.index) | set(test_dist.index))
        train_dist = train_dist.reindex(all_classes, fill_value=0)
        test_dist = test_dist.reindex(all_classes, fill_value=0)
        
        # Bar plots
        axes[idx, 0].bar(train_dist.index.astype(str), train_dist.values, alpha=0.8, label='Train')
        axes[idx, 0].bar(test_dist.index.astype(str), test_dist.values, alpha=0.5, label='Test')
        axes[idx, 0].set_title(f'{col} Distribution Comparison')
        axes[idx, 0].set_ylabel('Proportion')
        axes[idx, 0].set_xlabel('Class')
        axes[idx, 0].legend()
        axes[idx, 0].grid(True, alpha=0.3)
        
        # Add percentage labels on bars
        for i, v in enumerate(train_dist.values):
            axes[idx, 0].text(i, v, f'{v:.1%}', ha='center', va='bottom')
        
        # Pie charts
        train_sizes = y_train[col].value_counts()
        
        # Training set pie
        axes[idx, 1].pie(train_sizes.values, labels=[f'Class {c}\n{v/len(y_train):.1%}' 
                        for c, v in zip(train_sizes.index, train_sizes.values)],
                        autopct='%1.1f%%', 
                        colors=plt.cm.Pastel1(np.linspace(0, 1, len(train_sizes))),
                        wedgeprops={'alpha': 0.8})
        axes[idx, 1].set_title(f'{col} Training Set Distribution')
    
    plt.tight_layout()
    plt.show()
    
    # If there are multiple targets, try to plot joint distribution heatmap
    if len(target_columns) > 1:
        try:
            # Training set joint distribution
            joint_train = pd.crosstab(y_train[target_columns[0]], 
                                    y_train[target_columns[1]], 
                                    normalize='all')
            
            # Test set joint distribution
            joint_test = pd.crosstab(y_test[target_columns[0]], 
                                   y_test[target_columns[1]], 
                                   normalize='all')
            
            # Only plot if we have valid data
            if not (joint_train.empty or joint_test.empty):
                plt.figure(figsize=(15, 6))
                
                plt.subplot(121)
                sns.heatmap(joint_train, annot=True, fmt='.2%', cmap='YlOrRd')
                plt.title(f'Joint Distribution in Training Set\n{target_columns[0]} vs {target_columns[1]}')
                
                plt.subplot(122)
                sns.heatmap(joint_test, annot=True, fmt='.2%', cmap='YlOrRd')
                plt.title(f'Joint Distribution in Test Set\n{target_columns[0]} vs {target_columns[1]}')
                
                plt.tight_layout()
                plt.show()
            else:
                print("\nWarning: Could not create joint distribution heatmap due to insufficient data overlap between target variables.")
        except Exception as e:
            print(f"\nWarning: Could not create joint distribution heatmap: {str(e)}")

def perform_stratified_split(df, target_columns, test_size=0.2, random_state=42, plot=False):
    """
    Performs a stratified train-test split on a pandas DataFrame with multiple target columns.
    Uses a combined hash of all target columns for stratification.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame
    target_columns : str or list
        Name(s) of the target column(s) containing the classes
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split (0.0 to 1.0)
    random_state : int, default=42
        Random seed for reproducibility
    plot : bool, default=True
        Whether to plot the distributions of the split
        
    Returns:
    --------
    X_train, X_test, y_train, y_test : numpy arrays or pandas DataFrames
        The split data. y_train and y_test will be DataFrames if multiple targets
    """
    # Convert single target column to list for consistent processing
    if isinstance(target_columns, str):
        target_columns = [target_columns]
    
    # Verify all target columns exist in the DataFrame
    missing_cols = [col for col in target_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Target columns not found in DataFrame: {missing_cols}")
    
    # Separate features and targets
    X = df.drop(target_columns, axis=1)
    y = df[target_columns]
    
    # Check for empty DataFrame
    if len(df) == 0:
        raise ValueError("Input DataFrame is empty")
    
    # Create a combined category for stratification
    combined_targets = y.apply(lambda row: '_'.join(row.astype(str)), axis=1)
    
    try:
        # Perform stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=combined_targets
        )
    except ValueError as e:
        if "The least populated class in y has only 1 member" in str(e):
            print("\nWarning: Some classes have too few samples for stratification. "
                  "Performing regular train-test split instead.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
        else:
            raise e
    
    # Print distribution for each target column
    print("\nClass distributions:")
    for col in target_columns:
        print(f"\nTarget: {col}")
        print("Training set distribution:")
        print(y_train[col].value_counts(normalize=True))
        print("\nTest set distribution:")
        print(y_test[col].value_counts(normalize=True))
    
    # Plot distributions if requested
    if plot:
        plot_distributions(y_train, y_test, target_columns)
    
    return X_train, X_test, y_train, y_test

# Example usage:
"""
# Create sample DataFrame with multiple target columns
df = pd.DataFrame({
    'feature1': np.random.randn(1000),
    'feature2': np.random.randn(1000),
    'target1': np.random.choice([0, 1], size=1000, p=[0.7, 0.3]),
    'target2': np.random.choice([0, 1, 2], size=1000, p=[0.5, 0.3, 0.2])
})

# Perform split with multiple targets
X_train, X_test, y_train, y_test = perform_stratified_split(
    df,
    target_columns=['target1', 'target2'],
    test_size=0.25,
    random_state=42,
    plot=True
)
"""