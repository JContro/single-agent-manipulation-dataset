import json
import os
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Union, Tuple
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
import scipy.stats as stats

def setup_logging() -> logging.Logger:
    """Configure and return logger with consistent formatting."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('data_processing.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def handle_data_files(logger: logging.Logger, download_flag: bool = False) -> Dict[str, Union[dict, pd.DataFrame]]:
    """Handle data files and convert to appropriate format."""
    data_folder = Path('data')
    data_folder.mkdir(exist_ok=True)
    
    files = {
        'manipulation_definitions': 'manipulation-definitions.json',
        'conversations': 'conversations.json',
        'human_responses': 'human_responses.json',
        'user_scores': 'user_scores.json',
        'user_timing': 'user_timing.json'
    }
    
    if download_flag:
        logger.info("Downloading files from GCS bucket")
        from data_connection import create_gcs_file_handler
        file_handler = create_gcs_file_handler('manipulation-dataset-kcl')
        
        for filename in files.values():
            data = file_handler(filename)
            with open(data_folder / filename, 'w') as f:
                json.dump(data, f)
            logger.debug(f"Downloaded and saved {filename}")

    data = {}
    for key, filename in files.items():
        try:
            data[key] = json.load(open(data_folder / filename))
            logger.debug(f"Loaded {filename}")
        except Exception as e:
            logger.error(f"Error loading {filename}: {str(e)}")
            raise
    
    return data

def calculate_statistics(row: pd.Series, manipulation_cols: List[str]) -> Tuple[float, float]:
    """Calculate variance and mean score for a row of manipulation scores."""
    scores = []
    for col in manipulation_cols:
        if isinstance(row[col], list) and row[col]:
            scores.extend(row[col])
    
    if not scores:
        return np.nan, np.nan
        
    return np.var(scores), np.mean(scores)

def analyze_correlations(series1, series2, series1_name="Series 1", series2_name="Series 2"):
    """
    Calculate different types of correlations between two pandas Series.
    
    Parameters:
    series1 (pd.Series): First series of data
    series2 (pd.Series): Second series of data
    series1_name (str): Name of first series for output
    series2_name (str): Name of second series for output
    
    Returns:
    dict: Dictionary containing different correlation metrics
    """
    # Remove any rows where either series has NaN values
    clean_data = pd.DataFrame({
        series1_name: series1,
        series2_name: series2
    }).dropna()
    
    series1_clean = clean_data[series1_name]
    series2_clean = clean_data[series2_name]
    
    # Calculate different correlation coefficients
    correlations = {
        'pearson': {
            'coefficient': series1_clean.corr(series2_clean, method='pearson'),
            'pvalue': stats.pearsonr(series1_clean, series2_clean)[1]
        },
        'spearman': {
            'coefficient': series1_clean.corr(series2_clean, method='spearman'),
            'pvalue': stats.spearmanr(series1_clean, series2_clean)[1]
        },
        'kendall': {
            'coefficient': series1_clean.corr(series2_clean, method='kendall'),
            'pvalue': stats.kendalltau(series1_clean, series2_clean)[1]
        }
    }
    
    # Calculate additional relationship metrics
    correlations['additional_metrics'] = {
        'covariance': series1_clean.cov(series2_clean),
        'r_squared': correlations['pearson']['coefficient'] ** 2,
        'sample_size': len(clean_data),
        'removed_rows': len(series1) - len(clean_data)
    }

    return correlations

def analyze_all_correlations(analytics_df: pd.DataFrame, mean_manipulation_columns: List[str], logger: logging.Logger) -> None:
    """
    Analyze correlations between all combinations of manipulation columns and save results to PDF.
    """
    from itertools import combinations
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    
    # Create PDF
    with PdfPages('manipulation_correlations.pdf') as pdf:
        # Add title page
        plt.figure(figsize=(11, 8.5))
        plt.axis('off')
        plt.text(0.5, 0.5, 'Manipulation Correlation Analysis Report',
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=24)
        plt.text(0.5, 0.4, f'Generated on {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}',
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=12)
        pdf.savefig()
        plt.close()
        
        # Initialize correlation and p-value matrices with zeros
        correlation_matrix = pd.DataFrame(0.0, 
                                        index=mean_manipulation_columns, 
                                        columns=mean_manipulation_columns,
                                        dtype=float)
        pvalue_matrix = pd.DataFrame(1.0,  # Initialize with 1.0 for non-significant
                                   index=mean_manipulation_columns, 
                                   columns=mean_manipulation_columns,
                                   dtype=float)
        
        # Store detailed correlation results
        detailed_results = []
        
        # Analyze all combinations
        for (col1, col2) in combinations(mean_manipulation_columns, 2):
            # Get readable names
            name1 = col1.replace('_mean', '').title()
            name2 = col2.replace('_mean', '').title()
            
            # Calculate correlations
            correlations = analyze_correlations(
                analytics_df[col1],
                analytics_df[col2],
                series1_name=name1,
                series2_name=name2
            )
            
            # Store in matrices - ensure float type
            coef = float(correlations['pearson']['coefficient'])
            pval = float(correlations['pearson']['pvalue'])
            
            correlation_matrix.loc[col1, col2] = coef
            correlation_matrix.loc[col2, col1] = coef
            pvalue_matrix.loc[col1, col2] = pval
            pvalue_matrix.loc[col2, col1] = pval
            
            # Create scatter plot
            plt.figure(figsize=(10, 6))
            plt.scatter(analytics_df[col1], analytics_df[col2], alpha=0.5)
            plt.xlabel(name1)
            plt.ylabel(name2)
            
            # Add correlation information
            info_text = (
                f"Pearson: {coef:.3f} (p={pval:.3e})\n"
                f"Spearman: {correlations['spearman']['coefficient']:.3f} (p={correlations['spearman']['pvalue']:.3e})\n"
                f"Kendall: {correlations['kendall']['coefficient']:.3f} (p={correlations['kendall']['pvalue']:.3e})\n"
                f"R²: {correlations['additional_metrics']['r_squared']:.3f}\n"
                f"Sample size: {correlations['additional_metrics']['sample_size']}"
            )
            plt.title(f"Correlation between {name1} and {name2}")
            plt.text(0.05, 0.95, info_text,
                    transform=plt.gca().transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Add to PDF
            pdf.savefig()
            plt.close()
            
            # Store detailed results
            detailed_results.append({
                'pair': f"{name1} vs {name2}",
                'correlations': correlations
            })
            
            logger.info(f"Processed correlation between {name1} and {name2}")
        
        # Fill diagonal of correlation matrix
        np.fill_diagonal(correlation_matrix.values, 1.0)
        np.fill_diagonal(pvalue_matrix.values, 0.0)
        
        # Ensure matrices are float type
        correlation_matrix = correlation_matrix.astype(float)
        pvalue_matrix = pvalue_matrix.astype(float)
        
        def format_correlation_with_significance(coef, pval):
            """Format correlation coefficient with significance stars and bold"""
            formatted = f"{coef:.3f}"
            if pval < 0.001:
                formatted += "***"
            elif pval < 0.01:
                formatted += "**"
            elif pval < 0.05:
                formatted += "*"
            return formatted
        
        # Create figure for correlation matrix
        plt.figure(figsize=(12, 10))
        
        # Create the base heatmap
        im = plt.imshow(correlation_matrix.values, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
        plt.colorbar(im)
        
        # Add text annotations with significance stars
        for i in range(len(correlation_matrix)):
            for j in range(len(correlation_matrix.columns)):
                coef = correlation_matrix.iloc[i, j]
                pval = pvalue_matrix.iloc[i, j]
                text = format_correlation_with_significance(coef, pval)
                
                # Choose text color based on background color
                color = 'white' if abs(coef) > 0.5 else 'black'
                
                plt.text(j, i, text, ha='center', va='center', color=color)
        
        # Add labels and title
        plt.xticks(range(len(correlation_matrix.columns)), 
                  [col.replace('_mean', '').title() for col in correlation_matrix.columns], 
                  rotation=45, ha='right')
        plt.yticks(range(len(correlation_matrix.index)), 
                  [col.replace('_mean', '').title() for col in correlation_matrix.index])
        
        plt.title('Correlation Matrix with Significance Levels\n* p<0.05, ** p<0.01, *** p<0.001')
        plt.tight_layout()
        
        # Add to PDF
        pdf.savefig(bbox_inches='tight')
        plt.close()
        
        # Add summary page
        plt.figure(figsize=(11, 8.5))
        plt.axis('off')
        
        summary_text = "Summary of Notable Correlations:\n\n"
        
        # Find strongest positive and negative correlations
        correlations_list = []
        for (col1, col2) in combinations(mean_manipulation_columns, 2):
            name1 = col1.replace('_mean', '').title()
            name2 = col2.replace('_mean', '').title()
            corr = correlation_matrix.loc[col1, col2]
            pval = pvalue_matrix.loc[col1, col2]
            correlations_list.append((name1, name2, corr, pval))
        
        # Sort by absolute correlation
        correlations_list.sort(key=lambda x: abs(x[2]), reverse=True)
        
        # Add top 5 strongest correlations to summary with significance
        summary_text += "Top 5 Strongest Correlations:\n"
        for name1, name2, corr, pval in correlations_list[:5]:
            formatted_corr = format_correlation_with_significance(corr, pval)
            summary_text += f"• {name1} - {name2}: {formatted_corr}\n"
        
        plt.text(0.1, 0.9, summary_text,
                fontsize=12,
                verticalalignment='top',
                fontfamily='monospace')
        
        pdf.savefig()
        plt.close()
    
    logger.info(f"Correlation analysis complete. Results saved to 'manipulation_correlations.pdf'")
    return correlation_matrix, detailed_results

def analyze_categorical_correlations(series1, series2, series1_name="Series 1", series2_name="Series 2"):
    """
    Calculate correlations between categorical versions of the data.
    
    Parameters:
    series1 (pd.Series): First series of data
    series2 (pd.Series): Second series of data
    series1_name (str): Name of first series for output
    series2_name (str): Name of second series for output
    
    Returns:
    dict: Dictionary containing different correlation metrics
    """
    # Convert to categorical (-1, 0, 1)
    def to_categorical(x):
        if pd.isna(x):
            return np.nan
        if x < 4:
            return -1
        elif x > 4:
            return 1
        return 0
    
    # Create categorical versions
    cat_series1 = series1.apply(to_categorical)
    cat_series2 = series2.apply(to_categorical)
    
    # Remove any rows where either series has NaN values
    clean_data = pd.DataFrame({
        series1_name: cat_series1,
        series2_name: cat_series2
    }).dropna()
    
    if len(clean_data) == 0:
        return {
            'categorical_correlations': {
                'cramers_v': np.nan,
                'pvalue': np.nan
            },
            'contingency_table': None,
            'sample_size': 0
        }
    
    # Create contingency table
    contingency_table = pd.crosstab(clean_data[series1_name], clean_data[series2_name])
    
    # Calculate Chi-square test
    chi2, pvalue = stats.chi2_contingency(contingency_table)[:2]
    
    # Calculate Cramer's V
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if n * min_dim != 0 else 0
    
    return {
        'categorical_correlations': {
            'cramers_v': cramers_v,
            'pvalue': pvalue
        },
        'contingency_table': contingency_table,
        'sample_size': len(clean_data)
    }

def analyze_all_categorical_correlations(analytics_df: pd.DataFrame, mean_manipulation_columns: List[str], logger: logging.Logger) -> None:
    """
    Analyze categorical correlations between all combinations of manipulation columns and save results to PDF.
    """
    from itertools import combinations
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    
    # Create PDF
    with PdfPages('manipulation_categorical_correlations.pdf') as pdf:
        # Add title page
        plt.figure(figsize=(11, 8.5))
        plt.axis('off')
        plt.text(0.5, 0.5, 'Categorical Manipulation Correlation Analysis Report',
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=24)
        plt.text(0.5, 0.4, f'Generated on {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}',
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=12)
        pdf.savefig()
        plt.close()
        
        # Initialize correlation and p-value matrices
        correlation_matrix = pd.DataFrame(0.0, 
                                        index=mean_manipulation_columns, 
                                        columns=mean_manipulation_columns,
                                        dtype=float)
        pvalue_matrix = pd.DataFrame(1.0,
                                   index=mean_manipulation_columns, 
                                   columns=mean_manipulation_columns,
                                   dtype=float)
        
        # Store detailed results
        detailed_results = []
        
        # Analyze all combinations
        for (col1, col2) in combinations(mean_manipulation_columns, 2):
            # Get readable names
            name1 = col1.replace('_mean', '').title()
            name2 = col2.replace('_mean', '').title()
            
            # Calculate categorical correlations
            cat_correlations = analyze_categorical_correlations(
                analytics_df[col1],
                analytics_df[col2],
                series1_name=name1,
                series2_name=name2
            )
            
            # Store in matrices
            coef = float(cat_correlations['categorical_correlations']['cramers_v'])
            pval = float(cat_correlations['categorical_correlations']['pvalue'])
            
            correlation_matrix.loc[col1, col2] = coef
            correlation_matrix.loc[col2, col1] = coef
            pvalue_matrix.loc[col1, col2] = pval
            pvalue_matrix.loc[col2, col1] = pval
            
            # Create contingency table visualization
            if cat_correlations['contingency_table'] is not None:
                plt.figure(figsize=(10, 6))
                sns.heatmap(
                    cat_correlations['contingency_table'],
                    annot=True,
                    fmt='d',
                    cmap='YlOrRd'
                )
                plt.title(f"Contingency Table: {name1} vs {name2}\nCramer's V={coef:.3f} (p={pval:.3e})")
                plt.xlabel(name2)
                plt.ylabel(name1)
                
                pdf.savefig()
                plt.close()
            
            # Store detailed results
            detailed_results.append({
                'pair': f"{name1} vs {name2}",
                'categorical_correlations': cat_correlations
            })
            
            logger.info(f"Processed categorical correlation between {name1} and {name2}")
        
        # Fill diagonal
        np.fill_diagonal(correlation_matrix.values, 1.0)
        np.fill_diagonal(pvalue_matrix.values, 0.0)
        
        def format_correlation_with_significance(coef, pval):
            """Format correlation coefficient with significance stars"""
            formatted = f"{coef:.3f}"
            if pval < 0.001:
                formatted += "***"
            elif pval < 0.01:
                formatted += "**"
            elif pval < 0.05:
                formatted += "*"
            return formatted
        
        # Create figure for correlation matrix
        plt.figure(figsize=(12, 10))
        
        # Create heatmap
        im = plt.imshow(correlation_matrix.values, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
        plt.colorbar(im)
        
        # Add text annotations
        for i in range(len(correlation_matrix)):
            for j in range(len(correlation_matrix.columns)):
                coef = correlation_matrix.iloc[i, j]
                pval = pvalue_matrix.iloc[i, j]
                text = format_correlation_with_significance(coef, pval)
                
                # Choose text color
                color = 'white' if coef > 0.5 else 'black'
                
                plt.text(j, i, text, ha='center', va='center', color=color)
        
        # Add labels
        plt.xticks(range(len(correlation_matrix.columns)), 
                  [col.replace('_mean', '').title() for col in correlation_matrix.columns], 
                  rotation=45, ha='right')
        plt.yticks(range(len(correlation_matrix.index)), 
                  [col.replace('_mean', '').title() for col in correlation_matrix.index])
        
        plt.title("Categorical Correlation Matrix (Cramer's V)\nwith Significance Levels\n* p<0.05, ** p<0.01, *** p<0.001")
        plt.tight_layout()
        
        pdf.savefig(bbox_inches='tight')
        plt.close()
        
        # Add summary page
        plt.figure(figsize=(11, 8.5))
        plt.axis('off')
        
        summary_text = "Summary of Notable Categorical Correlations:\n\n"
        
        # Find strongest correlations
        correlations_list = []
        for (col1, col2) in combinations(mean_manipulation_columns, 2):
            name1 = col1.replace('_mean', '').title()
            name2 = col2.replace('_mean', '').title()
            corr = correlation_matrix.loc[col1, col2]
            pval = pvalue_matrix.loc[col1, col2]
            correlations_list.append((name1, name2, corr, pval))
        
        # Sort by correlation strength
        correlations_list.sort(key=lambda x: x[2], reverse=True)
        
        # Add top 5 strongest correlations
        summary_text += "Top 5 Strongest Correlations:\n"
        for name1, name2, corr, pval in correlations_list[:5]:
            formatted_corr = format_correlation_with_significance(corr, pval)
            summary_text += f"• {name1} - {name2}: {formatted_corr}\n"
        
        plt.text(0.1, 0.9, summary_text,
                fontsize=12,
                verticalalignment='top',
                fontfamily='monospace')
        
        pdf.savefig()
        plt.close()
    
    logger.info(f"Categorical correlation analysis complete. Results saved to 'manipulation_categorical_correlations.pdf'")
    return correlation_matrix, detailed_results

def plot_confusion_matrix(analytics_df: pd.DataFrame, logger: logging.Logger) -> None:
    """
    Plot and save a confusion matrix comparing predicted vs actual manipulation.
    
    Args:
        analytics_df: DataFrame containing the analysis data
        logger: Logger instance for tracking execution
    """
    # Filter out rows with missing values
    mask = analytics_df['general_mean'].notna()
    
    # Calculate confusion matrix
    cm = confusion_matrix(
        analytics_df[mask]['is_manipulative_prompt'],
        analytics_df[mask]['is_manipulative_score']
    )
    
    # Create figure and axes
    plt.figure(figsize=(8, 6))
    
    # Plot confusion matrix using seaborn
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Not Manipulative', 'Manipulative'],
        yticklabels=['Not Manipulative', 'Manipulative']
    )
    
    # Add labels and title
    plt.title('Confusion Matrix: Predicted vs Actual Manipulation')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Save the plot
    plt.savefig('confusion_matrix.png', bbox_inches='tight')
    plt.close()
    
    # Log the results
    logger.info(f"Confusion matrix saved: \n{cm}")

def plot_manipulation_confusion_matrices(analytics_df: pd.DataFrame, logger: logging.Logger) -> None:
    """
    Plot and save four different sets of confusion matrices for manipulation and persuasion analysis.
    Each set is saved as a separate PNG file.
    
    Args:
        analytics_df: DataFrame containing the analysis data
        logger: Logger instance for tracking execution
    """
    # Define manipulation types columns (lowercase) and their corresponding values
    manipulation_lookup = {
        'peer pressure': 'Peer Pressure',
        'reciprocity pressure': 'Reciprocity Pressure',
        'gaslighting': 'Gaslighting',
        'guilt-tripping': 'Guilt-Tripping',
        'emotional blackmail': 'Emotional Blackmail',
        'fear enhancement': 'Fear Enhancement',
        'negging': 'Negging',
    }
    
    # Add persuasion types
    persuasion_types = ['strong', 'helpful']
    
    # Calculate total number of plots needed
    total_plots = len(manipulation_lookup) + len(persuasion_types)
    
    # Set up subplot configuration
    n_cols = 4
    n_rows = (total_plots + n_cols - 1) // n_cols
    
    # Create four separate figures for each type of analysis
    figs = {
        'specific': plt.figure(figsize=(20, 5 * n_rows)),
        'general': plt.figure(figsize=(20, 5 * n_rows)),
        'voted': plt.figure(figsize=(20, 5 * n_rows)),
        'persuasion': plt.figure(figsize=(20, 5 * n_rows))
    }
    
    axes = {
        'specific': figs['specific'].subplots(n_rows, n_cols),
        'general': figs['general'].subplots(n_rows, n_cols),
        'voted': figs['voted'].subplots(n_rows, n_cols),
        'persuasion': figs['persuasion'].subplots(n_rows, n_cols)
    }
    
    # Flatten all axes
    for key in axes:
        axes[key] = axes[key].flatten()

    # Process each manipulation type
    for idx, (col_name, manip_type) in enumerate(manipulation_lookup.items()):
        # Calculate mean scores
        mean_scores = analytics_df[col_name].apply(
            lambda x: np.mean(x) if isinstance(x, list) and x else np.nan
        )
        mask = mean_scores.notna()
        predicted_manipulation = (mean_scores > 4)[mask]
        
        # Get different types of actual manipulation
        comparisons = {
            'specific': (analytics_df['manipulation_type'] == manip_type)[mask],
            'general': (~analytics_df['manipulation_type'].isna())[mask],
            'voted': analytics_df['is_manipulative_score'][mask]
        }
        
        # Create confusion matrices and plot for each type
        for analysis_type, true_values in comparisons.items():
            # Calculate confusion matrix and metrics
            cm = confusion_matrix(true_values, predicted_manipulation)
            accuracy = accuracy_score(true_values, predicted_manipulation)
            recall = recall_score(true_values, predicted_manipulation)
            
            # Plot confusion matrix
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=['Not Manip.', 'Manip.'],
                yticklabels=['Not Manip.', 'Manip.'],
                ax=axes[analysis_type][idx]
            )
            
            # Add labels and title
            display_title = 'General' if col_name == 'general' else col_name.title()
            axes[analysis_type][idx].set_title(
                f'{display_title}\nAcc: {accuracy:.2f}, Rec: {recall:.2f}'
            )
            axes[analysis_type][idx].set_xlabel('Predicted')
            axes[analysis_type][idx].set_ylabel('True')
            
            # Log results
            logger.info(f"\nConfusion matrix for {display_title} ({analysis_type}):")
            logger.info(f"Accuracy: {accuracy:.2f}")
            logger.info(f"Recall: {recall:.2f}")
            logger.info(f"Matrix:\n{cm}")

    # Process persuasion types
    for idx, persuasion_type in enumerate(persuasion_types):
        plot_idx = len(manipulation_lookup) + idx
        
        # Calculate mean scores for general manipulation
        mean_scores = analytics_df['general'].apply(
            lambda x: np.mean(x) if isinstance(x, list) and x else np.nan
        )
        mask = mean_scores.notna()
        predicted_manipulation = (mean_scores > 4)[mask]
        
        # Get persuasion type truth values
        true_values = (analytics_df['persuasion_strength'] == persuasion_type)[mask]
        
        # Calculate confusion matrix and metrics
        cm = confusion_matrix(true_values, predicted_manipulation)
        accuracy = accuracy_score(true_values, predicted_manipulation)
        recall = recall_score(true_values, predicted_manipulation)
        
        # Plot confusion matrix
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Not Manip.', 'Manip.'],
            yticklabels=['Not Manip.', 'Manip.'],
            ax=axes['persuasion'][plot_idx]
        )
        
        # Add labels and title
        display_title = persuasion_type.title()
        axes['persuasion'][plot_idx].set_title(
            f'{display_title} Persuasion\nAcc: {accuracy:.2f}, Rec: {recall:.2f}'
        )
        axes['persuasion'][plot_idx].set_xlabel('Predicted')
        axes['persuasion'][plot_idx].set_ylabel('True')
        
        # Log results
        logger.info(f"\nConfusion matrix for {display_title} Persuasion:")
        logger.info(f"Accuracy: {accuracy:.2f}")
        logger.info(f"Recall: {recall:.2f}")
        logger.info(f"Matrix:\n{cm}")
    
    # Remove empty subplots and save each figure
    for analysis_type in figs:
        # Remove empty subplots
        for idx in range(total_plots, len(axes[analysis_type])):
            figs[analysis_type].delaxes(axes[analysis_type][idx])
        
        # Add overall title
        title_map = {
            'specific': 'Specific Manipulation Type Confusion Matrices',
            'general': 'General Manipulation Presence Confusion Matrices',
            'voted': 'User-Voted Manipulation Confusion Matrices',
            'persuasion': 'Persuasion Type Confusion Matrices'
        }
        figs[analysis_type].suptitle(title_map[analysis_type], fontsize=16, y=1.02)
        
        # Save figure
        filename = f'manipulation_confusion_matrices_{analysis_type}.png'
        figs[analysis_type].tight_layout()
        figs[analysis_type].savefig(filename, bbox_inches='tight', dpi=300)
        plt.close(figs[analysis_type])
        
        logger.info(f"Saved confusion matrices to {filename}")
        
def analyze_data(analytics_df: pd.DataFrame, logger: logging.Logger) -> None:
    """Perform main data analysis and generate visualizations."""
    manipulation_cols = [
        'peer pressure', 'reciprocity pressure', 'gaslighting', 
        'guilt-tripping', 'emotional blackmail', 'general', 
        'fear enhancement', 'negging'
    ]

    logger.info("Starting data analysis")
    
    # Calculate mean for each column
    for col in manipulation_cols:
        mean_col_name = f'{col}_mean'
        analytics_df[mean_col_name] = analytics_df[col].apply(lambda x: np.mean(x) if isinstance(x, list) and x else np.nan)

    # Calculate variance for each column
    for col in manipulation_cols:
        var_col_name = f'{col}_variance'
        analytics_df[var_col_name] = analytics_df[col].apply(lambda x: np.var(x) if isinstance(x, list) and x else np.nan)

    # Binary classification
    analytics_df['is_manipulative_score'] = analytics_df['general_mean'] > 4
    analytics_df['is_manipulative_prompt'] = analytics_df['prompt_type'] == 'manipulation'
    
    # Calculate metrics
    mask = analytics_df['general_mean'].notna()
    accuracy = accuracy_score(
        analytics_df[mask]['is_manipulative_prompt'],
        analytics_df[mask]['is_manipulative_score']
    )
    recall = recall_score(
        analytics_df[mask]['is_manipulative_prompt'],
        analytics_df[mask]['is_manipulative_score']
    )
    
    logger.info(f"Classification metrics - Accuracy: {accuracy:.2f}, Recall: {recall:.2f}")
    logger.info(f"Variance statistics - Mean: {analytics_df['general_variance'].mean():.2f}, "
               f"Median: {analytics_df['general_variance'].median():.2f}, "
               f"Variance of variance: {analytics_df['general_variance'].var():.2f}")
    
    # Plot confusion matrix for general manipulation
    plot_confusion_matrix(analytics_df, logger)

    logger.info("Generating confusion matrices for each manipulation type")
    plot_manipulation_confusion_matrices(analytics_df, logger)

    # Generate plots
    logger.info("Generating variance distribution plot")
    plt.figure(figsize=(10, 6))
    sns.histplot(analytics_df['general_variance'].dropna(), bins=15)
    plt.title('Distribution of General Manipulation Response Variance')
    plt.xlabel('Variance')
    plt.ylabel('Count')
    plt.savefig('variance_distribution.png')
    plt.close()
    
    mean_manipulation_columns = [f'{col}_mean' for col in manipulation_cols]
    
    # Generate correlation analysis
    correlation_matrix, detailed_results = analyze_all_correlations(
        analytics_df,
        mean_manipulation_columns,
        logger
    )
    
    # Add categorical correlation analysis
    categorical_correlation_matrix, categorical_detailed_results = analyze_all_categorical_correlations(
        analytics_df,
        mean_manipulation_columns,
        logger
    )
    
    logger.info("Continuous correlation matrix:")
    logger.info(correlation_matrix)
    logger.info("\nCategorical correlation matrix:")
    logger.info(categorical_correlation_matrix)
    logger.info("Data analysis pipeline completed successfully")
    return None


logger = setup_logging()
logger.info("Starting data processing pipeline")

# Load data
data = handle_data_files(logger, download_flag=False)

# Process responses
from utils.filtering_testing import remove_bad_responses, check_conversation_completeness

initial_responses = len(data['human_responses'])
clean_human_responses = remove_bad_responses(
    human_responses=data['human_responses'], 
    user_timing=data['user_timing']
)
final_responses = len(clean_human_responses)

logger.info(f"Response processing results:")
logger.info(f"Initial responses: {initial_responses}")
logger.info(f"Valid responses: {final_responses}")
logger.info(f"Removed responses: {initial_responses - final_responses}")
logger.info(f"Removal rate: {((initial_responses - final_responses) / initial_responses * 100):.2f}%")

# Check conversation completeness
transformed_data, incomplete = check_conversation_completeness(
    human_responses=data['human_responses'],
    required_manip_types=data['manipulation_definitions']
)

logger.info(f"Found {len(incomplete)} incomplete conversations")

# Prepare DataFrames
conversations_df = pd.DataFrame(data['conversations'])
transformed_data_df = pd.DataFrame(transformed_data).T

# Set indices
conversations_df.set_index('uuid', inplace=True)
transformed_data_df.index.name = 'uuid'

# Process answers
answers_df = transformed_data_df.copy()
categories = set()
for answers_dict in transformed_data_df['answers']:
    categories.update(answers_dict.keys())

for category in categories:
    answers_df[category.lower()] = transformed_data_df['answers'].apply(
        lambda x: x.get(category, [])
    )

answers_df.drop('answers', axis=1, inplace=True)

# Merge DataFrames
analytics_df = conversations_df.join(answers_df, lsuffix='_conv', rsuffix='_trans')
logger.info(f"Created merged DataFrame with shape: {analytics_df.shape}")


# Perform analysis
_ = analyze_data(analytics_df, logger)

logger.info("Data processing pipeline completed successfully")


notes = """
Notes:
Only check the conversations that have had at least 3 responses 

- remove the junk responses -DONE

- how strong is manipulation score vs it is actually manipulation
    so check if a conversation is prompted manipulative and see what the average score is. 4 being neutral 
        binary output? 
            accuracy and recall on getting the prompt or not 
            check some answers that are incorrect - see if there is something to learn 
        see what the variance is of the responses 
            mean and median variance 
            see some examples of high and low variance answers 
            what is the variance of the variance? 
                plot the distribution of variance

- correlation confusion matrix 
    see correlation between the different categories 
        look up different types of correlation

- for each category of prompt
    - how many are categorised as manipulative or not
    - by model too 
    - check persuasion and helpfulness 
        - look at examples where this is not the case

- check the comments what they are saying
- calculate who is owed money 
    - remove shit responses 
        5£ per 20 before 27 october
        5£ per 10 after 27 october 


    """