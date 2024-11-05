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

def create_correlation_matrix(df: pd.DataFrame, manipulation_cols: List[str], logger: logging.Logger) -> Dict[str, pd.DataFrame]:
    """Create correlation matrices using different methods."""
    logger.info("Calculating correlation matrices")
    correlation_df = pd.DataFrame()
    
    for col in manipulation_cols:
        correlation_df[col] = df[col].apply(
            lambda x: np.mean(x) if isinstance(x, list) and x else np.nan
        )
    
    correlation_methods = ['pearson', 'spearman', 'kendall']
    correlations = {
        method: correlation_df.corr(method=method) 
        for method in correlation_methods
    }
    
    logger.info(f"Calculated correlations using methods: {', '.join(correlation_methods)}")
    return correlations

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
    Plot and save three different sets of confusion matrices for manipulation analysis.
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
    
    # Set up subplot configuration
    n_cols = 4
    n_rows = (len(manipulation_lookup) + n_cols - 1) // n_cols
    
    # Create three separate figures for each type of analysis
    figs = {
        'specific': plt.figure(figsize=(20, 5 * n_rows)),
        'general': plt.figure(figsize=(20, 5 * n_rows)),
        'voted': plt.figure(figsize=(20, 5 * n_rows))
    }
    
    axes = {
        'specific': figs['specific'].subplots(n_rows, n_cols),
        'general': figs['general'].subplots(n_rows, n_cols),
        'voted': figs['voted'].subplots(n_rows, n_cols)
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
    
    # Remove empty subplots and save each figure
    for analysis_type in figs:
        # Remove empty subplots
        for idx in range(len(manipulation_lookup), len(axes[analysis_type])):
            figs[analysis_type].delaxes(axes[analysis_type][idx])
        
        # Add overall title
        title_map = {
            'specific': 'Specific Manipulation Type Confusion Matrices',
            'general': 'General Manipulation Presence Confusion Matrices',
            'voted': 'User-Voted Manipulation Confusion Matrices'
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
    
    manipulation_cols = [
        'peer pressure', 'reciprocity pressure', 'gaslighting',
        'guilt-tripping', 'emotional blackmail', 'general',
        'fear enhancement', 'negging'
    ]

    # Calculate mean for each column
    for col in manipulation_cols:
        mean_col_name = f'{col}_mean'
        analytics_df[mean_col_name] = analytics_df[col].apply(lambda x: np.mean(x))

    # Calculate variance for each column
    for col in manipulation_cols:
        var_col_name = f'{col}_variance'
        analytics_df[var_col_name] = analytics_df[col].apply(lambda x: np.var(x))

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
    
    # plot a confusion matrix for general  manipulation
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
    
    # # Correlation analysis
    # correlations = create_correlation_matrix(analytics_df, manipulation_cols, logger)
    
    # # Prompt type analysis
    # logger.info("Analyzing prompt types")
    # prompt_analysis = analytics_df.groupby(['prompt_type', 'model']).agg({
    #     'is_manipulative_score': ['count', 'mean'],
    #     'persuasion_strength': lambda x: x.value_counts().to_dict()
    # }).round(2)
    
    # logger.info("\nPrompt Type Analysis by Model:")
    # logger.info("\n" + str(prompt_analysis))
    
    # # Analysis of incorrect classifications
    # incorrect_classifications = analytics_df[
    #     (analytics_df['is_manipulative_prompt'] != analytics_df['is_manipulative_score']) & 
    #     mask
    # ]
    
    # logger.info(f"\nFound {len(incorrect_classifications)} incorrect classifications")
    
    # # Variance examples
    # high_variance = analytics_df.nlargest(3, 'variance')
    # low_variance = analytics_df.nsmallest(3, 'variance')
    
    # logger.info("\nAnalysis complete")
    return None 
# correlations, prompt_analysis, incorrect_classifications, high_variance, low_variance

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