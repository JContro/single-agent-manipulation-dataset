import json
import os
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Union, Tuple
import scipy.stats as stats
from matplotlib.backends.backend_pdf import PdfPages

def setup_logging() -> logging.Logger:
    """Configure and return logger with consistent formatting."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('model_analysis.log'),
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

def preprocess_data(data: Dict, logger: logging.Logger) -> pd.DataFrame:
    """Preprocess the data using steps from original code."""
    from utils.filtering_testing import remove_bad_responses, check_conversation_completeness
    
    # Process responses
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
    
    # Check conversation completeness
    transformed_data, incomplete = check_conversation_completeness(
        human_responses=data['human_responses'],
        required_manip_types=data['manipulation_definitions']
    )

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
    
    # Calculate means for manipulation types
    manipulation_cols = [
        'peer pressure', 'reciprocity pressure', 'gaslighting', 
        'guilt-tripping', 'emotional blackmail', 'general', 
        'fear enhancement', 'negging'
    ]

    for col in manipulation_cols:
        mean_col_name = f'{col}_mean'
        analytics_df[mean_col_name] = analytics_df[col].apply(
            lambda x: np.mean(x) if isinstance(x, list) and x else np.nan
        )

        var_col_name = f'{col}_variance'
        analytics_df[var_col_name] = analytics_df[col].apply(
            lambda x: np.var(x) if isinstance(x, list) and x else np.nan
        )

    # Binary classification
    analytics_df['is_manipulative_score'] = analytics_df['general_mean'] > 4
    analytics_df['is_manipulative_prompt'] = analytics_df['prompt_type'] == 'manipulation'
    
    return analytics_df


def main():
    """Main execution function."""
    logger = setup_logging()
    logger.info("Starting model comparison analysis")
    
    try:
        # Load and preprocess data
        data = handle_data_files(logger, download_flag=False)
        analytics_df = preprocess_data(data, logger)
        
        print(analytics_df.head)

        print(analytics_df.columns)

        print(set(analytics_df.persuasion_strength.values))
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
