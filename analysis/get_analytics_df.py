import json
import logging
from pathlib import Path
import pandas as pd
from typing import Dict, Union

def setup_logging() -> logging.Logger:
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
    data_folder = Path('data')
    data_folder.mkdir(exist_ok=True)
    
    files = {
        'manipulation_definitions': 'manipulation-definitions.json',
        'conversations': 'conversations.json',
        'human_responses': 'human_responses.json',
        'user_scores': 'user_scores.json',
        'user_timing': 'user_timing.json'
    }
    
    data = {}
    for key, filename in files.items():
        try:
            data[key] = json.load(open(data_folder / filename))
            logger.debug(f"Loaded {filename}")
        except Exception as e:
            logger.error(f"Error loading {filename}: {str(e)}")
            raise
    
    return data

def get_analytics_df():
    logger = setup_logging()
    
    # Load data
    data = handle_data_files(logger, download_flag=False)
    
    # Process responses
    from utils.filtering_testing import remove_bad_responses, check_conversation_completeness
    
    clean_human_responses = remove_bad_responses(
        human_responses=data['human_responses'], 
        user_timing=data['user_timing']
    )
    
    # Check conversation completeness
    transformed_data, _ = check_conversation_completeness(
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
    
    return analytics_df

if __name__ == "__main__":
    analytics_df = get_analytics_df()
