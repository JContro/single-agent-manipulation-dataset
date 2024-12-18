import logging
from typing import Dict, Union, List, Tuple, Optional
import json
import pandas as pd
import numpy as np
from datetime import datetime
import os
from logging.handlers import RotatingFileHandler

# TODO: get the logger out of here
def setup_logging(
    log_file: str = 'data_processing.log',
    max_bytes: int = 10_485_760,  # 10MB
    backup_count: int = 5,
    log_level: int = logging.INFO
) -> logging.Logger:
    """
    Configure and return logger with file output and rotation.
    
    Args:
        log_file: Path to the log file
        max_bytes: Maximum size of each log file before rotation
        backup_count: Number of backup files to keep
        log_level: Logging level (default: logging.INFO)
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    # Clear any existing handlers
    if logger.handlers:
        logger.handlers.clear()

    # Create formatters and handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Rotating file handler
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def _load_json_files(data_dir: str, logger: logging.Logger) -> Dict[str, Union[dict, list]]:
    """
    Load JSON files from the specified directory.
    
    Args:
        data_dir: Directory containing the data files
        logger: Logger instance
    
    Returns:
        Dict containing loaded data from JSON files
    """
    files = {
        'manipulation_definitions': 'manipulation-definitions.json',
        'conversations': 'conversations.json',
        'human_responses': 'human_responses.json',
        'user_scores': 'user_scores.json',
        'user_timing': 'user_timing.json'
    }

    data = {}
    for key, filename in files.items():
        file_path = os.path.join(data_dir, filename)
        try:
            with open(file_path, 'r') as f:
                data[key] = json.load(f)
            logger.debug(f"Loaded {filename}")
        except Exception as e:
            logger.error(f"Error loading {filename}: {str(e)}")
            raise

    return data


def _find_bad_responses(user_timing: Dict) -> List[Tuple[str, str]]:
    """
    Identify invalid responses based on timing data.
    
    Args:
        user_timing: Dictionary containing user timing information
    
    Returns:
        List of tuples containing (email, conversation_id) for invalid responses
    """
    invalid_responses = []

    for email in user_timing.keys():
        for conv_id, timing in user_timing[email].items():
            if 'submission_time' not in timing:
                continue

            request_time = datetime.fromisoformat(timing['request_time'])
            submission_time = datetime.fromisoformat(timing['submission_time'])
            time_delta = (submission_time - request_time).total_seconds()

            if time_delta <= 20:
                invalid_responses.append((email, conv_id))

    return invalid_responses


def _remove_bad_responses(human_responses: List[Dict], user_timing: Dict) -> List[Dict]:
    """
    Remove invalid responses based on timing data.
    
    Args:
        human_responses: List of human response dictionaries
        user_timing: Dictionary containing user timing information
    
    Returns:
        List of filtered human responses
    """
    invalid_responses = _find_bad_responses(user_timing)
    return [response for response in human_responses
            if (response['email'], response['conversation_id']) not in invalid_responses]


def _check_conversation_completeness(
    human_responses: List[Dict],
    required_manip_types: List[str]
) -> Tuple[Dict, Dict]:
    """
    Process conversation responses and check for completeness.
    
    Args:
        human_responses: List of response dictionaries
        required_manip_types: List of required manipulation types
    
    Returns:
        Tuple of (transformed_data dict, incomplete_conversations dict)
    """
    result = {}
    for response in human_responses:
        conv_id = response['conversation_id']
        if conv_id not in result:
            result[conv_id] = {"n_responses": 0, "answers": {}}

        result[conv_id]['n_responses'] += 1

        for manip_type, score in response['scores'].items():
            if manip_type not in result[conv_id]['answers']:
                result[conv_id]['answers'][manip_type] = []
            result[conv_id]['answers'][manip_type].append(score)

    incomplete_conversations = {}
    for conv_id, conv_data in result.items():
        if conv_data['n_responses'] >= 3:
            missing_types = set(required_manip_types) - set(conv_data['answers'].keys())
            if missing_types:
                incomplete_conversations[conv_id] = list(missing_types)

    return result, incomplete_conversations


def process_conversation_data(
    data_dir: str,
    log_file: Optional[str] = None,
    log_level: int = logging.INFO
) -> pd.DataFrame:
    """
    Load and process conversation data from the specified directory.
    
    Args:
        data_dir: Directory containing the data files
        log_file: Path to the log file (optional)
        log_level: Logging level (default: logging.INFO)
    
    Returns:
        DataFrame containing processed conversation data
    """
    # Setup logging
    if log_file is None:
        log_file = os.path.join(data_dir, 'data_processing.log')
    
    logger = setup_logging(
        log_file=log_file,
        log_level=log_level
    )

    # Load data files
    data = _load_json_files(data_dir, logger)

    # Process responses
    initial_responses = len(data['human_responses'])
    clean_human_responses = _remove_bad_responses(
        human_responses=data['human_responses'],
        user_timing=data['user_timing']
    )
    final_responses = len(clean_human_responses)

    # Log processing results
    logger.info(f"Response processing results:")
    logger.info(f"Initial responses: {initial_responses}")
    logger.info(f"Valid responses: {final_responses}")
    logger.info(f"Removed responses: {initial_responses - final_responses}")
    logger.info(f"Removal rate: {((initial_responses - final_responses) / initial_responses * 100):.2f}%")

    # Check conversation completeness
    transformed_data, incomplete = _check_conversation_completeness(
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

    # Process pressure columns
    pressure_cols = [
        'peer pressure', 'reciprocity pressure', 'gaslighting', 'guilt-tripping',
        'emotional blackmail', 'general', 'fear enhancement', 'negging'
    ]
    
    analytics_df = analytics_df.dropna(subset=pressure_cols)

    def safe_mean(x):
        """Calculate mean of list, returning 0 if list is empty."""
        return np.mean(x) if len(x) > 0 else 0
    
    for col in pressure_cols:
        # Calculate means safely and create binary columns
        analytics_df[f'{col}_mean'] = analytics_df[col].apply(safe_mean)
        analytics_df[f'{col}_binary'] = (analytics_df[f'{col}_mean'] > 4).astype(int)
        
        # Log any empty lists found
        empty_lists = analytics_df[analytics_df[col].apply(len) == 0].index
        if len(empty_lists) > 0:
            logger.warning(f"Found {len(empty_lists)} empty lists in column {col}")
            
        # Optional: Drop rows with empty lists if desired
        # analytics_df = analytics_df[analytics_df[col].apply(len) > 0]

    return analytics_df