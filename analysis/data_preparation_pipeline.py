import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import json

def setup_logging(log_file: str = 'data_processing.log') -> logging.Logger:
    """
    Configure and return a logger with consistent formatting.
    
    Args:
        log_file: Name of the log file to write to
        
    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_data_files(data_folder: str = 'data', 
                   logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    Load required data files from the specified folder.
    
    Args:
        data_folder: Path to folder containing data files
        logger: Logger instance for tracking execution
        
    Returns:
        Dictionary containing loaded data from each file
    """
    data_path = Path(data_folder)
    files = {
        'manipulation_definitions': 'manipulation-definitions.json',
        'conversations': 'conversations.json',
        'human_responses': 'human_responses.json',
        'user_timing': 'user_timing.json'
    }
    
    data = {}
    for key, filename in files.items():
        try:
            file_path = data_path / filename
            with open(file_path) as f:
                data[key] = json.load(f)
            if logger:
                logger.debug(f"Loaded {filename}")
        except Exception as e:
            if logger:
                logger.error(f"Error loading {filename}: {str(e)}")
            raise
            
    return data

def remove_invalid_responses(human_responses: List[Dict], 
                           user_timing: Dict,
                           logger: Optional[logging.Logger] = None) -> List[Dict]:
    """
    Filter out invalid responses based on timing and completion criteria.
    
    Args:
        human_responses: List of human response data dictionaries
        user_timing: Dictionary of user timing data
        logger: Logger instance for tracking execution
        
    Returns:
        Filtered list of valid human responses
    """
    initial_count = len(human_responses)
    
    # Remove responses without timing data or incomplete responses
    valid_responses = [
        response for response in human_responses
        if response.get('uuid') in user_timing and len(response.get('answers', {})) >= 3
    ]
    
    final_count = len(valid_responses)
    
    if logger:
        logger.info(f"Response filtering results:")
        logger.info(f"Initial responses: {initial_count}")
        logger.info(f"Valid responses: {final_count}")
        logger.info(f"Removed responses: {initial_count - final_count}")
        logger.info(f"Removal rate: {((initial_count - final_count) / initial_count * 100):.2f}%")
    
    return valid_responses

def check_conversation_completeness(human_responses: List[Dict],
                                  required_manip_types: List[str]) -> Tuple[Dict[str, Dict], List[str]]:
    """
    Check completeness of conversations and identify incomplete ones.
    
    Args:
        human_responses: List of human response dictionaries
        required_manip_types: List of required manipulation types
        
    Returns:
        Tuple containing transformed data dictionary and list of incomplete conversation IDs
    """
    transformed_data = {}
    incomplete_conversations = []
    
    for response in human_responses:
        uuid = response.get('uuid')
        answers = response.get('answers', {})
        
        if not all(manip_type in answers for manip_type in required_manip_types):
            incomplete_conversations.append(uuid)
        else:
            transformed_data[uuid] = response
            
    return transformed_data, incomplete_conversations

def prepare_analytics_dataframe(conversations: List[Dict],
                              transformed_data: Dict[str, Dict],
                              logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Prepare the final analytics DataFrame by merging and processing data.
    
    Args:
        conversations: List of conversation dictionaries
        transformed_data: Dictionary of transformed response data
        logger: Logger instance for tracking execution
        
    Returns:
        Processed DataFrame ready for analysis
    """
    # Convert conversations list to DataFrame and set index
    conversations_df = pd.DataFrame(conversations)
    conversations_df.set_index('uuid', inplace=True)
    
    # Create DataFrame from transformed data
    transformed_data_df = pd.DataFrame.from_dict(transformed_data, orient='index')
    transformed_data_df.index.name = 'uuid'
    
    # Process answers
    answers_df = transformed_data_df.copy()
    categories = set()
    
    # Get all unique categories from answers
    for response in transformed_data.values():
        answers = response.get('answers', {})
        categories.update(answers.keys())
    
    # Extract answer categories into separate columns
    for category in categories:
        answers_df[category.lower()] = transformed_data_df['answers'].apply(
            lambda x: x.get(category, []) if isinstance(x, dict) else []
        )
    
    answers_df.drop('answers', axis=1, inplace=True)
    
    # Merge DataFrames
    analytics_df = conversations_df.join(answers_df, lsuffix='_conv', rsuffix='_trans')
    
    if logger:
        logger.info(f"Created merged DataFrame with shape: {analytics_df.shape}")
    
    return analytics_df

def prepare_data_for_analysis(data_folder: str = 'data') -> Tuple[pd.DataFrame, logging.Logger]:
    """
    Main function to prepare data for analysis.
    
    Args:
        data_folder: Path to folder containing data files
        
    Returns:
        Tuple containing prepared DataFrame and logger instance
    """
    # Setup logging
    logger = setup_logging()
    logger.info("Starting data preparation pipeline")
    
    # Load data files
    data = load_data_files(data_folder, logger)
    
    # Remove invalid responses
    clean_responses = remove_invalid_responses(
        data['human_responses'],
        data['user_timing'],
        logger
    )
    
    # Check conversation completeness
    transformed_data, incomplete = check_conversation_completeness(
        clean_responses,
        data['manipulation_definitions']
    )
    
    logger.info(f"Found {len(incomplete)} incomplete conversations")
    
    # Prepare final DataFrame
    analytics_df = prepare_analytics_dataframe(
        data['conversations'],
        transformed_data,
        logger
    )
    
    logger.info("Data preparation pipeline completed successfully")
    
    return analytics_df, logger

if __name__ == "__main__":
    analytics_df, logger = prepare_data_for_analysis()