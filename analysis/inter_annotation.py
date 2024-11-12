import json
import os
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Tuple
import krippendorff

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


import numpy as np
from typing import List, Dict, Union, Tuple
import krippendorff
from sklearn.metrics import cohen_kappa_score
import pandas as pd



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

def create_krippendorff_matrix(data):
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame([{
        'email': d['email'],
        'conversation_id': d['conversation_id'],
        'score': d['scores']['General']
    } for d in data])
    
    # Create matrix where rows=conversations, columns=annotators
    matrix = pd.pivot(
        df,
        index='conversation_id',
        columns='email',
        values='score'
    ).to_numpy()
    
    return df, matrix
df, matrix = create_krippendorff_matrix(clean_human_responses)

print(df.head())
print(matrix.shape)

print(len(set(df.email.values)))
print(len(set(df.conversation_id.values)))

print(matrix)

# Calculate alpha
# level_of_measurement options: 'nominal', 'ordinal', 'interval', 'ratio'
alpha = krippendorff.alpha(reliability_data=matrix, level_of_measurement='ordinal')

print(f"Krippendorff's alpha: {alpha:.3f}")

def transform_matrix_binary(matrix):
    # Create a copy to avoid modifying the original
    transformed = matrix.copy()
    
    # Apply conditions using numpy's where
    # First handle < 4
    transformed = np.where(transformed < 4, -1, transformed)
    # Then handle == 4
    transformed = np.where(transformed == 4, 0, transformed)
    # Finally handle > 4
    transformed = np.where(transformed > 4, 1, transformed)
    
    return transformed

binary_data = transform_matrix_binary(matrix)
binary_alpha = krippendorff.alpha(reliability_data=binary_data, level_of_measurement='ordinal')


print(f"Binary Krippendorff's alpha: {binary_alpha:.3f}")

