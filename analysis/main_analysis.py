import json
import os
import logging
from pathlib import Path
import pandas as pd
from typing import Dict, Union
from utils.filtering_testing import remove_bad_responses, check_conversation_completeness

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def handle_data_files(download_flag=False) -> Dict[str, Union[dict, pd.DataFrame]]:
    """
    Handle data files and convert to appropriate format.
    Returns manipulation_definitions as dict, others as pandas DataFrames.
    """
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
        logger.debug("Downloading files from GCS bucket")
        from data_connection import create_gcs_file_handler
        file_handler = create_gcs_file_handler('manipulation-dataset-kcl')
        
        for filename in files.values():
            data = file_handler(filename)
            with open(data_folder / filename, 'w') as f:
                json.dump(data, f)
            logger.debug(f"Downloaded and saved {filename}")

    # Load and convert data
    logger.info("Loading data files")
    data = {}
    for key, filename in files.items():
        try:
            data[key] = json.load(open(data_folder / filename))
            logger.debug(f"Loaded {filename}")
        except Exception as e:
            logger.error(f"Error loading {filename}: {str(e)}")
            raise
    
    return data

# Usage
(
    manipulation_definitions,
    conversations,
    human_responses,
    user_scores,
    user_timing
) = handle_data_files(download_flag=False).values()

data = handle_data_files(download_flag=False)
(
    manipulation_definitions,
    conversations,
    human_responses,
    user_scores,
    user_timing
) = data.values()

initial_responses = len(human_responses)
clean_human_responses = remove_bad_responses(human_responses=human_responses, user_timing=user_timing)
final_responses = len(clean_human_responses)

logger.info("---------------------")
logger.info(f"Responses processed:")
logger.info(f"Initial responses: {initial_responses}")
logger.info(f"Valid responses: {final_responses}")
logger.info(f"Removed responses: {initial_responses - final_responses}")
logger.info(f"Removal rate: {((initial_responses - final_responses) / initial_responses * 100):.2f}%")
logger.info("---------------------")
    
transformed_data, incomplete = check_conversation_completeness(human_responses=human_responses, required_manip_types=manipulation_definitions)

logger.info("Checking incompleteness of manipulation questions")
logger.info(f"Number of incomplete questions: {len(incomplete)}")
logger.info("---------------------")

# print(transformed_data)
# print(type(transformed_data))
conversations_df = pd.DataFrame(conversations) 
transformed_data_df = pd.DataFrame(transformed_data).T


# Set 'uuid' as the index for both DataFrames
conversations_df.set_index('uuid', inplace=True)
transformed_data_df.index.name = 'uuid'

# Join the DataFrames on 'uuid'
analytics_df = conversations_df.join(transformed_data_df, lsuffix='_conv', rsuffix='_trans')

# Log the result of the merge
logger.info("DataFrames merged successfully")
logger.info(f"Merged DataFrame shape: {analytics_df.shape}")
logger.info(f"The columns are: {analytics_df.columns}")
logger.info("---------------------")

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