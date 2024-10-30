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


answers_df = transformed_data_df.copy()

# Get all unique categories across all rows
categories = set()
for answers_dict in transformed_data_df['answers']:
    categories.update(answers_dict.keys())

# Create a column for each category
for category in categories:
    # Extract the list for this category from each row
    answers_df[category.lower()] = transformed_data_df['answers'].apply(
        lambda x: x.get(category, [])
    )

# Drop the original answers column
answers_df.drop('answers', axis=1, inplace=True)
    

# Join the DataFrames on 'uuid'
analytics_df = conversations_df.join(answers_df, lsuffix='_conv', rsuffix='_trans')

# Log the result of the merge
logger.info("DataFrames merged successfully")
logger.info(f"Merged DataFrame shape: {analytics_df.shape}")
logger.info(f"The columns are: {analytics_df.columns}")
logger.info("---------------------")


pd.set_option('display.max_columns', None)
print(analytics_df.head())


# ----------------------------
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
import scipy.stats as stats

# First, let's create helper functions to calculate variance of manipulation scores
def calculate_variance(row):
    scores = []
    for col in ['peer pressure', 'reciprocity pressure', 'gaslighting', 
                'guilt-tripping', 'emotional blackmail', 'general', 'fear enhancement', 
                'negging']:
        if isinstance(row[col], list) and row[col] != []:
            scores.extend(row[col])
    return np.var(scores) if scores else np.nan

def flatten_list(lst):
    return [item for sublist in lst if isinstance(sublist, list) for item in sublist]

def calculate_mean_score(row):
    scores = []
    for col in ['peer pressure', 'reciprocity pressure', 'gaslighting', 
                'guilt-tripping', 'emotional blackmail', 'general', 'fear enhancement', 
                'negging']:
        if isinstance(row[col], list) and row[col] != []:
            scores.extend(row[col])
    return np.mean(scores) if scores else np.nan

# Calculate variance and mean scores
analytics_df['variance'] = analytics_df.apply(calculate_variance, axis=1)
analytics_df['mean_score'] = analytics_df.apply(calculate_mean_score, axis=1)

# 1. Manipulation Score Analysis
print("\n=== Manipulation Score Analysis ===")
# Binary classification: mean_score > 4 is considered manipulative
analytics_df['is_manipulative_score'] = analytics_df['mean_score'] > 4
analytics_df['is_manipulative_prompt'] = analytics_df['prompt_type'] == 'manipulation'

# Calculate accuracy and recall
mask = analytics_df['mean_score'].notna()
accuracy = accuracy_score(analytics_df[mask]['is_manipulative_prompt'], 
                         analytics_df[mask]['is_manipulative_score'])
recall = recall_score(analytics_df[mask]['is_manipulative_prompt'], 
                     analytics_df[mask]['is_manipulative_score'])

print(f"Accuracy: {accuracy:.2f}")
print(f"Recall: {recall:.2f}")

# Variance analysis
print("\n=== Variance Analysis ===")
print(f"Mean variance: {analytics_df['variance'].mean():.2f}")
print(f"Median variance: {analytics_df['variance'].median():.2f}")
print(f"Variance of variance: {analytics_df['variance'].var():.2f}")

# Plot distribution of variance
plt.figure(figsize=(10, 6))
sns.histplot(analytics_df['variance'].dropna(), bins=20)
plt.title('Distribution of Response Variance')
plt.xlabel('Variance')
plt.ylabel('Count')
plt.savefig('variance_distribution.png')
plt.close()

# 2. Correlation Analysis
# Create correlation matrix for numerical columns
manipulation_cols = ['peer pressure', 'reciprocity pressure', 'gaslighting', 
                'guilt-tripping', 'emotional blackmail', 'general', 'fear enhancement', 
                'negging']

# Convert lists to mean values for correlation
correlation_df = pd.DataFrame()
for col in manipulation_cols:
    correlation_df[col] = analytics_df[col].apply(
        lambda x: np.mean(x) if isinstance(x, list) and x else np.nan
    )

# Calculate correlations using different methods
correlation_methods = ['pearson', 'spearman', 'kendall']
correlations = {}
for method in correlation_methods:
    correlations[method] = correlation_df.corr(method=method)

# 3. Prompt Type Analysis
print("\n=== Prompt Type Analysis ===")
prompt_analysis = analytics_df.groupby(['prompt_type', 'model']).agg({
    'is_manipulative_score': ['count', 'mean'],
    'persuasion_strength': lambda x: x.value_counts().to_dict()
}).round(2)

print("\nPrompt Type Analysis by Model:")
print(prompt_analysis)

# Find examples of incorrect classifications
incorrect_classifications = analytics_df[
    (analytics_df['is_manipulative_prompt'] != analytics_df['is_manipulative_score']) & 
    mask
]

print("\n=== Examples of Incorrect Classifications ===")
for _, row in incorrect_classifications.head(3).iterrows():
    print(f"\nContext: {row['context']}")
    print(f"Prompt Type: {row['prompt_type']}")
    print(f"Mean Score: {row['mean_score']:.2f}")
    print(f"Variance: {row['variance']:.2f}")

# Examples of high and low variance
print("\n=== Examples of High and Low Variance ===")
high_variance = analytics_df.nlargest(3, 'variance')
low_variance = analytics_df.nsmallest(3, 'variance')

print("\nHigh Variance Examples:")
for _, row in high_variance.iterrows():
    print(f"\nContext: {row['context']}")
    print(f"Variance: {row['variance']:.2f}")
    print(f"Mean Score: {row['mean_score']:.2f}")

print("\nLow Variance Examples:")
for _, row in low_variance.iterrows():
    print(f"\nContext: {row['context']}")
    print(f"Variance: {row['variance']:.2f}")
    print(f"Mean Score: {row['mean_score']:.2f}")

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