from utils.load_data import process_conversation_data 
from utils.stratified_splitter import perform_stratified_split, plot_distributions

from datetime import datetime
import logging

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = f'logs/processing_{timestamp}.log'

df = process_conversation_data(
    data_dir='data',
    log_file=log_file,
    log_level=logging.INFO
)


# Perform split with multiple targets
X_train, X_test, y_train, y_test = perform_stratified_split(
    df,
    target_columns=['manipulation_type', 'persuasion_strength'],
    test_size=0.25,
    random_state=42,
    plot=False
)

# TODO: check y_train and y_test - these are not actually what my targets actually are
print(y_test.head())

# TODO: train_test split with balanced classes 
# TODO: create a script that turns the df into a torch datasset class that can be used for training 
