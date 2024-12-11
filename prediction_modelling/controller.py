from utils.load_data import process_conversation_data 
from utils.stratified_splitter import perform_stratified_split
from utils.custom_dataloader import ManipulationDataset

from datetime import datetime
import logging
from torch.utils.data import DataLoader

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = f'logs/processing_{timestamp}.log'

df = process_conversation_data(
    data_dir='data',
    log_file=log_file,
    log_level=logging.INFO
)

target_columns = ['peer pressure_binary', 'reciprocity pressure_binary',
                 'gaslighting_binary', 'guilt-tripping_binary',
                 'emotional blackmail_binary', 'general_binary',
                 'fear enhancement_binary', 'negging_binary']

# Perform split with multiple targets
X_train, X_test, y_train, y_test = perform_stratified_split(
    df,
    stratify_columns=['manipulation_type', 'persuasion_strength'],
    target_columns=target_columns,
    test_size=0.25,
    random_state=42,
    plot=False
)


# Create datasets using the ManipulationDataset class
model_name = "google/long-t5-local-base" 
text_column = "chat_completion" 

train_dataset = ManipulationDataset(
    X=X_train,
    y=y_train,
    text_column=text_column,
    model=model_name,
    target_columns=target_columns
)

test_dataset = ManipulationDataset(
    X=X_test,
    y=y_test,
    text_column=text_column,
    model=model_name,
    target_columns=target_columns
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


# Test the data loading
batch = next(iter(train_loader))
print("Batch keys:", batch.keys())
print("Input shape:", batch['input_ids'].shape)
print("Label shape:", batch['labels'].shape)
print("\nSample input ids:", batch['input_ids'][0][:10])  # First 10 tokens of first item
print("Sample label:", batch['labels'][0])

# Check if labels match expected format
print("\nLabel values distribution:")
for i, col in enumerate(target_columns):
    print(f"{col}: {batch['labels'][:, i].sum().item()}/{len(batch['labels'])}")

# Verify attention mask
print("\nAttention mask sample:", batch['attention_mask'][0][:10])
print("Max sequence length:", batch['input_ids'].size(1))

