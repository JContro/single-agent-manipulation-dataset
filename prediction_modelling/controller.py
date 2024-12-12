from utils.load_data import process_conversation_data
from utils.stratified_splitter import perform_stratified_split
from utils.custom_dataloader import ManipulationDataset
from utils.train_classifier import setup_trainer, save_predictions
from torch.utils.data import DataLoader
import t5_encoder

from datetime import datetime
import logging
import os

def main():
    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/processing_{timestamp}.log'
    
    # Process data
    df = process_conversation_data(
        data_dir='data',
        log_file=log_file,
        log_level=logging.INFO
    )

    # Define target columns
    target_columns = [
        'peer pressure_binary', 'reciprocity pressure_binary',
        'gaslighting_binary', 'guilt-tripping_binary',
        'emotional blackmail_binary', 'general_binary',
        'fear enhancement_binary', 'negging_binary'
    ]

    # Perform split
    X_train, X_test, y_train, y_test = perform_stratified_split(
        df,
        stratify_columns=['manipulation_type', 'persuasion_strength'],
        target_columns=target_columns,
        test_size=0.25,
        random_state=42,
        plot=False
    )

    # Create datasets
    model_name = "google/t5-v1_1-base"
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

    # Setup output directory
    output_dir = f"outputs/manipulation_classifier_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize trainer
    trainer = setup_trainer(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        model_name=model_name,
        output_dir=output_dir,
        num_labels=len(target_columns),
        batch_size=8,
        num_epochs=10,
        device="cuda"  
    )

    # Train model
    print("Starting training...")
    trainer.train()
    trainer.model.gradient_checkpointing_enable()

    # Save best model
    best_model_path = os.path.join(output_dir, 'best-model')
    trainer.save_model(best_model_path)
    print(f"Best model saved to {best_model_path}")

    # Evaluate and save predictions
    test_results = trainer.evaluate(test_dataset)
    print(f"Test results: {test_results}")
    
    predictions, results_df = save_predictions(
        trainer=trainer,
        test_dataset=test_dataset,
        output_dir=output_dir,
        target_columns=target_columns
    )
    print(f"Predictions saved to {output_dir}/predictions.csv")

if __name__ == "__main__":
    main()