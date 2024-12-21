from utils.load_data import process_conversation_data
from utils.stratified_splitter import perform_stratified_split
from utils.custom_dataloader import ManipulationDataset
from utils.train_classifier import setup_trainer, save_predictions
from torch.utils.data import DataLoader
import torch
from datetime import datetime
import logging
import os
import argparse
import wandb
from dotenv import load_dotenv

load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser(description='Train manipulation detection model')
    
    # Required arguments
    parser.add_argument('--base-path', type=str, required=True,
                        help='Base path for all saved files (e.g., /scratch_tmp/users/k23108295)')
    parser.add_argument('--model-name', type=str, required=True,
                        help='Name or path of the model to use (e.g., meta-llama/Llama-3.2-1B)')
    
    # Directory paths
    parser.add_argument('--data-dir', type=str, default="data",
                        help='Directory containing the training data')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Base directory for saving models and outputs')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Directory for saving logs')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Training batch size')
    parser.add_argument('--num-epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--test-size', type=float, default=0.25,
                        help='Proportion of data to use for testing')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # W&B configuration
    parser.add_argument('--wandb-project', type=str, default='manipulation-detection',
                        help='W&B project name')
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help='W&B entity/username')
    parser.add_argument('--disable-wandb', action='store_true',
                        help='Disable W&B logging')
    
    return parser.parse_args()


def setup_wandb(args, timestamp):
    if not args.disable_wandb:
        wandb_api_key = os.getenv('WANDB_API_KEY')
        
        if not wandb_api_key:
            raise ValueError("WANDB_API_KEY not found in .env file")
        
        wandb.login(key=wandb_api_key)
        print("Successfully logged into Weights & Biases!")

        run_name = f'training_run_{timestamp}'
        
        # Log key model and training info but avoid heavy parameter logging
        config = {
            'model_name': args.model_name,
            'batch_size': args.batch_size,
            'num_epochs': args.num_epochs,
            'test_size': args.test_size,
            'random_seed': args.random_seed,
            'gpu_available': torch.cuda.is_available(),
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'
        }
        
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=config
        )

def setup_logging(args, timestamp):
    # Create full path using base_path
    log_dir = os.path.join(args.base_path, args.log_dir)
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup main log file
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return log_file

def main():
    args = parse_args()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Setup logging
    log_file = setup_logging(args, timestamp)
    
    # Setup W&B
    setup_wandb(args, timestamp)
    
    # Set device
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Target columns definition
    target_columns = [
        'peer pressure_binary', 'reciprocity pressure_binary',
        'gaslighting_binary', 'guilt-tripping_binary',
        'emotional blackmail_binary', 'general_binary',
        'fear enhancement_binary', 'negging_binary'
    ]
    
    # Process data
    data_dir = args.data_dir
    df = process_conversation_data(
        data_dir=data_dir,
        log_file=log_file,
        log_level=logging.INFO
    )
    
    # Perform split
    X_train, X_test, y_train, y_test = perform_stratified_split(
        df,
        stratify_columns=['manipulation_type', 'persuasion_strength'],
        target_columns=target_columns,
        test_size=args.test_size,
        random_state=args.random_seed,
        plot=False
    )
    
    # Create datasets
    text_column = "chat_completion"
    train_dataset = ManipulationDataset(
        X=X_train,
        y=y_train,
        text_column=text_column,
        model=args.model_name,
        target_columns=target_columns
    )
    
    test_dataset = ManipulationDataset(
        X=X_test,
        y=y_test,
        text_column=text_column,
        model=args.model_name,
        target_columns=target_columns
    )
    
    # Setup output directory
    model_output_dir = os.path.join(args.base_path, args.output_dir, f'manipulation_dataset_{timestamp}')
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Initialize trainer
    trainer = setup_trainer(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        model_name=args.model_name,
        output_dir=model_output_dir,
        num_labels=len(target_columns),
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        device=device
    )
    
    # Train model
    logging.info("Starting training...")
    trainer.model.gradient_checkpointing_enable()
    trainer.train()
    
    # Save best model
    best_model_path = os.path.join(model_output_dir, 'best-model')
    trainer.save_model(best_model_path)
    logging.info(f"Best model saved to {best_model_path}")
    
    # Evaluate and save predictions
    test_results = trainer.evaluate(test_dataset)
    logging.info(f"Test results: {test_results}")
    
    predictions, results_df = save_predictions(
        trainer=trainer,
        test_dataset=test_dataset,
        output_dir=model_output_dir,
        target_columns=target_columns
    )
    logging.info(f"Predictions saved to {model_output_dir}/predictions.csv")
    
    if not args.disable_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()