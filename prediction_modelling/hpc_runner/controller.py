from utils.load_data import process_conversation_data
from utils.stratified_splitter import perform_kfold_stratified_split
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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
from sklearn.model_selection import StratifiedKFold

load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser(description='Train manipulation detection model')
    
    # Required arguments
    parser.add_argument('--base-path', type=str, required=True,
                        help='Base path for all saved files (e.g., /scratch_tmp/users/k23108295)')
    parser.add_argument('--model-name', type=str, required=True,
                        help='Name or path of the model to use (e.g., meta-llama/Llama-3.2-1B)')
    parser.add_argument('--model-cache-dir', type=str, required=True,
                        help='Path of the directory where to save the models (e.g. /mnt/ssd/transformers-cache/)')

    # K-fold validation
    parser.add_argument('--plots-dir', type=str, default='plots',
                        help='Directory for saving plots')
    parser.add_argument('--n-folds', type=int, default=5,
                        help='Number of folds for cross-validation')
    parser.add_argument('--plot-figsize', nargs=2, type=int, default=[12, 6],
                        help='Figure size for plots (width height)')
    
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

def setup_wandb_for_fold(args, timestamp, fold_idx):
    """Setup W&B for a specific fold"""
    if not args.disable_wandb:
        if fold_idx == 0:  # Only login once
            wandb_api_key = os.getenv('WANDB_API_KEY')
            if not wandb_api_key:
                raise ValueError("WANDB_API_KEY not found in .env file")
            wandb.login(key=wandb_api_key)
            print("Successfully logged into Weights & Biases!")

        run_name = f'fold_{fold_idx}_training_run_{timestamp}'
        
        # Initialize group for this k-fold experiment
        group_name = f'kfold_experiment_{timestamp}'
        
        config = {
            'model_name': args.model_name,
            'batch_size': args.batch_size,
            'num_epochs': args.num_epochs,
            'n_folds': args.n_folds,
            'fold': fold_idx,
            'random_seed': args.random_seed,
            'gpu_available': torch.cuda.is_available(),
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'
        }
        
        # Initialize new run for this fold
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            group=group_name,  # Group all folds together
            job_type='fold_training',
            config=config,
            reinit=True  # Allow multiple runs in the same script
        )
        
        return run
    return None

def log_fold_metrics(trainer, fold_idx, metrics_history):
    """Log training metrics for a specific fold"""
    if not wandb.run:
        return
    
    # Log available training metrics
    for key, value in metrics_history.items():
        wandb.log({
            f'fold_{fold_idx}/{key}': value,
            'epoch': metrics_history.get('epoch', 0)
        })

    try:
        # Log final evaluation metrics
        final_metrics = trainer.evaluate()
        wandb.log({
            f'fold_{fold_idx}/final_metrics': final_metrics,
            'fold': fold_idx
        })
        return final_metrics
    except Exception as e:
        logging.error(f"Error evaluating model for fold {fold_idx}: {e}")
        return metrics_history  # Return training metrics if evaluation fails

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

def aggregate_cv_results(fold_results):
    """Aggregate metrics across all folds."""
    all_metrics = {}
    for metric in fold_results[0].keys():
        values = [fold[metric] for fold in fold_results]
        all_metrics[f'mean_{metric}'] = np.mean(values)
        all_metrics[f'std_{metric}'] = np.std(values)
    
    return all_metrics

def setup_plot_directories(base_path, plots_dir, timestamp):
    """Create directories for saving plots."""
    # Create main plots directory
    plots_base = os.path.join(base_path, plots_dir, f'run_{timestamp}')
    plots_kfold = os.path.join(plots_base, 'kfold_distributions')
    plots_folds = os.path.join(plots_base, 'fold_specific')
    
    # Create all directories
    os.makedirs(plots_kfold, exist_ok=True)
    os.makedirs(plots_folds, exist_ok=True)
    
    return plots_base, plots_kfold, plots_folds

def plot_and_save_fold_metrics(metrics_history, output_dir, fold_idx):
    """Plot and save training metrics for a specific fold."""
    if not isinstance(metrics_history, dict):
        logging.error(f"Metrics history for fold {fold_idx} is not a dictionary")
        return
        
    plt.figure(figsize=(12, 6))
    
    try:
        # Plot training metrics
        numerical_metrics = {k: v for k, v in metrics_history.items() 
                           if isinstance(v, (int, float)) and k not in ['epoch']}
        
        for i, (metric_name, metric_value) in enumerate(numerical_metrics.items()):
            plt.plot([metric_value], label=metric_name, marker='o')
        
        plt.title(f'Fold {fold_idx} - Training Metrics')
        plt.xlabel('Current Step')
        plt.ylabel('Value')
        plt.legend()
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'fold_{fold_idx}_metrics.png')
        plt.savefig(output_path)
        plt.close()
        
        # Log plot to W&B
        if wandb.run:
            wandb.log({
                f"fold_{fold_idx}_metrics_plot": wandb.Image(output_path),
                f"fold_{fold_idx}_metrics": numerical_metrics
            })
        
    except Exception as e:
        logging.error(f"Failed to save metrics plot for fold {fold_idx}: {e}")
        plt.close()

def log_final_cv_results(cv_results, plots_base):
    """Log final cross-validation results to W&B"""
    if not wandb.run:
        return
        
    # Log mean and std metrics
    wandb.log({"final_cv_results": cv_results})
    
    # Create and log CV results plot
    plt.figure(figsize=(12, 6))
    metrics = list(cv_results.keys())
    means = [cv_results[m] for m in metrics if m.startswith('mean_')]
    stds = [cv_results[m] for m in metrics if m.startswith('std_')]
    
    try:
        plt.errorbar(
            range(len(means)), 
            means, 
            yerr=stds, 
            fmt='o', 
            capsize=5
        )
        
        plt.xticks(
            range(len(means)), 
            [m.replace('mean_', '') for m in metrics if m.startswith('mean_')], 
            rotation=45
        )
        plt.title('Cross-Validation Results with Standard Deviation')
        plt.tight_layout()
        
        cv_plot_path = os.path.join(plots_base, 'cv_results.png')
        plt.savefig(cv_plot_path)
        wandb.log({"cv_results_plot": wandb.Image(cv_plot_path)})
        plt.close()
        
    except Exception as e:
        logging.error(f"Failed to save CV results plot: {e}")
        plt.close()

def main():
    args = parse_args()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Setup logging
    log_file = setup_logging(args, timestamp)
    
    # Setup plot directories
    plots_base, plots_kfold, plots_folds = setup_plot_directories(
        args.base_path, args.plots_dir, timestamp
    )
    
    # Set device
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
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
    
    # Perform k-fold split with plot saving
    fold_splits = perform_kfold_stratified_split(
        df,
        stratify_columns=['manipulation_type', 'persuasion_strength'],
        target_columns=target_columns,
        n_splits=args.n_folds,
        random_state=args.random_seed,
        plot=True,
        plot_output_dir=plots_kfold
    )
    
    # Store results for each fold
    fold_results = []
    all_predictions = []
    
    # Train and evaluate on each fold
    for fold_idx, split in fold_splits.items():
        logging.info(f"\nTraining Fold {fold_idx + 1}/{args.n_folds}")
        
        # Setup W&B for this fold
        run = setup_wandb_for_fold(args, timestamp, fold_idx)
        
        # Create fold-specific plot directory
        fold_plot_dir = os.path.join(plots_folds, f'fold_{fold_idx}')
        os.makedirs(fold_plot_dir, exist_ok=True)
        
        X_train, X_test = split['X_train'], split['X_test']
        y_train, y_test = split['y_train'], split['y_test']
        
        # Create datasets for this fold
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
        
        # Setup output directory for this fold
        fold_output_dir = os.path.join(
            args.base_path, 
            args.output_dir, 
            f'manipulation_dataset_{timestamp}',
            f'fold_{fold_idx}'
        )
        os.makedirs(fold_output_dir, exist_ok=True)
        
        # Setup trainer for this fold
        trainer = setup_trainer(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            model_name=args.model_name,
            output_dir=fold_output_dir,
            num_labels=len(target_columns),
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            device=device,
            model_cache_dir=args.model_cache_dir,
        )
        
        # Train model for this fold
        trainer.model.gradient_checkpointing_enable()
        metrics_history = trainer.train()
        
        # Plot and save fold metrics
        plot_and_save_fold_metrics(
            metrics_history=metrics_history.metrics,
            output_dir=fold_plot_dir,
            fold_idx=fold_idx
        )
        
        # Save fold's model
        trainer.save_model(os.path.join(fold_output_dir, 'model'))
        
        # Log metrics for this fold and get final results
        test_results = log_fold_metrics(trainer, fold_idx, metrics_history.metrics)
        fold_results.append(test_results)
        
        # Get and save fold's predictions
        predictions, results_df = save_predictions(
            trainer=trainer,
            test_dataset=test_dataset,
            output_dir=fold_output_dir,
            target_columns=target_columns
        )
        
        # Add fold information to results
        results_df['fold'] = fold_idx
        all_predictions.append(results_df)
        
        # Log artifacts to W&B
        if wandb.run:
            model_artifact = wandb.Artifact(
                f"model-fold-{fold_idx}", 
                type="model",
                description=f"Model checkpoint for fold {fold_idx}"
            )
            model_artifact.add_dir(os.path.join(fold_output_dir, 'model'))
            wandb.log_artifact(model_artifact)
            
            # Log predictions
            predictions_table = wandb.Table(dataframe=results_df)
            wandb.log({f"fold_{fold_idx}_predictions": predictions_table})
        
        if run:
            run.finish()  # Close this fold's run
    
    # Aggregate results across folds
    cv_results = aggregate_cv_results(fold_results)
    logging.info("\nCross-validation results:")
    for metric, value in cv_results.items():
        logging.info(f"{metric}: {value:.4f}")
    
    # Save combined predictions
    all_predictions_df = pd.concat(all_predictions, ignore_index=True)
    combined_output_dir = os.path.join(
        args.base_path, 
        args.output_dir, 
        f'manipulation_dataset_{timestamp}'
    )
    all_predictions_df.to_csv(
        os.path.join(combined_output_dir, 'all_fold_predictions.csv'),
        index=False
    )
    
    # Start a final W&B run for aggregated results if W&B is enabled
    if not args.disable_wandb:
        final_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f'final_cv_results_{timestamp}',
            group=f'kfold_experiment_{timestamp}',
            job_type='cv_summary',
            config=vars(args),
            reinit=True
        )
        
        # Log final CV results and plots
        log_final_cv_results(cv_results, plots_base)
        
        # Log combined predictions
        combined_predictions_table = wandb.Table(dataframe=all_predictions_df)
        wandb.log({"all_folds_predictions": combined_predictions_table})
        
        final_run.finish()

if __name__ == "__main__":
    main()