#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --mem=240G
#SBATCH --time=3:00:00
#SBATCH --output=/scratch_tmp/users/k23108295/experiments/logs/slurm-%j.out
#SBATCH --error=/scratch_tmp/users/k23108295/experiments/logs/slurm-%j.err

# Load required modules
module load cudnn/8.7.0.84-11.8-gcc-13.2.0
module load cuda/11.8.0-gcc-13.2.0

# Activate virtual environment
source /scratch_tmp/users/k23108295/pytorch-venv/bin/activate

# Print Python path for verification
which python3

# Run the main script
python3 controller.py --base-path /scratch_tmp/users/k23108295/experiments/llama32-3B --model-name meta-llama/Llama-3.2-3B