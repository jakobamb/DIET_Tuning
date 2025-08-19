#!/bin/bash
# Slurm script for DIET block training on Brown CS Hydra cluster
#
# This script trains DINOv2-small on OctMNIST with 2 transformer blocks
#
#SBATCH --partition=gpus
#SBATCH --exclude=gpu[1601-1605],gpu[1701-1708],gpu1801,gpu[1802,1905-1906]
#SBATCH --gres=gpu:1
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --mem=20G
#SBATCH --job-name=RC_2B_150E_OCT
#SBATCH --output=/home/jambsdor/projects/DIET_Tuning/logs/rc_2b_150e_oct_%j.out
#SBATCH --error=/home/jambsdor/projects/DIET_Tuning/logs/rc_2b_150e_oct_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jakob_ambsdorf@brown.edu


# Navigate to project directory
cd /home/jambsdor/projects/DIET_Tuning

# Activate virtual environment
source venv/bin/activate

PROJECT_DIR=/home/jambsdor/projects/DIET_Tuning

# Set environment variables
nvidia-smi
export WANDB_DATA_DIR=$PROJECT_DIR/wandb

# Run the training script
srun python -u main.py \
    --wandb-prefix "RC_2block_150e_da2" \
    --backbone dinov2 \
    --model-size small \
    --label-smoothing 0.2 \
    --da-strength 2 \
    --diet-head-only-epochs 0.05 \
    --dataset octmnist \
    --num-epochs 150 \
    --num-trained-blocks 2 \
    --data-root $PROJECT_DIR/data \
    --wandb-dir $PROJECT_DIR/wandb \
    --checkpoint-dir $PROJECT_DIR/checkpoints
