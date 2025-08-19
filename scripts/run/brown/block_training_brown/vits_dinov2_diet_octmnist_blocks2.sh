#!/bin/bash
# Slurm script for DIET block training on Brown CS Hydra cluster
#
# This script trains DINOv2-small on OctMNIST with 2 transformer blocks
#
#SBATCH --partition=gpus
#SBATCH --exclude=gpu[1601-1605],gpu[1701-1708],gpu1801,gpu[1802,1905-1906]
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --mem=20G
#SBATCH --job-name=DIET_BLOCKS2_OCT
#SBATCH --output=/data/people/jambsdor/DIET_Tuning/logs/diet_blocks2_oct_%j.out
#SBATCH --error=/data/people/jambsdor/DIET_Tuning/logs/diet_blocks2_oct_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jakob_ambsdorf@brown.edu


# Navigate to project directory
cd /data/people/jambsdor/DIET_Tuning

# Activate virtual environment
source venv/bin/activate

PROJECT_DIR=/data/people/jambsdor/DIET_Tuning

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export WANDB_DATA_DIR=$PROJECT_DIR/wandb

# Run the training script
python main.py \
    --wandb-prefix "block_training_brown" \
    --backbone dinov2 \
    --model-size small \
    --label-smoothing 0.2 \
    --diet-head-only-epochs 0.05 \
    --dataset octmnist \
    --num-epochs 500 \
    --num-trained-blocks 2 \
    --data-root $PROJECT_DIR/data \
    --wandb-dir $PROJECT_DIR/wandb \
    --checkpoint-dir $PROJECT_DIR/checkpoints
