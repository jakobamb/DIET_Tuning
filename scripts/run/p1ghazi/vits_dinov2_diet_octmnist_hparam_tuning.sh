cd /home/ja-am/projects/DIET_Tuning

# Activate conda environment
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=0

# Run both experiments in parallel
echo "Starting experiment 1: Low lr (1e-5) on GPU 0"

python main.py \
    --backbone dinov2 \
    --model-size small \
    --label-smoothing 0.2 \
    --dataset octmnist \
    --lr 1e-5 \
    --num-epochs 500\
    --data-root /home/ja-am/data \
    --wandb-dir /home/ja-am/projects/DIET_Tuning/wandb \
    --checkpoint-dir /home/ja-am/projects/DIET_Tuning/checkpoints &

EXP1_PID=$!
echo "Experiment 1 (Low lr) started with PID: $EXP1_PID"

export CUDA_VISIBLE_DEVICES=3

echo "Starting experiment 2: High batch size (512) on GPU 1"

python main.py \
    --backbone dinov2 \
    --model-size small \
    --label-smoothing 0.2 \
    --dataset octmnist \
    --batch-size 512 \
    --num-epochs 500 \
    --data-root /home/ja-am/data \
    --wandb-dir /home/ja-am/projects/DIET_Tuning/wandb \
    --checkpoint-dir /home/ja-am/projects/DIET_Tuning/checkpoints &

EXP2_PID=$!
echo "Experiment 2 (High batch size) started with PID: $EXP2_PID"

echo "Both experiments are running in parallel..."
echo "Experiment 1 PID: $EXP1_PID"
echo "Experiment 2 PID: $EXP2_PID"

# Wait for both experiments to complete
echo "Waiting for both experiments to complete..."
wait $EXP1_PID
echo "Experiment 1 (Low lr) completed"

wait $EXP2_PID
echo "Experiment 2 (High batch size) completed"

echo "All experiments completed!"


