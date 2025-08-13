#!/bin/bash
### General options
### –- specify queue --
#BSUB -q p1
### -- set the job Name --
#BSUB -J DIET
### -- specify that the cores must be on the same host --
#BSUB -R "span[hosts=1]"
### -- ask for number of cores (default: 1) --
#BSUB -n 12
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 6:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=20GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u jaam@di.ku.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o logs/gpu_%J.out
# -- end of LSF options --

module load python3/3.11.9

# cd into project directory
cd /zhome/6a/3/198677/projects/DIET_Tuning

# Activate conda environment
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=0

# Run the training script
python main.py \
    --backbone ijepa \
    --model-size b16_22k \
    --training-mode diet_only \
    --label-smoothing 0.2 \
    --dataset dermamnist \
    --num-epochs 50 \
    --data-root /dtu/p1/jakambs/diet/data \
    --wandb-dir /dtu/p1/jakambs/diet/wandb \
    --checkpoint-dir /dtu/p1/jakambs/diet/checkpoints
