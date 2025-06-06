#!/bin/bash

# THIS IS AN EXAMPLE SCRIPT. 
# PLEASE CONFIGURE FOR YOUR SETUP.

export BUILDINGS_BENCH=/data/lx/buildings-bench/v2.0.0/BuildingsBench
export WANDB_ENTITY="jmdyz-"
export WANDB_PROJECT="train"
NUM_GPUS=4

torchrun \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    scripts/pretrain.py \
    --model TransformerWithGaussian-L \
    --disable_slurm \
    --num_workers 16 \
    --resume_from_checkpoint TransformerWithGaussian-L_last.pt \
    --wandb_run_id 2bt0e02i