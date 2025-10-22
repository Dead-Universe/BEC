#!/bin/bash

# THIS IS AN EXAMPLE SCRIPT. 
# PLEASE CONFIGURE FOR YOUR SETUP.

export BUILDINGS_BENCH=/home/hadoop/bec/buildings-bench/v2.0.0/BuildingsBench
export WANDB_ENTITY="jmdyz-"
export WANDB_PROJECT="train"
NUM_GPUS=1

# torchrun \
#     --nnodes=1 \
#     --nproc_per_node=$NUM_GPUS \
#     --rdzv-backend=c10d \
#     --rdzv-endpoint=localhost:0 \
#     scripts/pretrain.py \
#     --model TransformerWithGaussian-L \
#     --disable_slurm \
#     --num_workers 16 \
#     --resume_from_checkpoint TransformerWithGaussian-L_last.pt \
#     --wandb_run_id 2bt0e02i

torchrun \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    scripts/train_mixed.py \
    --model BuildMoE-top-k-2 \
    --disable_slurm \
    --num_workers 32 \

PYTHON="python -u"
CONTEXT=168
LOG_DIR=logs
MODE=BuildMoE-top-k-2
mkdir -p "$LOG_DIR"

for H in  1 6 12 24 48 96 168; do
  LOG_FILE="$LOG_DIR/${MODE}_${CONTEXT}_${H}.log"

  echo ">>> context=$CONTEXT, horizon=$H  (日志：$LOG_FILE)"
  # 用 |& 同时捕获 stdout 和 stderr；tee 负责边看边写
  $PYTHON scripts/zero_shot.py \
      --model "$MODE" \
      --benchmark real \
      --forecast_horizon "$H" \
      --variant_name "${CONTEXT}_${H}" \
      --checkpoint checkpoints/BuildMoE-top-k-2_best_val.pt \
      --apply_scaler_transform boxcox \
      --device cuda \
      --batch_size 1024 \
      --oov_path /home/hadoop/bec/BuildingsBench/oov_test.txt \
      --num_workers 32 \
  |& tee "$LOG_FILE"
done


