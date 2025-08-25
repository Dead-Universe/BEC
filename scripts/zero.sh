#!/usr/bin/env bash
set -euo pipefail               # 出错即停

export BUILDINGS_BENCH=/home/hadoop/bec/buildings-bench/v2.0.0/BuildingsBench
export HF_ENDPOINT=https://hf-mirror.com

# PYTHON="python -u"
# CONTEXT=168
# LOG_DIR=logs
# MODE=timemoe
# mkdir -p "$LOG_DIR"

# for H in 1 6 12 24 48 96 168; do
#   LOG_FILE="$LOG_DIR/${MODE}_${CONTEXT}_${H}.log"

#   echo ">>> context=$CONTEXT, horizon=$H  (日志：$LOG_FILE)"
#   # 用 |& 同时捕获 stdout 和 stderr；tee 负责边看边写
#   $PYTHON scripts/zero_shot.py \
#       --model $MODE \
#       --benchmark cofactor \
#       --forecast_horizon "$H" \
#       --variant_name "${CONTEXT}_${H}" \
#       --device cuda \
#       --batch_size 32 \
#       --oov_path /home/hadoop/bec/BuildingsBench/oov_test.txt \
#   |& tee "$LOG_FILE"
# done

PYTHON="python -u"
CONTEXT=168
LOG_DIR=logs
MODE=TimeMoE-L
mkdir -p "$LOG_DIR"

for H in 1 6 12 24 48 96 168; do
  LOG_FILE="$LOG_DIR/${MODE}_${CONTEXT}_${H}.log"

  echo ">>> context=$CONTEXT, horizon=$H  (日志：$LOG_FILE)"
  # 用 |& 同时捕获 stdout 和 stderr；tee 负责边看边写
  $PYTHON scripts/zero_shot.py \
      --model "$MODE" \
      --benchmark real \
      --forecast_horizon "$H" \
      --variant_name "${CONTEXT}_${H}" \
      --checkpoint checkpoints/TimeMoE-L_last.pt \
      --apply_scaler_transform boxcox \
      --device cuda \
      --batch_size 1024 \
      --oov_path /home/hadoop/bec/BuildingsBench/oov_test.txt \
      --stl_robust \
      --num_workers 32 \
  |& tee "$LOG_FILE"
done