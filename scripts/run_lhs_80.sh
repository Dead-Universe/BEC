#!/bin/bash
set -e

export BUILDINGS_BENCH=/home/hadoop/bec/buildings-bench/v2.0.0/BuildingsBench
NUM_GPUS=1

# ===== 固定路径（按你的环境） =====
SELECTED="/home/hadoop/bec/selected_buildings.txt"
PLAN="/mnt/data/clrs_sensitivity_plan_16runs.csv"

# ===== 训练配置 =====
MODEL="BuildMoE-top-k-2-without-shared-export"
CTX=168
PRED=168
EPOCHS=12
BS=32
LR=3e-5
WD=0.01

# ===== 输出路径（你可以改到任意位置） =====
RESULTS_DIR="/home/hadoop/bec/results_lhs"
CKPT_DIR="/home/hadoop/bec/checkpoints_lhs"

mkdir -p "$RESULTS_DIR" "$CKPT_DIR"

# ===== 读取选中的 5 栋楼 =====
if [[ ! -f "$SELECTED" ]]; then
  echo "找不到 selected_buildings.txt: $SELECTED"
  exit 1
fi

echo "==> 读取楼栋列表: $SELECTED"
mapfile -t LINES < "$SELECTED"

# 跳过空行和注释
FILTERED_LINES=()
for LINE in "${LINES[@]}"; do
  # 去掉首尾空白
  LINE_TRIM=$(echo "$LINE" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
  [[ -z "$LINE_TRIM" ]] && continue
  [[ "${LINE_TRIM:0:1}" == "#" ]] && continue
  FILTERED_LINES+=("$LINE_TRIM")
done

echo "==> 最终用于实验的楼："
printf ' - %s\n' "${FILTERED_LINES[@]}"

# ===== 每栋楼跑 16 组 LHS 参数 =====
for LINE in "${FILTERED_LINES[@]}"; do
  TYPE=$(echo "$LINE" | cut -d',' -f1)
  BID=$(echo "$LINE" | cut -d',' -f2)

  if [[ "$TYPE" == "university" ]]; then
    DATASET="university"
  else
    DATASET="cofactor"
  fi

  echo ""
  echo "==============================="
  echo " 微调楼栋：$TYPE , $BID"
  echo " 数据集  ：$DATASET"
  echo "==============================="

  # 16 组 LHS；若你计划里不是 16，请把 16 改成对应行数
  for ((i=0; i<16; i++)); do
    echo "---- LHS 组：$i ----"

    CUDA_VISIBLE_DEVICES=0 \
    python scripts/lhs.py \
      --dataset "$DATASET" \
      --single_building "$BID" \
      --model "$MODEL" \
      --ctx_len $CTX \
      --pred_len $PRED \
      --epochs $EPOCHS \
      --patience 0 \
      --batch_size $BS \
      --lr $LR \
      --weight_decay $WD \
      --clrs_plan_csv "$PLAN" \
      --lhs_run $i \
      --eval_horizons 168 \
      --results_path "$RESULTS_DIR" \
      --ckpt_dir "$CKPT_DIR" \
      --experiment_tag "LHS" \
      --pretrained_path /home/hadoop/bec/BuildingsBench/checkpoints/BuildMoE-top-k-2-without-shared-export_best_val.pt \
      --variant_name "${TYPE}_${BID}_run${i}"
  done
done

echo ""
echo "✅ 全部实验完成：结果在 $RESULTS_DIR ，权重在 $CKPT_DIR"
