#!/bin/bash
set -e

export BUILDINGS_BENCH=/home/hadoop/bec/buildings-bench/v2.0.0/BuildingsBench
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# python scripts/finetune_university_cofactor.py \
#   --dataset cofactor:Office \
#   --model BuildMoE-top-k-2 \
#   --pretrained_path /home/hadoop/bec/BuildingsBench/checkpoints/BuildMoE-top-k-2_best_val.pt \
#   --ctx_len 168 \
#   --pred_len 24 \
#   --batch_size 32 \
#   --epochs 10 \
#   --lr 3e-5 \
#   --weight_decay 1e-2 \
#   --apply_scaler_transform boxcox \
#   --results_path results_finetune \

# python scripts/finetune_university_cofactor.py \
#   --dataset cofactor:Office \
#   --model DLinearRegression \
#   --ctx_len 168 \
#   --pred_len 24 \
#   --batch_size 64 \
#   --epochs 36 \
#   --lr 1e-3 \
#   --apply_scaler_transform boxcox \
#   --results_path results_finetune \

# python scripts/finetune_university_cofactor.py \
#   --dataset university \
#   --model NLinearRegression \
#   --ctx_len 168 \
#   --pred_len 24 \
#   --batch_size 64 \
#   --epochs 200 \
#   --lr 1e-3 \
#   --apply_scaler_transform boxcox \
#   --results_path results_finetune \

# python scripts/finetune_university_cofactor.py \
#   --dataset university \
#   --model LinearRegression \
#   --ctx_len 168 \
#   --pred_len 24 \
#   --batch_size 64 \
#   --epochs 200 \
#   --lr 1e-3 \
#   --apply_scaler_transform boxcox \
#   --results_path results_finetune \

# python scripts/finetune_university_cofactor.py \
#   --dataset cofactor \
#   --model AutoformerBB \
#   --ctx_len 168 \
#   --pred_len 168 \
#   --batch_size 64 \
#   --epochs 200 \
#   --lr 1e-4 \
#   --apply_scaler_transform boxcox \
#   --results_path results_finetune \

# python scripts/finetune_university_cofactor.py \
#   --dataset university \
#   --model PatchTSTBB \
#   --ctx_len 168 \
#   --pred_len 24 \
#   --batch_size 128 \
#   --epochs 200 \
#   --lr 1e-4 \
#   --apply_scaler_transform boxcox \
#   --results_path results_finetune \

# 日季节（24）
# python scripts/finetune_university_cofactor.py \
#   --dataset university \
#   --model SeasonalNaive24 \
#   --ctx_len 168 \
#   --pred_len 24 \
#   --batch_size 64 \
#   --epochs 1 \
#   --lr 1e-4 \
#   --apply_scaler_transform boxcox \
#   --results_path results_finetune \

# # 周季节（168）
# python scripts/finetune_university_cofactor.py \
#   --dataset university \
#   --model SeasonalNaive168 \
#   --ctx_len 168 \
#   --pred_len 24 \
#   --batch_size 64 \
#   --epochs 1 \
#   --lr 1e-4 \
#   --apply_scaler_transform boxcox \
#   --results_path results_finetune \

# python scripts/finetune_university_cofactor.py \
#   --dataset university \
#   --model NHITSBB \
#   --batch_size 128 \
#   --lr 1e-3 \
#   --weight_decay 1e-4 \
#   --epochs 200 \
#   --apply_scaler_transform boxcox \
#   --results_path results_finetune \

# python scripts/finetune_university_cofactor.py \
#   --dataset cofactor:Office \
#   --model TimeMixerBB \
#   --ctx_len 168 \
#   --pred_len 24 \
#   --batch_size 32 \
#   --epochs 1000 \
#   --lr 1e-3 \
#   --patience 5 \
#   --apply_scaler_transform boxcox \
#   --results_path results_finetune \

PYTHON="python -u"
CONTEXT=168
LOG_DIR="logs_all"
RESULTS_DIR="results_finetune"

mkdir -p "$LOG_DIR"
mkdir -p "$RESULTS_DIR"

# 数据集
DATASETS=(
  "cofactor:Kindergarten"
  "cofactor:School"
  "cofactor:NursingHome"
  "cofactor:Office"
  "university"
)

# 模型列表
MODELS=(
  "TimeMixerBB"
  "BuildMoE-top-k-2"
  "DLinearRegression"
  "NLinearRegression"
  "LinearRegression"
  "AutoformerBB"
  "PatchTSTBB"
  "SeasonalNaive24"
  "SeasonalNaive168"
  "NHITSBB"
)

# 预测长度
HORIZONS=(1 6 12 24 48 96 168)

# 设置并发上限（根据显存大小调整，比如 2 或 3）
MAX_JOBS=10

# for DATASET in "${DATASETS[@]}"; do
#   for H in "${HORIZONS[@]}"; do
#     LOG_FILE="$LOG_DIR/${DATASET}_ctx${CONTEXT}_pred${H}.log"
#     echo "=== Running dataset=$DATASET, context=$CONTEXT, horizon=$H ==="
#     echo "日志保存: $LOG_FILE"

#     JOBS=0
#     for MODEL in "${MODELS[@]}"; do
#       echo ">>> 模型: $MODEL, 数据集=$DATASET, horizon=$H"

#       # 默认超参
#       EPOCHS=200
#       LR=1e-3
#       PRETRAIN=""
#       WD=""

#       # 特殊情况
#       if [[ "$MODEL" == "BuildMoE-top-k-2" ]]; then
#         EPOCHS=1
#         LR=3e-5
#         PRETRAIN="--pretrained_path /home/hadoop/bec/BuildingsBench/checkpoints/BuildMoE-top-k-2_best_val.pt"
#       elif [[ "$MODEL" == "AutoformerBB" || "$MODEL" == "PatchTSTBB" ]]; then
#         LR=1e-4
#       elif [[ "$MODEL" == "SeasonalNaive24" || "$MODEL" == "SeasonalNaive168" ]]; then
#         EPOCHS=1
#         LR=1e-4
#       fi

#       # 并行执行：后台任务
#       (
#         $PYTHON scripts/finetune_university_cofactor.py \
#           --dataset "$DATASET" \
#           --model "$MODEL" \
#           --ctx_len "$CONTEXT" \
#           --pred_len "$H" \
#           --batch_size 32 \
#           --epochs "$EPOCHS" \
#           --lr "$LR" \
#           --apply_scaler_transform boxcox \
#           --results_path "$RESULTS_DIR" \
#           --variant_name "${CONTEXT}_${H}" \
#           --patience 5 \
#           $PRETRAIN $WD \
#         |& tee -a "$LOG_FILE"
#       ) &

#       JOBS=$((JOBS+1))
#       if [[ $JOBS -ge $MAX_JOBS ]]; then
#         wait
#         JOBS=0
#       fi
#     done

#     # 等待这一轮 horizon 的所有模型跑完
#     wait
#   done
# done

PYTHON="python -u"
CONTEXT=168
LOG_DIR="logs_all"
RESULTS_DIR="results_finetune"
LOG_FILE="$LOG_DIR/BuildMoE-top-k-2-without-shared-export_finetune_ctx${CONTEXT}.log"

mkdir -p "$LOG_DIR" "$RESULTS_DIR"

# 数据集
DATASETS=(
  "cofactor:Kindergarten"
  "cofactor:School"
  "cofactor:NursingHome"
  "cofactor:Office"
  "university"
)

# 预测长度
HORIZONS=(1 6 12 24 48 96 168)
# HORIZONS=(24)


# 并发上限（按显存调整）
MAX_JOBS=2

# echo "========== $(date) START BuildMoE-top-k-2-without-shared-export FINETUNE ==========" |& tee -a "$LOG_FILE"

# JOBS=0
# for DATASET in "${DATASETS[@]}"; do
#   for H in "${HORIZONS[@]}"; do
#     echo "--- dataset=${DATASET}, ctx=${CONTEXT}, horizon=${H} ---" |& tee -a "$LOG_FILE"
#     (
#       $PYTHON scripts/finetune_university_cofactor.py \
#         --dataset "$DATASET" \
#         --model "BuildMoE-top-k-2-without-shared-export" \
#         --pretrained_path "/home/hadoop/bec/BuildingsBench/checkpoints/BuildMoE-top-k-2-without-shared-export_best_val.pt" \
#         --ctx_len "$CONTEXT" \
#         --pred_len "$H" \
#         --batch_size 32 \
#         --epochs 10 \
#         --lr 3e-5 \
#         --apply_scaler_transform boxcox \
#         --results_path "$RESULTS_DIR" \
#         --variant_name "${CONTEXT}_${H}" \
#         --patience 5 \
#       |& tee -a "$LOG_FILE"
#     ) &

#     JOBS=$((JOBS+1))
#     if [[ $JOBS -ge $MAX_JOBS ]]; then
#       wait
#       JOBS=0
#     fi
#   done
# done

# wait
# echo "========== $(date) ALL DONE ==========" |& tee -a "$LOG_FILE"


# 数据集
DATASETS=(
  # "cofactor:Kindergarten"
  # "cofactor:School"
  "cofactor:NursingHome"
  "cofactor:Office"
  "university"
  # "university30t"
  # "university15t"
  # "university1t"
)

# 模型列表
MODELS=(
  # "TimeMixerBB"
  # "DLinearRegression"
  # "NLinearRegression"
  # "LinearRegression"
  # "AutoformerBB"
  # "PatchTSTBB"
  # "SeasonalNaive24"
  # "SeasonalNaive168"
  # "NHITSBB"
  "BuildMoE-top-k-2-without-shared-export"
)

# 预测长度
# HORIZONS=(1 6 12 24 48 96 168)
HORIZONS=(168)

# 设置并发上限（根据显存大小调整，比如 2 或 3）
MAX_JOBS=10

# 定义单个日志文件路径
SINGLE_LOG_FILE="$LOG_DIR/all_models_$(date +%Y%m%d_%H%M%S).log"
echo "=== 开始执行所有模型训练，日志保存到: $SINGLE_LOG_FILE ===" | tee "$SINGLE_LOG_FILE"

for DATASET in "${DATASETS[@]}"; do
  for H in "${HORIZONS[@]}"; do
    echo "=== Running dataset=$DATASET, context=$CONTEXT, horizon=$H ===" | tee -a "$SINGLE_LOG_FILE"

    JOBS=0
    for MODEL in "${MODELS[@]}"; do
      echo ">>> 模型: $MODEL, 数据集=$DATASET, horizon=$H" | tee -a "$SINGLE_LOG_FILE"

      # 默认超参
      EPOCHS=20
      LR=1e-3
      PRETRAIN=""
      WD=""

      # 特殊情况
      if [[ "$MODEL" == "BuildMoE-top-k-2-without-shared-export" ]]; then
        LR=3e-5
        PRETRAIN="--pretrained_path /home/hadoop/bec/BuildingsBench/checkpoints/BuildMoE-top-k-2-without-shared-export_best_val.pt"
      elif [[ "$MODEL" == "AutoformerBB" || "$MODEL" == "PatchTSTBB" ]]; then
        LR=1e-4
      elif [[ "$MODEL" == "SeasonalNaive24" || "$MODEL" == "SeasonalNaive168" ]]; then
        EPOCHS=1
      fi

      # 并行执行：后台任务
      (
        echo ">>> 开始训练: $MODEL on $DATASET (horizon=$H)" | tee -a "$SINGLE_LOG_FILE"
        $PYTHON scripts/finetune_university_cofactor.py \
          --dataset "$DATASET" \
          --model "$MODEL" \
          --ctx_len "$CONTEXT" \
          --pred_len "$H" \
          --batch_size 32 \
          --epochs "$EPOCHS" \
          --lr "$LR" \
          --apply_scaler_transform boxcox \
          --results_path "$RESULTS_DIR" \
          --variant_name "${CONTEXT}_${H}" \
          --patience 0 \
          $PRETRAIN $WD \
        |& tee -a "$SINGLE_LOG_FILE"
        echo ">>> 完成训练: $MODEL on $DATASET (horizon=$H)" | tee -a "$SINGLE_LOG_FILE"
      ) &

      JOBS=$((JOBS+1))
      if [[ $JOBS -ge $MAX_JOBS ]]; then
        wait
        JOBS=0
      fi
    done

    # 等待这一轮 horizon 的所有模型跑完
    wait
    echo "=== 完成 dataset=$DATASET, horizon=$H 的所有模型训练 ===" | tee -a "$SINGLE_LOG_FILE"
  done
done

echo "=== 所有模型训练执行完成 ===" | tee -a "$SINGLE_LOG_FILE"



# 数据集
DATASETS=(
  "cofactor:Kindergarten"
  "cofactor:School"
  "cofactor:NursingHome"
  "cofactor:Office"
  "university"
  # "university30t"
  # "university15t"
  # "university1t"
)

# 模型列表
MODELS=(
  "TimeMixerBB"
  "DLinearRegression"
  "NLinearRegression"
  "LinearRegression"
  "AutoformerBB"
  "PatchTSTBB"
  "SeasonalNaive24"
  "SeasonalNaive168"
  "NHITSBB"
  # "BuildMoE-top-k-2-without-shared-export"
)

# 预测长度
# HORIZONS=(1 6 12 24 48 96 168)
HORIZONS=(168)

# 设置并发上限（根据显存大小调整，比如 2 或 3）
MAX_JOBS=10

# 定义单个日志文件路径
SINGLE_LOG_FILE="$LOG_DIR/all_models_$(date +%Y%m%d_%H%M%S).log"
echo "=== 开始执行所有模型训练，日志保存到: $SINGLE_LOG_FILE ===" | tee "$SINGLE_LOG_FILE"

for DATASET in "${DATASETS[@]}"; do
  for H in "${HORIZONS[@]}"; do
    echo "=== Running dataset=$DATASET, context=$CONTEXT, horizon=$H ===" | tee -a "$SINGLE_LOG_FILE"

    JOBS=0
    for MODEL in "${MODELS[@]}"; do
      echo ">>> 模型: $MODEL, 数据集=$DATASET, horizon=$H" | tee -a "$SINGLE_LOG_FILE"

      # 默认超参
      EPOCHS=20
      LR=1e-3
      PRETRAIN=""
      WD=""

      # 特殊情况
      if [[ "$MODEL" == "BuildMoE-top-k-2-without-shared-export" ]]; then
        LR=3e-5
        PRETRAIN="--pretrained_path /home/hadoop/bec/BuildingsBench/checkpoints/BuildMoE-top-k-2-without-shared-export_best_val.pt"
      elif [[ "$MODEL" == "AutoformerBB" || "$MODEL" == "PatchTSTBB" ]]; then
        LR=1e-4
      elif [[ "$MODEL" == "SeasonalNaive24" || "$MODEL" == "SeasonalNaive168" ]]; then
        EPOCHS=1
      fi

      # 并行执行：后台任务
      (
        echo ">>> 开始训练: $MODEL on $DATASET (horizon=$H)" | tee -a "$SINGLE_LOG_FILE"
        $PYTHON scripts/finetune_university_cofactor.py \
          --dataset "$DATASET" \
          --model "$MODEL" \
          --ctx_len "$CONTEXT" \
          --pred_len "$H" \
          --batch_size 32 \
          --epochs "$EPOCHS" \
          --lr "$LR" \
          --apply_scaler_transform boxcox \
          --results_path "$RESULTS_DIR" \
          --variant_name "${CONTEXT}_${H}" \
          --patience 0 \
          $PRETRAIN $WD \
        |& tee -a "$SINGLE_LOG_FILE"
        echo ">>> 完成训练: $MODEL on $DATASET (horizon=$H)" | tee -a "$SINGLE_LOG_FILE"
      ) &

      JOBS=$((JOBS+1))
      if [[ $JOBS -ge $MAX_JOBS ]]; then
        wait
        JOBS=0
      fi
    done

    # 等待这一轮 horizon 的所有模型跑完
    wait
    echo "=== 完成 dataset=$DATASET, horizon=$H 的所有模型训练 ===" | tee -a "$SINGLE_LOG_FILE"
  done
done

echo "=== 所有模型训练执行完成 ===" | tee -a "$SINGLE_LOG_FILE"

