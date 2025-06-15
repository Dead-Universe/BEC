#!/usr/bin/env bash
#===============================================================================
# File: run_zero_shot_all.sh
# Description: 用于一键运行 zero-shot 评估，并把结果聚合成中位数
# Usage: bash run_zero_shot_all.sh
#===============================================================================

# 如果你尚未激活 conda 环境，可以在这里加：
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate bec

# 1. 设置 BUILDINGS_BENCH 环境变量（指向你本地克隆的仓库根目录）
export BUILDINGS_BENCH=/data/lx/buildings-bench/v2.0.0/BuildingsBench

# 2. 指定 Python 可执行文件（可选，若 you want 用绝对路径）
PYTHON=python

# 3. 配置要使用的 GPU
export CUDA_VISIBLE_DEVICES=0

# 4. 运行 zero-shot 脚本
$PYTHON scripts/zero_shot.py \
    --model TransformerWithGaussianAndMoEs-S \
    --benchmark all \
    --checkpoint checkpoints/TransformerWithGaussianAndMoEs-S_best.pt \
    --apply_scaler_transform boxcox

# $PYTHON scripts/zero_shot.py \
#     --model chronos \
#     --benchmark all

# 5. 如果想顺便打印出“中位数 + 95% CI”，不用额外再写 BASH，因为脚本最后已内置了
#    aggregate.return_aggregate_median(...) + pretty_print_aggregates()。
#    所以到这里脚本运行完就会自动把：
#         BuildingsBench (real)
#         TransformerWithGaussian-L residential cvrmse: XX.XXX (lo,hi)
#         TransformerWithGaussian-L commercial cvrmse: YY.YYY (lo,hi)
#         …
#    打印到终端。

