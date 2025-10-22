export BUILDINGS_BENCH=/home/hadoop/bec/buildings-bench/v2.0.0/BuildingsBench

python scripts/balanced_moe_activation_experiment.py \
    --dataset sceaux \
    --data-root /home/hadoop/bec/buildings-bench/v2.0.0/BuildingsBench \
    --samples 4000 \
    --ckpt /home/hadoop/bec/BuildingsBench/checkpoints/BuildMoE-top-k-2-without-shared-export_best_val.pt \
    --outdir ../exp_out/sceaux \

python scripts/balanced_moe_activation_experiment.py \
    --dataset borealis \
    --data-root /home/hadoop/bec/buildings-bench/v2.0.0/BuildingsBench \
    --samples 4000 \
    --ckpt /home/hadoop/bec/BuildingsBench/checkpoints/BuildMoE-top-k-2-without-shared-export_best_val.pt \
    --outdir ../exp_out/borealis \

python scripts/balanced_moe_activation_experiment.py \
    --dataset ideal \
    --data-root /home/hadoop/bec/buildings-bench/v2.0.0/BuildingsBench \
    --samples 4000 \
    --ckpt /home/hadoop/bec/BuildingsBench/checkpoints/BuildMoE-top-k-2-without-shared-export_best_val.pt \
    --outdir ../exp_out/ideal \

python scripts/balanced_moe_activation_experiment.py \
    --dataset bdg-2:panther \
    --data-root /home/hadoop/bec/buildings-bench/v2.0.0/BuildingsBench \
    --samples 4000 \
    --ckpt /home/hadoop/bec/BuildingsBench/checkpoints/BuildMoE-top-k-2-without-shared-export_best_val.pt \
    --outdir ../exp_out/panther \

python scripts/balanced_moe_activation_experiment.py \
    --dataset bdg-2:fox \
    --data-root /home/hadoop/bec/buildings-bench/v2.0.0/BuildingsBench \
    --samples 4000 \
    --ckpt /home/hadoop/bec/BuildingsBench/checkpoints/BuildMoE-top-k-2-without-shared-export_best_val.pt \
    --outdir ../exp_out/fox \

python scripts/balanced_moe_activation_experiment.py \
    --dataset bdg-2:rat \
    --data-root /home/hadoop/bec/buildings-bench/v2.0.0/BuildingsBench \
    --samples 4000 \
    --ckpt /home/hadoop/bec/BuildingsBench/checkpoints/BuildMoE-top-k-2-without-shared-export_best_val.pt \
    --outdir ../exp_out/rat \

python scripts/balanced_moe_activation_experiment.py \
    --dataset bdg-2:bear \
    --data-root /home/hadoop/bec/buildings-bench/v2.0.0/BuildingsBench \
    --samples 4000 \
    --ckpt /home/hadoop/bec/BuildingsBench/checkpoints/BuildMoE-top-k-2-without-shared-export_best_val.pt \
    --outdir ../exp_out/bear \

python scripts/balanced_moe_activation_experiment.py \
    --dataset electricity \
    --data-root /home/hadoop/bec/buildings-bench/v2.0.0/BuildingsBench \
    --samples 4000 \
    --ckpt /home/hadoop/bec/BuildingsBench/checkpoints/BuildMoE-top-k-2-without-shared-export_best_val.pt \
    --outdir ../exp_out/electricity \

python scripts/balanced_moe_activation_experiment.py \
    --dataset smart \
    --data-root /home/hadoop/bec/buildings-bench/v2.0.0/BuildingsBench \
    --samples 4000 \
    --ckpt /home/hadoop/bec/BuildingsBench/checkpoints/BuildMoE-top-k-2-without-shared-export_best_val.pt \
    --outdir ../exp_out/smart \

python scripts/balanced_moe_activation_experiment.py \
    --dataset lcl \
    --data-root /home/hadoop/bec/buildings-bench/v2.0.0/BuildingsBench \
    --samples 4000 \
    --ckpt /home/hadoop/bec/BuildingsBench/checkpoints/BuildMoE-top-k-2-without-shared-export_best_val.pt \
    --outdir ../exp_out/lcl \

python scripts/balanced_moe_activation_experiment.py \
    --dataset university \
    --data-root /home/hadoop/bec/buildings-bench/v2.0.0/BuildingsBench \
    --samples 4000 \
    --ckpt /home/hadoop/bec/BuildingsBench/checkpoints/BuildMoE-top-k-2-without-shared-export_best_val.pt \
    --outdir ../exp_out/university \

python scripts/balanced_moe_activation_experiment.py \
    --dataset cofactor \
    --data-root /home/hadoop/bec/buildings-bench/v2.0.0/BuildingsBench \
    --category Office \
    --samples 4000 \
    --ckpt /home/hadoop/bec/BuildingsBench/checkpoints/BuildMoE-top-k-2-without-shared-export_best_val.pt \
    --outdir ../exp_out/office \

python scripts/balanced_moe_activation_experiment.py \
    --dataset cofactor \
    --data-root /home/hadoop/bec/buildings-bench/v2.0.0/BuildingsBench \
    --category School \
    --samples 4000 \
    --ckpt /home/hadoop/bec/BuildingsBench/checkpoints/BuildMoE-top-k-2-without-shared-export_best_val.pt \
    --outdir ../exp_out/school \

python scripts/balanced_moe_activation_experiment.py \
    --dataset cofactor \
    --data-root /home/hadoop/bec/buildings-bench/v2.0.0/BuildingsBench \
    --category Kindergarten \
    --samples 4000 \
    --ckpt /home/hadoop/bec/BuildingsBench/checkpoints/BuildMoE-top-k-2-without-shared-export_best_val.pt \
    --outdir ../exp_out/kindergarten \

python scripts/balanced_moe_activation_experiment.py \
    --dataset cofactor \
    --data-root /home/hadoop/bec/buildings-bench/v2.0.0/BuildingsBench \
    --category NursingHome \
    --samples 4000 \
    --ckpt /home/hadoop/bec/BuildingsBench/checkpoints/BuildMoE-top-k-2-without-shared-export_best_val.pt \
    --outdir ../exp_out/nursing_home \

python scripts/balanced_moe_activation_experiment.py \
    --dataset buildings-900k-test \
    --data-root /home/hadoop/bec/buildings-bench/v2.0.0/BuildingsBench \
    --samples 4000 \
    --ckpt /home/hadoop/bec/BuildingsBench/checkpoints/BuildMoE-top-k-2-without-shared-export_best_val.pt \
    --outdir ../exp_out/buildings-900k-test \

python scripts/balanced_moe_activation_experiment.py \
    --registry-all \
    --data-root /home/hadoop/bec/buildings-bench/v2.0.0/BuildingsBench \
    --samples 4000 \
    --ckpt /home/hadoop/bec/BuildingsBench/checkpoints/BuildMoE-top-k-2-without-shared-export_best_val.pt \
    --outdir ../exp_out/new_all \

