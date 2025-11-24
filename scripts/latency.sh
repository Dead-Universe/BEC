export BUILDINGS_BENCH=/home/hadoop/bec/buildings-bench/v2.0.0/BuildingsBench

python scripts/latency_eval.py \
  --model BuildMoE-top-k-2-without-shared-export \
  --dataset cofactor \
  --L 168 --H 24 \
  --samples 200 \
  --ckpt checkpoints/BuildMoE-top-k-2-without-shared-export_best_val.pt \
  --outdir ./latency_out_buildmoe \
  --device cuda:0 \
  --warmup 10 \

python scripts/latency_eval.py \
  --model Moment-L \
  --dataset cofactor \
  --L 168 --H 24 \
  --samples 200 \
  --ckpt checkpoints/Moment-L_best_val.pt \
  --outdir ./latency_out_moment \
  --device cuda:0 \
  --warmup 10 \

python scripts/latency_eval.py \
  --model Chronos-Base \
  --dataset cofactor \
  --L 168 --H 24 \
  --samples 200 \
  --ckpt checkpoints/Chronos-Base_best_val.pt \
  --outdir ./latency_out_chronos \
  --device cuda:0 \
  --warmup 10 \

python scripts/latency_eval.py \
  --model TimeMoE-200M \
  --dataset cofactor \
  --L 168 --H 24 \
  --samples 200 \
  --ckpt checkpoints/TimeMoE-200M_best_val.pt \
  --outdir ./latency_out_timemoe \
  --device cuda:0 \
  --warmup 10 \
