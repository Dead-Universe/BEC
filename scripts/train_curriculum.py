#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_curriculum.py
从短 horizon → 长 horizon 的课程学习训练脚本
"""

# ---- 标准库 & 第三方 ----
import argparse, datetime, os, tomli
from bisect import bisect_right
from pathlib import Path
from timeit import default_timer as timer

import torch, torch.distributed
import torch.utils.data.distributed
import torch.amp
import transformers
import swanlab

# ---- 本项目 ----
from buildings_bench import utils
from buildings_bench.data.new import build_datasets, worker_init_fn
from buildings_bench.models import model_factory
from buildings_bench.tokenizer import LoadQuantizer
from buildings_bench.transforms import BoxCoxTransform, StandardScalerTransform

# ---------- 常量 ----------
DATASET_FULL = True
SCRIPT_PATH = Path(os.path.realpath(__file__)).parent


# ──────────────────────────────────────────────────────────────
# 0.  辅助：课程阶段边界计算
# ──────────────────────────────────────────────────────────────
def build_stage_boundaries(
    mode: str, value: float, num_stage: int, train_steps: int, tokens_per_step: int
):
    """返回 [stage0_end, stage1_end, ...]（不含最后一段）"""
    if num_stage == 1:
        return []

    if mode == "steps":
        step_per_stage = int(train_steps * value) if value < 1 else int(value)
        return [(i + 1) * step_per_stage for i in range(num_stage - 1)]

    if mode == "tokens":
        tok_per_stage = int(value)
        steps = [
            int((i + 1) * tok_per_stage // tokens_per_step)
            for i in range(num_stage - 1)
        ]
        return steps

    if mode == "epochs":
        # 简单按 dataset 大小近似；这里先返回空，需要外层按 epoch 切
        return []

    raise ValueError(f"Unknown curriculum switch mode: {mode}")


# ──────────────────────────────────────────────────────────────
# 1.  主函数
# ──────────────────────────────────────────────────────────────
def main(args, model_args):

    # ------------------- 1.1  几个随机 & CuDNN 设置 -------------------
    utils.set_seed(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ------------------- 1.2  DDP 初始化 -----------------------------
    gpus_per_node = torch.cuda.device_count()
    args.world_size = int(os.environ["WORLD_SIZE"])
    if args.disable_slurm:
        local_rank = int(os.environ["LOCAL_RANK"])
        args.rank = local_rank
    else:
        args.rank = int(os.environ["SLURM_PROCID"])
        local_rank = args.rank - gpus_per_node * (args.rank // gpus_per_node)

    print(f"[rank {args.rank}] init_process_group (local {local_rank})", flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        timeout=datetime.timedelta(hours=1),
        rank=args.rank,
    )
    torch.cuda.set_device(local_rank)

    # ------------------- 1.3  跑 W&B / SwanLab -----------------------
    checkpoint_dir = SCRIPT_PATH / ".." / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_name = f"{args.model}"
    if args.note:
        checkpoint_name += f"_{args.note}"

    if args.rank == 0:
        wandb_project = os.getenv("WANDB_PROJECT", "")
        if wandb_project == "":
            args.disable_wandb = True
        cfg_dump = vars(args).copy()
        cfg_dump["model_args"] = model_args
        if args.disable_wandb:
            run_swanlab = swanlab.init(mode="disabled")
        else:
            run_swanlab = swanlab.init(
                project=wandb_project,
                experiment_name=args.model,
                tags=[args.model, "curriculum"],
                config=cfg_dump,
            )

    # =================================================================
    # 2.  Model & Dataset
    # =================================================================
    model, loss_fn, predict_fn = model_factory(args.model, model_args)
    model = model.to(local_rank)
    print(f"[rank {args.rank}] #params={sum(p.numel() for p in model.parameters()):,}")

    # ----------------- 2.1 数据 -----------------
    if DATASET_FULL:
        train_dataset, val_dataset = build_datasets()
    else:
        raise NotImplementedError("只展示 full-dataset 分支，mini 数据集同理")

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=worker_init_fn,
    )

    # ----------- 2.2 负荷变换（Box-Cox / Standard / Tokenizer）---------
    transform_path = Path(os.getenv("BUILDINGS_BENCH", "")) / "metadata" / "transforms"
    if not model.continuous_loads:
        load_transform = LoadQuantizer(
            with_merge=(not args.tokenizer_without_merge),
            num_centroids=model.vocab_size,
            device=f"cuda:{local_rank}",
        )
        load_transform.load(transform_path)
        transform, inverse = load_transform.transform, load_transform.undo_transform
    else:
        if args.apply_scaler_transform == "boxcox":
            load_transform = BoxCoxTransform()
        elif args.apply_scaler_transform == "standard":
            load_transform = StandardScalerTransform()
        else:
            load_transform = None
        if load_transform:
            load_transform.load(transform_path)
        transform = lambda x: x
        inverse = load_transform.undo_transform if load_transform else (lambda x: x)

    # =================================================================
    # 3.  Optimizer & DDP 包装
    # =================================================================
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=0.01,
    )

    # ------------- 3.1 课程参数 -------------
    from utils_autobatch import autotune_batch_size

    pred_lens = [int(x) for x in args.curriculum_pred_lens.split(",")]
    ctx_lens = (
        [int(x) for x in args.curriculum_ctx_len.split(",")]
        if args.curriculum_ctx_len
        else [model.max_context_len] * len(pred_lens)
    )
    try:
        new_bs = autotune_batch_size(
            model,
            ctx_lens,
            pred_lens,
            args.batch_size,
            torch.device(f"cuda:{local_rank}"),
        )
        if new_bs < args.batch_size and args.rank == 0:
            print(
                f"[AutoBatch] batch_size {args.batch_size} → {new_bs} (fit GPU memory)"
            )
        args.batch_size = new_bs
    except RuntimeError as e:
        if args.rank == 0:
            print("[AutoBatch] ERROR:", e)
        torch.distributed.destroy_process_group()
        exit(1)
    assert len(pred_lens) == len(ctx_lens)

    # 训练以「总 token 数」为准，用最大 pred_len 估算上限步数
    global_bs = args.world_size * args.batch_size
    train_steps = args.train_tokens // (global_bs * max(pred_lens))
    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.02 * train_steps),
        num_training_steps=train_steps,
    )

    tokens_per_step = global_bs * max(pred_lens)
    mode, value = args.curriculum_switch.split(":")
    stage_bounds = build_stage_boundaries(
        mode, float(value), len(pred_lens), train_steps, tokens_per_step
    )

    scaler = torch.amp.GradScaler()
    best_train_loss = float("inf")

    # ---------- 修改开始：早停 / 自动切课变量 ----------
    steps_no_improve = 0  # 连续无改进计数器
    current_stage = 0  # 当前 curriculum stage（手动可调）
    # ---------- 修改结束 --------------------------------

    step = 0
    seen_tokens = 0

    # =================================================================
    # 4.  训练循环
    # =================================================================
    print(f"[rank {args.rank}] start training, total steps≈{train_steps}")
    model.train()
    train_sampler.set_epoch(0)

    for batch in train_loader:
        # ---------------- 4.1 课程阶段决定 ----------------
        stage_idx_auto = bisect_right(stage_bounds, step)
        current_stage = max(current_stage, stage_idx_auto)

        cur_pred_len = pred_lens[current_stage]
        cur_ctx_len = ctx_lens[current_stage]
        seq_len = cur_ctx_len + cur_pred_len

        start_time = timer()
        optimizer.zero_grad()

        # ----------- 裁剪 batch 到动态长度 -----------
        for k, v in batch.items():
            if v.ndim >= 2 and v.size(1) >= seq_len:
                batch[k] = v[:, :seq_len].to(model.device)
            else:
                batch[k] = v.to(model.device)

        batch["load"] = transform(batch["load"])

        with torch.amp.autocast("cuda"):
            preds = model(batch, context_len=cur_ctx_len, pred_len=cur_pred_len)
            targets = batch["load"][:, cur_ctx_len:seq_len]
            loss = loss_fn(preds, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # ---------------- 4.2 统计 & 控制逻辑 ----------------
        step += 1
        seen_tokens += global_bs * cur_pred_len
        train_sampler.set_epoch(step)  # 打乱

        # ----- rank‑0: 评估改进、保存 ckpt、日志 -----
        # ---------- 修改开始：改进判断 / 计数 ----------
        improved = False
        if args.rank == 0:
            if loss.item() + 1e-9 < best_train_loss:
                best_train_loss = loss.item()
                steps_no_improve = 0
                improved = True
            else:
                steps_no_improve += 1
        # ---------- 修改结束 --------------------------

        # ---------- rank‑0: 日志 & checkpoint ----------
        if args.rank == 0:
            if improved:
                utils.save_model_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    step,
                    best_train_loss,
                    checkpoint_dir / f"{checkpoint_name}_best_train.pt",
                )
                print(
                    f"[rank 0] step {step} - New best train loss: {best_train_loss:.4f}"
                )
            if step % 500 == 0:
                elapsed = timer() - start_time
                swanlab.log(
                    {
                        "train/loss": loss.item(),
                        "train/seen_tok(M)": seen_tokens / 1e6,
                        "train/pred_len": cur_pred_len,
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/step_time": elapsed,
                    },
                    step=step,
                )
                print(
                    f"[rank 0] step {step} "
                    f"loss={loss:.4f} pred_len={cur_pred_len} "
                    f"tok_seen={seen_tokens/1e6:.1f}M",
                    flush=True,
                )

        # ---------- 修改开始：切课 / 早停 ------------
        if args.rank == 0:
            switch_stage = -1
            early_stop = False
            if steps_no_improve >= args.no_improve_steps:
                if current_stage < len(pred_lens) - 1:  # 还有下一阶段
                    current_stage += 1
                    best_train_loss = float("inf")
                    steps_no_improve = 0
                    switch_stage = current_stage
                    print(
                        f"[Curriculum] {args.no_improve_steps} steps no gain → "
                        f"switch to stage {current_stage} "
                        f"(pred_len={pred_lens[current_stage]})"
                    )
                else:  # 最后一阶段 → 早停
                    early_stop = True
                    print(
                        f"[EarlyStop] final stage still no improvement for "
                        f"{args.no_improve_steps} steps, stop training."
                    )
        else:
            switch_stage = -1
            early_stop = False

        # 把控制信号广播到所有进程
        ctrl_msg = [switch_stage, early_stop]
        torch.distributed.broadcast_object_list(ctrl_msg, src=0)

        # 所有 rank 根据信号同步
        if ctrl_msg[0] >= 0:  # 切换 stage
            current_stage = ctrl_msg[0]
            best_train_loss = float("inf")
            steps_no_improve = 0
        if ctrl_msg[1]:  # 早停
            break
        # ---------- 修改结束 --------------------------

        if step >= train_steps:  # 达到既定总步数
            break

    # -------- 训练结束 --------
    if args.rank == 0:
        utils.save_model_checkpoint(
            model,
            optimizer,
            scheduler,
            step,
            best_train_loss,
            checkpoint_dir / f"{checkpoint_name}_last.pt",
        )
        run_swanlab.finish()

    torch.distributed.destroy_process_group()


# ──────────────────────────────────────────────────────────────
# 5.  argparse & 启动
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()

    # ---------- 基础 ----------
    p.add_argument("--model", required=True)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--warmup_steps", type=int, default=20000)
    p.add_argument("--train_tokens", type=int, default=1_000_000_000)
    p.add_argument("--random_seed", type=int, default=99)
    p.add_argument("--resume_from_checkpoint", type=str, default="")
    p.add_argument("--note", type=str, default="")
    p.add_argument("--disable_wandb", action="store_true")

    # ---------- DDP ----------
    p.add_argument("--disable_slurm", action="store_true")
    p.add_argument("--world-size", default=-1, type=int)
    p.add_argument("--rank", default=-1, type=int)
    p.add_argument("--dist-url", default="env://", type=str)
    p.add_argument("--dist-backend", default="nccl", type=str)
    p.add_argument("--num_workers", type=int, default=8)

    # ---------- 数据 / 变换 ----------
    p.add_argument(
        "--apply_scaler_transform", choices=["", "standard", "boxcox"], default="boxcox"
    )
    p.add_argument("--tokenizer_without_merge", action="store_true")

    # ---------- Curriculum ----------
    p.add_argument(
        "--curriculum_pred_lens",
        type=str,
        default="24,48,96,168",
        help="Comma-sep list of pred_len for each stage",
    )
    p.add_argument(
        "--curriculum_ctx_len",
        type=str,
        default="168,224,280,336",
        help="Optional comma-sep context_len list; leave blank to keep fixed",
    )
    p.add_argument(
        "--curriculum_switch",
        type=str,
        default="steps:0.25",
        help="Switch condition <mode>:<value>, mode∈steps|tokens|epochs",
    )
    # ---------- 修改开始：新增无改进步数阈值 ----------
    p.add_argument(
        "--no_improve_steps",
        type=int,
        default=20000,
        help="连续无 loss 改进达到该步数后：若非最后阶段则切课；若已是最后阶段则提前结束",
    )
    # ---------- 修改结束 --------------------------------

    # ---------- 其他（省略部分原 flags，可自行补回） ----------
    args = p.parse_args()

    # -------------- 读取 TOML 配置覆盖 ----------------
    cfg_dir = SCRIPT_PATH / ".." / "buildings_bench" / "configs"
    toml_file = cfg_dir / f"{args.model}.toml"
    if not toml_file.exists():
        raise FileNotFoundError(toml_file)
    with toml_file.open("rb") as f:
        toml_cfg = tomli.load(f)
    model_args = toml_cfg["model"]
    # 训练超参覆盖
    for k, v in toml_cfg.get("pretrain", {}).items():
        if hasattr(args, k):
            setattr(args, k, v)

    if not torch.cuda.is_available():
        raise ValueError("CUDA not available")

    main(args, model_args)
