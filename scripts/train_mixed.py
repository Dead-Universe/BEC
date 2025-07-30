#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_mixed.py  –  变长输入 / 输出混合训练 + 验证 RMSE/MAE
------------------------------------------------------------
• 每个 mini‑batch 随机采样 (ctx_len, pred_len)
• Loss 先对序列平均再对 batch 平均
• Optimizer / Scheduler 按真实 token 数更新
• 每 N 更新在验证集上计算 “平均 RMSE / MAE”
"""

from __future__ import annotations
from functools import partial
import argparse, datetime, math, os, random, tomli, numpy as np
from pathlib import Path
from typing import Dict

import torch, torch.distributed
import torch.utils.data.distributed
import torch.amp
import transformers
import swanlab

# ---------- 项目依赖 ----------
from buildings_bench import utils
from buildings_bench.data.new import build_datasets, worker_init_fn
from buildings_bench.models import model_factory
from buildings_bench.tokenizer import LoadQuantizer
from buildings_bench.transforms import BoxCoxTransform, StandardScalerTransform

# ════════════════════════════════════════════════════════════
# 1. collate_fn   (ctx/pred 上下限来自 CLI)
# ════════════════════════════════════════════════════════════
CTX_MIN, CTX_MAX = 24, 336
PRED_MIN, PRED_MAX = 1, 168


def mixed_len_collate(batch):
    ctx_len = random.randint(CTX_MIN, CTX_MAX)
    pred_len = random.randint(PRED_MIN, PRED_MAX)
    seq_len = ctx_len + pred_len

    # 裁剪
    for s in batch:
        for k, v in s.items():
            if isinstance(v, (torch.Tensor, np.ndarray)) and v.ndim >= 1:
                if v.shape[0] < seq_len:
                    raise ValueError(f"样本长度 {v.shape[0]} < 期望 {seq_len}")
                s[k] = v[:seq_len]

    def _stack(key, dtype=None):
        ts = [torch.as_tensor(s[key]) for s in batch]
        out = torch.stack(ts)
        return out if dtype is None else out.to(dtype)

    collated = {
        "latitude": _stack("latitude", torch.float32),
        "longitude": _stack("longitude", torch.float32),
        "day_of_year": _stack("day_of_year", torch.float32),
        "day_of_week": _stack("day_of_week", torch.float32),
        "hour_of_day": _stack("hour_of_day", torch.float32),
        "building_type": _stack("building_type", torch.long),
        "load": _stack("load", torch.float32),
    }
    for k in batch[0]:
        if k not in collated:
            collated[k] = _stack(k, torch.float32)

    collated["context_len"] = ctx_len
    collated["pred_len"] = pred_len
    return collated


# === NEW
def fixed_len_collate(batch, ctx_len: int, pred_len: int):
    """
    把样本裁剪/填充到固定 (ctx_len + pred_len)，
    不再随机长度，专供验证。
    """
    seq_len = ctx_len + pred_len
    for s in batch:
        for k, v in s.items():
            if isinstance(v, (torch.Tensor, np.ndarray)) and v.ndim >= 1:
                if v.shape[0] < seq_len:
                    raise ValueError(f"val 样本过短: {v.shape[0]} < {seq_len}")
                s[k] = v[:seq_len]

    def _stack(key, dtype=None):
        ts = [torch.as_tensor(s[key]) for s in batch]
        out = torch.stack(ts)
        return out if dtype is None else out.to(dtype)

    merged = {
        "latitude": _stack("latitude", torch.float32),
        "longitude": _stack("longitude", torch.float32),
        "day_of_year": _stack("day_of_year", torch.float32),
        "day_of_week": _stack("day_of_week", torch.float32),
        "hour_of_day": _stack("hour_of_day", torch.float32),
        "building_type": _stack("building_type", torch.long),
        "load": _stack("load", torch.float32),
        "context_len": torch.tensor(ctx_len),
        "pred_len": torch.tensor(pred_len),
    }
    return merged


# ════════════════════════════════════════════════════════════
# 2. 验证循环 – 平均 RMSE & MAE             # === NEW
# ════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(model, val_loader, transform, inverse, device):
    model.eval()
    se_sum = 0.0  # squared error 累计
    ae_sum = 0.0  # abs   error 累计
    n_tok = 0

    for batch in val_loader:
        ctx_len = int(batch.pop("context_len"))
        pred_len = int(batch.pop("pred_len"))
        seq_len = ctx_len + pred_len

        for k in batch:
            batch[k] = batch[k].to(device)
        batch["load"] = transform(batch["load"])

        preds = model(batch, context_len=ctx_len, pred_len=pred_len)  # [B,L,1]
        tgt = batch["load"][:, ctx_len:seq_len]

        # 反标准化到真实量纲
        preds_real = inverse(preds)
        tgt_real = inverse(tgt)

        err = preds_real - tgt_real
        se_sum += (err**2).sum().item()
        ae_sum += err.abs().sum().item()
        n_tok += err.numel()

    rmse = math.sqrt(se_sum / n_tok)
    mae = ae_sum / n_tok
    model.train()
    return rmse, mae


# === NEW
@torch.no_grad()
def evaluate_all(model, loaders_dict, transform, inverse, device):
    """
    返回 dict { (ctx,pred): (rmse, mae) }
    """
    model.eval()
    out = {}
    for (c, p), loader in loaders_dict.items():
        se, ae, n = 0.0, 0.0, 0
        for batch in loader:
            for k in batch:
                batch[k] = batch[k].to(device)
            batch["load"] = transform(batch["load"])

            preds = model(batch, context_len=c, pred_len=p)
            tgt = batch["load"][:, c : c + p]

            err = inverse(preds) - inverse(tgt)
            se += (err**2).sum().item()
            ae += err.abs().sum().item()
            n += err.numel()
        out[(c, p)] = (math.sqrt(se / n), ae / n)
    model.train()
    return out


# ════════════════════════════════════════════════════════════
# 3. 主函数
# ════════════════════════════════════════════════════════════
SCRIPT_PATH = Path(os.path.realpath(__file__)).parent
DATASET_FULL = True
# === NEW: 固定验证窗口 (ctx_len, pred_len) 组合 ===
VAL_COMBOS = [
    (168, 24),
    (224, 48),
    (280, 96),
    (336, 168),
]


def main(args, model_args):
    global CTX_MIN, CTX_MAX, PRED_MIN, PRED_MAX
    CTX_MIN, CTX_MAX = args.ctx_min, args.ctx_max
    PRED_MIN, PRED_MAX = args.pred_min, args.pred_max

    utils.set_seed(args.random_seed)
    torch.backends.cudnn.deterministic = False  # 更快；如需复现设 True
    torch.backends.cudnn.benchmark = True

    # ---------- DDP ----------
    gpus_per_node = torch.cuda.device_count()
    args.world_size = int(os.environ["WORLD_SIZE"])
    if args.disable_slurm:
        local_rank = int(os.environ["LOCAL_RANK"])
        args.rank = local_rank
    else:
        args.rank = int(os.environ["SLURM_PROCID"])
        local_rank = args.rank - gpus_per_node * (args.rank // gpus_per_node)
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
        timeout=datetime.timedelta(hours=1),
    )

    # ---------- 日志 ----------
    checkpoint_dir = SCRIPT_PATH / ".." / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = f"{args.model}{'_' + args.note if args.note else ''}"

    if args.rank == 0:
        wandb_project = os.getenv("WANDB_PROJECT", "")
        if wandb_project == "":
            args.disable_wandb = True
        run_swanlab = swanlab.init(
            project=wandb_project or "disabled",
            experiment_name=ckpt_name,
            tags=[args.model, "mixed_len"],
            config={**vars(args), "model_args": model_args},
            mode="disabled" if args.disable_wandb else "cloud",
            resume="allow",
            id=args.swanlab_id,
        )

    # ════════════════════════════════════════════════
    # 4. Model & Dataset
    # ════════════════════════════════════════════════
    model, loss_fn, _ = model_factory(args.model, model_args)
    model = model.to(local_rank)

    train_ds, val_ds, _ = build_datasets(
        context_len=model_args["max_context_len"],
        pred_len=model_args["max_pred_len"],
        apply_scaler_transform=args.apply_scaler_transform,
        split="val",
        oov_val=Path("/home/hadoop/bec/BuildingsBench/oov_val.txt"),
        oov_test=Path("/home/hadoop/bec/BuildingsBench/oov_test.txt"),
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_ds, num_replicas=args.world_size, rank=args.rank, shuffle=True
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=mixed_len_collate,
        worker_init_fn=worker_init_fn,
    )
    if args.rank == 0:

        val_loaders = {
            (c, p): torch.utils.data.DataLoader(
                val_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                persistent_workers=True,
                collate_fn=partial(fixed_len_collate, ctx_len=c, pred_len=p),
                worker_init_fn=worker_init_fn,
            )
            for (c, p) in VAL_COMBOS
        }
    else:
        val_loaders = {}

    # ---------- 负荷变换 ----------
    tpath = Path(os.getenv("BUILDINGS_BENCH", "")) / "metadata" / "transforms"
    if not model.continuous_loads:
        quantizer = LoadQuantizer(
            with_merge=not args.tokenizer_without_merge,
            num_centroids=model.vocab_size,
            device=f"cuda:{local_rank}",
        )
        quantizer.load(tpath)
        transform, inverse = quantizer.transform, quantizer.undo_transform
    else:
        if args.apply_scaler_transform == "boxcox":
            scaler = BoxCoxTransform()
        elif args.apply_scaler_transform == "standard":
            scaler = StandardScalerTransform()
        else:
            scaler = None
        if scaler:
            scaler.load(tpath)
        transform = lambda x: x  # 数据集已经做过变换
        inverse = scaler.undo_transform if scaler else (lambda x: x)

    # ════════════════════════════════════════════════
    # 5. Optimizer, Scheduler, DDP
    # ════════════════════════════════════════════════
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

    total_updates = math.ceil(args.train_tokens / args.tokens_per_update)
    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.02 * total_updates),
        num_training_steps=total_updates,
    )
    scaler = torch.amp.GradScaler()

    # ════════════════════════════════════════════════
    # 6. 训练循环
    # ════════════════════════════════════════════════
    # ════════════════════════════════════════════════
    # 6‑0. 可选：从检查点恢复
    # ════════════════════════════════════════════════
    best_val_rmse = float("inf")
    tokens_buf = seen_tokens = update_idx = 0

    if args.resume_from_checkpoint:
        ckpt_path = checkpoint_dir / args.resume_from_checkpoint
        if not ckpt_path.exists():
            raise FileNotFoundError(ckpt_path)

        (model, optimizer, scheduler, update_idx, best_val_rmse) = (
            utils.load_model_checkpoint(
                ckpt_path,
                model,
                optimizer,
                scheduler,
                local_rank,
            )
        )

        # 让 dataloader 的 shuffle 状态与之前保持⼀致
        train_sampler.set_epoch(update_idx // args.shuffle_every)

        # 已看过的 token 数，用 tokens_per_update 近似即可
        seen_tokens = update_idx * args.tokens_per_update

        if args.rank == 0:
            print(
                f"[rank 0] Resumed from {ckpt_path.name}  "
                f"update_idx={update_idx}  seen_tokens≈{seen_tokens/1e6:.1f}M  "
                f"best_val_RMSE={best_val_rmse:.4f}",
                flush=True,
            )

    for step, batch in enumerate(train_loader, start=1):
        ctx_len = int(batch.pop("context_len"))
        pred_len = int(batch.pop("pred_len"))
        seq_len = ctx_len + pred_len
        bs = batch["load"].size(0)

        if tokens_buf == 0:
            optimizer.zero_grad(set_to_none=True)

        for k in batch:
            batch[k] = batch[k].to(local_rank)
        batch["load"] = transform(batch["load"])

        with torch.amp.autocast("cuda"):
            preds = model(batch, context_len=ctx_len, pred_len=pred_len)
            tgt = batch["load"][:, ctx_len:seq_len]
            loss = loss_fn(preds, tgt)

        scaler.scale(loss).backward()

        tok_step = bs * pred_len * preds.size(-1)  # === MOD: out_dim
        tokens_buf += tok_step
        seen_tokens += tok_step

        # 梯度累积到阈值
        while tokens_buf >= args.tokens_per_update:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            optimizer.zero_grad(set_to_none=True)

            tokens_buf -= args.tokens_per_update
            update_idx += 1
            if update_idx % args.shuffle_every == 0:  # === NEW
                train_sampler.set_epoch(update_idx // args.shuffle_every)

            # ----------- 验证 ----------
            if update_idx % args.eval_every == 0:
                if args.world_size > 1:
                    torch.distributed.barrier()  # 所有卡停
                if args.rank == 0:
                    metrics = evaluate_all(
                        model, val_loaders, transform, inverse, local_rank
                    )
                    # 日志 & 打印
                    for (c, p), (rmse, mae) in metrics.items():
                        tag = f"{c}_{p}"
                        swanlab.log(
                            {
                                f"val/RMSE_{tag}": rmse,
                                f"val/MAE_{tag}": mae,
                                "train/update_idx": update_idx,
                            },
                            step=update_idx,
                        )
                        print(
                            f"[rank 0] Eval@upd{update_idx}  "
                            f"(ctx{c},pred{p})  RMSE={rmse:.4f}  MAE={mae:.4f}"
                        )
                    # 用最长 horizon 336→168 做早停/最佳模型
                    main_rmse = metrics[(336, 168)][0]
                    if main_rmse < best_val_rmse - 1e-9:
                        best_val_rmse = main_rmse
                        utils.save_model_checkpoint(
                            model,
                            optimizer,
                            scheduler,
                            update_idx,
                            best_val_rmse,
                            checkpoint_dir / f"{ckpt_name}_best_val.pt",
                        )
                if args.world_size > 1:
                    # 等待 rank 0 完成验证
                    torch.distributed.barrier()

        # if tokens_buf == 0:
        #     optimizer.zero_grad(set_to_none=True)
        # elif tokens_buf > 0:
        #     scaler.unscale_(optimizer)
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        #     scaler.step(optimizer)
        #     scaler.update()
        #     scheduler.step()

        #     update_idx += 1

        # rank‑0 训练日志
        if args.rank == 0 and step % args.log_every == 0:
            swanlab.log(
                {
                    "train/loss": loss.item(),
                    "train/pred_len": pred_len,
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/seen_tok(M)": seen_tokens / 1e6,
                    "train/update_idx": update_idx,
                },
                step=step,
            )
            print(
                f"[rank 0] iter {step} upd {update_idx}/{total_updates} "
                f"loss={loss:.4f} H={pred_len} tok_seen={seen_tokens/1e6:.1f}M",
                flush=True,
            )
        if args.rank == 0 and step % 5000 == 0:
            print(
                f"[rank 0] 已训练 {seen_tokens / 1e6:.1f}M tokens "
                f"（{update_idx}/{total_updates} 更新）",
                flush=True,
            )
            utils.save_model_checkpoint(
                model,
                optimizer,
                scheduler,
                update_idx,
                best_val_rmse,
                checkpoint_dir / f"{ckpt_name}_iter{step}.pt",
            )

        if update_idx >= total_updates:
            break

    # ════════════════════════════════════════════════
    # 7. 收尾
    # ════════════════════════════════════════════════
    if args.rank == 0:
        utils.save_model_checkpoint(
            model,
            optimizer,
            scheduler,
            update_idx,
            best_val_rmse,
            checkpoint_dir / f"{ckpt_name}_last.pt",
        )
        run_swanlab.finish()
    torch.distributed.destroy_process_group()


# ════════════════════════════════════════════════════════════
# 8. CLI
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # 基础
    p.add_argument("--model", required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--train_tokens", type=int, default=4_000_000_000)
    p.add_argument("--tokens_per_update", type=int, default=1_000_000)
    p.add_argument("--random_seed", type=int, default=99)
    p.add_argument("--note", type=str, default="")
    p.add_argument("--disable_wandb", action="store_true")
    p.add_argument("--log_every", type=int, default=500)
    # 长度上下限
    p.add_argument("--ctx_min", type=int, default=24)
    p.add_argument("--ctx_max", type=int, default=336)
    p.add_argument("--pred_min", type=int, default=1)
    p.add_argument("--pred_max", type=int, default=168)
    # 评估/Shuffle 频率        # === NEW
    p.add_argument(
        "--eval_every", type=int, default=400, help="每多少次参数更新做一次验证"
    )
    p.add_argument(
        "--shuffle_every",
        type=int,
        default=100,
        help="多少次参数更新后对 sampler 重新 shuffle",
    )
    # DDP
    p.add_argument("--disable_slurm", action="store_true")
    p.add_argument("--world-size", default=-1, type=int)
    p.add_argument("--rank", default=-1, type=int)
    p.add_argument("--dist-url", default="env://")
    p.add_argument("--dist-backend", default="nccl")
    p.add_argument("--num_workers", type=int, default=8)
    # 数据 / 变换
    p.add_argument(
        "--apply_scaler_transform", choices=["", "standard", "boxcox"], default="boxcox"
    )
    p.add_argument("--tokenizer_without_merge", action="store_true")
    # === NEW ===  允许指定检查点文件名（相对 checkpoints/）
    p.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="",
        help="如需断点续训，填入 *.pt 文件名（位于 checkpoints/ 目录下）",
    )
    p.add_argument(
        "--swanlab_id",
        type=str,
        default="",
        help="SwanLab 实验 ID（用于实验追踪）",
    )

    args = p.parse_args()

    # 读取 TOML
    cfg_dir = SCRIPT_PATH / ".." / "buildings_bench" / "configs"
    toml_file = cfg_dir / f"{args.model}.toml"
    if not toml_file.exists():
        raise FileNotFoundError(toml_file)
    with toml_file.open("rb") as f:
        toml_cfg = tomli.load(f)
    model_args: Dict = toml_cfg["model"]
    for k, v in toml_cfg.get("pretrain", {}).items():
        if hasattr(args, k):
            setattr(args, k, v)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    main(args, model_args)
