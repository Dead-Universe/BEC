#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_mixed.py  –  变长输入 / 输出混合训练 + 验证 RMSE/MAE
------------------------------------------------------------
• 每个 mini-batch 随机采样 (ctx_len, pred_len)
• Loss 先对序列平均再对 batch 平均
• Optimizer / Scheduler 按真实 token 数更新
• 每 N 更新在验证集上计算 “平均 RMSE / MAE”
"""

from __future__ import annotations

import argparse
import datetime
import math
import os
import random
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Dict

import numpy as np
import swanlab
import tomli
import torch
import torch.amp
import torch.distributed
import torch.utils.data.distributed
import transformers

# ---------- 项目依赖 ----------
from buildings_bench import utils
from buildings_bench.data.new import build_datasets, worker_init_fn
from buildings_bench.models import model_factory
from buildings_bench.tokenizer import LoadQuantizer
from buildings_bench.transforms import BoxCoxTransform, StandardScalerTransform
from statsmodels.tsa.seasonal import STL

# 建议在训练脚本入口处限制底层 BLAS 线程，避免和批内并行抢核
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# 复用的线程池（每个 DataLoader worker 会各自持有一份）
_STL_POOL = None


def _get_pool(n_workers: int):
    global _STL_POOL
    if _STL_POOL is None:
        _STL_POOL = ThreadPoolExecutor(max_workers=n_workers)
    return _STL_POOL


def _stl_one_series(x_ctx_fp64: np.ndarray, period: int = 24, robust: bool = False):
    """
    在 Box-Cox 尺度上对一条 context 段做 STL。
    返回 (S_ctx, T_ctx, R_ctx)，都是 float32，长度 = len(x_ctx_fp64)。
    失败时回退到简易移动平均趋势。
    """
    try:
        stl = STL(x_ctx_fp64, period=period, robust=robust)  # 可按需加 seasonal/trend
        res = stl.fit()
        S = res.seasonal.astype(np.float32)
        T = res.trend.astype(np.float32)
    except Exception:
        # 回退：奇数窗口的移动平均作为趋势；季节取残差
        win = max(3, (period // 2) * 2 + 1)  # 约等于 period/2，保证奇数
        kern = np.ones(win, dtype=np.float64) / win
        T = np.convolve(x_ctx_fp64, kern, mode="same").astype(np.float32)
        S = x_ctx_fp64.astype(np.float32) - T
    R = x_ctx_fp64.astype(np.float32) - S - T
    return S, T, R


# ════════════════════════════════════════════════════════════
# 1. collate_fn   (ctx/pred 上下限来自 CLI)
# ════════════════════════════════════════════════════════════

CTX_MIN, CTX_MAX = 24, 336
PRED_MIN, PRED_MAX = 1, 168

STL_ENABLED = False


def mixed_len_collate(
    batch,
    stl_period: int = 24,
    stl_robust: bool = True,
    max_workers: int | None = None,
):
    """
    - 随机 ctx_len/pred_len，堆叠基础特征；
    - 在 Box-Cox 尺度上对 load[:ctx_len] 做 STL，得到 stl_S/T/R；
    - stl_* 的预测段填 0（不泄漏）。
    """
    ctx_len = random.randint(CTX_MIN, CTX_MAX)
    pred_len = random.randint(PRED_MIN, PRED_MAX)
    seq_len = ctx_len + pred_len

    # 裁剪到随机序列长度
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

    # 先把你原有的字段堆好
    collated = {
        "latitude": _stack("latitude", torch.float32),
        "longitude": _stack("longitude", torch.float32),
        "day_of_year": _stack("day_of_year", torch.float32),
        "day_of_week": _stack("day_of_week", torch.float32),
        "hour_of_day": _stack("hour_of_day", torch.float32),
        "building_type": _stack("building_type", torch.long),
        "load": _stack("load", torch.float32),  # 注意：这里已是 Box-Cox 尺度
    }
    for k in batch[0]:
        v0 = batch[0][k]
        if k not in collated:
            if isinstance(v0, (torch.Tensor, np.ndarray)):
                collated[k] = _stack(k, torch.float32)
            elif isinstance(v0, (int, float)):
                collated[k] = torch.tensor([s[k] for s in batch], dtype=torch.float32)
            # 如果是 str，就直接跳过，或者保存在一个 list 中
            elif isinstance(v0, str):
                # collated["meta"][k] = [s.get(k, "") for s in batch]
                pass
    if STL_ENABLED:

        # ─────────────────────────────────────────────────────────────
        # 在 collate_fn 里做 STL（仅 context 段），跨样本并行
        # ─────────────────────────────────────────────────────────────
        load = collated["load"]  # [B, seq_len, 1], float32, CPU
        B = load.shape[0]

        # 取 context 段并转 numpy float64（statsmodels 偏好）
        X_ctx = [
            load[i, :ctx_len, 0].numpy().astype(np.float64, copy=False)
            for i in range(B)
        ]

        S = torch.zeros(B, seq_len, 1, dtype=torch.float32)
        T = torch.zeros(B, seq_len, 1, dtype=torch.float32)
        R = torch.zeros(B, seq_len, 1, dtype=torch.float32)

        n_workers = max_workers or min(os.cpu_count() or 1, B)
        pool = _get_pool(n_workers)

        futs = [pool.submit(_stl_one_series, x, stl_period, stl_robust) for x in X_ctx]

        for i, f in enumerate(futs):
            s_ctx, t_ctx, r_ctx = f.result()
            S[i, :ctx_len, 0] = torch.from_numpy(s_ctx)
            T[i, :ctx_len, 0] = torch.from_numpy(t_ctx)
            R[i, :ctx_len, 0] = torch.from_numpy(r_ctx)

        collated["stl_S"] = S  # Box-Cox 尺度
        collated["stl_T"] = T
        collated["stl_R"] = R

    collated["context_len"] = torch.tensor(ctx_len)
    collated["pred_len"] = torch.tensor(pred_len)
    return collated


# === NEW
def fixed_len_collate(
    batch,
    ctx_len: int,
    pred_len: int,
    *,
    stl_period: int = 24,
    stl_robust: bool = True,
    max_workers: int | None = None,
):
    """
    把样本裁剪到固定 (ctx_len + pred_len)，并在 Box-Cox 尺度对 load[:ctx_len] 做 STL。
    生成 stl_S/T/R（预测段填 0）。
    """
    seq_len = ctx_len + pred_len

    # 裁剪
    for s in batch:
        for k, v in s.items():
            if isinstance(v, (torch.Tensor, np.ndarray)) and getattr(v, "ndim", 0) >= 1:
                if v.shape[0] < seq_len:
                    raise ValueError(f"val 样本过短: {v.shape[0]} < {seq_len}")
                s[k] = v[:seq_len]

    def _stack(key, dtype=None):
        ts = [torch.as_tensor(s[key]) for s in batch]
        out = torch.stack(ts)
        return out if dtype is None else out.to(dtype)

    # 先堆基础字段
    merged = {
        "latitude": _stack("latitude", torch.float32),
        "longitude": _stack("longitude", torch.float32),
        "day_of_year": _stack("day_of_year", torch.float32),
        "day_of_week": _stack("day_of_week", torch.float32),
        "hour_of_day": _stack("hour_of_day", torch.float32),
        "building_type": _stack("building_type", torch.long),
        "load": _stack("load", torch.float32),  # 注意：应为 Box-Cox 尺度
    }
    # 任何额外键（例如天气列）也一并堆起来
    for k in batch[0]:
        v0 = batch[0][k]
        if k not in merged:
            if isinstance(v0, (torch.Tensor, np.ndarray)):
                merged[k] = _stack(k, torch.float32)
            elif isinstance(v0, (int, float)):
                merged[k] = torch.tensor([s[k] for s in batch], dtype=torch.float32)
            # 如果是 str，就直接跳过，或者保存在一个 list 中
            elif isinstance(v0, str):
                # merged["meta"][k] = [s.get(k, "") for s in batch]
                pass

    if STL_ENABLED:
        # ── STL 分解：仅 context 段 ─────────────────────────────────────
        load = merged["load"]  # [B, seq_len, 1]
        B = load.shape[0]

        X_ctx = [
            load[i, :ctx_len, 0].numpy().astype(np.float64, copy=False)
            for i in range(B)
        ]
        S = torch.zeros(B, seq_len, 1, dtype=torch.float32)
        T = torch.zeros(B, seq_len, 1, dtype=torch.float32)
        R = torch.zeros(B, seq_len, 1, dtype=torch.float32)

        n_workers = max_workers or min(os.cpu_count() or 1, B)
        pool = _get_pool(n_workers)
        futs = [pool.submit(_stl_one_series, x, stl_period, stl_robust) for x in X_ctx]
        for i, f in enumerate(futs):
            s_ctx, t_ctx, r_ctx = f.result()
            S[i, :ctx_len, 0] = torch.from_numpy(s_ctx)
            T[i, :ctx_len, 0] = torch.from_numpy(t_ctx)
            R[i, :ctx_len, 0] = torch.from_numpy(r_ctx)

        merged["stl_S"] = S  # Box-Cox 尺度
        merged["stl_T"] = T
        merged["stl_R"] = R
    merged["context_len"] = torch.tensor(ctx_len)
    merged["pred_len"] = torch.tensor(pred_len)
    return merged


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


@torch.no_grad()
def evaluate_all(model, loaders_dict, transform, inverse, device):
    """
    返回 dict { (ctx,pred): (rmse, mae) }
    """
    model.eval()
    out = {}

    for (c, p), loader in loaders_dict.items():
        i = 0
        se, ae, n = 0.0, 0.0, 0
        for batch in loader:
            i += 1
            for k in batch:
                batch[k] = batch[k].to(device)
            batch["load"] = transform(batch["load"])

            preds = model(batch, context_len=c, pred_len=p)
            tgt = batch["load"][:, c : c + p]

            err = inverse(preds) - inverse(tgt)
            se += (err**2).sum().item()
            ae += err.abs().sum().item()
            n += err.numel()
            # if i >= 100:
            #     break
        out[(c, p)] = (math.sqrt(se / n), ae / n)

    model.train()
    return out


# ════════════════════════════════════════════════════════════
# 3. 主函数
# ════════════════════════════════════════════════════════════
SCRIPT_PATH = Path(os.path.realpath(__file__)).parent
# === NEW: 固定验证窗口 (ctx_len, pred_len) 组合 ===
VAL_COMBOS = [
    (168, 24),
    (224, 48),
    (280, 96),
    (336, 168),
]


def main(args, model_args):
    global CTX_MIN, CTX_MAX, PRED_MIN, PRED_MAX, STL_ENABLED
    CTX_MIN, CTX_MAX = args.ctx_min, args.ctx_max
    PRED_MIN, PRED_MAX = args.pred_min, args.pred_max
    STL_ENABLED = args.stl_enabled

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
            resume="allow" if not args.disable_wandb else "never",
            id=args.swanlab_id,
        )

    # ════════════════════════════════════════════════
    # 4. Model & Dataset
    # ════════════════════════════════════════════════
    model, loss_fn, _ = model_factory(args.model, model_args)
    model = model.to(local_rank)

    # 打印模型参数
    if args.rank == 0:
        count_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"[rank 0] Model: {args.model}  "
            f"ctx_len={model_args['max_context_len']}  "
            f"pred_len={model_args['max_pred_len']}  "
            f"params={count_parameters:,}  "
            f"continuous_loads={model.continuous_loads}  "
            f"continuous_head={model.continuous_head}",
            flush=True,
        )

    train_ds, val_ds = build_datasets(
        context_len=model_args["max_context_len"],
        pred_len=model_args["max_pred_len"],
        apply_scaler_transform=args.apply_scaler_transform,
        split="train",
        # oov_val=Path("./oov_val.txt"),
        # oov_test=Path("./oov_test.txt"),
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
    # 6-0. 可选：从检查点恢复
    # ════════════════════════════════════════════════
    best_val_rmse = float("inf")
    tokens_buf = seen_tokens = update_idx = 0

    if args.from_checkpoint:
        ckpt_path = checkpoint_dir / args.from_checkpoint
        if not ckpt_path.exists():
            raise FileNotFoundError(ckpt_path)
        checkpoint = torch.load(ckpt_path, map_location=f"cuda:{local_rank}")
        model.load_state_dict(checkpoint["model"], strict=False)
        print("Success load from checkpoint")

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
            try:
                progress = min(1.0, float(seen_tokens) / max(1, args.train_tokens))
                loss = model.module.loss(preds, tgt, progress=progress)
            except Exception as e:
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

                    # 计算四种长度的 RMSE 均值作为主要指标
                    rmse_values = []

                    for combo in VAL_COMBOS:
                        if combo in metrics:
                            rmse_values.append(metrics[combo][0])
                        else:
                            print(f"警告: 组合 {combo} 不在 metrics 中")
                            # 如果某个组合缺失，可以用0或其他默认值，但最好确保它存在
                            rmse_values.append(0.0)

                    # 计算平均值（只计算存在的值）
                    valid_rmse_values = [rmse for rmse in rmse_values if rmse > 0]
                    if valid_rmse_values:
                        avg_rmse = sum(valid_rmse_values) / len(valid_rmse_values)
                    else:
                        avg_rmse = float("inf")  # 如果没有有效的RMSE值，设为无穷大

                    print(f"[rank 0] 四种组合平均RMSE: {avg_rmse:.4f}")

                    # 用平均 RMSE 作为最佳模型判断标准
                    if avg_rmse < best_val_rmse - 1e-9:
                        best_val_rmse = avg_rmse
                        utils.save_model_checkpoint(
                            model,
                            optimizer,
                            scheduler,
                            update_idx,
                            best_val_rmse,
                            checkpoint_dir / f"{ckpt_name}_best_val.pt",
                        )
                        print(f"[rank 0] 保存最佳模型，平均RMSE: {best_val_rmse:.4f}")
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

        # rank-0 训练日志
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
        # if args.rank == 0 and step % 10000 == 0:
        #     utils.save_model_checkpoint(
        #         model,
        #         optimizer,
        #         scheduler,
        #         update_idx,
        #         best_val_rmse,
        #         checkpoint_dir / f"{ckpt_name}_iter{step}.pt",
        #     )
        #     print(f"[rank 0] 已保存迭代 {step} 的检查点", flush=True)

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
        "--from_checkpoint",
        type=str,
        default="",
        help="从权重预加载",
    )
    p.add_argument(
        "--swanlab_id",
        type=str,
        default=None,
        help="SwanLab 实验 ID（用于实验追踪）",
    )
    # STL相关
    p.add_argument("--stl_enabled", action="store_true", help="是否启用STL拆分")

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
