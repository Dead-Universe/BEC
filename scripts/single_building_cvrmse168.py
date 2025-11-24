# -*- coding: utf-8 -*-
"""
single_building_cvrmse168.py

目标：仅计算“单栋楼”的 CVRMSE@168（最终一个均值）。
- 支持从 LHS CSV 注入 CLRS 超参（--clrs_plan_csv, --lhs_run）
- 支持可选按楼短程微调（--do_finetune）
- 评估：对该楼的数据集窗口按 pred_len=168 做预测，将预测与目标都反变换到原始量纲后
        仅使用“有效目标点(valid=True)”参与累计与分母均值，输出 CVRMSE（百分比）

输出：
- 控制台打印：building_id, CVRMSE_168
- CSV：{output_dir}/{dataset_name}_{building_id}_CVRMSE168.csv
"""

import argparse
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import random
import numpy as np

import torch
from torch.utils.data import ConcatDataset, DataLoader, random_split

from importlib import resources as ir
import tomli
import pandas as pd

from buildings_bench.models import model_factory
from buildings_bench.data import load_torch_dataset
from buildings_bench.transforms import BoxCoxTransform


# -----------------------
# 基础工具
# -----------------------
def set_global_seed(seed: int):
    """统一设置 Python / NumPy / PyTorch 的随机种子，以提高复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 尽可能让 CUDA 算子确定性（有些算子会报错，可视情况关掉）
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # 这句可能会因为某些非确定性算子报错，如果遇到可以注释掉
        # torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def load_model_config(model_name: str) -> dict:
    try:
        with ir.files("buildings_bench.configs").joinpath(f"{model_name}.toml").open(
            "rb"
        ) as f:
            toml_cfg = tomli.load(f)
    except FileNotFoundError:
        cfg_dir = Path(__file__).parent / "buildings_bench" / "configs"
        toml_file = cfg_dir / f"{model_name}.toml"
        with toml_file.open("rb") as f:
            toml_cfg = tomli.load(f)
    return toml_cfg["model"]


def make_safe_transformers(transform, inverse_transform):
    def safe_transform(x: torch.Tensor) -> torch.Tensor:
        if transform is None:
            return x
        x_np = x.detach().cpu().numpy()
        out = transform(x_np)
        return torch.from_numpy(out).to(x.device)

    def safe_inverse_transform(x: torch.Tensor) -> torch.Tensor:
        if inverse_transform is None:
            return x
        x_np = x.detach().cpu().numpy()
        out = inverse_transform(x_np)
        return torch.from_numpy(out).to(x.device)

    return safe_transform, safe_inverse_transform


def fixed_len_collate(batch, ctx_len: int, pred_len: int) -> Dict[str, torch.Tensor]:
    """
    组装 batch：
    - 必带: load
    - 可选: valid（如样本无 valid 字段，则用 isfinite(load) 近似生成）
    - 其他：context_len/pred_len/building_id
    """
    seq_len = ctx_len + pred_len

    def _stack(key, dtype=None):
        ts = [torch.as_tensor(s[key][:seq_len]) for s in batch]
        out = torch.stack(ts)
        return out if dtype is None else out.to(dtype)

    # load 保证为 float32，形状 [B, T, ...]
    load = _stack("load", torch.float32)
    if load.ndim == 2:  # [B, T] -> [B, T, 1]
        load = load.unsqueeze(-1)

    # valid：优先用样本提供；否则用 isfinite(load) 近似
    if "valid" in batch[0]:
        valid_list = []
        for s in batch:
            v = torch.as_tensor(s["valid"][:seq_len])
            if v.ndim == 2:
                v = v.squeeze(-1)
            valid_list.append(v.bool())
        valid = torch.stack(valid_list)  # [B, T]
    else:
        valid = torch.isfinite(load.squeeze(-1))  # [B, T]

    merged: Dict[str, torch.Tensor] = {
        "load": load,  # [B, T, 1]
        "valid": valid.to(torch.bool),  # [B, T]
        "context_len": torch.tensor(ctx_len),
        "pred_len": torch.tensor(pred_len),
    }
    # building_id 仅保留字符串列表，不参与张量运算
    merged["building_id"] = [s.get("building_id", "") for s in batch]
    return merged


def build_dataloaders_for_building(
    dataset_name: str,
    single_building: str,
    ctx_len: int,
    pred_len: int,
    batch_size: int,
    seed: int,
    num_workers: int,
    dataset_path: Optional[Path] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    gen = load_torch_dataset(
        dataset_name, dataset_path=dataset_path, context_len=ctx_len, pred_len=pred_len
    )
    ds_list = []
    for bid, bldg_ds in gen:
        if bid == single_building:
            ds_list = [bldg_ds]
            break
    if len(ds_list) == 0:
        raise RuntimeError(
            f"未找到 building_id={single_building} 的数据（dataset={dataset_name}）"
        )
    full_ds = ConcatDataset(ds_list)

    collate_fn = lambda batch: fixed_len_collate(batch, ctx_len, pred_len)
    N = len(full_ds)
    n_train = int(0.7 * N)
    n_val = int(0.1 * N)
    n_test = N - n_train - n_val

    print(
        f"[Data] building={single_building} | total windows={N} "
        f"(train={n_train}, val={n_val}, test={n_test}), "
        f"context_len={ctx_len}, pred_len={pred_len}, batch_size={batch_size}"
    )

    # 1) 划分 train/val/test，固定 seed
    g_split = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        full_ds, [n_train, n_val, n_test], generator=g_split
    )

    # 2) DataLoader 的 shuffle 也用固定 seed
    g_loader = torch.Generator().manual_seed(seed)

    worker_kwargs = dict(num_workers=max(0, int(num_workers)), pin_memory=True)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
        generator=g_loader,  # ⭐ 关键：固定 DataLoader shuffle 的随机性
        **worker_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
        **worker_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
        **worker_kwargs,
    )
    return train_loader, val_loader, test_loader


def save_model_checkpoint(
    model,
    optimizer,
    scheduler,
    step,
    best_val_loss,
    path,
    extra: Optional[Dict[str, Any]] = None,
):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else {},
        "scheduler": scheduler.state_dict() if scheduler is not None else {},
        "step": int(step),
        "best_val_loss": float(best_val_loss if math.isfinite(best_val_loss) else 1e30),
        "extra": extra or {},
    }
    torch.save(checkpoint, str(path))


def load_model_checkpoint(path: str | Path, model: torch.nn.Module) -> None:
    ckpt = torch.load(str(path), map_location="cpu")
    if "model" in ckpt:
        sd = ckpt["model"]
    else:
        sd = ckpt
    # 去掉/加上 module. 前缀的简单自适配
    model_has_module = next(iter(model.state_dict().keys()), "").startswith("module.")
    ckpt_has_module = any(k.startswith("module.") for k in sd.keys())
    if model_has_module and not ckpt_has_module:
        sd = {f"module.{k}": v for k, v in sd.items()}
    if (not model_has_module) and ckpt_has_module:
        sd = {k[len("module.") :]: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)


def inject_clrs_from_lhs_if_any(
    model, clrs_plan_csv: Optional[Path], lhs_run: Optional[int]
) -> Optional[pd.Series]:
    if clrs_plan_csv is None:
        return None
    plan = pd.read_csv(clrs_plan_csv)
    idx = 0 if lhs_run is None else lhs_run
    if idx < 0 or idx >= len(plan):
        raise IndexError(f"--lhs_run 超出范围：{idx} / {len(plan)}")
    row = plan.iloc[idx]

    phi1 = row.get("phi1", None)
    phi2 = row.get("phi2", None)
    if (phi1 is None) and ("fixed_phi1" in row):
        phi1 = row["fixed_phi1"]
    if (phi2 is None) and ("fixed_phi2" in row):
        phi2 = row["fixed_phi2"]

    if hasattr(model, "update_clrs_config"):
        model.update_clrs_config(
            rho_star=row.get("rho_star", None),
            k_fb=row.get("k_fb", None),
            c_pulse=row.get("c_pulse", None),
            tau_hi=row.get("tau_hi", None),
            sigma_hi=row.get("sigma_hi", None),
            phi1=phi1,
            phi2=phi2,
        )
        print("[CLRS] LHS 注入完成。")
    else:
        print("⚠️ 模型没有 update_clrs_config()，跳过 CLRS 超参注入。")
    return row


@torch.no_grad()
def eval_cvrmse_168(
    model, loader: DataLoader, device, transform, inverse_transform
) -> float:
    """
    在 loader 上评估：将预测与目标都反变换到原始标度后，
    仅用“有效目标点(valid=True 且 isfinite)”参与累计；
    分母 global_mean 也仅由这些有效目标点决定。
    """
    model.eval()
    se_sum, tgt_sum, n_tok = 0.0, 0.0, 0
    for batch in loader:
        # to device
        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        ctx_len = int(batch.pop("context_len"))
        pred_len = int(batch.pop("pred_len"))

        # 原始 load & valid
        load_raw = batch["load"]  # [B, T, 1]
        if load_raw.ndim == 2:
            load_raw = load_raw.unsqueeze(-1)
        valid = batch.get("valid", torch.isfinite(load_raw.squeeze(-1)))  # [B, T]
        # 目标段掩码（仅目标期, 长度 pred_len）
        valid_pred = valid[:, ctx_len : ctx_len + pred_len]  # [B, pred]
        # 安全起见，与反变换后的有限性再相与
        batch["load"] = transform(load_raw)

        preds = model(
            batch, context_len=ctx_len, pred_len=pred_len
        )  # [B, pred, 1] or [B, pred]
        if preds.ndim == 2:
            preds = preds.unsqueeze(-1)

        tgt = batch["load"][:, ctx_len : ctx_len + pred_len, :]  # 仍在“变换域”

        # 反变换回原始量纲
        preds_raw = inverse_transform(preds).squeeze(-1)  # [B, pred]
        tgt_raw = inverse_transform(tgt).squeeze(-1)  # [B, pred]

        # 最终有效掩码：valid_pred 且 isfinite
        m = valid_pred & torch.isfinite(tgt_raw) & torch.isfinite(preds_raw)
        if m.any():
            err = preds_raw - tgt_raw
            se_sum += (err[m] ** 2).sum().item()
            tgt_sum += tgt_raw[m].sum().item()
            n_tok += int(m.sum().item())

    if n_tok == 0:
        return float("nan")
    rmse = math.sqrt(se_sum / n_tok)
    global_mean = tgt_sum / n_tok
    if not math.isfinite(global_mean) or global_mean == 0.0:
        return float("nan")
    cvrmse = 100.0 * rmse / global_mean
    return float(cvrmse)


def finetune_one_building(
    model,
    loss_fn,
    device,
    transform,
    ctx_len: int,
    pred_len: int,
    train_loader,
    val_loader,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    clip_norm: float = 1.0,
):
    import inspect

    params = model.unfreeze_and_get_parameters_for_finetuning()
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    criterion = getattr(model, "loss", None) or loss_fn

    total_steps = max(1, epochs * len(train_loader))
    global_step = 0

    best_val, no_improve = float("inf"), 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            for k, v in list(batch.items()):
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            ctx_len_b = int(batch.pop("context_len"))
            pred_len_b = int(batch.pop("pred_len"))
            batch["load"] = transform(batch["load"])

            preds = model(batch, context_len=ctx_len_b, pred_len=pred_len_b)
            tgt = batch["load"][:, ctx_len_b : ctx_len_b + pred_len_b]

            progress = global_step / total_steps
            if (
                inspect.signature(criterion).parameters.get("progress", None)
                is not None
            ):
                loss = criterion(preds, tgt, progress=progress)
            else:
                loss = criterion(preds, tgt)

            optimizer.zero_grad()
            loss.backward()
            for g in optimizer.param_groups:
                torch.nn.utils.clip_grad_norm_(g["params"], clip_norm)
            optimizer.step()
            global_step += 1

        scheduler.step()
        # 简易早停：用 val 上的 L2 作 proxy
        with torch.no_grad():
            model.eval()
            se, n = 0.0, 0
            for batch in val_loader:
                for k, v in list(batch.items()):
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)
                ctx_len_b = int(batch.pop("context_len"))
                pred_len_b = int(batch.pop("pred_len"))
                batch["load"] = transform(batch["load"])
                preds = model(batch, context_len=ctx_len_b, pred_len=pred_len_b)
                tgt = batch["load"][:, ctx_len_b : ctx_len_b + pred_len_b]
                err = preds - tgt
                se += (err**2).sum().item()
                n += err.numel()
            val_rmse = math.sqrt(se / n) if n > 0 else float("inf")

        print(f"[Finetune] epoch={epoch:03d} val_RMSE(norm)={val_rmse:.6f}")
        if val_rmse + 1e-9 < best_val:
            best_val = val_rmse
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            no_improve = 0
        else:
            no_improve += 1

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
    model.train()


# -----------------------
# 主程序
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    # 模型
    ap.add_argument(
        "--baseline_model_name",
        type=str,
        default="BuildMoE-top-k-2-without-shared-export",
    )
    ap.add_argument("--baseline_ckpt", type=Path, required=True)
    # 数据/楼
    ap.add_argument(
        "--dataset_name", type=str, default="cofactor", help="如 cofactor / university"
    )
    ap.add_argument("--building_id", type=str, required=True)
    ap.add_argument("--context_len", type=int, default=168)
    ap.add_argument("--pred_len", type=int, default=168)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--random_seed", type=int, default=42)
    # LHS 注入
    ap.add_argument("--clrs_plan_csv", type=Path, default=None)
    ap.add_argument("--lhs_run", type=int, default=None)
    # CLRS / LBL 消融设置
    ap.add_argument(
        "--no_clrs",
        action="store_true",
        help="禁用 CLRS 路由调度（w/o CLRS）。",
    )
    ap.add_argument(
        "--no_clrs_stage",
        action="store_true",
        help="关闭 CLRS 的阶段式温度/噪声调度（固定 tau/sigma）。",
    )
    ap.add_argument(
        "--no_clrs_pulse",
        action="store_true",
        help="关闭 CLRS 的 pulse 机制（不再对 collapse 层做脉冲升温）。",
    )
    ap.add_argument(
        "--no_clrs_revive",
        action="store_true",
        help="关闭 CLRS 的 revive_dead_experts（不再对冷门专家加 bias）。",
    )
    ap.add_argument(
        "--no_lbl",
        action="store_true",
        help="关闭 MoE LBL 辅助损失（lambda_layer=lambda_global=0）。",
    )
    # 可选微调
    ap.add_argument("--do_finetune", action="store_true")
    ap.add_argument("--ft_epochs", type=int, default=10)
    ap.add_argument("--ft_patience", type=int, default=0)
    ap.add_argument("--ft_batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    # 设备/输出
    ap.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    ap.add_argument("--output_dir", type=Path, default=Path("./single_eval"))
    args = ap.parse_args()

    set_global_seed(args.random_seed)

    # 变换
    boxcox = BoxCoxTransform()
    tpath = Path(os.environ["BUILDINGS_BENCH"]) / "metadata" / "transforms"
    boxcox.load(tpath)
    transform, inverse_transform = make_safe_transformers(
        boxcox.transform, boxcox.undo_transform
    )

    # 模型
    model_args = load_model_config(args.baseline_model_name)
    model_args["context_len"] = args.context_len
    model_args["pred_len"] = args.pred_len
    model, loss_fn, _ = model_factory(args.baseline_model_name, model_args)
    model = model.to(args.device).eval()
    # baseline 权重
    model.load_from_checkpoint(str(args.baseline_ckpt))

    # LHS 注入（可选）
    inject_clrs_from_lhs_if_any(model, args.clrs_plan_csv, args.lhs_run)

    # CLRS / LBL 消融配置（若模型支持）
    if hasattr(model, "configure_clrs_ablation"):
        model.configure_clrs_ablation(
            use_clrs=False if args.no_clrs else None,
            disable_stage_schedule=args.no_clrs_stage,
            disable_pulse=args.no_clrs_pulse,
            disable_revive=args.no_clrs_revive,
            disable_lbl=args.no_lbl,
        )
    else:
        print("⚠️ 模型没有 configure_clrs_ablation()，跳过消融配置。")

    # 数据（仅该楼）
    train_loader, val_loader, test_loader = build_dataloaders_for_building(
        dataset_name=args.dataset_name,
        single_building=args.building_id,
        ctx_len=args.context_len,
        pred_len=args.pred_len,
        batch_size=args.ft_batch_size if args.do_finetune else args.batch_size,
        seed=args.random_seed,
        num_workers=args.num_workers,
        dataset_path=(
            Path(os.environ.get("BUILDINGS_BENCH", ""))
            if os.environ.get("BUILDINGS_BENCH")
            else None
        ),
    )

    # 可选微调
    if args.do_finetune:
        print(f"🛠 Finetune {args.building_id} ...")
        finetune_one_building(
            model=model,
            loss_fn=loss_fn,
            device=args.device,
            transform=transform,
            ctx_len=args.context_len,
            pred_len=args.pred_len,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.ft_epochs,
            patience=args.ft_patience,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        model.eval()

    # 评估（仅输出 CVRMSE@168 单个均值；使用 test split 更贴近泛化）
    print("🔎 Evaluating CVRMSE@168 (test split, valid-only) ...")
    cvrmse = eval_cvrmse_168(
        model, test_loader, args.device, transform, inverse_transform
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.output_dir / f"{args.dataset_name}_{args.building_id}_CVRMSE168.csv"
    pd.DataFrame(
        [
            {
                "dataset": args.dataset_name,
                "building_id": args.building_id,
                "pred_len": args.pred_len,
                "CVRMSE_168": round(cvrmse, 4),
            }
        ]
    ).to_csv(out_csv, index=False)

    print(f"\n✅ building_id={args.building_id} | CVRMSE@168 = {cvrmse:.4f}%")
    print(f"📄 已写出: {out_csv}")


if __name__ == "__main__":
    main()
