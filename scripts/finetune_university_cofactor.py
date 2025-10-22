#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
finetune_university_cofactor.py

微调 BuildMoE / DLinearRegression（或其他基于 BaseModel 的模型）
在 BuildingsBench 的 'university' / 'cofactor' 数据集上，并使用
DatasetMetricsManager + aggregate 在测试集评估、导出 CSV、打印聚合结果。

修正：
- DataLoader 不再应用 Box-Cox 转换
- 训练/验证时手动 transform 到 Box-Cox 空间
- 测试/评估时 inverse_transform 回到实量纲
"""

import argparse
import math
import os
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset

# ---- tomli/tomllib 兼容 ----
try:
    import tomllib as tomli  # py3.11+
except ModuleNotFoundError:
    import tomli  # py3.10-

import importlib.resources as ir

# BuildingsBench 组件
from buildings_bench import utils
from buildings_bench.data import load_torch_dataset
from buildings_bench.models import model_factory
from buildings_bench.transforms import BoxCoxTransform
from buildings_bench.evaluation import scoring_rule_factory, aggregate
from buildings_bench.evaluation.managers import BuildingTypes, DatasetMetricsManager


# =========================
#  Checkpoint I/O（健壮版）
# =========================
def _adapt_module_prefix(
    model: torch.nn.Module, state_dict: Dict[str, Any]
) -> Dict[str, Any]:
    model_has_module = next(iter(model.state_dict().keys()), "").startswith("module.")
    ckpt_has_module = any(k.startswith("module.") for k in state_dict.keys())
    if model_has_module and not ckpt_has_module:
        return {f"module.{k}": v for k, v in state_dict.items()}
    if (not model_has_module) and ckpt_has_module:
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict


def save_model_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    step: int,
    best_val_loss: float,
    path: str | Path,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    import math as _m

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else {},
        "scheduler": scheduler.state_dict() if scheduler is not None else {},
        "step": int(step),
        "best_val_loss": float(best_val_loss if _m.isfinite(best_val_loss) else 1e30),
        "extra": extra or {},
    }
    torch.save(checkpoint, str(path))


def load_model_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    *,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    local_rank: Optional[int] = None,
    strict: bool = False,
    model_key_candidates: Tuple[str, ...] = ("model", "model_state", "state_dict"),
) -> Tuple[int, float]:
    if isinstance(path, Path):
        path = str(path)

    if device.startswith("cuda") and torch.cuda.is_available():
        map_loc = f"cuda:{local_rank if local_rank is not None else torch.cuda.current_device()}"
    else:
        map_loc = "cpu"

    ckpt = torch.load(path, map_location=map_loc)

    # 模型权重
    sd = None
    if isinstance(ckpt, dict):
        for k in model_key_candidates:
            if k in ckpt and isinstance(ckpt[k], dict):
                sd = ckpt[k]
                break
        if sd is None:
            if ckpt and all(hasattr(v, "shape") for v in ckpt.values()):
                sd = ckpt
    if sd is None:
        raise ValueError(f"Cannot find model state_dict in checkpoint: {path}")

    sd = _adapt_module_prefix(model, sd)
    model.load_state_dict(sd, strict=strict)

    # 可选恢复优化器/调度器
    if optimizer is not None:
        opt_sd = ckpt.get("optimizer", None) if isinstance(ckpt, dict) else None
        if isinstance(opt_sd, dict) and len(opt_sd) > 0:
            try:
                optimizer.load_state_dict(opt_sd)
            except Exception as e:
                print(f"[load_model_checkpoint] optimizer state not loaded: {e}")

    if scheduler is not None:
        sch_sd = ckpt.get("scheduler", None) if isinstance(ckpt, dict) else None
        if isinstance(sch_sd, dict) and len(sch_sd) > 0:
            try:
                scheduler.load_state_dict(sch_sd)
            except Exception as e:
                print(f"[load_model_checkpoint] scheduler state not loaded: {e}")

    step = int(ckpt.get("step", 0)) if isinstance(ckpt, dict) else 0
    best_val_loss = (
        float(ckpt.get("best_val_loss", float("inf")))
        if isinstance(ckpt, dict)
        else float("inf")
    )
    return step, best_val_loss


def load_pretrained_if_any(
    model: torch.nn.Module, pretrained_path: Optional[str | Path], device: str
) -> None:
    if not pretrained_path:
        return
    try:
        if hasattr(model, "load_from_checkpoint"):
            model.load_from_checkpoint(str(pretrained_path))
            print(
                f"[Pretrained] loaded via model.load_from_checkpoint: {pretrained_path}"
            )
        else:
            _ = load_model_checkpoint(
                pretrained_path, model, optimizer=None, scheduler=None, device=device
            )
            print(f"[Pretrained] loaded state_dict from: {pretrained_path}")
    except Exception as e:
        print(f"[Pretrained] failed to load ({pretrained_path}): {e}")


# =========================
#  Data / Collate / Eval
# =========================
def fixed_len_collate(batch, ctx_len: int, pred_len: int) -> Dict[str, Any]:
    seq_len = ctx_len + pred_len

    def _stack(key, dtype=None):
        ts = [torch.as_tensor(s[key][:seq_len]) for s in batch]
        out = torch.stack(ts)
        return out if dtype is None else out.to(dtype)

    merged: Dict[str, Any] = {"load": _stack("load", torch.float32)}
    for k in batch[0]:
        if k != "load":
            v = batch[0][k]
            if isinstance(v, torch.Tensor) or hasattr(v, "shape"):
                merged[k] = _stack(k, torch.float32)
    merged["context_len"] = torch.tensor(ctx_len)
    merged["pred_len"] = torch.tensor(pred_len)
    merged["building_id"] = [s["building_id"] for s in batch]
    return merged


def make_safe_transformers(transform, inverse_transform):
    """包装 transform / inverse_transform，自动处理 torch.Tensor / GPU"""

    def safe_transform(x: torch.Tensor) -> torch.Tensor:
        if transform is None:
            return x
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
            out = transform(x_np)
            return torch.from_numpy(out).to(x.device)
        else:
            return transform(x)

    def safe_inverse_transform(x: torch.Tensor) -> torch.Tensor:
        if inverse_transform is None:
            return x
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
            out = inverse_transform(x_np)
            return torch.from_numpy(out).to(x.device)
        else:
            return inverse_transform(x)

    return safe_transform, safe_inverse_transform


@torch.no_grad()
def quick_eval_rmse_mae(model, loader, device, transform) -> Tuple[float, float]:
    """在 transform 空间上的 RMSE/MAE"""
    model.eval()
    se_sum, ae_sum, n_tok = 0.0, 0.0, 0
    for batch in loader:
        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        ctx_len = int(batch.pop("context_len"))
        pred_len = int(batch.pop("pred_len"))

        batch["load"] = transform(batch["load"])

        preds = model(batch, context_len=ctx_len, pred_len=pred_len)
        tgt = batch["load"][:, ctx_len : ctx_len + pred_len]

        err = preds - tgt
        se_sum += (err**2).sum().item()
        ae_sum += err.abs().sum().item()
        n_tok += err.numel()

    rmse = math.sqrt(se_sum / n_tok)
    mae = ae_sum / n_tok
    model.train()
    return rmse, mae


def build_dataloaders(
    full_ds,
    ctx_len: int,
    pred_len: int,
    batch_size: int,
    seed: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    N = len(full_ds)
    n_train = int(0.7 * N)
    n_val = int(0.1 * N)
    n_test = N - n_train - n_val
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        full_ds, [n_train, n_val, n_test], generator=g
    )

    collate_fn = lambda batch: fixed_len_collate(batch, ctx_len, pred_len)
    worker_kwargs = dict(num_workers=max(0, int(num_workers)), pin_memory=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
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


# =========================
#  正式测试：出表 + 聚合
# =========================
@torch.no_grad()
def test_with_metrics_manager_and_aggregate(
    args,
    model,
    test_loader: DataLoader,
    device: str,
    results_dir: Path,
    transform=lambda x: x,
    inverse_transform=lambda x: x,
):
    """支持多个 horizon（切片）独立评估，结果命名为 model:predlen_h"""

    # === 为每个 horizon 准备一个 metrics_manager ===
    managers: dict[int, DatasetMetricsManager] = {}
    for h in args.eval_horizons:
        if args.ignore_scoring_rules:
            mm = DatasetMetricsManager()
        elif getattr(model, "continuous_loads", True):
            use_crps = getattr(model, "continuous_head", "") == "gaussian_nll"
            mm = DatasetMetricsManager(
                scoring_rule=(scoring_rule_factory("crps") if use_crps else None)
            )
        else:
            mm = DatasetMetricsManager(scoring_rule=scoring_rule_factory("rps"))
        managers[h] = mm

    model.eval()
    dataset_name = args.dataset
    building_id_counter = 0

    for batch in test_loader:
        # building_type 掩码
        if "building_type" in batch and isinstance(
            batch["building_type"], torch.Tensor
        ):
            bt = batch["building_type"]
            while bt.ndim > 1:
                bt = bt.select(-1, 0)
            try:
                commercial_val = getattr(BuildingTypes, "COMMERCIAL_INT", 1)
                building_types_mask = (bt == commercial_val).bool().cpu()
            except Exception:
                building_types_mask = (bt > 0).bool().cpu()
        else:
            building_types_mask = torch.ones(batch["load"].shape[0], dtype=torch.bool)

        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        ctx_len = int(batch.pop("context_len"))
        pred_len = int(batch.pop("pred_len"))

        # === 输入用 transform 过的 load ===
        trans_batch = {k: v for k, v in batch.items()}
        trans_batch["load"] = transform(batch["load"])

        out = model(trans_batch, context_len=ctx_len, pred_len=pred_len)
        if isinstance(out, tuple) and len(out) == 2:
            preds, distribution_params = out
        else:
            preds, distribution_params = out, None

        y_true_full = (
            batch["load"][:, ctx_len : ctx_len + pred_len, 0]
            if batch["load"].ndim == 3
            else batch["load"][:, ctx_len : ctx_len + pred_len]
        )

        preds = preds.detach().float().cpu()
        y_true_full = y_true_full.detach().float().cpu()

        # === 遍历 horizon，切片评估 ===
        for horizon in args.eval_horizons:
            if horizon > preds.shape[1]:
                continue

            preds_h = inverse_transform(preds[:, :horizon])
            y_true_h = y_true_full[:, :horizon]

            if preds_h.ndim == 3 and preds_h.size(-1) == 1:
                preds_h = preds_h.squeeze(-1)
            if y_true_h.ndim == 3 and y_true_h.size(-1) == 1:
                y_true_h = y_true_h.squeeze(-1)

            bsz = preds_h.size(0)
            for i in range(bsz):
                building_id_counter += 1
                if "building_id" in batch:
                    building_name = f"{batch['building_id'][i]}:h{horizon}"
                else:
                    building_name = (
                        f"{dataset_name}-b{building_id_counter:06d}:h{horizon}"
                    )
                mask_i = building_types_mask[i]
                managers[horizon](
                    dataset_name,
                    building_name,
                    y_true_h[i : i + 1],
                    preds_h[i : i + 1],
                    mask_i.unsqueeze(0),
                )

    # === 每个 horizon 独立 summary + aggregate ===
    results_dir.mkdir(parents=True, exist_ok=True)

    for horizon, mm in managers.items():
        # 保持 {model}:{predlen}_{horizon} 命名
        model_tag = f"{args.model}:{args.pred_len}_{horizon}"
        metrics_file = results_dir / f"metrics_{model_tag}.csv"
        scoring_file = results_dir / f"scoring_rule_{model_tag}.csv"

        if mm.scoring_rule:
            metrics_df, scoring_df = mm.summary()
            metrics_df.to_csv(metrics_file, index=False)
            scoring_df.to_csv(scoring_file, index=False)
            print(
                f"[Results:h{horizon}] metrics -> {metrics_file}\n"
                f"[Results:h{horizon}] scoring_rule -> {scoring_file}"
            )
        else:
            metrics_df = mm.summary()
            metrics_df.to_csv(metrics_file, index=False)
            print(f"[Results:h{horizon}] metrics -> {metrics_file}")

        # === aggregate 输出 ===
        metric_names = [m.name for m in mm.metrics_list]
        if mm.scoring_rule:
            metric_names += [mm.scoring_rule.name]

        print(f"\nAggregates (real, {model_tag})")
        real_med = aggregate.return_aggregate(
            model_list=[model_tag],
            results_dir=str(results_dir),
            experiment=args.experiment_tag,
            metrics=metric_names,
            aggregate="median",
            exclude_simulated=True,
        )
        aggregate.pretty_print(real_med, aggregate="median")

        real_mean = aggregate.return_aggregate(
            model_list=[model_tag],
            results_dir=str(results_dir),
            experiment=args.experiment_tag,
            metrics=metric_names,
            aggregate="mean",
            exclude_simulated=True,
        )
        ans = aggregate.pretty_print(real_mean, aggregate="mean")

        # 追加写入文件
        log_file = results_dir / "aggregates_summary.txt"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(ans + "\n\n")


# =========================
#  Cofactor 子集映射
# =========================
cofactor_type: dict[str, list[str]] = {
    "Kindergarten": [
        "building6396",
        "building6398",
        "building6402",
        "building6405",
        "building6406",
        "building6407",
        "building6409",
        "building6415",
        "building6419",
        "building6421",
        "building6422",
        "building6425",
        "building6426",
        "building6428",
        "building6429",
        "building6433",
        "building6434",
        "building6437",
        "building6439",
        "building6443",
    ],
    "School": [
        "building6397",
        "building6400",
        "building6404",
        "building6408",
        "building6413",
        "building6414",
        "building6416",
        "building6418",
        "building6420",
        "building6424",
        "building6431",
        "building6432",
        "building6438",
        "building6440",
        "building6444",
        "building6445",
    ],
    "NursingHome": [
        "building6399",
        "building6410",
        "building6412",
        "building6417",
        "building6423",
        "building6436",
        "building6442",
    ],
    "Office": ["building6411", "building6441"],
}


# =========================
#  主流程：训练 + 测试
# =========================
def main(args, model_args: Dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    utils.set_seed(args.random_seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    _bb = os.environ.get("BUILDINGS_BENCH")
    dataset_path = Path(_bb) if _bb else None

    if args.dataset.startswith("cofactor:"):
        dataset_name = "cofactor"
    else:
        dataset_name = args.dataset
    gen = load_torch_dataset(
        dataset_name,
        dataset_path=dataset_path,
        context_len=args.ctx_len,
        pred_len=args.pred_len,
    )
    ds_list = []
    for bid, bldg_ds in gen:
        if args.dataset.startswith("cofactor:"):
            subtype = args.dataset.split(":", 1)[1]
            allow = set(cofactor_type.get(subtype, []))
            if bid not in allow:
                continue
        ds_list.append(bldg_ds)
    if len(ds_list) == 0:
        raise RuntimeError("空数据集：未收集到任何楼栋子数据集")
    full_ds = ConcatDataset(ds_list)

    sample0 = full_ds[0]
    assert "load" in sample0, "样本缺少 'load' 键"
    assert len(sample0["load"]) >= args.ctx_len + args.pred_len, "序列长度不足"

    train_loader, val_loader, test_loader = build_dataloaders(
        full_ds,
        args.ctx_len,
        args.pred_len,
        args.batch_size,
        args.random_seed,
        args.num_workers,
    )

    # === 准备 BoxCox transform/inverse ===
    transform = lambda x: x
    inverse_transform = lambda x: x
    if args.apply_scaler_transform == "boxcox":
        try:
            tpath = Path(os.getenv("BUILDINGS_BENCH", "")) / "metadata" / "transforms"
            scaler = BoxCoxTransform()
            scaler.load(tpath)
            transform = scaler.transform
            inverse_transform = scaler.undo_transform
            transform, inverse_transform = make_safe_transformers(
                transform, inverse_transform
            )
            print(">> BoxCoxTransform 已加载")
        except Exception as e:
            print(f"!! BoxCoxTransform 加载失败：{e}")

    # 构建模型
    model, loss_fn, _ = model_factory(args.model, model_args)
    model = model.to(device)
    load_pretrained_if_any(model, args.pretrained_path, device)

    params = model.unfreeze_and_get_parameters_for_finetuning()
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    criterion = getattr(model, "loss", None) or loss_fn

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / f"{args.dataset}_{args.model}_{args.pred_len}_best.pt"
    best_val, best_epoch, no_improve = float("inf"), -1, 0

    print(
        f"==> Train | dataset={args.dataset} model={args.model} ctx={args.ctx_len} "
        f"pred={args.pred_len} bs={args.batch_size} lr={args.lr} wd={args.weight_decay} "
        f"scaler={args.apply_scaler_transform} seed={args.random_seed} pretrained={bool(args.pretrained_path)}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            for k, v in list(batch.items()):
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            ctx_len = int(batch.pop("context_len"))
            pred_len = int(batch.pop("pred_len"))

            batch["load"] = transform(batch["load"])

            preds = model(batch, context_len=ctx_len, pred_len=pred_len)
            tgt = batch["load"][:, ctx_len : ctx_len + pred_len]

            loss = criterion(preds, tgt)
            optimizer.zero_grad()
            loss.backward()
            for g in optimizer.param_groups:
                torch.nn.utils.clip_grad_norm_(g["params"], 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / max(1, len(train_loader))
        val_rmse, val_mae = quick_eval_rmse_mae(model, val_loader, device, transform)
        print(
            f"[Epoch {epoch:03d}] train_loss={avg_loss:.6f} val_RMSE={val_rmse:.6f} val_MAE={val_mae:.6f}"
        )

        if val_rmse < best_val - 1e-9:
            best_val, best_epoch, no_improve = val_rmse, epoch, 0
            save_model_checkpoint(
                model, optimizer, scheduler, epoch, best_val, best_ckpt
            )
            print(f"  >> Saved best -> {best_ckpt} (val_RMSE={best_val:.6f})")
        else:
            no_improve += 1
            if args.patience > 0 and no_improve >= args.patience:
                print(
                    f"==> Early stop at epoch {epoch}, best@{best_epoch}={best_val:.6f}"
                )
                break

    _step, _best = load_model_checkpoint(
        best_ckpt,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        local_rank=0,
        strict=False,
    )

    test_rmse, test_mae = quick_eval_rmse_mae(model, test_loader, device, transform)
    print(f"[Test@Best] RMSE={test_rmse:.6f} MAE={test_mae:.6f}")

    results_dir = Path(args.results_path)
    test_with_metrics_manager_and_aggregate(
        args=args,
        model=model,
        test_loader=test_loader,
        device=device,
        results_dir=results_dir,
        transform=transform,
        inverse_transform=inverse_transform,
    )


# =========================
#  CLI
# =========================
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    valid_datasets = ["university", "cofactor"] + [
        f"cofactor:{k}" for k in cofactor_type.keys()
    ]
    p.add_argument("--dataset", required=True, choices=valid_datasets)
    p.add_argument(
        "--model", required=True, help="BuildMoE / DLinearRegression / NLinear 等"
    )
    p.add_argument("--ctx_len", type=int, default=168)
    p.add_argument("--pred_len", type=int, default=168)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--ckpt_dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--results_path", type=Path, default=Path("results"))
    p.add_argument("--variant_name", type=str, default="")
    p.add_argument(
        "--experiment_tag",
        type=str,
        default="zero_shot",
        help="aggregate 用的 experiment 标签",
    )
    p.add_argument(
        "--eval_horizons",
        type=int,
        nargs="+",
        default=[1, 6, 12, 24, 48, 96, 168],
        help="测试时切片长度列表",
    )
    p.add_argument(
        "--apply_scaler_transform",
        choices=["", "standard", "boxcox"],
        default="boxcox",
        help="Box-Cox 时会 transform 训练目标，inverse 测试输出",
    )
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--pretrained_path", type=str, default="")
    p.add_argument("--ignore_scoring_rules", action="store_true")

    args = p.parse_args()

    try:
        with ir.files("buildings_bench.configs").joinpath(f"{args.model}.toml").open(
            "rb"
        ) as f:
            toml_cfg = tomli.load(f)
    except FileNotFoundError:
        cfg_dir = Path(__file__).parent / "buildings_bench" / "configs"
        toml_file = cfg_dir / f"{args.model}.toml"
        with toml_file.open("rb") as f:
            toml_cfg = tomli.load(f)

    model_args: Dict = toml_cfg["model"]
    model_args["context_len"] = args.ctx_len
    model_args["pred_len"] = args.pred_len

    main(args, model_args)
