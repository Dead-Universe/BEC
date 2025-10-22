#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Balanced-MoE 统一实验脚本（多数据集聚合版）：
- 支持单数据集 (--dataset) 或一次性跑 benchmark_registry 全部 (--registry-all)
- 统一通过 load_torch_dataset 迭代“楼宇级 Dataset”
- 仅 COFACTOR 做按类别/ID 的过滤；其余数据集直接使用生成器
- Box–Cox 反变换，输出 matched-horizon=H 的 NRMSE/NMAE、预测明细与 MoE 路由统计
- 当 --registry-all 打开时，所有数据集的路由统计、热图与指标均“统计到一块”

依赖：
- buildings_bench.data.load_torch_dataset（见 new.py）
- buildings_bench.transforms.BoxCoxTransform
- 你的模型 LoadForecastingTransformerMoE（需提供 predict() 或等价接口）

注意：
- 请设置 BUILDINGS_BENCH 环境变量，或通过 --transforms-dir 指定 Box–Cox 参数目录
"""

from __future__ import annotations
import argparse, os, json, random
from pathlib import Path
import re
from typing import List, Tuple, Optional, Dict, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ==== 你的项目内模块 ====
from buildings_bench.data import load_torch_dataset  # ← 统一入口
from buildings_bench.transforms import BoxCoxTransform
from buildings_bench.models.transformer_moes_update_20 import (
    LoadForecastingTransformerMoE,
)
from buildings_bench.models.buildmoe import BuildMoE


def is_rank0() -> bool:
    return (
        not (torch.distributed.is_available() and torch.distributed.is_initialized())
        or torch.distributed.get_rank() == 0
    )


# ──────────────────────────────────────────────────────────────────────────────
# 统一基准注册表：当 --registry-all 打开时按此顺序逐一评测
# ──────────────────────────────────────────────────────────────────────────────
benchmark_registry: List[str] = [
    "buildings-900k-test",
    "sceaux",
    "borealis",
    "ideal",
    "bdg-2:panther",
    "bdg-2:fox",
    "bdg-2:rat",
    "bdg-2:bear",
    "electricity",
    "smart",
    "lcl",
    "university",
    "cofactor",
]

# ──────────────────────────────────────────────────────────────────────────────
# COFACTOR 建筑类型到楼宇 ID 的映射（仅 COFACTOR 需要）
# ──────────────────────────────────────────────────────────────────────────────
COFACTOR_TYPE: Dict[str, List[str]] = {
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


# ──────────────────────────────────────────────────────────────────────────────
# MoE 激活统计 Hook
# ──────────────────────────────────────────────────────────────────────────────
class MoEActivationHook:
    def __init__(self, model: nn.Module, sync_ddp: bool = True):
        self.model = model
        self.sync_ddp = (
            sync_ddp
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.gates: List[nn.Module] = []
        self.gate_names: List[str] = []
        self.n_experts: Optional[int] = None
        self.stats: Optional[torch.Tensor] = None  # [n_gates, E], float

        # 收集 Gate
        for name, m in model.named_modules():
            if m.__class__.__name__ == "Gate":
                self.gates.append(m)
                self.gate_names.append(name)
        if len(self.gates) == 0:
            raise RuntimeError("未在模型中找到 Gate 模块。")

        g0 = self.gates[0]
        if hasattr(g0, "args") and hasattr(g0.args, "n_routed_experts"):
            self.n_experts = int(g0.args.n_routed_experts)
        elif hasattr(g0, "router") and hasattr(g0.router, "out_features"):
            self.n_experts = int(g0.router.out_features)
        else:
            raise RuntimeError("无法确定专家数量（n_experts）。")

        self.stats = torch.zeros(
            len(self.gates), self.n_experts, dtype=torch.float64, device="cpu"
        )

        for li, g in enumerate(self.gates):
            h = g.register_forward_hook(self._make_hook(li), with_kwargs=False)
            self.handles.append(h)

    def _make_hook(self, layer_idx: int):
        @torch.no_grad()
        def _hook(module, inputs, output):
            if not hasattr(module, "last_logits"):
                return
            logits = module.last_logits.detach()
            E = logits.shape[-1]
            logits = logits.reshape(-1, E)
            # 判断logits是否存在异常
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                raise ValueError(
                    f"Layer {layer_idx} logits contain NaN or Inf values. Please check the model."
                )
            # 判断logits是否存在0
            score_func = getattr(module, "score_func", "softmax")
            probs = (
                torch.softmax(logits, dim=-1)
                if score_func == "softmax"
                else torch.sigmoid(logits)
            )
            k = getattr(
                module,
                "topk",
                getattr(getattr(module, "args", object()), "n_activated_experts", 1),
            )
            top_idx = torch.topk(probs, k, dim=-1).indices  # [N,k]
            counts = torch.bincount(top_idx.reshape(-1), minlength=E).to(torch.float32)

            if self.sync_ddp:
                backend = torch.distributed.get_backend()
                if backend == "nccl":
                    counts = counts.to(logits.device)
                    torch.distributed.all_reduce(
                        counts, op=torch.distributed.ReduceOp.SUM
                    )
                    counts = counts.to(self.stats.device)
                else:
                    counts = counts.to(self.stats.device)
                    torch.distributed.all_reduce(
                        counts, op=torch.distributed.ReduceOp.SUM
                    )
            else:
                counts = counts.to(self.stats.device)

            self.stats[layer_idx] += counts.to(self.stats.dtype)

        return _hook

    def reset(self):
        if self.stats is not None:
            self.stats.zero_()

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    @torch.no_grad()
    def usage(self, normalize: bool = True) -> torch.Tensor:
        mat = self.stats.clone()
        if normalize:
            row_sum = mat.sum(-1, keepdim=True).clamp_min(1.0)
            mat = mat / row_sum
        return mat

    def plot(self, normalize: bool = True, figsize: Tuple[int, int] = (10, 4)):
        mat = self.usage(normalize=normalize).numpy()
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(mat, aspect="auto")
        ax.set_xlabel("Expert ID")
        ax.set_ylabel("Gate (depth)")
        ax.set_xticks(range(self.n_experts))
        ax.set_yticks(range(len(self.gates)))
        short_names = [n if len(n) <= 40 else n[-40:] for n in self.gate_names]
        ax.set_yticklabels(short_names)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Activation share" if normalize else "Activation count")
        ax.set_title("MoE Expert Activation (aggregated)")
        fig.tight_layout()
        return fig


# ──────────────────────────────────────────────────────────────────────────────
# 指标
# ──────────────────────────────────────────────────────────────────────────────
def aggregated_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true 与 y_pred 形状不一致")
    denom = float(np.mean(y_true))
    if denom <= 0:
        raise ValueError("归一化分母均值 <= 0；请检查数据")
    mse = float(np.mean((y_pred - y_true) ** 2))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    nrmse = 100.0 * np.sqrt(mse) / denom
    nmae = 100.0 * mae / denom
    return {"NRMSE(%)": nrmse, "NMAE(%)": nmae, "denom_mean": denom}


# ──────────────────────────────────────────────────────────────────────────────
# 从 load_torch_dataset 生成器中过滤/收集楼宇
# ──────────────────────────────────────────────────────────────────────────────
def iter_buildings(
    dataset_name: str,
    context_len: int,
    pred_len: int,
    scaler_transform_path: Path,
    apply_scaler_transform: str = "boxcox",
    split: str = "",
    oov_path: Path | None = None,
    cofactor_category: Optional[str] = None,
    cofactor_buildings: Optional[List[str]] = None,
) -> Iterable[Tuple[str, torch.utils.data.Dataset]]:
    """
    统一返回 (building_id, torch Dataset) 对。
    - 仅 dataset_name == 'cofactor' 时按类别或自定义 ID 过滤
    """
    gen = load_torch_dataset(
        dataset_name,
        context_len=context_len,
        pred_len=pred_len,
        apply_scaler_transform=apply_scaler_transform,
        scaler_transform_path=scaler_transform_path,
        split=split,
        oov_path=oov_path,
    )

    for item in gen:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            bid, bds = item
        else:
            bid, bds = f"{dataset_name}_0", item

        # 仅 COFACTOR 做过滤（如果未给 category/buildings，则意味着“全量 cofactor”）
        if dataset_name.lower() == "cofactor":
            allow: Optional[set] = None
            if cofactor_buildings:
                allow = set(x.strip().lower() for x in cofactor_buildings)
            elif cofactor_category:
                if cofactor_category not in COFACTOR_TYPE:
                    raise ValueError(f"未知 COFACTOR 类别：{cofactor_category}")
                allow = set(x.lower() for x in COFACTOR_TYPE[cofactor_category])
            if allow is not None and bid.lower() not in allow:
                continue

        yield str(bid), bds


def safe_for_path(s: str) -> str:
    # 把路径分隔符和不安全字符都替换掉，得到扁平且可移植的文件名片段
    s = s.replace("\\", "/")
    parts = [p for p in s.split("/") if p]
    s = "_".join(parts)
    s = re.sub(r"[^0-9A-Za-z._-]+", "_", s).strip("._")
    return s[:150]  # 可选：限制长度，避免超长文件名


# ──────────────────────────────────────────────────────────────────────────────
# 评测一个数据集（可被多次调用；hook 不 reset，即跨数据集累加）
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_one_dataset(
    dataset_name: str,
    args,
    transforms_dir: Path,
    device: str,
    model: nn.Module,
    moe_hook: MoEActivationHook,
    boxcox: BoxCoxTransform,
    outdir: Path,
    rank0: bool,
    routing_rows: List[dict],
    entropy_rows: List[dict],
    preds_rows: List[dict],
    first_single_heatmap_saved: List[bool],
):
    # 收集楼宇
    bldg_list: List[Tuple[str, torch.utils.data.Dataset]] = []
    for bid, bds in iter_buildings(
        dataset_name=dataset_name,
        context_len=args.L,
        pred_len=args.H,
        scaler_transform_path=transforms_dir,
        split="",
        oov_path=None,
        cofactor_category=(
            args.category if dataset_name.lower() == "cofactor" else None
        ),
        cofactor_buildings=(
            args.buildings.split(",")
            if (args.buildings and dataset_name.lower() == "cofactor")
            else None
        ),
    ):
        try:
            n = len(bds)
        except TypeError:
            n = bds.__len__()
        if n <= 0:
            if rank0:
                print(
                    f"[Skip] {dataset_name}:{bid} has 0 samples for L={args.L}, H={args.H}."
                )
            continue
        bldg_list.append((bid, bds))

    if len(bldg_list) == 0:
        if rank0:
            print(f"[Warn] 数据集 {dataset_name} 未加载到任何楼宇。")
        return

    # 每栋分配的采样数（向上取整）
    per_bldg = max(1, (args.samples + len(bldg_list) - 1) // len(bldg_list))

    # 逐楼宇评测
    with torch.no_grad():
        for bid, bds in bldg_list:
            dl = DataLoader(
                bds,
                batch_size=1,
                shuffle=True,
                num_workers=args.loader_workers,
                pin_memory=True,
                collate_fn=torch.utils.data.default_collate,
            )
            taken = 0
            for batch in dl:
                if "load" not in batch:
                    raise KeyError("样本中缺少 'load' 键；请检查数据管线。")
                x = batch["load"].squeeze(0)  # [L+H, 1] or [L+H]
                if x.ndim == 1:
                    x = x.unsqueeze(-1)

                x_with_future = x.clone()
                x_with_future[args.L :] = 0.0
                sample_dict = {"load": x_with_future.unsqueeze(0).to(device)}

                prev_stats = moe_hook.stats.clone()
                out = model.predict(sample_dict)
                pred_norm = out[0] if isinstance(out, (tuple, list)) else out
                pred_norm = pred_norm.to("cpu").squeeze().float().numpy()  # (H,)
                y_norm_true = x[args.L : args.L + args.H, 0].cpu().numpy()

                # 反变换
                y_pred_raw = boxcox.undo_transform(pred_norm)
                y_true_raw = boxcox.undo_transform(y_norm_true)

                # 落 preds_rows（带 dataset 字段）
                for h in range(args.H):
                    preds_rows.append(
                        {
                            "dataset": dataset_name,
                            "series": bid,
                            "sample_idx": int(taken),
                            "h": int(h + 1),
                            "y_true": float(y_true_raw[h]),
                            "y_pred": float(y_pred_raw[h]),
                        }
                    )

                # 本样本路由份额（差分）
                delta = (moe_hook.stats - prev_stats).clamp_min(0)  # [n_gates, E]
                row_sum = delta.sum(-1, keepdim=True).clamp_min(1.0)
                usage_step = (delta / row_sum).cpu().numpy()

                max_share_step = usage_step.max(axis=1)  # [n_gates]
                top_expert_step = usage_step.argmax(axis=1)  # [n_gates]
                eps = 1e-12
                H_raw_step = -np.sum(
                    (usage_step + eps) * np.log(usage_step + eps), axis=1
                )
                E = usage_step.shape[1]
                H_norm_step = H_raw_step / np.log(E)
                N_eff_step = np.exp(H_raw_step)

                if rank0:
                    for g in range(usage_step.shape[0]):
                        routing_rows.append(
                            {
                                "dataset": dataset_name,
                                "series": bid,
                                "sample_idx": int(taken),
                                "gate": int(g),
                                "max_share": float(max_share_step[g]),
                                "top_expert": int(top_expert_step[g]),
                            }
                        )
                        entropy_rows.append(
                            {
                                "dataset": dataset_name,
                                "series": bid,
                                "sample_idx": int(taken),
                                "gate": int(g),
                                "H_raw": float(H_raw_step[g]),
                                "H_norm": float(H_norm_step[g]),
                                "N_eff": float(N_eff_step[g]),
                            }
                        )

                # 仅保存一次“单样本热力图”（第一个样本）
                if (not first_single_heatmap_saved[0]) and rank0:
                    try:
                        fig_single = moe_hook.plot(normalize=True)
                        safe_ds = safe_for_path(dataset_name)
                        safe_bid = safe_for_path(bid)
                        sp = (
                            outdir
                            / f"moe_activation_heatmap_single_{safe_ds}_{safe_bid}.png"
                        )
                        sp.parent.mkdir(
                            parents=True, exist_ok=True
                        )  # 保险：确保父目录存在
                        fig_single.savefig(sp, dpi=200)
                        print(f"[Save] 单样本热力图: {sp}")
                        first_single_heatmap_saved[0] = True
                    except Exception as e:
                        print(f"[Warn] 单样本热力图保存失败：{e}")

                taken += 1
                if taken >= per_bldg:
                    break


# ──────────────────────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Balanced-MoE: 多数据集统一评测脚本（聚合版）"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        default=None,
        help="单个数据集名（如 cofactor / lcl / smart / electricity / bdg-2:bear / university 等）",
    )
    parser.add_argument(
        "--registry-all",
        action="store_true",
        help="遍历 benchmark_registry 中的全部数据集并聚合统计",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="BuildingsBench 根目录（仅用于检查/提示；真正的数据路径由环境变量 BUILDINGS_BENCH 提供）",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="仅 cofactor 可用：School/Kindergarten/NursingHome/Office（默认全量）",
    )
    parser.add_argument(
        "--buildings",
        type=str,
        default=None,
        help="仅 cofactor 可用：逗号分隔 building id 列表（优先于 --category）",
    )
    parser.add_argument("--L", type=int, default=168, help="历史窗口长度")
    parser.add_argument(
        "--H", type=int, default=24, help="预测地平线长度（matched-horizon）"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=200,
        help="总采样窗口数：单数据集=该数据集总窗口；--registry-all=每个数据集分配的窗口",
    )
    parser.add_argument("--seed", type=int, default=2025, help="随机种子")
    parser.add_argument("--ckpt", type=str, required=True, help="模型 checkpoint 路径")
    parser.add_argument(
        "--transforms-dir",
        type=str,
        default=None,
        help="Box–Cox 变换目录；默认从 $BUILDINGS_BENCH/metadata/transforms 推断",
    )
    parser.add_argument(
        "--outdir", type=str, required=True, help="输出目录：图、指标与路由统计 CSV"
    )
    parser.add_argument(
        "--ddp-sync", action="store_true", help="在 DDP 下 all-reduce 统计全局激活"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="cuda:0 / cpu；默认自动"
    )
    parser.add_argument(
        "--loader-workers", type=int, default=0, help="DataLoader num_workers"
    )
    args = parser.parse_args()

    # 设备与随机性
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 输出目录
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Box–Cox 变换目录
    if args.transforms_dir is None:
        bb_root = Path(os.environ.get("BUILDINGS_BENCH", ""))
        transforms_dir = (
            (bb_root / "metadata" / "transforms") if bb_root.exists() else None
        )
    else:
        transforms_dir = Path(args.transforms_dir)
    if transforms_dir is None or not transforms_dir.exists():
        raise FileNotFoundError(
            "找不到 Box–Cox 变换目录。请通过 --transforms-dir 指定，或设置 BUILDINGS_BENCH 环境变量。"
        )
    boxcox = BoxCoxTransform()
    boxcox.load(transforms_dir)

    # 模型
    cfg = dict(
        context_len=args.L,
        pred_len=args.H,
        num_encoder_layers=8,
        num_decoder_layers=10,
        d_model=768,
        dim_feedforward=2048,
        num_experts=8,
        top_k=2,
        nhead=12,
        dropout=0.0,
        continuous_loads=True,
        continuous_head="huber",
        n_shared_experts=0,
    )
    # model = LoadForecastingTransformerMoE(**cfg).to(device).eval()
    model = BuildMoE(**cfg).to(device).eval()
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint 不存在：{args.ckpt}")
    state = torch.load(args.ckpt, map_location="cpu")
    if hasattr(model, "load_from_checkpoint"):
        model.load_from_checkpoint(args.ckpt)
    else:
        model.load_state_dict(state.get("state_dict", state), strict=False)

    # Hook（贯穿全流程，不 reset，跨数据集累加）
    moe_hook = MoEActivationHook(model, sync_ddp=args.ddp_sync)

    # 要处理的数据集列表
    if args.registry_all:
        datasets_to_run = list(benchmark_registry)
    else:
        if not args.dataset:
            raise ValueError("未指定 --dataset，且未开启 --registry-all。")
        datasets_to_run = [args.dataset]

    # 汇总容器（跨全部数据集）
    routing_rows: List[dict] = []  # 逐样本 max_share/top_expert
    entropy_rows: List[dict] = []  # 逐样本 H_raw/H_norm/N_eff
    preds_rows: List[dict] = []  # 逐样本预测明细（反变换后）
    first_single_heatmap_saved = [False]
    rank0 = is_rank0()

    # 依次评测每个数据集（hook 统计不 reset）
    if rank0:
        print(f"[Info] 将顺序评测并聚合以下数据集：{datasets_to_run}")
    for ds in datasets_to_run:
        if rank0:
            print(f"\n===== Evaluating dataset: {ds} =====")
        evaluate_one_dataset(
            dataset_name=ds,
            args=args,
            transforms_dir=transforms_dir,
            device=device,
            model=model,
            moe_hook=moe_hook,
            boxcox=boxcox,
            outdir=outdir,
            rank0=rank0,
            routing_rows=routing_rows,
            entropy_rows=entropy_rows,
            preds_rows=preds_rows,
            first_single_heatmap_saved=first_single_heatmap_saved,
        )

    # ====== 全部样本的误差指标（micro-average；跨全部数据集）======
    if len(preds_rows) == 0:
        raise RuntimeError("没有产生任何预测样本，无法计算指标。")
    preds_df = pd.DataFrame(preds_rows)
    # 组装为 [N, H] 以便重用 aggregated_metrics
    # 注意：这里是按窗口等权的 micro-average；若要“按楼宇等权”，需另行分组再平均
    y_true_all = (
        preds_df.pivot_table(
            index=["dataset", "series", "sample_idx"], columns="h", values="y_true"
        )
        .sort_index(axis=1)
        .to_numpy()
    )
    y_pred_all = (
        preds_df.pivot_table(
            index=["dataset", "series", "sample_idx"], columns="h", values="y_pred"
        )
        .sort_index(axis=1)
        .to_numpy()
    )
    metrics = aggregated_metrics(y_true_all, y_pred_all)

    # ====== 专家激活热力图（跨全部数据集的聚合）======
    try:
        fig = moe_hook.plot(normalize=True)
        heatmap_path = outdir / (
            "moe_activation_heatmap_aggregated_ALL.png"
            if args.registry_all
            else "moe_activation_heatmap_aggregated.png"
        )
        fig.savefig(heatmap_path, dpi=240)
        if rank0:
            print(f"[Save] 聚合热力图: {heatmap_path}")
        usage_mat = moe_hook.usage(normalize=True)
        usage_df = pd.DataFrame(
            usage_mat.numpy(),
            columns=[f"expert_{i}" for i in range(usage_mat.shape[1])],
        )
        usage_csv_path = outdir / (
            "moe_activation_usage_ALL.csv"
            if args.registry_all
            else "moe_activation_usage.csv"
        )
        usage_df.to_csv(usage_csv_path, index_label="gate")
        if rank0:
            print(f"[Save] 激活占比矩阵 CSV: {usage_csv_path}")
    except Exception as e:
        if rank0:
            print(f"[Warn] 聚合热力图保存失败：{e}")

    # ====== 预测明细与指标落盘 ======
    preds_csv = outdir / (
        "predictions_detail_ALL.csv" if args.registry_all else "predictions_detail.csv"
    )
    preds_df.to_csv(preds_csv, index=False)
    if rank0:
        print(f"[Save] 预测明细: {preds_csv}")

    metrics_path = outdir / (
        "metrics_ALL.json" if args.registry_all else "metrics.json"
    )
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    if rank0:
        print(f"[Save] 指标: {metrics_path}")
        print(
            f"[Metrics] NRMSE(%)={metrics['NRMSE(%)']:.2f}, NMAE(%)={metrics['NMAE(%)']:.2f}, denom_mean={metrics['denom_mean']:.4f}"
        )

    # ====== 逐样本路由 & 带状图（跨全部数据集）======
    if rank0:
        if len(routing_rows):
            rt_df = pd.DataFrame(routing_rows)
            rt_df.to_csv(
                outdir
                / (
                    "routing_timeseries_ALL.csv"
                    if args.registry_all
                    else "routing_timeseries.csv"
                ),
                index=False,
            )

            def q10(s):
                return s.quantile(0.10)

            def q90(s):
                return s.quantile(0.90)

            rib = (
                rt_df.groupby("gate")["max_share"]
                .agg(
                    mean="mean", std="std", p10=q10, p90=q90, median="median", n="count"
                )
                .reset_index()
            )
            rib.to_csv(
                outdir
                / (
                    "routing_ribbon_stats_ALL.csv"
                    if args.registry_all
                    else "routing_ribbon_stats.csv"
                ),
                index=False,
            )

            x = rib["gate"].to_numpy()
            plt.figure(figsize=(6.0, 3.0))
            plt.plot(x, rib["mean"].to_numpy(), label="mean max_share")
            plt.fill_between(
                x,
                rib["p10"].to_numpy(),
                rib["p90"].to_numpy(),
                alpha=0.25,
                label="p10–p90",
            )
            plt.axvline(cfg["num_encoder_layers"] - 0.5, linestyle="--", linewidth=1)
            plt.xlabel("Gate (depth)")
            plt.ylabel("max_share")
            plt.legend(frameon=False)
            plt.tight_layout()
            plt.savefig(
                outdir
                / (
                    "maxshare_ribbon_ALL.png"
                    if args.registry_all
                    else "maxshare_ribbon.png"
                ),
                dpi=240,
            )
            plt.close()

        if len(entropy_rows):
            et_df = pd.DataFrame(entropy_rows)
            et_df.to_csv(
                outdir
                / (
                    "routing_entropy_timeseries_ALL.csv"
                    if args.registry_all
                    else "routing_entropy_timeseries.csv"
                ),
                index=False,
            )

            prof = et_df.groupby("gate")[["H_norm", "N_eff"]].mean().reset_index()
            prof.to_csv(
                outdir
                / (
                    "entropy_neff_by_gate_ALL.csv"
                    if args.registry_all
                    else "entropy_neff_by_gate.csv"
                ),
                index=False,
            )

            x = prof["gate"].to_numpy()
            plt.figure(figsize=(6.0, 3.0))
            plt.plot(x, prof["H_norm"].to_numpy(), label="H (normalized)")
            ax2 = plt.gca().twinx()
            ax2.plot(x, prof["N_eff"].to_numpy(), label="N_eff", linestyle="-")
            plt.axvline(cfg["num_encoder_layers"] - 0.5, linestyle="--", linewidth=1)
            plt.xlabel("Gate (depth)")
            plt.tight_layout()
            plt.savefig(
                outdir
                / (
                    "entropy_neff_curve_ALL.png"
                    if args.registry_all
                    else "entropy_neff_curve.png"
                ),
                dpi=240,
            )
            plt.close()

    # ====== 路由统计导出（由“聚合热图矩阵”直接计算；跨全部数据集）======
    mat = moe_hook.usage(normalize=True).numpy()  # [n_gates, E]
    eps = 1e-12
    m = mat.max(axis=1)
    Hn = (-(mat + eps) * np.log(mat + eps)).sum(1) / np.log(mat.shape[1])
    N_eff = np.exp((-(mat + eps) * np.log(mat + eps)).sum(1))
    enc_layers = cfg["num_encoder_layers"]
    role = ["enc"] * enc_layers + ["dec"] * (mat.shape[0] - enc_layers)
    df = pd.DataFrame(
        {
            "layer": np.arange(mat.shape[0]),
            "role": role,
            "max_share": m,
            "entropy": Hn,
            "N_eff": N_eff,
        }
    )
    stats_csv = outdir / (
        "routing_stats_ALL.csv" if args.registry_all else "routing_stats.csv"
    )
    df.to_csv(stats_csv, index=False)
    if rank0:
        print(f"[Save] 路由统计: {stats_csv}")
        print(
            "[Routing] enc/dec median max_share:",
            df[df.role == "enc"].max_share.median(),
            df[df.role == "dec"].max_share.median(),
        )

    # 记录本次跑过的数据集清单
    if rank0:
        with open(outdir / "datasets_processed.json", "w", encoding="utf-8") as f:
            json.dump({"datasets": datasets_to_run}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
