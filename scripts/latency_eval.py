#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Building-MoE 推理延迟统计脚本（BMS端到端 + 理论模型时延）
- 端到端计时：Box-Cox 正/逆、未来段置零、to(device)、predict()、cpu()、逆变换
- 理论模型计时：仅围住 model.predict()（输入已在 device 上），CUDA 用事件计时、CPU 用 perf_counter
- e2e 吞吐/elapsed 基于端到端计时和（不含 DataLoader/遍历等离线开销）
- distribution 模式：分布统计仅用于报告，不改变窗口吞吐定义
"""

from __future__ import annotations
import argparse, os, json, time, random
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Iterable

import numpy as np
import pandas as pd
import tomli
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from buildings_bench.data import load_torch_dataset
from buildings_bench.models import model_factory
from buildings_bench.transforms import BoxCoxTransform

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


def is_rank0() -> bool:
    return (
        not (torch.distributed.is_available() and torch.distributed.is_initialized())
        or torch.distributed.get_rank() == 0
    )


def iter_buildings(
    dataset_name: str,
    context_len: int,
    pred_len: int,
    split: str = "",
    cofactor_category: Optional[str] = None,
    cofactor_buildings: Optional[List[str]] = None,
) -> Iterable[Tuple[str, torch.utils.data.Dataset]]:
    gen = load_torch_dataset(
        dataset_name,
        context_len=context_len,
        pred_len=pred_len,
        scaler_transform_path=None,
        split=split,
        oov_path=None,
    )
    for item in gen:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            bid, bds = item
        else:
            bid, bds = f"{dataset_name}_0", item

        if dataset_name.lower() == "cofactor":
            allow: Optional[set] = None
            if cofactor_buildings:
                allow = set(x.strip().lower() for x in cofactor_buildings)
            elif cofactor_category:
                if cofactor_category not in COFACTOR_TYPE:
                    raise ValueError(f"Unknown COFACTOR category: {cofactor_category}")
                allow = set(x.lower() for x in COFACTOR_TYPE[cofactor_category])
            if allow is not None and bid.lower() not in allow:
                continue
        yield str(bid), bds


# ------------------ 计时工具 ------------------
def _cuda_model_only_ms(model, batch) -> float:
    """CUDA Events 统计纯 model.predict() 时延（毫秒）"""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    out = model.predict(batch)
    end.record()
    end.synchronize()
    _ = out[0] if (isinstance(out, (list, tuple)) and len(out) >= 1) else out
    return float(start.elapsed_time(end))


def _cpu_model_only_ms(model, batch) -> float:
    t0 = time.perf_counter()
    out = model.predict(batch)
    t1 = time.perf_counter()
    _ = out[0] if (isinstance(out, (list, tuple)) and len(out) >= 1) else out
    return (t1 - t0) * 1000.0


@torch.no_grad()
def time_once_e2e_and_model_only(
    model: nn.Module,
    *,
    y_seq: torch.Tensor,  # [L+H,1] or [L+H]
    L: int,
    H: int,
    device: str,
    boxcox: BoxCoxTransform,
) -> Tuple[float, float]:
    """
    返回 (e2e_ms, model_only_ms)
    e2e_ms：含 Box-Cox 正/逆、置零、搬运、predict、回拷、逆变换（不含 DataLoader）
    model_only_ms：仅 model.predict() 的时延（输入已在 device 上）
    """
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    # CPU 上 Box-Cox 正变换
    y_raw = y_seq.squeeze(-1).cpu().numpy().astype("float32")
    y_norm = boxcox.transform(y_raw)
    y_norm[L:] = 0.0
    x_norm_t = (
        torch.from_numpy(y_norm)
        .unsqueeze(0)
        .unsqueeze(-1)
        .to(device, non_blocking=True)
    )
    batch = {"load": x_norm_t}

    # 模型-only
    if device.startswith("cuda"):
        model_only_ms = _cuda_model_only_ms(model, batch)
    else:
        model_only_ms = _cpu_model_only_ms(model, batch)

    # 端到端需要完整一次预测与逆变换
    out = model.predict(batch)
    preds_norm_t = out[0] if (isinstance(out, (list, tuple)) and len(out) >= 1) else out
    preds_norm = preds_norm_t.squeeze().detach().cpu().numpy().astype("float32")
    _ = boxcox.undo_transform(preds_norm)

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    e2e_ms = (t1 - t0) * 1000.0
    return float(e2e_ms), float(model_only_ms)


def _reduce_stats(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(arr.min()),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }


# ------------------ 主流程 ------------------
def main():
    parser = argparse.ArgumentParser(
        description="Building-MoE 推理延迟统计（端到端 + 模型-only，不含DataLoader时间）"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--registry-all", action="store_true")
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--buildings", type=str, default=None)
    parser.add_argument("--L", type=int, default=168)
    parser.add_argument("--H", type=int, default=24)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--loader-workers", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--measure-mode", type=str, default="e2e", choices=["e2e", "distribution"]
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="distribution 模式下的重复次数（e2e 模式忽略）",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    SCRIPT_PATH = Path(os.path.realpath(__file__)).parent
    cfg_dir = SCRIPT_PATH / ".." / "buildings_bench" / "configs"
    toml_file = cfg_dir / f"{args.model}.toml"
    if not toml_file.exists():
        raise FileNotFoundError(toml_file)
    with toml_file.open("rb") as f:
        toml_cfg = tomli.load(f)
    model_args: Dict = toml_cfg["model"]
    model, loss_fn, _ = model_factory(model_name=args.model, model_args=model_args)
    model.to(device).eval()
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint 不存在：{args.ckpt}")
    state = torch.load(args.ckpt, map_location="cpu")
    if hasattr(model, "load_from_checkpoint"):
        model.load_from_checkpoint(args.ckpt)
    else:
        model.load_state_dict(state.get("state_dict", state), strict=False)

    # 数据集选择
    if args.registry_all:
        datasets_to_run = list(benchmark_registry)
    else:
        if not args.dataset:
            raise ValueError("未指定 --dataset，且未开启 --registry-all。")
        datasets_to_run = [args.dataset]

    rank0 = is_rank0()
    if rank0:
        print(f"[Info] device={device} ; datasets={datasets_to_run}")
        if device.startswith("cuda"):
            print(
                f"[CUDA] name={torch.cuda.get_device_name()}, capability={torch.cuda.get_device_capability()}"
            )

    # 全局 warmup（只针对模型）
    if args.warmup > 0:
        if rank0:
            print(f"[Warmup] {args.warmup} runs ...")
        dummy = torch.zeros((1, args.L + args.H, 1), device=device)
        sample = {"load": dummy}
        with torch.inference_mode():
            for _ in range(args.warmup):
                _ = model.predict(sample)
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
        if rank0:
            print("[Warmup done]")

    # Box-Cox 参数
    bb_env = os.environ.get("BUILDINGS_BENCH", None)
    if not bb_env:
        raise KeyError("环境变量 BUILDINGS_BENCH 未设置，无法加载 Box-Cox 变换。")
    boxcox = BoxCoxTransform()
    tpath = Path(bb_env) / "metadata" / "transforms"
    boxcox.load(tpath)

    # ====== 测量（累计端到端时间，不含 DataLoader）======
    detail_rows: List[dict] = []
    total_measured_windows = 0
    sum_e2e_ms_for_windows = 0.0  # 窗口吞吐/elapsed 的时间基（不含 DataLoader）
    sum_e2e_ms_for_all_calls = (
        0.0  # 所有 predict 调用（distribution 模式）累计端到端时间
    )

    with torch.inference_mode():
        for ds in datasets_to_run:
            if rank0:
                print(f"\n===== Measuring latency on dataset: {ds} =====")

            bldg_list: List[Tuple[str, torch.utils.data.Dataset]] = []
            for bid, bds in iter_buildings(
                dataset_name=ds,
                context_len=args.L,
                pred_len=args.H,
                split="",
                cofactor_category=(args.category if ds.lower() == "cofactor" else None),
                cofactor_buildings=(
                    args.buildings.split(",")
                    if (args.buildings and ds.lower() == "cofactor")
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
                            f"[Skip] {ds}:{bid} has 0 samples for L={args.L}, H={args.H}."
                        )
                    continue
                bldg_list.append((bid, bds))

            if not bldg_list:
                if rank0:
                    print(f"[Warn] dataset {ds} contains no buildings.")
                continue

            per_bldg_cap = max(1, (args.samples + len(bldg_list) - 1) // len(bldg_list))

            for bid, bds in bldg_list:
                dl = DataLoader(
                    bds,
                    batch_size=1,
                    shuffle=True,
                    num_workers=args.loader_workers,
                    pin_memory=True,
                    collate_fn=torch.utils.data.default_collate,
                    persistent_workers=(args.loader_workers > 0),
                    prefetch_factor=(2 if args.loader_workers > 0 else None),
                )
                taken = 0
                for batch in dl:
                    if "load" not in batch:
                        raise KeyError("样本缺少 'load' 键；请检查数据管线。")
                    y_seq = batch["load"].squeeze(0)
                    if y_seq.ndim == 1:
                        y_seq = y_seq.unsqueeze(-1)

                    rep = 1 if args.measure_mode == "e2e" else max(1, args.repeat)
                    e2e_times, model_only_times = [], []

                    for r in range(rep):
                        e2e_ms, model_only_ms = time_once_e2e_and_model_only(
                            model,
                            y_seq=y_seq,
                            L=args.L,
                            H=args.H,
                            device=device,
                            boxcox=boxcox,
                        )
                        e2e_times.append(e2e_ms)
                        model_only_times.append(model_only_ms)
                        # 统计所有 predict 调用的端到端时间和（用于 predict_calls 吞吐）
                        sum_e2e_ms_for_all_calls += e2e_ms

                        # 窗口吞吐仅按“每窗一次”来累计（distribution 模式不加倍）
                        if r == 0:
                            sum_e2e_ms_for_windows += e2e_ms

                    e2e_stats = _reduce_stats(e2e_times)
                    mo_stats = _reduce_stats(model_only_times)

                    detail_rows.append(
                        {
                            "dataset": ds,
                            "series": bid,
                            "sample_idx": int(taken),
                            # E2E 分位
                            "latency_ms_min": e2e_stats["min"],
                            "latency_ms_mean": e2e_stats["mean"],
                            "latency_ms_p50": e2e_stats["p50"],
                            "latency_ms_p90": e2e_stats["p90"],
                            "latency_ms_p95": e2e_stats["p95"],
                            "latency_ms_p99": e2e_stats["p99"],
                            "latency_ms_max": e2e_stats["max"],
                            # 模型-only 分位
                            "model_only_ms_min": mo_stats["min"],
                            "model_only_ms_mean": mo_stats["mean"],
                            "model_only_ms_p50": mo_stats["p50"],
                            "model_only_ms_p90": mo_stats["p90"],
                            "model_only_ms_p95": mo_stats["p95"],
                            "model_only_ms_p99": mo_stats["p99"],
                            "model_only_ms_max": mo_stats["max"],
                            "repeat_used": rep,
                        }
                    )

                    taken += 1
                    total_measured_windows += 1
                    if taken >= per_bldg_cap:
                        break

    # ====== 落盘 ======
    if not detail_rows:
        raise RuntimeError("没有任何样本被测量，无法生成延迟结果。")
    detail_df = pd.DataFrame(detail_rows)
    detail_csv = Path(args.outdir) / (
        "latency_detail_ALL.csv" if args.registry_all else "latency_detail.csv"
    )
    detail_df.to_csv(detail_csv, index=False)

    # 汇总（全部基于端到端计时和，不含 DataLoader）
    mean_latency_ms_e2e = float(detail_df["latency_ms_mean"].mean())
    mean_latency_ms_mo = float(detail_df["model_only_ms_mean"].mean())
    mean_latency_s_e2e = mean_latency_ms_e2e / 1000.0
    mean_latency_s_mo = mean_latency_ms_mo / 1000.0

    # 时间基：端到端计时和
    measured_elapsed_seconds = max(1e-9, sum_e2e_ms_for_windows / 1000.0)

    total_predict_calls = (
        total_measured_windows
        if args.measure_mode == "e2e"
        else total_measured_windows * max(1, args.repeat)
    )
    predict_calls_elapsed_seconds = max(1e-9, sum_e2e_ms_for_all_calls / 1000.0)

    agg = {
        "count_samples": int(len(detail_df)),
        "total_measured_windows": int(total_measured_windows),
        # 端到端计时和（不含 DataLoader）
        "measured_elapsed_seconds": float(measured_elapsed_seconds),
        # 窗口吞吐（按每窗一次端到端）
        "throughput_end2end_windows_per_sec": float(
            total_measured_windows / measured_elapsed_seconds
        ),
        # predict 调用吞吐（distribution 模式下=窗口数*repeat / 所有端到端时间和）
        "throughput_predict_calls_per_sec": float(
            total_predict_calls / predict_calls_elapsed_seconds
        ),
        # 由均值反推的理论吞吐（E2E & 模型-only）
        "throughput_from_mean_latency": (
            float(1.0 / mean_latency_s_e2e) if mean_latency_s_e2e > 0 else float("inf")
        ),
        "throughput_from_model_only_mean_latency": (
            float(1.0 / mean_latency_s_mo) if mean_latency_s_mo > 0 else float("inf")
        ),
        # E2E 分布均值
        "latency_ms_min_mean": float(detail_df["latency_ms_min"].mean()),
        "latency_ms_mean_mean": mean_latency_ms_e2e,
        "latency_ms_p50_mean": float(detail_df["latency_ms_p50"].mean()),
        "latency_ms_p90_mean": float(detail_df["latency_ms_p90"].mean()),
        "latency_ms_p95_mean": float(detail_df["latency_ms_p95"].mean()),
        "latency_ms_p99_mean": float(detail_df["latency_ms_p99"].mean()),
        "latency_ms_max_mean": float(detail_df["latency_ms_max"].mean()),
        # 模型-only 分布均值
        "model_only_ms_min_mean": float(detail_df["model_only_ms_min"].mean()),
        "model_only_ms_mean_mean": mean_latency_ms_mo,
        "model_only_ms_p50_mean": float(detail_df["model_only_ms_p50"].mean()),
        "model_only_ms_p90_mean": float(detail_df["model_only_ms_p90"].mean()),
        "model_only_ms_p95_mean": float(detail_df["model_only_ms_p95"].mean()),
        "model_only_ms_p99_mean": float(detail_df["model_only_ms_p99"].mean()),
        "model_only_ms_max_mean": float(detail_df["model_only_ms_max"].mean()),
        "by_dataset": {},
        "measure_mode": args.measure_mode,
    }

    for ds, sub in detail_df.groupby("dataset"):
        agg["by_dataset"][ds] = {
            "count_samples": int(len(sub)),
            # E2E
            "latency_ms_min_mean": float(sub["latency_ms_min"].mean()),
            "latency_ms_mean_mean": float(sub["latency_ms_mean"].mean()),
            "latency_ms_p50_mean": float(sub["latency_ms_p50"].mean()),
            "latency_ms_p90_mean": float(sub["latency_ms_p90"].mean()),
            "latency_ms_p95_mean": float(sub["latency_ms_p95"].mean()),
            "latency_ms_p99_mean": float(sub["latency_ms_p99"].mean()),
            "latency_ms_max_mean": float(sub["latency_ms_max"].mean()),
            # 模型-only
            "model_only_ms_min_mean": float(sub["model_only_ms_min"].mean()),
            "model_only_ms_mean_mean": float(sub["model_only_ms_mean"].mean()),
            "model_only_ms_p50_mean": float(sub["model_only_ms_p50"].mean()),
            "model_only_ms_p90_mean": float(sub["model_only_ms_p90"].mean()),
            "model_only_ms_p95_mean": float(sub["model_only_ms_p95"].mean()),
            "model_only_ms_p99_mean": float(sub["model_only_ms_p99"].mean()),
            "model_only_ms_max_mean": float(sub["model_only_ms_max"].mean()),
        }

    summary_json = Path(args.outdir) / (
        "latency_summary_ALL.json" if args.registry_all else "latency_summary.json"
    )
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(agg, f, ensure_ascii=False, indent=2)

    if is_rank0():
        print(f"[Save] 细节: {detail_csv}")
        print(f"[Save] 汇总: {summary_json}")
        print(
            "[Done] mode={mode} ; windows={w} ; "
            "E2E_elapsed={sec:.2f}s ; "
            "E2E_throughput={e2e:.2f} win/s ; "
            "predict_calls_throughput={pc:.2f} calls/s ; "
            "from_E2E_mean_latency={e2e_t:.2f} calls/s ; "
            "from_model_only_mean_latency={mo_t:.2f} calls/s".format(
                mode=args.measure_mode,
                w=total_measured_windows,
                sec=agg["measured_elapsed_seconds"],
                e2e=agg["throughput_end2end_windows_per_sec"],
                pc=agg["throughput_predict_calls_per_sec"],
                e2e_t=agg["throughput_from_mean_latency"],
                mo_t=agg["throughput_from_model_only_mean_latency"],
            )
        )


if __name__ == "__main__":
    main()
