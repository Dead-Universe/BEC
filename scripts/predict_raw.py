#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
predict_and_plot.py

- 使用 baseline 模型 (BuildMoE-top-k-2) 在数据集最多 max_trials 个序列中搜索 RMSE 最低的序列
- 在该序列上，用多个模型预测
- 绘制每个模型的预测图（包含上下文、预测、真实序列）
"""

import os
from pathlib import Path
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

from buildings_bench.data import load_torch_dataset
from buildings_bench.transforms import BoxCoxTransform
from buildings_bench.models import model_factory

try:
    import tomllib as tomli
except ModuleNotFoundError:
    import tomli
import importlib.resources as ir

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

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


# 色系
colors = ["#9BC6DC", "#3492A9", "#216075", "#0B2D41", "#DFAA24", "#DF9722", "#DB841F"]


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


def rmse(pred, true):
    return float(np.sqrt(((pred - true) ** 2).mean()))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # === 配置 ===
    dataset = "cofactor:Office"  # cofactor:Kindergarten, cofactor:School, cofactor:NursingHome, cofactor:Office, cofactor:All, e.g., "cofactor:Kindergarten"
    ctx_len, pred_len = 168, 168
    max_trials = 20  # 最多探索多少个序列来找最佳楼宇样本

    baseline_model = {
        "name": "BuildMoE-top-k-2",
        "ckpt": "/home/hadoop/bec/BuildingsBench/checkpoints/BuildMoE-top-k-2-without-shared-export_best_val.pt",
    }
    other_models = [
        {
            "name": "Moment-L",
            "ckpt": "/home/hadoop/bec/BuildingsBench/checkpoints/Moment-L_best_val.pt",
        },
        {
            "name": "Chronos-Base",
            "ckpt": "/home/hadoop/bec/BuildingsBench/checkpoints/Chronos-Base_best_val.pt",
        },
        {
            "name": "TimeMoE-200M",
            "ckpt": "/home/hadoop/bec/BuildingsBench/checkpoints/TimeMoE-200M_best_val.pt",
        },
    ]

    # === 加载数据集 ===
    if dataset.startswith("cofactor:"):
        dataset_name = "cofactor"
        subtype = dataset.split(":", 1)[1]
    else:
        dataset_name = dataset
        subtype = None

    gen = load_torch_dataset(dataset_name, context_len=ctx_len, pred_len=pred_len)
    ds_list = []
    for bid, bldg_ds in gen:
        if subtype is not None:
            allow = set(cofactor_type.get(subtype, []))
            if bid not in allow:
                continue
        ds_list.append((bid, bldg_ds))
    if not ds_list:
        raise RuntimeError(f"空数据集：{dataset}")

    # === BoxCox ===
    boxcox = BoxCoxTransform()
    tpath = Path(os.environ["BUILDINGS_BENCH"]) / "metadata" / "transforms"
    boxcox.load(tpath)

    # === baseline 模型配置 ===
    model_args = load_model_config(baseline_model["name"])
    model_args["context_len"] = ctx_len
    model_args["pred_len"] = pred_len

    model, _, _ = model_factory(baseline_model["name"], model_args)
    model = model.to(device).eval()
    model.load_from_checkpoint(baseline_model["ckpt"])

    # === 搜索最佳样本 ===
    best_bid, best_rmse, best_seq = None, float("inf"), None
    trials = 0
    # 总次数
    total_trials = len(ds_list) * max_trials
    sum_trials = 0

    for building_id, ds in ds_list:
        for idx in range(len(ds)):
            sample = ds[idx]
            y_raw = sample["load"].astype("float32").squeeze()
            target_raw = y_raw[-pred_len:]

            y_norm = torch.from_numpy(boxcox.transform(y_raw))
            batch = {"load": y_norm.unsqueeze(0).unsqueeze(-1).to(device)}

            with torch.no_grad():
                preds_norm, _ = model.predict(batch, ctx_len, pred_len)
            preds_raw = boxcox.undo_transform(preds_norm.cpu()).squeeze().numpy()

            score = rmse(preds_raw, target_raw)

            if score < best_rmse:
                best_rmse = score
                best_bid = building_id
                best_seq = y_raw

            trials += 1
            sum_trials += 1
            if trials >= max_trials:
                break  # 只跳出当前 building 的样本循环
        if trials >= total_trials:
            break  # 跳出所有建筑循环

    print(f"✅ Best sequence found in building={best_bid}, RMSE={best_rmse:.4f}")

    # === 在最佳序列上跑所有模型 ===
    results = []
    target_raw = best_seq[-pred_len:]
    context_raw = best_seq[:ctx_len]

    for cfg in [baseline_model] + other_models:
        model_name, ckpt_path = cfg["name"], cfg["ckpt"]

        model_args = load_model_config(model_name)
        model_args["context_len"] = ctx_len
        model_args["pred_len"] = pred_len

        model, _, _ = model_factory(model_name, model_args)
        model = model.to(device).eval()
        model.load_from_checkpoint(ckpt_path)

        y_norm = torch.from_numpy(boxcox.transform(best_seq))
        batch = {"load": y_norm.unsqueeze(0).unsqueeze(-1).to(device)}

        with torch.no_grad():
            preds_norm, _ = model.predict(batch, ctx_len, pred_len)
        preds_raw = boxcox.undo_transform(preds_norm.cpu()).squeeze().numpy()

        # 保存结果
        for i, (p_pred, p_true) in enumerate(zip(preds_raw, target_raw), 1):
            results.append(
                {
                    "model": model_name,
                    "building": best_bid,
                    "timestep": i,
                    "predict": float(p_pred),
                    "actual": float(p_true),
                }
            )

        # === 绘图 ===
        plt.figure(figsize=(10, 4))

        # Ground Truth (context + target)
        full_truth = np.concatenate([context_raw, target_raw])
        plt.plot(
            range(1, ctx_len + pred_len + 1),
            full_truth,
            color=colors[0],
            label="Ground Truth",
            linewidth=2,
        )

        # Prediction (only pred_len 部分)
        plt.plot(
            range(ctx_len + 1, ctx_len + pred_len + 1),
            preds_raw,
            color=colors[-1],
            label="Prediction",
            linewidth=2,
        )

        plt.xlabel("Timestep")
        plt.ylabel("Load (kWh)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"plot_{model_name}_{best_bid}.png", dpi=300)
        plt.close()
        print(f"Saved plot_{model_name}_{best_bid}.png")

    # === 保存 CSV ===
    df = pd.DataFrame(results)
    out_file = f"predictions_compare_{dataset.replace(':','_')}_{best_bid}.csv"
    df.to_csv(out_file, index=False)
    print(f"📄 Saved all results to {out_file}")


if __name__ == "__main__":
    main()
