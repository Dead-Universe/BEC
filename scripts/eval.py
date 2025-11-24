# -*- coding: utf-8 -*-
"""
extreme_and_normal_eval_batched.py

功能：
- 对每栋楼的极端事件窗口（来自 events_dir 下的 CSV）与非极端窗口（normal）做预测评估
- 使用批次加速（可调 batch_size）
- 输出：
  1) 每栋楼的明细：extreme + normal -> ./extreme_eval_plots/{dataset_name}_{building_id}_summary.csv
  2) 全部合并：./extreme_eval_plots/{dataset_name}_all_buildings_summary.csv
  3) 分类对比汇总（extreme vs normal）：./extreme_eval_plots/{dataset_name}_category_summary.csv
"""

import os
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from importlib import resources as ir
import tomli

# ===== BuildingsBench =====
from buildings_bench.models import model_factory
from buildings_bench.transforms import BoxCoxTransform


# ------------------------------
# 配置：COFACTOR 建筑类型映射
# ------------------------------
COFACTOR_TYPE = {
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


# ------------------------------
# 工具函数
# ------------------------------
def load_model_config(model_name: str) -> dict:
    """读取 buildings_bench/configs 下的模型配置文件"""
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


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    denom_mean: float,
    mask: np.ndarray | None = None,
) -> Dict[str, float]:
    """
    输出两组指标：
      - 百分比：CVRMSE/NMAE/NMBE（分母=denom_mean）
      - 绝对值：RMSE_kWh/MAE_kWh（不归一化）
    """
    if mask is not None:
        y_true = y_true[mask]
        y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {
            "CVRMSE": np.nan,
            "NMAE": np.nan,
            "NMBE": np.nan,
            "RMSE_kWh": np.nan,
            "MAE_kWh": np.nan,
        }

    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    mbe = float(np.mean(y_pred - y_true))

    if (not np.isfinite(denom_mean)) or denom_mean == 0.0:
        cvrmse = np.nan
        nmae = np.nan
        nmbe = np.nan
    else:
        cvrmse = 100.0 * rmse / denom_mean
        nmae = 100.0 * mae / denom_mean
        nmbe = 100.0 * mbe / denom_mean

    return {
        "CVRMSE": cvrmse,
        "NMAE": nmae,
        "NMBE": nmbe,
        "RMSE_kWh": rmse,
        "MAE_kWh": mae,
    }


def extract_building_id_from_events_filename(fname: str) -> str:
    """
    例如：building_6396_events.csv -> building6396
    """
    stem = Path(fname).stem
    digits = "".join([c for c in stem if c.isdigit()])
    return f"building{digits}" if digits else stem


def assemble_windows_for_events(
    df: pd.DataFrame,
    events: pd.DataFrame,
    context_len: int,
    pred_len: int,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, pd.Timestamp, pd.Timestamp]]:
    """
    返回 [(x_context, y_true, y_mask, t0, tend), ...]
    y_mask: 目标段有效性掩码（基于 df['valid']）
    """
    windows = []
    for _, e in events.iterrows():
        t0 = pd.to_datetime(e["start"])
        tend = pd.to_datetime(e["end"])
        df_in = df[
            (df["timestamp"] >= t0 - pd.Timedelta(hours=context_len))
            & (df["timestamp"] < t0)
        ]
        df_out = df[
            (df["timestamp"] >= t0)
            & (df["timestamp"] < t0 + pd.Timedelta(hours=pred_len))
        ]

        if len(df_in) < context_len or len(df_out) < pred_len:
            continue

        x_ctx = df_in["power"].values.astype("float32")
        y_true = df_out["power"].values.astype("float32")
        y_mask = df_out["valid"].values.astype(bool)

        windows.append((x_ctx, y_true, y_mask, t0, tend))
    return windows


def build_extreme_mask(df: pd.DataFrame, events: pd.DataFrame) -> np.ndarray:
    """
    构造布尔掩码，标记 df['timestamp'] 中哪些小时处于任一极端事件内
    """
    ts = df["timestamp"].values
    mask = np.zeros(len(ts), dtype=bool)
    for _, e in events.iterrows():
        t0 = pd.to_datetime(e["start"])
        tend = pd.to_datetime(e["end"])
        idx = (df["timestamp"] >= t0) & (df["timestamp"] < tend)
        mask[idx.values] = True
    return mask


def assemble_windows_for_normal(
    df: pd.DataFrame,
    extreme_mask: np.ndarray,
    context_len: int,
    pred_len: int,
    stride: int = 168,
    exclude_context: bool = True,
    max_windows: int | None = None,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, pd.Timestamp, pd.Timestamp]]:
    """
    返回 [(x_context, y_true, y_mask, t0, t1), ...]
    y_mask: 目标段有效性掩码（基于 df['valid']）
    """
    N = len(df)
    windows = []
    i = 0
    while i + context_len + pred_len <= N:
        ctx_start = i
        ctx_end = i + context_len
        pred_end = ctx_end + pred_len

        pred_overlap = extreme_mask[ctx_end:pred_end].any()
        ctx_overlap = (
            extreme_mask[ctx_start:ctx_end].any() if exclude_context else False
        )

        if (not pred_overlap) and (not ctx_overlap):
            x_ctx = df.iloc[ctx_start:ctx_end]["power"].values.astype("float32")
            y_true = df.iloc[ctx_end:pred_end]["power"].values.astype("float32")
            y_mask = df.iloc[ctx_end:pred_end]["valid"].values.astype(bool)

            t0 = df.iloc[ctx_end]["timestamp"]
            t1 = df.iloc[pred_end - 1]["timestamp"]
            windows.append((x_ctx, y_true, y_mask, t0, t1))

            if (max_windows is not None) and (len(windows) >= max_windows):
                break

        i += stride
    return windows


def predict_batched(
    model,
    boxcox: BoxCoxTransform,
    device: str,
    contexts: List[np.ndarray],
    context_len: int,
    pred_len: int,
    batch_size: int = 64,
) -> np.ndarray:
    """
    批量预测：输入多个 context（每个长度 context_len），
    内部拼接 0 填充到 context_len+pred_len，并一次性送入 model.predict。
    返回：y_pred_all, shape [B, pred_len]
    """
    if len(contexts) == 0:
        return np.empty((0, pred_len), dtype="float32")

    B = len(contexts)
    y_preds = []

    # 分块批次
    for s in range(0, B, batch_size):
        e = min(s + batch_size, B)
        ctx_batch = contexts[s:e]

        # 组装 [b, context_len+pred_len, 1]
        loads_np = np.zeros((e - s, context_len + pred_len), dtype="float32")
        for k, ctx in enumerate(ctx_batch):
            loads_np[k, :context_len] = ctx  # 未来段填 0

        # BoxCox 逐样本变换（安全）
        loads_norm = np.stack([boxcox.transform(x) for x in loads_np], axis=0)

        batch = {
            "load": torch.from_numpy(loads_norm).unsqueeze(-1).to(device)  # [b, T, 1]
        }

        with torch.no_grad():
            preds_norm, _ = model.predict(
                batch, context_len, pred_len
            )  # 形状依模型而定

        preds_norm = (
            preds_norm.detach().cpu().numpy()
        )  # 可能是 [b, T, 1] 或 [b, pred_len, 1]
        preds_denorm = []
        for k in range(preds_norm.shape[0]):
            arr = preds_norm[k].squeeze()  # [T] 或 [pred_len]
            # 统一反变换 & 取最后 pred_len
            inv = boxcox.undo_transform(arr)
            if inv.shape[0] >= pred_len:
                inv = inv[-pred_len:]
            preds_denorm.append(inv.astype("float32"))
        y_preds.append(np.stack(preds_denorm, axis=0))

    return np.concatenate(y_preds, axis=0)  # [B, pred_len]


def evaluate_building_extreme_and_normal(
    building_id: str,
    events_csv: Path,
    base_dir: Path,
    model,
    boxcox: BoxCoxTransform,
    *,
    dataset_name: str = "cofactor_eval",
    context_len: int = 168,
    pred_len: int = 168,
    batch_size: int = 64,
    normal_stride: int = 168,
    normal_exclude_context: bool = True,
    normal_max_windows_per_building: int | None = None,
    output_dir: Path = Path("./extreme_eval_plots"),
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    make_plots: bool = False,
) -> pd.DataFrame:
    """
    对单栋楼同时评估极端与非极端窗口（均使用批次加速），返回明细 DataFrame。
    评价分母：使用本次评估所有目标点(所有窗口×pred_len)的均值 \bar{y}_{b,τ}
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 读取多年度 CSV 并补齐小时节点
    pattern = f"{building_id}_clean=*.csv"
    csvs = sorted(base_dir.glob(pattern))
    if not csvs:
        print(f"❌ 未找到匹配文件: {base_dir}/{pattern}")
        return pd.DataFrame()

    all_df = []
    for f in csvs:
        d = pd.read_csv(f, parse_dates=["timestamp"]).sort_values("timestamp")
        all_df.append(d)
    df = (
        pd.concat(all_df, ignore_index=True)
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    full_index = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="H")
    df = (
        df.set_index("timestamp")
        .reindex(full_index)
        .rename_axis("timestamp")
        .reset_index()
    )
    df["valid"] = ~df["power"].isna()
    df["power"] = df["power"].fillna(0.0)

    # ===== 读取事件 =====
    if (not events_csv.exists()) or (events_csv.stat().st_size == 0):
        print(f"⚠️ {events_csv.name} 空或不存在，跳过极端评估，只做 normal。")
        events = pd.DataFrame(columns=["start", "end"])
    else:
        try:
            events = pd.read_csv(events_csv)
            if not {"start", "end"}.issubset(set(events.columns)):
                print(
                    f"⚠️ {events_csv.name} 缺少 start/end 列，按空事件处理，只做 normal。"
                )
                events = pd.DataFrame(columns=["start", "end"])
            else:
                events["start"] = pd.to_datetime(events["start"], errors="coerce")
                events["end"] = pd.to_datetime(events["end"], errors="coerce")
                events = events.dropna(subset=["start", "end"])
        except pd.errors.EmptyDataError:
            print(f"⚠️ {events_csv.name} 为空文件（无列），只做 normal。")
            events = pd.DataFrame(columns=["start", "end"])
        except Exception as e:
            print(f"⚠️ 无法读取 {events_csv.name}: {e}，只做 normal。")
            events = pd.DataFrame(columns=["start", "end"])

    # ===== 先抽取窗口，不立刻算指标 =====
    # 极端窗口
    extreme_windows = assemble_windows_for_events(df, events, context_len, pred_len)
    extreme_contexts = [w[0] for w in extreme_windows]
    extreme_y_trues = [w[1] for w in extreme_windows]
    extreme_masks = [w[2] for w in extreme_windows]
    extreme_t0t1 = [(w[3], w[4]) for w in extreme_windows]

    # normal 窗口
    extreme_mask = (
        build_extreme_mask(df, events)
        if not events.empty
        else np.zeros(len(df), dtype=bool)
    )
    normal_windows = assemble_windows_for_normal(
        df,
        extreme_mask,
        context_len,
        pred_len,
        stride=normal_stride,
        exclude_context=normal_exclude_context,
        max_windows=normal_max_windows_per_building,
    )
    normal_contexts = [w[0] for w in normal_windows]
    normal_y_trues = [w[1] for w in normal_windows]
    normal_masks = [w[2] for w in normal_windows]
    normal_t0t1 = [(w[3], w[4]) for w in normal_windows]

    # ===== 基于“本次评估的所有目标点”计算 \bar{y}_{b,τ} =====
    all_targets = []
    if len(extreme_y_trues) > 0:
        all_targets.append(np.concatenate(extreme_y_trues).astype("float32"))
        mean_ext = (
            float(
                np.mean(
                    np.concatenate(
                        [yt[m] for yt, m in zip(extreme_y_trues, extreme_masks)]
                    )
                )
            )
            if len(extreme_y_trues) > 0
            else np.nan
        )
    else:
        mean_ext = np.nan
    if len(normal_y_trues) > 0:
        all_targets.append(np.concatenate(normal_y_trues).astype("float32"))
        mean_nor = (
            float(
                np.mean(
                    np.concatenate(
                        [yt[m] for yt, m in zip(normal_y_trues, normal_masks)]
                    )
                )
            )
            if len(normal_y_trues) > 0
            else np.nan
        )
    else:
        mean_nor = np.nan

    print("mean_extreme(valid-only) =", mean_ext if np.isfinite(mean_ext) else "N/A")
    print("mean_normal(valid-only)  =", mean_nor if np.isfinite(mean_nor) else "N/A")

    if len(all_targets) == 0:
        print(f"⚠️ {building_id}: 评估窗口为空，跳过。")
        return pd.DataFrame()

    eval_mean = float(np.mean(np.concatenate(all_targets)))
    if eval_mean == 0.0:
        print(f"⚠️ {building_id}: 评估集目标点均值为 0，跳过。")
        return pd.DataFrame()

    # ===== 现在再去批量预测，并用 eval_mean 计算指标 =====
    rows = []

    # Extreme
    if len(extreme_contexts) > 0 and np.isfinite(mean_ext) and mean_ext != 0.0:
        extreme_preds = predict_batched(
            model, boxcox, device, extreme_contexts, context_len, pred_len, batch_size
        )
        for k in range(len(extreme_contexts)):
            met = compute_metrics(
                extreme_y_trues[k], extreme_preds[k], mean_ext, mask=extreme_masks[k]
            )
            met.update(
                {
                    "dataset": dataset_name,
                    "building_id": building_id,
                    "event_type": "extreme",
                    "event_start": extreme_t0t1[k][0],
                    "event_end": extreme_t0t1[k][1],
                }
            )
            rows.append(met)

    # Normal
    if len(normal_contexts) > 0 and np.isfinite(mean_nor) and mean_nor != 0.0:
        normal_preds = predict_batched(
            model, boxcox, device, normal_contexts, context_len, pred_len, batch_size
        )
        for k in range(len(normal_contexts)):
            met = compute_metrics(
                normal_y_trues[k], normal_preds[k], mean_nor, mask=normal_masks[k]
            )
            met.update(
                {
                    "dataset": dataset_name,
                    "building_id": building_id,
                    "event_type": "normal",
                    "event_start": normal_t0t1[k][0],
                    "event_end": normal_t0t1[k][1],
                }
            )
            rows.append(met)

    df_out = pd.DataFrame(rows)

    # 保存单楼明细
    single_out = output_dir / f"{dataset_name}_{building_id}_summary.csv"
    df_out.to_csv(single_out, index=False)

    # 可选画图（保持原逻辑）
    if make_plots and len(extreme_windows) > 0:
        kmax = max(3, len(extreme_windows))
        for k in range(kmax):
            x_ctx = extreme_windows[k][0]
            y_true = extreme_windows[k][1]
            x_full = np.zeros(context_len + pred_len, dtype="float32")
            x_full[:context_len] = x_ctx
            x_norm = (
                torch.from_numpy(boxcox.transform(x_full))
                .unsqueeze(0)
                .unsqueeze(-1)
                .to(device)
            )
            with torch.no_grad():
                preds_norm, _ = model.predict({"load": x_norm}, context_len, pred_len)
            preds_raw = preds_norm.squeeze().detach().cpu().numpy()
            preds_raw = boxcox.undo_transform(preds_raw)
            y_pred = preds_raw[-pred_len:]

            x_input = np.arange(-context_len, 0)
            x_output = np.arange(0, pred_len)
            y_full_actual = np.concatenate([x_ctx, y_true])

            plt.figure(figsize=(10, 4))
            plt.plot(
                np.concatenate([x_input, x_output]),
                y_full_actual,
                label="Actual",
                linewidth=2.0,
            )
            plt.plot(x_output, y_pred, label="Predicted", linewidth=1.2)
            plt.axvline(0, color="gray", linestyle=":", linewidth=1)
            plt.title(
                f"{building_id}: {extreme_t0t1[k][0].date()} → {extreme_t0t1[k][1].date()} (extreme)"
            )
            plt.xlabel("Hour Relative to Forecast Start")
            plt.ylabel("Load (kWh)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                output_dir
                / f"{building_id}_{extreme_t0t1[k][0].date()}_{extreme_t0t1[k][1].date()}.pdf",
            )
            plt.close()

    print(
        f"✅ {building_id} 完成：extreme={len(extreme_contexts)}，normal={len(normal_contexts)}"
    )
    return df_out


def summarize_by_category_extreme_vs_normal(
    df_all: pd.DataFrame,
    cofactor_type: Dict[str, List[str]],
    out_csv: Path,
) -> pd.DataFrame:
    """
    生成分类对比汇总（extreme vs normal），并保存到 out_csv
    列：
      Type, Count, Extreme_CVRMSE, Normal_CVRMSE, Extreme_NMAE, Normal_NMAE, Extreme_NMBE, Normal_NMBE
    追加 Overall 行
    """
    rows = []
    for cat, ids in cofactor_type.items():
        sub = df_all[df_all["building_id"].isin(ids)]

        sub_ext = sub[sub["event_type"] == "extreme"]
        sub_nor = sub[sub["event_type"] == "normal"]

        if sub.empty:
            continue

        rows.append(
            {
                "Type": cat,
                "Count": int(sub["building_id"].nunique()),
                "Extreme_CVRMSE": (
                    float(sub_ext["CVRMSE"].mean()) if not sub_ext.empty else np.nan
                ),
                "Normal_CVRMSE": (
                    float(sub_nor["CVRMSE"].mean()) if not sub_nor.empty else np.nan
                ),
                "Extreme_NMAE": (
                    float(sub_ext["NMAE"].mean()) if not sub_ext.empty else np.nan
                ),
                "Normal_NMAE": (
                    float(sub_nor["NMAE"].mean()) if not sub_nor.empty else np.nan
                ),
                "Extreme_NMBE": (
                    float(sub_ext["NMBE"].mean()) if not sub_ext.empty else np.nan
                ),
                "Normal_NMBE": (
                    float(sub_nor["NMBE"].mean()) if not sub_nor.empty else np.nan
                ),
                # 新增绝对误差汇总
                "Extreme_RMSE_kWh": (
                    float(sub_ext["RMSE_kWh"].mean()) if not sub_ext.empty else np.nan
                ),
                "Normal_RMSE_kWh": (
                    float(sub_nor["RMSE_kWh"].mean()) if not sub_nor.empty else np.nan
                ),
                "Extreme_MAE_kWh": (
                    float(sub_ext["MAE_kWh"].mean()) if not sub_ext.empty else np.nan
                ),
                "Normal_MAE_kWh": (
                    float(sub_nor["MAE_kWh"].mean()) if not sub_nor.empty else np.nan
                ),
            }
        )

    # Overall
    sub = df_all
    sub_ext = sub[sub["event_type"] == "extreme"]
    sub_nor = sub[sub["event_type"] == "normal"]
    rows.append(
        {
            "Type": "Overall",
            "Count": int(sub["building_id"].nunique()),
            "Extreme_CVRMSE": (
                float(sub_ext["CVRMSE"].mean()) if not sub_ext.empty else np.nan
            ),
            "Normal_CVRMSE": (
                float(sub_nor["CVRMSE"].mean()) if not sub_nor.empty else np.nan
            ),
            "Extreme_NMAE": (
                float(sub_ext["NMAE"].mean()) if not sub_ext.empty else np.nan
            ),
            "Normal_NMAE": (
                float(sub_nor["NMAE"].mean()) if not sub_nor.empty else np.nan
            ),
            "Extreme_NMBE": (
                float(sub_ext["NMBE"].mean()) if not sub_ext.empty else np.nan
            ),
            "Normal_NMBE": (
                float(sub_nor["NMBE"].mean()) if not sub_nor.empty else np.nan
            ),
            # 新增四列：绝对误差
            "Extreme_RMSE_kWh": (
                float(sub_ext["RMSE_kWh"].mean()) if not sub_ext.empty else np.nan
            ),
            "Normal_RMSE_kWh": (
                float(sub_nor["RMSE_kWh"].mean()) if not sub_nor.empty else np.nan
            ),
            "Extreme_MAE_kWh": (
                float(sub_ext["MAE_kWh"].mean()) if not sub_ext.empty else np.nan
            ),
            "Normal_MAE_kWh": (
                float(sub_nor["MAE_kWh"].mean()) if not sub_nor.empty else np.nan
            ),
        }
    )

    out = pd.DataFrame(rows)
    out_rounded = out.copy()
    for col in [
        "Extreme_CVRMSE",
        "Normal_CVRMSE",
        "Extreme_NMAE",
        "Normal_NMAE",
        "Extreme_NMBE",
        "Normal_NMBE",
        "Extreme_RMSE_kWh",
        "Normal_RMSE_kWh",
        "Extreme_MAE_kWh",
        "Normal_MAE_kWh",
    ]:
        out_rounded[col] = out_rounded[col].round(3)
    out_rounded.to_csv(out_csv, index=False)
    return out_rounded


# ------------------------------
# 主流程
# ------------------------------
if __name__ == "__main__":
    # ===== 路径与超参 =====
    baseline_model = {
        "name": "BuildMoE-top-k-2-without-shared-export",
        "ckpt": "/home/hadoop/bec/BuildingsBench/checkpoints/BuildMoE-top-k-2-without-shared-export_best_val.pt",
    }

    base_dir = Path(
        "/home/hadoop/bec/buildings-bench/v2.0.0/BuildingsBench/cofactor"
    )  # 楼宇时序 CSV（*_clean=YYYY.csv）
    events_dir = Path(
        "/home/hadoop/bec/BuildingsBench/scripts/events"
    )  # 每栋楼的事件 CSV（含 start/end）
    output_dir = Path("./extreme_eval_plots")
    dataset_name = "cofactor_eval"

    context_len = 168
    pred_len = 168
    batch_size = 64
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # normal 抽样设置
    normal_stride = 168  # 每 168 小时取一个 normal 窗口
    normal_exclude_context = True
    normal_max_windows_per_building = None  # 可设为比如 512 限制数量

    make_plots = True  # True 则为每栋楼画少量示例图

    # ===== 模型加载 =====
    print("🔧 Loading transforms & model...")
    boxcox = BoxCoxTransform()
    tpath = Path(os.environ["BUILDINGS_BENCH"]) / "metadata" / "transforms"
    boxcox.load(tpath)

    model_args = load_model_config(baseline_model["name"])
    model, _, _ = model_factory(baseline_model["name"], model_args)
    model = model.to(device).eval()
    model.load_from_checkpoint(baseline_model["ckpt"])

    # ===== 遍历事件文件（每栋楼一个）=====
    all_dfs = []
    event_files = sorted(events_dir.glob("*.csv"))
    if not event_files:
        print(f"❌ 事件目录为空：{events_dir}")
        raise SystemExit(1)

    for ev in event_files:
        building_id = extract_building_id_from_events_filename(ev.name)
        print("\n==============================")
        print(f"开始评估 {building_id} ({ev.name})")

        df_b = evaluate_building_extreme_and_normal(
            building_id=building_id,
            events_csv=ev,
            base_dir=base_dir,
            model=model,
            boxcox=boxcox,
            dataset_name=dataset_name,
            context_len=context_len,
            pred_len=pred_len,
            batch_size=batch_size,
            normal_stride=normal_stride,
            normal_exclude_context=normal_exclude_context,
            normal_max_windows_per_building=normal_max_windows_per_building,
            output_dir=output_dir,
            device=device,
            make_plots=make_plots,
        )
        if not df_b.empty:
            all_dfs.append(df_b)

    # ===== 合并 & 输出全局明细 =====
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        all_path = output_dir / f"{dataset_name}_all_buildings_summary.csv"
        final_df.to_csv(all_path, index=False)
        print(f"\n📊 所有建筑评估完成，总计 {len(final_df)} 条窗口（extreme+normal）。")
        print(f"✅ 汇总文件已保存：{all_path}")

        # ===== 分类对比（extreme vs normal）=====
        cat_path = output_dir / f"{dataset_name}_category_summary.csv"
        cat_df = summarize_by_category_extreme_vs_normal(
            final_df, COFACTOR_TYPE, cat_path
        )
        print("\n===== 分类对比（extreme vs normal）=====")
        print(cat_df.to_string(index=False))
        print(f"\n✅ 分类对比已保存：{cat_path}")
    else:
        print("⚠️ 未得到任何评估结果，请检查输入路径与数据。")
