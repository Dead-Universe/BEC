# -*- coding: utf-8 -*-
"""
analyze_clrs_sensitivity.py

独立分析脚本：
- 读取 LHS 计划 (plan_csv) 与结果汇总 (wide_csv)
- 目标变量：run_mean（若缺失则按各栋楼列均值自动生成）
- 特征：rho_star, k_fb, c_pulse, tau_hi, sigma_hi, phi1, phi2
- 产出：
  1) Tornado（按 |SRC| 排序） -> out_dir/*.png + pdf_fig_dir/clrs_tornado.pdf
  2) SRC（带符号条形图，按 |SRC| 排序） -> pdf_fig_dir/clrs_src_signed.pdf
  3) PDP（Top-k） -> out_dir/*.png
  4) 导出 SRC 数值 CSV -> out_dir/tornado_SRC_runmean.csv

用法示例：
python analyze_clrs_sensitivity.py \
  --plan_csv ./lhs_screen_results/clrs_sensitivity_plan_16runs.csv \
  --wide_csv ./lhs_screen_results/all_results_wide.csv \
  --out_dir ./lhs_screen_results/tornado_pdp \
  --top_k 3 \
  --pdf_fig_dir ./figures
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PARAMS = ["rho_star", "k_fb", "c_pulse", "tau_hi", "sigma_hi", "phi1", "phi2"]


def _standardize_vec(y: np.ndarray) -> Tuple[np.ndarray, float, float]:
    mu = np.nanmean(y)
    sd = np.nanstd(y, ddof=1)
    if not np.isfinite(sd) or sd < 1e-12:
        sd = 1.0
    return (y - mu) / sd, mu, sd


def _standardize_mat(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0, ddof=1)
    sd = np.where(~np.isfinite(sd) | (sd < 1e-12), 1.0, sd)
    Xz = (X - mu) / sd
    return Xz, mu, sd


def _fit_src_linear(Xz: np.ndarray, yz: np.ndarray) -> np.ndarray:
    # 标准化线性回归（无截距）；SRC 即系数
    mask = np.isfinite(Xz).all(axis=1) & np.isfinite(yz)
    if mask.sum() < Xz.shape[1]:
        raise ValueError("有效样本过少，无法拟合 SRC。")
    beta, *_ = np.linalg.lstsq(Xz[mask], yz[mask], rcond=None)
    return beta


def _pdp_curve(
    param_name: str,
    grid: np.ndarray,
    params: List[str],
    mu_x: np.ndarray,
    sd_x: np.ndarray,
    mu_y: float,
    sd_y: float,
    beta: np.ndarray,
    med_x: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    j = params.index(param_name)
    Xg = np.tile(med_x, (len(grid), 1))
    Xg[:, j] = grid
    Xgz = (Xg - mu_x) / np.where(sd_x < 1e-12, 1.0, sd_x)
    yhat_z = Xgz @ beta
    yhat = yhat_z * sd_y + mu_y
    return grid, yhat


def _ensure_run_mean(wide_df: pd.DataFrame) -> pd.DataFrame:
    df = wide_df.copy()
    if "run_mean" in df.columns:
        return df
    val_cols = [c for c in df.columns if c != "run_id"]
    val_cols = [c for c in val_cols if np.issubdtype(df[c].dtype, np.number)]
    df["run_mean"] = df[val_cols].mean(axis=1, skipna=True)
    return df


def _export_src_pdfs(src_df_sorted_desc: pd.DataFrame, pdf_fig_dir: Path) -> None:
    """输出两份 PDF：clrs_tornado.pdf 与 clrs_src_signed.pdf"""
    pdf_fig_dir.mkdir(parents=True, exist_ok=True)

    # 1) clrs_tornado.pdf —— 绝对值龙卷风（按 |SRC| 排序，横向条形）
    fig1 = plt.figure(figsize=(7.2, 4.8))
    y_pos = np.arange(len(src_df_sorted_desc))
    # 仅取绝对值，颜色统一；为清晰把条从小到大自下而上绘制
    src_abs_asc = src_df_sorted_desc.sort_values("abs_SRC", ascending=True)
    y_pos1 = np.arange(len(src_abs_asc))
    plt.barh(y_pos1, src_abs_asc["abs_SRC"].values)
    plt.yticks(y_pos1, src_abs_asc["param"].values)
    plt.xlabel("|SRC| on CVRMSE@168")
    plt.title("Tornado (|SRC|) — CLRS sensitivity")
    plt.tight_layout()
    (pdf_fig_dir / "clrs_tornado.pdf").parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(pdf_fig_dir / "clrs_tornado.pdf")
    plt.close(fig1)

    # 2) clrs_src_signed.pdf —— 带符号 SRC（按 |SRC| 排序，横向条形）
    fig2 = plt.figure(figsize=(7.2, 4.8))
    # 关键：按 |SRC| 升序排，让小的在下、最大的在最上
    src_for_signed = src_df_sorted_desc.sort_values("abs_SRC", ascending=True)
    y_pos2 = np.arange(len(src_for_signed))
    plt.barh(y_pos2, src_for_signed["SRC"].values)
    plt.yticks(y_pos2, src_for_signed["param"].values)
    plt.axvline(0.0, linewidth=0.8)
    plt.xlabel("Standardized Regression Coefficient (SRC)")
    plt.title("Signed SRC — CLRS sensitivity")
    plt.tight_layout()
    plt.savefig(pdf_fig_dir / "clrs_src_signed.pdf")
    plt.close(fig2)

    print(f"🖨  PDF 导出：{pdf_fig_dir/'clrs_tornado.pdf'}")
    print(f"🖨  PDF 导出：{pdf_fig_dir/'clrs_src_signed.pdf'}")


def plot_tornado_src(
    plan_df: pd.DataFrame, wide_df: pd.DataFrame, out_dir: Path, pdf_fig_dir: Path
) -> Path:
    df = wide_df.merge(plan_df[["run_id"] + PARAMS], on="run_id", how="left")
    df = df[np.isfinite(df["run_mean"])]

    X = df[PARAMS].to_numpy(float)
    y = df["run_mean"].to_numpy(float)

    Xz, mu_x, sd_x = _standardize_mat(X)
    yz, mu_y, sd_y = _standardize_vec(y)

    beta = _fit_src_linear(Xz, yz)
    src_df = pd.DataFrame({"param": PARAMS, "SRC": beta, "abs_SRC": np.abs(beta)})
    # 降序（大到小）用于“主—次”从上到下显示
    src_df_sorted_desc = src_df.sort_values("abs_SRC", ascending=False)

    out_dir.mkdir(parents=True, exist_ok=True)

    # 原 PNG（保留）
    plt.figure(figsize=(7, 4.6))
    y_pos = np.arange(len(src_df_sorted_desc))
    plt.barh(y_pos, src_df_sorted_desc["SRC"].values)
    plt.yticks(y_pos, src_df_sorted_desc["param"].values)
    plt.axvline(0.0, linewidth=0.8)
    plt.xlabel("Standardized Regression Coefficient (SRC)")
    plt.title("Tornado Plot — SRC on CVRMSE@168 (run_mean)")
    plt.tight_layout()
    tornado_png = out_dir / "tornado_SRC_runmean.png"
    plt.savefig(tornado_png, dpi=150)
    plt.close()

    # CSV（保留）
    src_csv = out_dir / "tornado_SRC_runmean.csv"
    src_df_sorted_desc.to_csv(src_csv, index=False)

    # PDF（新增）：clrs_tornado.pdf（|SRC|）与 clrs_src_signed.pdf（带符号）
    _export_src_pdfs(src_df_sorted_desc, pdf_fig_dir)

    # 保存用于 PDP 的标准化器与回归参数（保留）
    np.savez(
        out_dir / "src_fit_artifacts.npz",
        mu_x=mu_x,
        sd_x=sd_x,
        mu_y=np.array([mu_y]),
        sd_y=np.array([sd_y]),
        beta=beta,
        med_x=np.median(X, axis=0),
    )

    print(f"🖼  Tornado PNG：{tornado_png}")
    print(f"📄  Tornado 数据：{src_csv}")
    return tornado_png


def plot_topk_pdp(
    plan_df: pd.DataFrame, wide_df: pd.DataFrame, out_dir: Path, top_k: int = 3
) -> None:
    art = np.load(out_dir / "src_fit_artifacts.npz")
    mu_x = art["mu_x"]
    sd_x = art["sd_x"]
    mu_y = float(art["mu_y"][0])
    sd_y = float(art["sd_y"][0])
    beta = art["beta"]
    med_x = art["med_x"]

    src_df = pd.DataFrame({"param": PARAMS, "SRC": beta, "abs_SRC": np.abs(beta)})
    top_params = (
        src_df.sort_values("abs_SRC", ascending=False)["param"].head(top_k).tolist()
    )

    X = plan_df[PARAMS].to_numpy(float)

    for p in top_params:
        j = PARAMS.index(p)
        p_min = np.nanmin(X[:, j])
        p_max = np.nanmax(X[:, j])

        zero_pt = None
        if p == "sigma_hi":
            has_zero = np.any(X[:, j] == 0.0)
            pos = X[X[:, j] > 0, j]
            if pos.size == 0:
                gx = np.array([0.0])
            else:
                lo = max(pos.min(), 1e-5)
                hi = max(lo * 1.0001, max(pos.max(), 1e-5 * 1.0001))
                gx = np.exp(np.linspace(np.log(lo), np.log(hi), 120))
                if has_zero:
                    zero_pt = 0.0
        else:
            gx = np.linspace(p_min, p_max, 120)

        gy_x, gy_y = _pdp_curve(
            param_name=p,
            grid=gx,
            params=PARAMS,
            mu_x=mu_x,
            sd_x=sd_x,
            mu_y=mu_y,
            sd_y=sd_y,
            beta=beta,
            med_x=med_x,
        )

        plt.figure(figsize=(6.2, 4.2))
        plt.plot(gy_x, gy_y, linewidth=2)
        if p == "sigma_hi" and np.all(gy_x > 0):
            plt.xscale("log")
        if zero_pt is not None:
            gx0 = max(1e-6, np.min(gy_x[gy_x > 0]) * 0.5) if np.any(gy_x > 0) else 1e-6
            gx_tmp = np.array([gx0])
            _, y0 = _pdp_curve(p, gx_tmp, PARAMS, mu_x, sd_x, mu_y, sd_y, beta, med_x)
            plt.scatter([zero_pt + 1e-12], [float(y0)], s=35, marker="x")

        plt.xlabel(p)
        plt.ylabel("Predicted CVRMSE@168 (%)")
        plt.title(f"PDP — {p} (others at median)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_png = out_dir / f"pdp_{p}.png"
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"🖼  PDP：{out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--plan_csv",
        type=Path,
        required=True,
        help="LHS 计划：clrs_sensitivity_plan_16runs.csv",
    )
    ap.add_argument(
        "--wide_csv", type=Path, required=True, help="结果宽表：all_results_wide.csv"
    )
    ap.add_argument("--out_dir", type=Path, default=Path("./tornado_pdp"))
    ap.add_argument("--top_k", type=int, default=3)
    ap.add_argument(
        "--pdf_fig_dir",
        type=Path,
        default=Path("./figures"),
        help="PDF 图导出目录，默认 ./figures（输出 clrs_tornado.pdf 与 clrs_src_signed.pdf）",
    )
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    args.pdf_fig_dir.mkdir(parents=True, exist_ok=True)

    # 读表
    plan_df = pd.read_csv(args.plan_csv)
    wide_df = pd.read_csv(args.wide_csv)

    # 确保 run_mean 可用
    wide_df = _ensure_run_mean(wide_df)

    # 基础字段检查
    for col in ["run_id", "run_mean"]:
        if col not in wide_df.columns:
            raise ValueError(f"结果宽表缺少列：{col}")
    if not all(p in plan_df.columns for p in PARAMS):
        missing = [p for p in PARAMS if p not in plan_df.columns]
        raise ValueError(f"计划表缺少参数列：{missing}")

    # 合并前先确保 run_id 为整数
    wide_df["run_id"] = wide_df["run_id"].astype(int)
    plan_df["run_id"] = plan_df["run_id"].astype(int)

    # Tornado（SRC）+ PDF 导出
    tornado_png = plot_tornado_src(plan_df, wide_df, out_dir, args.pdf_fig_dir)

    # PDP（Top-k）
    plot_topk_pdp(plan_df, wide_df, out_dir, top_k=args.top_k)

    print("\n🎯 完成：")
    print(f"  - Tornado PNG：{tornado_png}")
    print(f"  - PDF：{args.pdf_fig_dir/'clrs_tornado.pdf'}")
    print(f"  - PDF：{args.pdf_fig_dir/'clrs_src_signed.pdf'}")
    print(f"  - PDP × {args.top_k}：输出到 {out_dir}")


if __name__ == "__main__":
    main()
