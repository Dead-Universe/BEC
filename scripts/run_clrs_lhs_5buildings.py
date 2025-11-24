# -*- coding: utf-8 -*-
"""
run_clrs_lhs_5buildings.py

功能：
- 自动选择 5 栋楼（cofactor 则跨类型选，其他数据集顺序选 5 栋）
- 生成 16 组 CLRS 超参的 LHS 计划（固定随机种子；sigma_hi 用 log 采样 + 若干 run 置 0）
- 逐 run × 楼 调用 evaluator 脚本（single_building_cvrmse168.py）计算 CVRMSE@168（valid-only 口径）
- 汇总所有结果并绘图（热力图 + 每栋楼折线）

用法示例：
python run_clrs_lhs_5buildings.py \
  --baseline_ckpt /path/to/BuildMoE-top-k-2-without-shared-export_best_val.pt \
  --evaluator_py ./single_building_cvrmse168.py \
  --output_dir ./lhs_screen_results
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -------------------------
# 固定：cofactor 类型映射（用于多样化选楼）
# -------------------------
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


# -------------------------
# 选取 5 栋楼
# -------------------------
def pick_five_buildings(output_dir: Path, seed: int = 2025) -> List[dict]:
    """
    选 5 栋：
      - cofactor: Kindergarten/School/NursingHome/Office 各 1 栋（随机，固定种子）
      - university: 学生公寓 A/B 随机 1 栋
    返回：[{"dataset":"cofactor","building":"..."}, ..., {"dataset":"university","building":"A|B"}]
    同时写出 selected_buildings.txt 以便复现
    """
    import random

    random.seed(seed)

    # 四类各随机 1 栋
    chosen = []
    for cat in ["Kindergarten", "School", "NursingHome", "Office"]:
        pool = COFACTOR_TYPE[cat]
        bid = random.choice(pool)
        chosen.append({"dataset": "cofactor", "building": bid})

    # 学生公寓（内置，不依赖外部入参）
    uni_bid = random.choice(["A", "B"])
    chosen.append({"dataset": "university", "building": uni_bid})

    # 写出选择结果
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "selected_buildings.txt").open("w", encoding="utf-8") as f:
        for rec in chosen:
            f.write(f"{rec['dataset']},{rec['building']}\n")

    print("✅ 本次用于 LHS 微调实验的 5 栋楼：")
    for rec in chosen:
        print(f"  {rec['dataset']:<12} → {rec['building']}")
    print(f"已写出 → {output_dir / 'selected_buildings.txt'}")

    return chosen


# -------------------------
# 生成 16 组 LHS（sigma_hi: log 采样 + 零噪声对照）
# -------------------------
def latin_hypercube_1d(n: int) -> np.ndarray:
    """
    标准 [0,1) LHS：n 个等宽 bin 内各采 1 次，然后整体打乱。
    """
    bins = np.linspace(0.0, 1.0, n + 1)
    u = bins[:-1] + (bins[1:] - bins[:-1]) * np.random.rand(n)
    np.random.shuffle(u)
    return u


def make_16runs_clrs_plan(
    seed: int = 2025, n_runs: int = 16, n_sigma_zero: int = 2
) -> pd.DataFrame:
    """
    7 个 CLRS 旋钮：
      - rho_star:   0.75~0.95   (linear)
      - k_fb:       0.05~0.40   (linear)
      - c_pulse:    0.20~2.00   (linear)
      - tau_hi:     2.00~3.00   (linear)
      - sigma_hi:   0~0.06      (log 采样 + 显式 0；对照用 n_sigma_zero 条)
      - phi1:       0.10~0.25   (linear)
      - phi2:       0.55~0.75   (linear)
    固定项：
      - tau_lo=1.0, sigma_lo=0.0, alpha_ema=0.1, tau_min=1.0, tau_max=3.0,
        sigma_min=0.0, sigma_max=0.1
    """
    np.random.seed(seed)
    df = pd.DataFrame({"run_id": np.arange(1, n_runs + 1)})

    # 线性口径
    def lin_range(lo, hi):
        u = latin_hypercube_1d(n_runs)
        return lo + (hi - lo) * u

    df["rho_star"] = lin_range(0.75, 0.95)
    df["k_fb"] = lin_range(0.05, 0.40)
    df["c_pulse"] = lin_range(0.20, 2.00)
    df["tau_hi"] = lin_range(2.00, 3.00)
    df["phi1"] = lin_range(0.10, 0.25)
    df["phi2"] = lin_range(0.55, 0.75)

    # sigma_hi: log 采样（排除 0），再随机挑若干 run 置 0
    lo, hi = 1e-4, 6e-2
    u = latin_hypercube_1d(n_runs)
    sigma_vals = lo * (hi / lo) ** u  # 对应 log-uniform
    # 随机置零的索引
    zero_idx = np.random.choice(
        np.arange(n_runs), size=min(n_sigma_zero, n_runs), replace=False
    )
    sigma_vals[zero_idx] = 0.0
    df["sigma_hi"] = sigma_vals

    # 固定项（仅记录在 CSV 便于复现）
    df["tau_lo"] = 1.00
    df["sigma_lo"] = 0.00
    df["alpha_ema"] = 0.10
    df["tau_min"] = 1.00
    df["tau_max"] = 3.00
    df["sigma_min"] = 0.00
    df["sigma_max"] = 0.10
    df["comment"] = "16-run LHS; sigma_hi log + zeros; λ_aux fixed (not in plan)."

    # 列顺序友好化
    cols = [
        "run_id",
        "rho_star",
        "k_fb",
        "c_pulse",
        "tau_hi",
        "sigma_hi",
        "phi1",
        "phi2",
        "tau_lo",
        "sigma_lo",
        "alpha_ema",
        "tau_min",
        "tau_max",
        "sigma_min",
        "sigma_max",
        "comment",
    ]
    return df[cols]


# -------------------------
# 评测执行
# -------------------------
def run_one_eval(
    evaluator_py: Path,
    baseline_ckpt: Path,
    dataset_name: str,
    building_id: str,
    device: str,
    out_dir_for_building: Path,
    plan_csv: Path,
    lhs_run: int,
    context_len: int = 168,
    pred_len: int = 168,
    batch_size: int = 64,
    do_finetune: bool = True,
    ft_batch_size: int = 64,
    extra_env: Optional[Dict[str, str]] = None,
) -> Optional[Path]:
    """
    调用 single_building_cvrmse168.py，返回其输出 CSV 路径（或 None 表失败）。
    """
    out_dir_for_building.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(evaluator_py),
        "--baseline_ckpt",
        str(baseline_ckpt),
        "--dataset_name",
        dataset_name,
        "--building_id",
        building_id,
        "--context_len",
        str(context_len),
        "--pred_len",
        str(pred_len),
        "--batch_size",
        str(batch_size),
        "--device",
        device,
        "--output_dir",
        str(out_dir_for_building),
        "--clrs_plan_csv",
        str(plan_csv),
        "--lhs_run",
        str(lhs_run),
    ]
    if do_finetune:
        cmd += ["--do_finetune", "--ft_batch_size", str(ft_batch_size)]

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    print("▶️  Running:", " ".join(cmd))
    try:
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
        if p.returncode != 0:
            print("❌ evaluator 失败：", p.stderr.strip())
            return None
        # evaluator 会写出 {dataset}_{building}_CVRMSE168.csv 到 out_dir_for_building
        # 我们尝试读取那个文件
        out_csv = out_dir_for_building / f"{dataset_name}_{building_id}_CVRMSE168.csv"
        if not out_csv.exists():
            print(f"⚠️ 未找到输出文件：{out_csv}")
            print("stdout:", p.stdout[-5000:])
            print("stderr:", p.stderr[-5000:])
            return None
        return out_csv
    except Exception as e:
        print(f"❌ 子进程异常：{e}")
        return None


# -------------------------
# 画图
# -------------------------
def plot_heatmap(df_sum: pd.DataFrame, out_path: Path) -> None:
    """
    df_sum: 包含列 ['run_id', <building columns>]
    """
    bids = [c for c in df_sum.columns if c not in ["run_id", "run_mean", "run_std"]]
    mat = df_sum[bids].values
    plt.figure(figsize=(max(6, 0.6 * len(bids) + 2), 8))
    plt.imshow(mat, aspect="auto")
    plt.colorbar(label="CVRMSE_168 (%)")
    plt.yticks(np.arange(len(df_sum)), df_sum["run_id"].tolist())
    plt.xticks(np.arange(len(bids)), bids, rotation=45, ha="right")
    plt.title("CVRMSE@168 Heatmap (runs × buildings)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_per_building_lines(df_sum: pd.DataFrame, out_dir: Path) -> None:
    bids = [c for c in df_sum.columns if c not in ["run_id", "run_mean", "run_std"]]
    for b in bids:
        plt.figure(figsize=(7, 4))
        plt.plot(df_sum["run_id"], df_sum[b], marker="o")
        plt.xlabel("run_id")
        plt.ylabel("CVRMSE_168 (%)")
        plt.title(f"CVRMSE@168 vs run_id — {b}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"line_{b}.png", dpi=150)
        plt.close()


# -------------------------
# 主程序
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_ckpt", type=Path, required=True)
    ap.add_argument("--evaluator_py", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, default=Path("./lhs_screen_results"))
    ap.add_argument("--dataset_name", type=str, default="cofactor")
    ap.add_argument("--context_len", type=int, default=168)
    ap.add_argument("--pred_len", type=int, default=168)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--random_seed", type=int, default=2025)
    ap.add_argument("--do_finetune", action="store_true")
    ap.add_argument("--ft_batch_size", type=int, default=64)
    ap.add_argument("--n_runs", type=int, default=16)
    ap.add_argument(
        "--n_sigma_zero", type=int, default=2, help="sigma_hi 显式置 0 的 run 数量"
    )
    args = ap.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 自动选 5 栋楼
    dataset_path = (
        Path(os.environ.get("BUILDINGS_BENCH", ""))
        if os.environ.get("BUILDINGS_BENCH")
        else None
    )
    selected = pick_five_buildings(out_dir, seed=args.random_seed)
    print("📦 选中的 5 栋：", [f"{r['dataset']}:{r['building']}" for r in selected])

    # 2) 生成 16 组 LHS（含 sigma_hi 的 log + zeros）
    plan_df = make_16runs_clrs_plan(
        seed=args.random_seed, n_runs=args.n_runs, n_sigma_zero=args.n_sigma_zero
    )
    plan_csv = out_dir / "clrs_sensitivity_plan_16runs.csv"
    plan_df.to_csv(plan_csv, index=False)
    print(f"📝 已写出 LHS 计划：{plan_csv}")

    # 3) 逐 run × 楼 评测
    all_rows = []
    per_building_files: Dict[Tuple[int, str], Path] = {}

    for run_id in plan_df["run_id"].tolist():
        run_dir = out_dir / f"run_{int(run_id):02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        print("\n" + "=" * 60)
        print(f"▶▶ 开始 run {run_id} ...")

        for rec in selected:
            ds = rec["dataset"]
            bid = rec["building"]
            label = f"{ds}:{bid}"
            bdir = run_dir / label.replace(":", "_")

            out_csv = run_one_eval(
                evaluator_py=args.evaluator_py,
                baseline_ckpt=args.baseline_ckpt,
                dataset_name=ds,  # ← 关键：各自的数据集
                building_id=bid,
                device=args.device,
                out_dir_for_building=bdir,
                plan_csv=plan_csv,
                lhs_run=int(run_id - 1),  # evaluator 内按 0-based 行索引
                context_len=args.context_len,
                pred_len=args.pred_len,
                batch_size=64 if not args.do_finetune else args.ft_batch_size,
                do_finetune=args.do_finetune,
                ft_batch_size=args.ft_batch_size,
            )

            if out_csv is None:
                val = np.nan
            else:
                try:
                    rec_df = pd.read_csv(out_csv).iloc[0]
                    val = float(rec_df["CVRMSE_168"])
                    per_building_files[(int(run_id), label)] = out_csv
                except Exception as e:
                    print(f"⚠️ 读取 {out_csv} 失败：{e}")
                    val = np.nan

            all_rows.append(
                {"run_id": int(run_id), "building_id": label, "CVRMSE_168": val}
            )
    # 4) 汇总表（长表 → 宽表）
    long_df = pd.DataFrame(all_rows)
    long_csv = out_dir / "all_results_long.csv"
    long_df.to_csv(long_csv, index=False)
    print(f"\n✅ 已写出长表：{long_csv}")

    wide_df = long_df.pivot_table(
        index="run_id", columns="building_id", values="CVRMSE_168", aggfunc="mean"
    )
    wide_df = wide_df.reindex(sorted(wide_df.index)).reset_index()
    # 行均值/方差
    val_cols = [c for c in wide_df.columns if c != "run_id"]
    wide_df["run_mean"] = wide_df[val_cols].mean(axis=1, skipna=True)
    wide_df["run_std"] = wide_df[val_cols].std(axis=1, ddof=1, skipna=True)

    wide_csv = out_dir / "all_results_wide.csv"
    wide_df.to_csv(wide_csv, index=False)
    print(f"✅ 已写出宽表：{wide_csv}")

    # 5) 绘图
    heatmap_png = out_dir / "heatmap_runs_x_buildings.png"
    plot_heatmap(wide_df[["run_id"] + val_cols + ["run_mean", "run_std"]], heatmap_png)
    print(f"🖼  热力图：{heatmap_png}")

    lines_dir = out_dir / "per_building_lines"
    lines_dir.mkdir(parents=True, exist_ok=True)
    plot_per_building_lines(wide_df[["run_id"] + val_cols], lines_dir)
    print(f"🖼  折线图目录：{lines_dir}")

    # 6) 记录一次 config 摘要
    summary = {
        "dataset_name": args.dataset_name,
        "buildings": selected,
        "n_runs": int(args.n_runs),
        "n_sigma_zero": int(args.n_sigma_zero),
        "context_len": int(args.context_len),
        "pred_len": int(args.pred_len),
        "device": args.device,
        "do_finetune": bool(args.do_finetune),
        "ft_batch_size": int(args.ft_batch_size),
        "baseline_ckpt": str(args.baseline_ckpt),
        "evaluator_py": str(args.evaluator_py),
        "plan_csv": str(plan_csv),
        "BUILDINGS_BENCH": os.environ.get("BUILDINGS_BENCH", ""),
    }
    with (out_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n🧾 运行摘要：{out_dir / 'run_summary.json'}")

    print("\n🎯 全部完成。你可以查看：")
    print(f"  - LHS 计划：{plan_csv}")
    print(f"  - 结果长表：{long_csv}")
    print(f"  - 结果宽表：{wide_csv}")
    print(f"  - 热力图：  {heatmap_png}")
    print(f"  - 折线图：  {lines_dir}")


if __name__ == "__main__":
    main()
