# -*- coding: utf-8 -*-
"""
run_clrs_ablation_5buildings.py

功能：
- 自动选择 5 栋楼（cofactor 跨类型选 4 栋 + university 1 栋）
- 对 CLRS 的 3 个子模块：Stage-wise scheduling (S), Pulse (P), Revive (R)
  进行 2^3 = 8 组组合消融（CLRS 总体开启，LBL 固定开启）
- 每个配置 × 每栋楼，调用 evaluator 脚本 single_building_cvrmse168.py 做微调并评估 CVRMSE@168
- 汇总所有结果（长表 + 宽表），并绘制简单图

用法示例：
python run_clrs_ablation_5buildings.py \
  --baseline_ckpt /path/to/BuildMoE-top-k-2-without-shared-export_best_val.pt \
  --evaluator_py ./single_building_cvrmse168.py \
  --output_dir ./clrs_ablation_results \
  --do_finetune --ft_batch_size 32
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

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
# 选取 5 栋楼（沿用你原来的逻辑）
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

    chosen = []
    for cat in ["Kindergarten", "School", "NursingHome", "Office"]:
        pool = COFACTOR_TYPE[cat]
        bid = random.choice(pool)
        chosen.append({"dataset": "cofactor", "building": bid})

    uni_bid = random.choice(["A", "B"])
    chosen.append({"dataset": "university", "building": uni_bid})

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "selected_buildings.txt").open("w", encoding="utf-8") as f:
        for rec in chosen:
            f.write(f"{rec['dataset']},{rec['building']}\n")

    print("✅ 本次用于 CLRS 消融实验的 5 栋楼：")
    for rec in chosen:
        print(f"  {rec['dataset']:<12} → {rec['building']}")
    print(f"已写出 → {output_dir / 'selected_buildings.txt'}")

    return chosen


# -------------------------
# 定义 8 组 (S,P,R) 配置
# -------------------------
def build_ablation_configs() -> List[Dict[str, Any]]:
    """
    S: stage-wise scheduling
    P: pulse intervention
    R: revive_dead_experts

    S/P/R = 1 表示启用，对应命令行不加 --no_clrs_stage / --no_clrs_pulse / --no_clrs_revive
    S/P/R = 0 表示关闭，对应命令行加相应的 --no_* 标志
    """
    configs: List[Dict[str, Any]] = []

    # 编码规则：
    # 1: S=1, P=1, R=1 (Full CLRS)
    # 2: S=0, P=1, R=1
    # 3: S=1, P=0, R=1
    # 4: S=1, P=1, R=0
    # 5: S=0, P=0, R=1
    # 6: S=0, P=1, R=0
    # 7: S=1, P=0, R=0
    # 8: S=0, P=0, R=0

    combos = [
        (1, 1, 1, "S1P1R1_full"),
        (0, 1, 1, "S0P1R1_no_stage"),
        (1, 0, 1, "S1P0R1_no_pulse"),
        (1, 1, 0, "S1P1R0_no_revive"),
        (0, 0, 1, "S0P0R1_revive_only"),
        (0, 1, 0, "S0P1R0_pulse_only"),
        (1, 0, 0, "S1P0R0_stage_only"),
        (0, 0, 0, "S0P0R0_none"),
    ]

    for idx, (S, P, R, name) in enumerate(combos, start=1):
        configs.append(
            dict(
                config_id=idx,
                config_name=name,
                S=int(S),
                P=int(P),
                R=int(R),
                no_clrs_stage=(S == 0),
                no_clrs_pulse=(P == 0),
                no_clrs_revive=(R == 0),
            )
        )
    return configs


# -------------------------
# 评测执行（调用 single_building_cvrmse168.py）
# -------------------------
def run_one_eval(
    evaluator_py: Path,
    baseline_ckpt: Path,
    dataset_name: str,
    building_id: str,
    device: str,
    out_dir_for_building: Path,
    context_len: int,
    pred_len: int,
    batch_size: int,
    do_finetune: bool,
    ft_batch_size: int,
    no_clrs_stage: bool,
    no_clrs_pulse: bool,
    no_clrs_revive: bool,
    extra_env: Optional[Dict[str, str]] = None,
) -> Optional[Path]:
    """
    调用 single_building_cvrmse168.py，返回其输出 CSV 路径（或 None 表失败）。
    这里不再做 LHS 注入，只用消融开关 --no_clrs_stage / --no_clrs_pulse / --no_clrs_revive。
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
    ]

    if do_finetune:
        cmd += ["--do_finetune", "--ft_batch_size", str(ft_batch_size)]

    # CLRS 消融开关（CLRS 总体开启，所以不加 --no_clrs；只关子模块）
    if no_clrs_stage:
        cmd.append("--no_clrs_stage")
    if no_clrs_pulse:
        cmd.append("--no_clrs_pulse")
    if no_clrs_revive:
        cmd.append("--no_clrs_revive")

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    print("▶️  Running:", " ".join(cmd))
    try:
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
        if p.returncode != 0:
            print("❌ evaluator 失败：", p.stderr.strip())
            print("stdout:", p.stdout[-2000:])
            print("stderr:", p.stderr[-2000:])
            return None

        out_csv = out_dir_for_building / f"{dataset_name}_{building_id}_CVRMSE168.csv"
        if not out_csv.exists():
            print(f"⚠️ 未找到输出文件：{out_csv}")
            print("stdout:", p.stdout[-2000:])
            print("stderr:", p.stderr[-2000:])
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
    df_sum: 包含列 ['config_id', <building columns>]
    """
    bids = [
        c
        for c in df_sum.columns
        if c not in ["config_id", "config_name", "S", "P", "R", "row_mean", "row_std"]
    ]
    mat = df_sum[bids].values
    plt.figure(figsize=(max(6, 0.6 * len(bids) + 2), 6))
    plt.imshow(mat, aspect="auto")
    plt.colorbar(label="CVRMSE_168 (%)")
    plt.yticks(np.arange(len(df_sum)), df_sum["config_id"].tolist())
    plt.xticks(np.arange(len(bids)), bids, rotation=45, ha="right")
    plt.title("CVRMSE@168 Heatmap (configs × buildings)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_per_building_lines(df_sum: pd.DataFrame, out_dir: Path) -> None:
    bids = [
        c
        for c in df_sum.columns
        if c not in ["config_id", "config_name", "S", "P", "R", "row_mean", "row_std"]
    ]
    for b in bids:
        plt.figure(figsize=(7, 4))
        plt.plot(df_sum["config_id"], df_sum[b], marker="o")
        plt.xlabel("config_id")
        plt.ylabel("CVRMSE_168 (%)")
        plt.title(f"CVRMSE@168 vs config_id — {b}")
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
    ap.add_argument("--output_dir", type=Path, default=Path("./clrs_ablation_results"))
    ap.add_argument("--context_len", type=int, default=168)
    ap.add_argument("--pred_len", type=int, default=168)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--random_seed", type=int, default=2025)
    ap.add_argument("--do_finetune", action="store_true")
    ap.add_argument("--ft_batch_size", type=int, default=64)
    ap.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="评估时 batch_size；若 do_finetune，则对 train/val/test 统一为 ft_batch_size。",
    )
    args = ap.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 自动选 5 栋楼
    selected = pick_five_buildings(out_dir, seed=args.random_seed)
    print("📦 选中的 5 栋：", [f"{r['dataset']}:{r['building']}" for r in selected])

    # 2) 构造 8 组 (S,P,R) 消融配置
    configs = build_ablation_configs()
    cfg_df = pd.DataFrame(configs)
    cfg_csv = out_dir / "clrs_ablation_configs.csv"
    cfg_df.to_csv(cfg_csv, index=False)
    print(f"📝 已写出消融配置表：{cfg_csv}")

    # 3) 逐 config × 楼 评测
    all_rows = []

    for cfg in configs:
        cid = cfg["config_id"]
        cname = cfg["config_name"]
        S, P, R = cfg["S"], cfg["P"], cfg["R"]
        print("\n" + "=" * 70)
        print(f"▶▶ 开始 config {cid}: {cname} (S={S}, P={P}, R={R}) ...")

        cfg_dir = out_dir / f"cfg_{cid:02d}_{cname}"
        cfg_dir.mkdir(parents=True, exist_ok=True)

        for rec in selected:
            ds = rec["dataset"]
            bid = rec["building"]
            label = f"{ds}:{bid}"
            bdir = cfg_dir / label.replace(":", "_")

            out_csv = run_one_eval(
                evaluator_py=args.evaluator_py,
                baseline_ckpt=args.baseline_ckpt,
                dataset_name=ds,
                building_id=bid,
                device=args.device,
                out_dir_for_building=bdir,
                context_len=args.context_len,
                pred_len=args.pred_len,
                batch_size=(
                    args.ft_batch_size if args.do_finetune else args.batch_size
                ),
                do_finetune=args.do_finetune,
                ft_batch_size=args.ft_batch_size,
                no_clrs_stage=cfg["no_clrs_stage"],
                no_clrs_pulse=cfg["no_clrs_pulse"],
                no_clrs_revive=cfg["no_clrs_revive"],
            )

            if out_csv is None:
                val = np.nan
            else:
                try:
                    rec_df = pd.read_csv(out_csv).iloc[0]
                    val = float(rec_df["CVRMSE_168"])
                except Exception as e:
                    print(f"⚠️ 读取 {out_csv} 失败：{e}")
                    val = np.nan

            all_rows.append(
                {
                    "config_id": cid,
                    "config_name": cname,
                    "S": int(S),
                    "P": int(P),
                    "R": int(R),
                    "building_id": label,
                    "CVRMSE_168": val,
                }
            )

    # 4) 汇总表（长表 → 宽表）
    long_df = pd.DataFrame(all_rows)
    long_csv = out_dir / "all_results_long.csv"
    long_df.to_csv(long_csv, index=False)
    print(f"\n✅ 已写出长表：{long_csv}")

    wide_df = long_df.pivot_table(
        index="config_id", columns="building_id", values="CVRMSE_168", aggfunc="mean"
    )
    wide_df = wide_df.reindex(sorted(wide_df.index)).reset_index()

    # 合并上 config 元信息
    wide_df = cfg_df.merge(wide_df, on="config_id", how="left")

    # 行均值/方差（跨 5 栋）
    bid_cols = [
        c
        for c in wide_df.columns
        if c not in ["config_id", "config_name", "S", "P", "R"]
    ]
    wide_df["row_mean"] = wide_df[bid_cols].mean(axis=1, skipna=True)
    wide_df["row_std"] = wide_df[bid_cols].std(axis=1, ddof=1, skipna=True)

    wide_csv = out_dir / "all_results_wide.csv"
    wide_df.to_csv(wide_csv, index=False)
    print(f"✅ 已写出宽表：{wide_csv}")

    # 5) 绘图（热力图 + 每栋楼折线）
    heatmap_png = out_dir / "heatmap_configs_x_buildings.png"
    plot_heatmap(wide_df, heatmap_png)
    print(f"🖼  热力图：{heatmap_png}")

    lines_dir = out_dir / "per_building_lines"
    lines_dir.mkdir(parents=True, exist_ok=True)
    plot_per_building_lines(wide_df, lines_dir)
    print(f"🖼  折线图目录：{lines_dir}")

    # 6) 记录一次 config 摘要
    summary = {
        "buildings": selected,
        "n_configs": len(configs),
        "context_len": int(args.context_len),
        "pred_len": int(args.pred_len),
        "device": args.device,
        "do_finetune": bool(args.do_finetune),
        "ft_batch_size": int(args.ft_batch_size),
        "batch_size": int(args.batch_size),
        "baseline_ckpt": str(args.baseline_ckpt),
        "evaluator_py": str(args.evaluator_py),
        "BUILDINGS_BENCH": os.environ.get("BUILDINGS_BENCH", ""),
    }
    with (out_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n🧾 运行摘要：{out_dir / 'run_summary.json'}")

    print("\n🎯 CLRS 8 组 (S,P,R) 消融全部完成。你可以查看：")
    print(f"  - 配置表： {cfg_csv}")
    print(f"  - 结果长表：{long_csv}")
    print(f"  - 结果宽表：{wide_csv}")
    print(f"  - 热力图：  {heatmap_png}")
    print(f"  - 折线图：  {lines_dir}")


if __name__ == "__main__":
    main()
