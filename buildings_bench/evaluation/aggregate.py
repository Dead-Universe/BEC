import pandas as pd
from pathlib import Path
from rliable import library as rly
import numpy as np
from buildings_bench import BuildingTypes


def return_aggregate_median(
    model_list,
    results_dir,
    experiment="zero_shot",
    metrics=["cvrmse"],
    exclude_simulated=True,
    only_simulated=False,
    oov_list=[],
    reps=50000,
):
    """Compute the aggregate median for a list of models and metrics over all buildings.
    Also returns the stratified 95% boostrap CIs for the aggregate median.

    Args:
        model_list (list): List of models to compute aggregate median for.
        results_dir (str): Path to directory containing results.
        experiment (str, optional): Experiment type. Defaults to 'zero_shot'.
            Options: 'zero_shot', 'transfer_learning'.
        metrics (list, optional): List of metrics to compute aggregate median for. Defaults to ['cvrmse'].
        exclude_simulated (bool, optional): Whether to exclude simulated data. Defaults to True.
        only_simulated (bool, optional): Whether to only include simulated data. Defaults to False.
        oov_list (list, optional): List of OOV buildings to exclude. Defaults to [].
        reps (int, optional): Number of bootstrap replicates to use. Defaults to 50000.

    Returns:
        result_dict (Dict): Dictionary containing aggregate median and CIs for each metric and building type.
    """

    result_dict = {}
    metrics = list(set(metrics))
    aggregate_func = lambda x: np.array([np.median(x.reshape(-1))])
    for building_type in [BuildingTypes.RESIDENTIAL, BuildingTypes.COMMERCIAL]:
        result_dict[building_type] = {}
        for metric in metrics:
            result_dict[building_type][metric] = {}

            if experiment == "zero_shot" and (metric == "rps" or metric == "crps"):
                prefix = "scoring_rule"
            elif experiment == "transfer_learning" and (
                metric == "rps" or metric == "crps"
            ):
                prefix = "TL_scoring_rule"
            elif experiment == "zero_shot":
                prefix = "metrics"
            elif experiment == "transfer_learning":
                prefix = "TL_metrics"

            for model in model_list:
                df = pd.read_csv(Path(results_dir) / f"{prefix}_{model}.csv")

                if len(oov_list) > 0:
                    # Remove OOV buildings
                    df = df[~df["building_id"].str.contains("|".join(oov_list))]

                if exclude_simulated:
                    # Exclude synthetic data
                    df = df[
                        ~(
                            (df["dataset"] == "buildings-900k-test")
                            | (df["dataset"] == "buildings-1m-test")
                        )
                    ]
                elif only_simulated:
                    df = df[
                        (df["dataset"] == "buildings-900k-test")
                        | (df["dataset"] == "buildings-1m-test")
                    ]

                # if any df values are inf or nan
                if df.isnull().values.any() or np.isinf(df.value).values.any():
                    print(f"Warning: {model} has inf/nan values")
                # REmove inf/nan values
                df = df.replace(np.inf, np.nan)
                df = df.dropna()

                if metric != "rps" and metric != "crps":
                    # condition = df["metric"].str.startswith(metric + "_")
                    # 若需包含无后缀的基础名，改用：
                    condition = (df["metric"] == metric) | df["metric"].str.startswith(
                        metric + "_"
                    )
                    result_dict[building_type][metric][model] = df[
                        (condition) & (df["building_type"] == building_type)
                    ]["value"].values.reshape(-1, 1)
                else:
                    result_dict[building_type][metric][model] = df[
                        df["building_type"] == building_type
                    ]["value"].values.reshape(-1, 1)

            aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(
                result_dict[building_type][metric], aggregate_func, reps=reps
            )
            result_dict[building_type][metric] = (aggregate_scores, aggregate_score_cis)
    return result_dict


def pretty_print_aggregates(results_dict) -> None:
    """Pretty print the aggregate results.

    `model_name building_type metric_name: value (CI)`

    Args:
        result_dict (Dict): Dictionary of aggregate metrics for residential and commercial buildings.
    """
    print("model name\t building type\t metric name\t value (95% C.I.)")
    print("==========================================================")

    for building_type, v in results_dict.items():
        for metric_name, vv in v.items():
            agg_scores, agg_cis = vv
            for model_name, metric_value in agg_scores.items():
                cis = agg_cis[model_name]
                if metric_name in ["nrmse", "nmae", "nmbe", "cvrmse"]:
                    metric_value = metric_value[0] * 100
                else:
                    metric_value = metric_value[0]
                print(
                    f"{model_name} {building_type} {metric_name}: {metric_value:.3f} ({cis[0][0]:.3f},{cis[1][0]:.3f})"
                )


# if __name__ == "__main__":

#     import os

#     oov = []
#     with open(
#         Path(os.environ.get("BUILDINGS_BENCH", "")) / "metadata" / "oov.txt", "r"
#     ) as f:
#         for l in f:
#             oov += [l.strip().split(" ")[1]]

#     results_dir = "/home/hadoop/bec/BuildingsBench/results"
#     models = ["timemoe"]

#     pretty_print_aggregates(
#         return_aggregate_median(
#             models,
#             results_dir,
#             metrics=["nrmse", "nmae", "nmbe"],
#             experiment="zero_shot",
#             oov_list=oov,
#         )
#     )

from buildings_bench.evaluation.managers import BuildingTypes


import pandas as pd
import numpy as np
from pathlib import Path
from rliable import library as rly
from buildings_bench.evaluation.managers import BuildingTypes


def _aggregate_scores(
    df: pd.DataFrame,
    aggregate: str = "median",
    reps: int = 50_000,
):
    """给定同一 building_type & metric 的 DataFrame，计算聚合指标和 bootstrap 置信区间。"""
    if aggregate == "median":
        func = lambda x: np.array([np.median(x.reshape(-1))])
    elif aggregate == "mean":
        func = lambda x: np.array([np.mean(x.reshape(-1))])
    else:
        raise ValueError("aggregate must be 'median' or 'mean'")

    scores = {m: v.values.reshape(-1, 1) for m, v in df.groupby("model")["value"]}
    agg_scores, agg_cis = rly.get_interval_estimates(scores, func, reps=reps)
    return agg_scores, agg_cis


def return_aggregate(
    model_list,
    results_dir,
    experiment="zero_shot",
    metrics=("cvrmse",),
    aggregate="median",  # 'median' | 'mean'
    exclude_simulated=True,
    only_simulated=False,
    oov_list=(),
    reps=50_000,
):
    """
    计算指定模型在不同 building_type / metric / dataset 下的聚合（均值或中位数）及 95% bootstrap CI。
    - 遇到 `inf` / `nan` 会打印详细警告并删除对应行，避免最终结果为 `nan`。

    Returns
    -------
    result_dict : dict
        result_dict[building_type][metric][dataset_tag][model] = (score, [lo, hi])
        其中 dataset_tag == "_overall" 表示跨数据集总体结果。
    """
    metrics = list(set(metrics))
    BTYPES = [BuildingTypes.RESIDENTIAL, BuildingTypes.COMMERCIAL]

    result_dict = {bt: {m: {} for m in metrics} for bt in BTYPES}

    # ---------- 读取一次 CSV，避免重复 I/O ---------- #
    cache = {}

    def load_csv(prefix, model):
        key = (prefix, model)
        if key not in cache:
            cache[key] = pd.read_csv(Path(results_dir) / f"{prefix}_{model}.csv")
        return cache[key].copy()

    # ---------- 主循环 ---------- #
    for bt in BTYPES:
        for metric in metrics:
            concat_dfs = []
            for model in model_list:
                # 选择文件前缀
                prefix = "scoring_rule" if metric in ("rps", "crps") else "metrics"
                if experiment == "transfer_learning":
                    prefix = "TL_" + prefix

                df = load_csv(prefix, model)
                df["model"] = model

                # ----------- 基础过滤 ----------- #
                if oov_list:
                    df = df[~df["building_id"].str.contains("|".join(oov_list))]

                if exclude_simulated:
                    df = df[
                        ~df["dataset"].isin(
                            {"buildings-900k-test", "buildings-1m-test"}
                        )
                    ]
                elif only_simulated:
                    df = df[
                        df["dataset"].isin({"buildings-900k-test", "buildings-1m-test"})
                    ]

                # 只保留当前 metric
                if metric not in ("rps", "crps"):
                    cond = (df["metric"] == metric) | df["metric"].str.startswith(
                        metric + "_"
                    )
                    df = df[cond]

                # 只保留当前 building_type
                df = df[df["building_type"] == bt]

                # ----------- inf / nan 处理 ----------- #
                n_nan = df["value"].isna().sum()
                n_inf = np.isinf(df["value"]).sum()
                if n_nan or n_inf:
                    print(
                        f"Warning: model '{model}' ({bt}-{metric}) contains "
                        f"{n_nan} NaN, {n_inf} inf — rows will be dropped."
                    )
                df["value"] = df["value"].replace(np.inf, np.nan)
                df = df.dropna(subset=["value"])

                concat_dfs.append(df)

            # 合并所有模型数据用于整体聚合
            all_df = (
                pd.concat(concat_dfs, ignore_index=True)
                if concat_dfs
                else pd.DataFrame()
            )

            # ---- (1) 总体聚合 ---- #
            result_dict[bt][metric]["_overall"] = {}
            if not all_df.empty:
                overall_scores, overall_cis = _aggregate_scores(all_df, aggregate, reps)
                for m in model_list:
                    if m in overall_scores:
                        result_dict[bt][metric]["_overall"][m] = (
                            overall_scores[m][0],
                            overall_cis[m][:, 0],
                        )
                    else:
                        print(
                            f"[WARN] {m} has no valid samples for {bt}-{metric}-OVERALL"
                        )

            # ---- (2) 按 dataset 聚合 ---- #
            for dname, sub_df in all_df.groupby("dataset"):
                ds_scores, ds_cis = _aggregate_scores(sub_df, aggregate, reps)
                result_dict[bt][metric][dname] = {}
                for m in model_list:
                    if m in ds_scores:
                        result_dict[bt][metric][dname][m] = (
                            ds_scores[m][0],
                            ds_cis[m][:, 0],
                        )
                    else:
                        print(f"[WARN] {m} has no samples in dataset '{dname}'")

    return result_dict


# --------------- 打印辅助（Markdown 版本） ------------------ #
def pretty_print(result_dict, aggregate="median"):
    header = (
        f"| model | btype | metric | dataset | {aggregate} (95 % CI) |\n"
        "|-------|-------|--------|---------|--------------------|\n"
    )
    rows = []
    for bt, m_dict in result_dict.items():
        for metric, dsets in m_dict.items():
            for dname, scores in dsets.items():
                dataset_tag = "OVERALL" if dname == "_overall" else dname
                for model, (val, ci) in scores.items():
                    if metric in {"nrmse", "nmae", "nmbe", "cvrmse"}:
                        val *= 100
                        ci = ci * 100
                    lo, hi = ci
                    rows.append(
                        f"| {model} | {bt} | {metric} | {dataset_tag} | "
                        f"{val:.3f} ({lo:.3f}, {hi:.3f}) |"
                    )
    print(header + "\n".join(rows))


if __name__ == "__main__":
    import os

    results_dir = "/home/hadoop/bec/BuildingsBench/results"
    models = ["timemoe"]

    oov = []  # ← 你的 oov 列表
    with open(Path(os.environ["BUILDINGS_BENCH"]) / "metadata" / "oov.txt") as f:
        oov = [l.split()[1] for l in f]

    # ---- ① 全局中位数 + dataset 中位数 ----
    res_med = return_aggregate(
        models,
        results_dir,
        experiment="zero_shot",
        metrics=["nrmse", "nmae", "nmbe"],
        aggregate="median",
        oov_list=oov,
    )
    pretty_print(res_med, aggregate="median")

    # ---- ② 全局均值 + dataset 均值 ----
    res_mean = return_aggregate(
        models,
        results_dir,
        experiment="zero_shot",
        metrics=["nrmse", "nmae", "nmbe"],
        aggregate="mean",
        oov_list=oov,
    )
    pretty_print(res_mean, aggregate="mean")
