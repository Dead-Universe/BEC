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

import pandas as pd
import numpy as np
from pathlib import Path
from rliable import library as rly
from buildings_bench.evaluation.managers import BuildingTypes
import zlib


def drop_extreme_iqr(
    df: pd.DataFrame,
    value_col: str = "value",
    k: float = 1.5,
    extreme_k: float = 10.0,
) -> pd.DataFrame:
    """
    返回 *剔除极端异常行* 的 DataFrame 副本，原 df 不变。
    """
    s = pd.to_numeric(df[value_col], errors="coerce")  # 强转成数值
    finite = s[np.isfinite(s)]
    if len(finite) < 2:  # 样本太少无法建 IQR
        return df.copy()

    q1, q3 = finite.quantile([0.25, 0.75])
    iqr = q3 - q1 if q3 > q1 else 0.0
    lower, upper = q1 - k * iqr, q3 + k * iqr

    def _is_extreme(v):
        if not np.isfinite(v):
            return True  # NaN / ±inf → 极端
        if iqr == 0:  # IQR 为 0 → 没法算比例，保守地不删
            return False
        bound = lower if v < lower else upper
        dist_ratio = abs(v - bound) / iqr
        return dist_ratio >= extreme_k

    mask_extreme = s.apply(_is_extreme)
    return df.loc[~mask_extreme].copy()


def _ctx_seed(base_seed: int | None, *parts) -> int | None:
    """根据上下文(parts)从 base_seed 派生一个稳定子种子；base_seed 为 None 则返回 None。"""
    if base_seed is None:
        return None
    s = "|".join(str(p) for p in parts)
    h = zlib.adler32(s.encode("utf-8")) & 0xFFFFFFFF
    return (int(base_seed) ^ int(h)) % (2**32 - 1) or 1  # 避免 0


def _aggregate_scores(
    df: pd.DataFrame,
    aggregate: str = "mean",
    reps: int = 50_000,
    seed: int | None = 0,
    sort_cols: tuple[str, ...] = (
        "dataset",
        "building_id",
        "metric",
        "metric_type",
        "model",
    ),
):
    """
    给定同一 building_type & metric 的 DataFrame，计算聚合“均值/中位数”与 95% bootstrap CI（确定性）。
    - 对每个 model 独立做 bootstrap；使用 np.random.default_rng(seed)。
    - 行顺序用稳定排序固定，避免平台/读入差异导致的抖动。
    返回：
      agg_scores[m] -> shape (1,)
      agg_cis[m]    -> shape (2,1)  # [lo, hi]^T
    """
    # 稳定排序，确保行顺序可复现
    sort_cols = [c for c in sort_cols if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, kind="mergesort").copy()

    # 选择聚合函数
    if aggregate == "mean":
        reduce_fn = np.mean
    elif aggregate == "median":
        reduce_fn = np.median
    else:
        raise ValueError("aggregate must be 'mean' or 'median'")

    agg_scores, agg_cis = {}, {}

    # 对每个模型独立抽样，避免不同 group 之间的顺序影响
    for m, g in df.groupby("model", sort=False):
        arr = g["value"].to_numpy(dtype=float).reshape(-1)
        if arr.size == 0:
            continue

        point = reduce_fn(arr)

        # 太少无法 bootstrap（<2），CI 置为 NaN，但仍返回点估计
        if (reps <= 0) or (arr.size < 2):
            lo = hi = np.nan
        else:
            rng = np.random.default_rng(seed)
            # (reps, n) 的索引矩阵；对大 n 可改成分批以省内存
            idx = rng.integers(0, arr.size, size=(reps, arr.size), endpoint=False)
            boots = reduce_fn(arr[idx], axis=1)
            lo, hi = np.percentile(boots, [2.5, 97.5])

        agg_scores[m] = np.array([point], dtype=float)
        agg_cis[m] = np.array([[lo, hi]], dtype=float).T  # (2,1)

    return agg_scores, agg_cis


# def return_aggregate(
#     model_list,
#     results_dir,
#     experiment="zero_shot",
#     metrics=("cvrmse",),
#     aggregate="median",  # 'median' | 'mean'
#     exclude_simulated=True,
#     only_simulated=False,
#     oov_list=(),
#     reps=50_000,
# ):
#     """
#     计算指定模型在不同 building_type / metric / dataset 下的聚合（均值或中位数）及 95% bootstrap CI。
#     - 遇到 `inf` / `nan` 会打印详细警告并删除对应行，避免最终结果为 `nan`。

#     Returns
#     -------
#     result_dict : dict
#         result_dict[building_type][metric][dataset_tag][model] = (score, [lo, hi])
#         其中 dataset_tag == "_overall" 表示跨数据集总体结果。
#     """
#     metrics = list(set(metrics))
#     BTYPES = [BuildingTypes.RESIDENTIAL, BuildingTypes.COMMERCIAL]

#     result_dict = {bt: {m: {} for m in metrics} for bt in BTYPES}

#     # ---------- 读取一次 CSV，避免重复 I/O ---------- #
#     cache = {}

#     def load_csv(prefix, model):
#         key = (prefix, model)
#         if key not in cache:
#             cache[key] = pd.read_csv(Path(results_dir) / f"{prefix}_{model}.csv")
#         return cache[key].copy()

#     # ---------- 主循环 ---------- #
#     for bt in BTYPES:
#         for metric in metrics:
#             concat_dfs = []
#             for model in model_list:
#                 # 选择文件前缀
#                 prefix = "scoring_rule" if metric in ("rps", "crps") else "metrics"
#                 if experiment == "transfer_learning":
#                     prefix = "TL_" + prefix

#                 df = load_csv(prefix, model)
#                 df["model"] = model

#                 # ----------- 基础过滤 ----------- #
#                 if oov_list:
#                     df = df[~df["building_id"].str.contains("|".join(oov_list))]

#                 if exclude_simulated:
#                     df = df[
#                         ~df["dataset"].isin(
#                             {"buildings-900k-test", "buildings-1m-test"}
#                         )
#                     ]
#                 elif only_simulated:
#                     df = df[
#                         df["dataset"].isin({"buildings-900k-test", "buildings-1m-test"})
#                     ]

#                 # 只保留当前 metric
#                 if metric not in ("rps", "crps"):
#                     cond = (df["metric"] == metric) | df["metric"].str.startswith(
#                         metric + "_"
#                     )
#                     df = df[cond]

#                 # 只保留当前 building_type
#                 df = df[df["building_type"] == bt]

#                 # ----------- inf / nan 处理 ----------- #
#                 # n_nan = df["value"].isna().sum()
#                 # n_inf = np.isinf(df["value"]).sum()
#                 # if n_nan or n_inf:
#                 #     print(
#                 #         f"Warning: model '{model}' ({bt}-{metric}) contains "
#                 #         f"{n_nan} NaN, {n_inf} inf — rows will be dropped."
#                 #     )
#                 # df["value"] = df["value"].replace(np.inf, np.nan)
#                 # df = df.dropna(subset=["value"])

#                 before = len(df)
#                 df = drop_extreme_iqr(df, k=1.5, extreme_k=10)
#                 after = len(df)
#                 if before != after:
#                     print(
#                         f"[INFO] {model} ({bt}-{metric}) dropped {before - after} extreme outliers."
#                     )

#                 concat_dfs.append(df)

#             # 合并所有模型数据用于整体聚合
#             all_df = (
#                 pd.concat(concat_dfs, ignore_index=True)
#                 if concat_dfs
#                 else pd.DataFrame()
#             )

#             # ---- (1) 总体聚合 ---- #
#             result_dict[bt][metric]["_overall"] = {}
#             if not all_df.empty:
#                 overall_scores, overall_cis = _aggregate_scores(all_df, aggregate, reps)
#                 for m in model_list:
#                     if m in overall_scores:
#                         result_dict[bt][metric]["_overall"][m] = (
#                             overall_scores[m][0],
#                             overall_cis[m][:, 0],
#                         )
#                     else:
#                         print(
#                             f"[WARN] {m} has no valid samples for {bt}-{metric}-OVERALL"
#                         )

#             # ---- (2) 按 dataset 聚合 ---- #
#             for dname, sub_df in all_df.groupby("dataset"):
#                 ds_scores, ds_cis = _aggregate_scores(sub_df, aggregate, reps)
#                 result_dict[bt][metric][dname] = {}
#                 for m in model_list:
#                     if m in ds_scores:
#                         result_dict[bt][metric][dname][m] = (
#                             ds_scores[m][0],
#                             ds_cis[m][:, 0],
#                         )
#                     else:
#                         print(f"[WARN] {m} has no samples in dataset '{dname}'")

#     return result_dict


def _load_oov_allowlist(oov_file: str):
    """
    读取 oov.txt，返回两个集合：
      - allow_pairs: {(dataset, building_id)}（全小写），用于非 BDG-2
      - allow_bdg2_buildings: {building_id}（全小写），用于 BDG-2（仅按楼宇名匹配）
    oov.txt 每行格式建议为:
      DatasetName:BuildingID
    BDG-2 行示例：
      BDG-2:Bear_education_Marta
    空行或以 '#' 开头的行会被忽略。
    """
    allow_pairs = set()
    allow_bdg2_buildings = set()
    for raw in Path(oov_file).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = line.lower()
        # 允许两种格式：dataset:building 或 仅 building（仅在 BDG-2 下有用）
        if ":" in line:
            ds, bid = line.split(":", 1)
            ds, bid = ds.strip(), bid.strip()
            if ds == "bdg-2":
                allow_bdg2_buildings.add(bid)
            else:
                allow_pairs.add((ds, bid))
        else:
            # 无数据集前缀，只能用于 BDG-2
            allow_bdg2_buildings.add(line)
    return allow_pairs, allow_bdg2_buildings


def _dbg(tag, df):
    print(f"[DBG] {tag}: n={len(df)}")


def return_aggregate(
    model_list,
    results_dir,
    experiment="zero_shot",
    metrics=("cvrmse",),
    aggregate="mean",  # 固定为均值
    exclude_simulated=True,
    only_simulated=False,
    oov_list=(),  # 兼容旧参数：若提供且 oov_file=None，仍按“排除”逻辑
    oov_file=None,  # 新增：白名单文件路径；若提供，则仅保留文件中列出的楼宇
    reps=50_000,
    seed: int | None = 0,  # 新增：全局种子；None 表示保持随机
    cofactor_type: dict[str, list[str]] | None = {
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
    },  # ← 新增
):
    """
    计算指定模型在不同 building_type / metric / dataset 下的均值及 95% bootstrap CI。
    - 如果提供 oov_file：只保留 oov.txt 中列出的楼宇（区分逻辑：BDG-2 只按楼宇名匹配）。
    - 如果 oov_file 为 None：全部计算；若 oov_list 非空，按原逻辑“排除”匹配项。
    - 遇到 `inf` / `nan` 会打印警告并删除对应行。
    """
    metrics = list(set(m.lower() for m in metrics))
    BTYPES = [BuildingTypes.RESIDENTIAL, BuildingTypes.COMMERCIAL]

    # 读取 oov 白名单
    allow_pairs, allow_bdg2_buildings = (set(), set())
    use_allowlist = oov_file is not None
    if use_allowlist:
        allow_pairs, allow_bdg2_buildings = _load_oov_allowlist(oov_file)

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

                # ----------- 统一小写 ----------- #
                for col in (
                    "dataset",
                    "building_id",
                    "building_type",
                    "metric",
                    "metric_type",
                ):
                    if col in df.columns and df[col].dtype == object:
                        df[col] = df[col].str.lower().str.strip()

                # ----------- OOV 白名单（仅保留） ----------- #
                if use_allowlist:
                    # BDG-2：dataset 以 "bdg-2" 开头的，只按 building_id 白名单保留
                    is_bdg2 = df["dataset"].str.startswith("bdg-2", na=False)
                    keep_bdg2 = is_bdg2 & df["building_id"].isin(allow_bdg2_buildings)

                    # 非 BDG-2：按 (dataset, building_id) 成对匹配
                    pair_series = (
                        df["dataset"].fillna("") + ":" + df["building_id"].fillna("")
                    )
                    keep_non = (~is_bdg2) & pair_series.isin(
                        {f"{d}:{b}" for d, b in allow_pairs}
                    )

                    old_df_len = len(df)

                    df = df[keep_bdg2 | keep_non]

                    if len(df) < old_df_len:
                        print(
                            f"[INFO] {model} ({bt}-{metric}) dropped "
                            f"{old_df_len - len(df)} OOV buildings."
                        )

                    if df.empty:
                        continue
                if oov_list:
                    # 旧逻辑：排除包含这些子串的楼宇（大小写已统一）
                    df = df[
                        ~df["building_id"].str.contains(
                            "|".join([s.lower() for s in oov_list])
                        )
                    ]

                # ----------- 模拟数据过滤 ----------- #
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

                # ----------- 只保留当前 metric（标量口径）----------- #
                if metric not in ("rps", "crps"):
                    # df = df[(df["metric"] == metric) & (df["metric_type"] == "scalar")]
                    cond = (df["metric"] == metric) | df["metric"].str.startswith(
                        metric + "_"
                    )
                    df = df[cond]

                # ----------- 只保留当前 building_type ----------- #
                df = df[df["building_type"] == bt]  # bt 本身就是标准小写

                # ----------- cofactor 数据集细分为子数据集 ----------- #
                # 仅当提供了 cofactor_type 字典且存在 'dataset' / 'building_id' 列时启用
                if (
                    cofactor_type
                    and "dataset" in df.columns
                    and "building_id" in df.columns
                ):
                    # 构建 building_id -> Category 全称 的反向映射（全部转小写以对齐前面的统一小写）
                    id2cat = {}
                    for full_cat, bid_list in cofactor_type.items():
                        for bid in bid_list:
                            id2cat[str(bid).lower()] = (
                                full_cat  # e.g. 'building6396' -> 'Kindergarten'
                            )

                    # 仅处理 dataset == 'cofactor' 的行
                    is_cof = df["dataset"].eq("cofactor")
                    if is_cof.any():
                        # 映射函数：cofactor -> cofactor:【Category】，否则保留原样
                        def _cof_ds_map(row):
                            if row["dataset"] != "cofactor":
                                return row["dataset"]
                            bid = str(row["building_id"]).lower()
                            full_cat = id2cat.get(bid)
                            if not full_cat:
                                print(f"[WARN] cofactor building_id 未找到类型：{bid}")
                                return "cofactor:Unknown"
                            return f"cofactor:{full_cat}"

                        # 只对 cofactor 行做改名，保持大小写为可读全称
                        df.loc[is_cof, "dataset"] = df.loc[
                            is_cof, ["dataset", "building_id"]
                        ].apply(
                            lambda r: _cof_ds_map(
                                {
                                    "dataset": r["dataset"],
                                    "building_id": r["building_id"],
                                }
                            ),
                            axis=1,
                        )

                # ----------- inf / nan 处理 ----------- #
                n_nan = df["value"].isna().sum()
                n_inf = np.isinf(df["value"]).sum()
                if n_nan or n_inf:
                    print(
                        f"Warning: model '{model}' ({bt}-{metric}) contains "
                        f"{n_nan} NaN, {n_inf} inf — NaN will be replaced with inf."
                    )

                # 把 NaN 替换成 inf
                df["value"] = df["value"].replace(np.nan, np.inf)

                if not df.empty:
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
                seed_overall = _ctx_seed(seed, bt, metric, "_overall")
                overall_scores, overall_cis = _aggregate_scores(
                    all_df, aggregate=aggregate, reps=reps, seed=seed_overall
                )
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
            if all_df.empty or "dataset" not in all_df.columns:
                continue  # 无 dataset 列或无数据，跳过
            for dname, sub_df in all_df.groupby("dataset"):
                seed_ds = _ctx_seed(seed, bt, metric, dname)
                ds_scores, ds_cis = _aggregate_scores(
                    sub_df, aggregate=aggregate, reps=reps, seed=seed_ds
                )
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
    final_table = header + "\n".join(rows)
    print(final_table)
    return final_table


if __name__ == "__main__":
    import os

    results_dir = "/home/hadoop/bec/BuildingsBench/results"
    models = [
        "TimeMoE-L:168_1",
        "TimeMoE-L:168_6",
        "TimeMoE-L:168_12",
        "TimeMoE-L:168_24",
        "TimeMoE-L:168_48",
        "TimeMoE-L:168_96",
        "TimeMoE-L:168_168",
        "TransformerWithGaussianAndMoEs-L:168_1",
        "TransformerWithGaussianAndMoEs-L:168_6",
        "TransformerWithGaussianAndMoEs-L:168_12",
        "TransformerWithGaussianAndMoEs-L:168_24",
        "TransformerWithGaussianAndMoEs-L:168_48",
        "TransformerWithGaussianAndMoEs-L:168_96",
        "TransformerWithGaussianAndMoEs-L:168_168",
        "Chronos-L:168_1",
        "Chronos-L:168_6",
        "Chronos-L:168_12",
        "Chronos-L:168_24",
        "Chronos-L:168_48",
        "Chronos-L:168_96",
        "Chronos-L:168_168",
        "Transformer-L:168_1",
        "Transformer-L:168_6",
        "Transformer-L:168_12",
        "Transformer-L:168_24",
        "Transformer-L:168_48",
        "Transformer-L:168_96",
        "Transformer-L:168_168",
    ]

    oov = []  # ← 你的 oov 列表
    with open(Path(os.environ["BUILDINGS_BENCH"]) / "metadata" / "oov.txt") as f:
        oov = [l.split()[1] for l in f]

    # ---- ① 全局中位数 + dataset 中位数 ----
    res_med = return_aggregate(
        models,
        results_dir,
        experiment="zero_shot",
        metrics=["nrmse", "nmae"],
        aggregate="mean",
        oov_list=oov,
        # oov_file="/home/hadoop/bec/BuildingsBench/oov_test.txt",
    )
    pretty_print(res_med, aggregate="mean")

    # ---- ② 全局均值 + dataset 均值 ----
    res_mean = return_aggregate(
        models,
        results_dir,
        experiment="zero_shot",
        metrics=["nrmse", "nmae"],
        aggregate="mean",
        oov_list=oov,
        oov_file="/home/hadoop/bec/BuildingsBench/oov_test.txt",
    )
    pretty_print(res_mean, aggregate="mean")
