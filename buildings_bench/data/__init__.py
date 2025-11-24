from pathlib import Path
import torch
import tomli
import os, re
from buildings_bench.data.buildings900K_new import Buildings900K
from buildings_bench.data.datasets import TorchBuildingDatasetsFromCSV
from buildings_bench.data.datasets import TorchBuildingDatasetFromParquet
from buildings_bench.data.datasets import PandasBuildingDatasetsFromCSV
from buildings_bench import BuildingTypes
from buildings_bench import transforms
from typing import Dict, List, Set, Union


dataset_registry = [
    "buildings-900k-train",
    "buildings-900k-val",
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
    "university15t",
    "university1t",
    "university30t",
]

benchmark_registry = [
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
    "university15t",
    "university1t",
    "university30t",
]


def parse_building_years_metadata(datapath: Path, dataset_name: str) -> List[str]:
    """
    - 严格匹配数据集段（大小写无关），不会把 'university' 命中 'university15t' 等。
    - 兼容 'bdg-2:panther' 这类“主:子”数据集名：映射到 'BDG-2/Panther_clean=...'。
    - 不把冒号当分隔符；以文件真实分隔符 '/' 来解析。
    """
    ds_query = dataset_name.lower().strip()
    # 是否是“主:子”形式（例如 'bdg-2:panther'）
    if ":" in ds_query:
        base_ds, sub_ds = ds_query.split(":", 1)
        sub_ds = sub_ds.strip()
    else:
        base_ds, sub_ds = ds_query, None

    kept: List[str] = []
    with open(datapath / "metadata" / "building_years.txt", "r", encoding="utf-8") as f:
        for raw in f:
            ln = raw.strip()
            if not ln or "=" not in ln:
                continue
            path_part, _year = ln.split("=", 1)
            parts = path_part.split("/")  # 关键：按实际使用的 '/' 分段
            if not parts:
                continue

            first_seg = parts[
                0
            ].lower()  # 数据集段（如 'cofactor', 'university15t', 'bdg-2', ...)
            if sub_ds is None:
                # 简单数据集：首段精确等于 dataset_name
                if first_seg == base_ds:
                    kept.append(ln)
            else:
                # “主:子”数据集：首段匹配 base_ds，第二段去掉 '_clean' 后匹配 sub_ds
                if first_seg != base_ds or len(parts) < 2:
                    continue
                second_seg = parts[1].lower()
                # 去掉诸如 '_clean' 后缀以获得子类名（如 'panther_clean' -> 'panther'）
                second_core = second_seg.split("_clean", 1)[0]
                if second_core == sub_ds:
                    kept.append(ln)

    return kept


def load_pretraining(
    name: str,
    num_buildings_ablation: int = -1,
    apply_scaler_transform: str = "",
    scaler_transform_path: Path | None = None,
    weather_inputs: List[str] | None = None,
    custom_idx_filename: str = "",
    context_len=168,  # week
    pred_len=24,
) -> torch.utils.data.Dataset:
    r"""
    Pre-training datasets: buildings-900k-train, buildings-900k-val

    Args:
        name (str): Name of the dataset to load.
        num_buildings_ablation (int): Number of buildings to use for pre-training.
                                        If -1, use all buildings.
        apply_scaler_transform (str): If not using quantized load or unscaled loads,
                                 applies a {boxcox,standard} scaling transform to the load. Default: ''.
        scaler_transform_path (Path): Path to data for transform, e.g., pickled data for BoxCox transform.
        weather_inputs (List[str]): list of weather feature names to use as additional inputs. Default: None.
        custom_idx_filename (str): customized index filename. Default: ''
        context_len (int): Length of the context. Defaults to 168.
        pred_len (int): Length of the prediction horizon. Defaults to 24.

    Returns:
        torch.utils.data.Dataset: Dataset for pretraining.
    """
    dataset_path = Path(os.environ.get("BUILDINGS_BENCH", ""))
    if not dataset_path.exists():
        raise ValueError("BUILDINGS_BENCH environment variable not set")

    if num_buildings_ablation > 0:
        idx_file_suffix = f"_{num_buildings_ablation}"
    else:
        idx_file_suffix = ""
    if name.lower() == "buildings-900k-train":
        idx_file = (
            f"train_weekly{idx_file_suffix}.idx"
            if custom_idx_filename == ""
            else custom_idx_filename
        )
        dataset = Buildings900K(
            dataset_path,
            idx_file,
            context_len=context_len,
            pred_len=pred_len,
            apply_scaler_transform=apply_scaler_transform,
            scaler_transform_path=scaler_transform_path,
            weather_inputs=weather_inputs,
        )
    elif name.lower() == "buildings-900k-val":
        idx_file = (
            f"val_weekly{idx_file_suffix}.idx"
            if custom_idx_filename == ""
            else custom_idx_filename
        )
        dataset = Buildings900K(
            dataset_path,
            idx_file,
            context_len=context_len,
            pred_len=pred_len,
            apply_scaler_transform=apply_scaler_transform,
            scaler_transform_path=scaler_transform_path,
            weather_inputs=weather_inputs,
        )
    return dataset


def load_torch_dataset(
    name: str,
    dataset_path: Path | None = None,
    apply_scaler_transform: str = "",
    scaler_transform_path: Path | None = None,
    weather_inputs: List[str] | None = None,
    include_outliers: bool = False,
    context_len=168,
    pred_len=24,
    *,  # ⬅ 让新增参数只能用关键字调用，避免旧代码位置错位
    split: str = "",  # '', 'train', 'test'
    oov_path: Path | None = None,  # Path to oov.txt
) -> Union[TorchBuildingDatasetsFromCSV, TorchBuildingDatasetFromParquet]:
    r"""Load datasets by name.

    Args:
        name (str): Name of the dataset to load.
        dataset_path (Path): Path to the benchmark data. Optional.
        apply_scaler_transform (str): If not using quantized load or unscaled loads,
                                 applies a {boxcox,standard} scaling transform to the load. Default: ''.
        scaler_transform_path (Path): Path to data for transform, e.g., pickled data for BoxCox transform.
        weather_inputs (List[str]): list of weather feature names to use as additional inputs. Default: None.
        include_outliers (bool): Use version of BuildingsBench with outliers.
        context_len (int): Length of the context. Defaults to 168.
        pred_len (int): Length of the prediction horizon. Defaults to 24.

    Returns:
        dataset (Union[TorchBuildingDatasetsFromCSV, TorchBuildingDatasetFromParquet]): Dataset for benchmarking.
    """
    # ---- 0) 参数检查 ---------------------------------------------------------
    split = split.lower()
    if split not in {"", "train", "test"}:
        raise ValueError(f"split must be '', 'train' or 'test' (got '{split}')")
    if split and not oov_path:
        raise ValueError("When `split` is specified, `oov_path` must be given")

    # ---- 1) 解析 oov.txt -----------------------------------------------------
    oov_lookup: Dict[str, Set[str]] = {}
    if oov_path:
        with open(oov_path) as fp:
            for line in fp:
                if ":" not in line:
                    continue
                ds, bid = line.strip().lower().split(":", 1)
                oov_lookup.setdefault(ds, set()).add(bid)

    if not dataset_path:
        dataset_path = Path(os.environ.get("BUILDINGS_BENCH", ""))
        if not dataset_path.exists():
            raise ValueError("BUILDINGS_BENCH environment variable not set")

    with open(dataset_path / "metadata" / "benchmark.toml", "rb") as f:
        metadata = tomli.load(f)["buildings_bench"]

    def get_oov_set(ds_name: str) -> Set[str]:
        key = ds_name.lower()
        if key.startswith("bdg-2"):  # 处理 bdg‑2 子集
            key = "bdg-2"
        return oov_lookup.get(key, set())

    if name.lower() == "buildings-900k-test":
        spatial_lookup = transforms.LatLonTransform()
        puma_files = list(
            (dataset_path / "Buildings-900K-test" / "2021").glob(
                "*2018*/*/*/*/*/*.parquet"
            )
        )
        if len(puma_files) == 0:
            raise ValueError(
                f"Could not find any Parquet files in "
                f' {str(dataset_path / "Buildings-900K-test" / "2021")}'
            )
        # to string
        puma_files = [str(Path(pf).parent) for pf in puma_files]
        puma_ids = [pf.split("puma=")[1] for pf in puma_files]
        building_types = []
        for pf in puma_files:
            if "res" in pf:
                building_types += [BuildingTypes.RESIDENTIAL]
            elif "com" in pf:
                building_types += [BuildingTypes.COMMERCIAL]
        dataset_generator = TorchBuildingDatasetFromParquet(
            dataset_path,
            puma_files,
            [
                spatial_lookup.undo_transform(  # pass unnormalized lat lon coords
                    spatial_lookup.transform(pid)
                )
                for pid in puma_ids
            ],
            building_types,
            weather_inputs=weather_inputs,
            context_len=context_len,
            pred_len=pred_len,
            apply_scaler_transform=apply_scaler_transform,
            scaler_transform_path=scaler_transform_path,
            leap_years=metadata["leap_years"],
        )
    elif ":" in name.lower():
        name_base, subset = name.lower().split(":")
        dataset_metadata = metadata[name_base]
        all_by_files = parse_building_years_metadata(dataset_path, name_base)
        all_by_files = [bf for bf in all_by_files if subset in bf.lower()]
        if include_outliers:
            dataset_path = dataset_path / "buildingsbench_with_outliers"

        dataset_generator = TorchBuildingDatasetsFromCSV(
            dataset_path,
            all_by_files,
            dataset_metadata[subset]["latlon"],
            dataset_metadata[subset]["building_type"],
            context_len=context_len,
            pred_len=pred_len,
            apply_scaler_transform=apply_scaler_transform,
            scaler_transform_path=scaler_transform_path,
            leap_years=metadata["leap_years"],
            weather_inputs=weather_inputs,
            split=split,
            oov_buildings=get_oov_set(name_base),
        )
    elif name.lower() in benchmark_registry:
        name_base = name.lower()
        dataset_metadata = metadata[name_base]
        all_by_files = parse_building_years_metadata(dataset_path, name_base)
        if include_outliers:
            dataset_path = dataset_path / "buildingsbench_with_outliers"

        dataset_generator = TorchBuildingDatasetsFromCSV(
            dataset_path,
            all_by_files,
            dataset_metadata["latlon"],
            dataset_metadata["building_type"],
            context_len=context_len,
            pred_len=pred_len,
            apply_scaler_transform=apply_scaler_transform,
            scaler_transform_path=scaler_transform_path,
            leap_years=metadata["leap_years"],
            weather_inputs=weather_inputs,
            split=split,
            oov_buildings=get_oov_set(name_base),
        )
    else:
        raise ValueError(f"Unknown dataset {name}")

    return dataset_generator


def load_pandas_dataset(
    name: str,
    dataset_path: Path = None,
    feature_set: str = "engineered",
    weather_inputs: List[str] = None,
    apply_scaler_transform: str = "",
    scaler_transform_path: Path = None,
    include_outliers: bool = False,
) -> PandasBuildingDatasetsFromCSV:
    """
    Load datasets by name.

    Args:
        name (str): Name of the dataset to load.
        dataset_path (Path): Path to the benchmark data. Optional.
        feature_set (str): Feature set to use. Default: 'engineered'.
        weather_inputs (List[str]): list of weather feature names to use as additional inputs. Default: None.
        apply_scaler_transform (str): If not using quantized load or unscaled loads,
                                    applies a {boxcox,standard} scaling transform to the load. Default: ''.
        scaler_transform_path (Path): Path to data for transform, e.g., pickled data for BoxCox transform.
        include_outliers (bool): Use version of BuildingsBench with outliers.

    Returns:
        dataset (PandasBuildingDatasetsFromCSV): Generator of Pandas datasets for benchmarking.
    """
    if not dataset_path:
        dataset_path = Path(os.environ.get("BUILDINGS_BENCH", ""))
        if not dataset_path.exists():
            raise ValueError("BUILDINGS_BENCH environment variable not set")

    if name.lower() == "buildings-900k-test":
        raise ValueError(f"{name.lower()} unavailable for now as pandas dataset")

    with open(dataset_path / "metadata" / "benchmark.toml", "rb") as f:
        metadata = tomli.load(f)["buildings_bench"]

    if ":" in name.lower():
        name, subset = name.lower().split(":")
        dataset_metadata = metadata[name.lower()]
        all_by_files = parse_building_years_metadata(dataset_path, name.lower())
        all_by_files = [
            by_file for by_file in all_by_files if subset in by_file.lower()
        ]
        building_type = dataset_metadata[subset]["building_type"]
        building_latlon = dataset_metadata[subset]["latlon"]
    else:
        dataset_metadata = metadata[name.lower()]
        all_by_files = parse_building_years_metadata(dataset_path, name.lower())
        building_type = dataset_metadata["building_type"]
        building_latlon = dataset_metadata["latlon"]
    if include_outliers:
        dataset_path = dataset_path / "buildingsbench_with_outliers"

    return PandasBuildingDatasetsFromCSV(
        dataset_path,
        all_by_files,
        building_latlon,
        building_type,
        weather_inputs=weather_inputs,
        features=feature_set,
        apply_scaler_transform=apply_scaler_transform,
        scaler_transform_path=scaler_transform_path,
        leap_years=metadata["leap_years"],
    )


if __name__ == "__main__":
    ds = load_torch_dataset(
        "LCL",  # or "BDG-2:bear"
        dataset_path=Path("/home/hadoop/bec/buildings-bench/v2.0.0/BuildingsBench"),
        context_len=168,
        pred_len=24,
        split="test",  # 'train'  | 'test' | ''
        oov_path=Path("/home/hadoop/bec/oov.txt"),  # ← 你的 oov.txt
    )

    print(f"Loaded {len(ds)} buildings")
