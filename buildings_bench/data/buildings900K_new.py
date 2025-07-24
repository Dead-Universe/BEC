from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from buildings_bench.transforms import (
    TimestampTransform,
    LatLonTransform,
    StandardScalerTransform,
)
from buildings_bench.transforms import BoxCoxTransform
from buildings_bench.utils import get_puma_county_lookup_table


class Buildings900K(Dataset):
    r"""Indexed dataset for Buildings-900K with on-the-fly boundary checking."""

    # ────────────────────────────────────────────────────────────────────────────
    # init / housekeeping
    # ────────────────────────────────────────────────────────────────────────────
    def __init__(
        self,
        dataset_path: Path,
        index_file: str,
        context_len: int = 168,
        pred_len: int = 24,
        apply_scaler_transform: str = "",
        scaler_transform_path: Optional[Path] = None,
        weather_inputs: Optional[List[str]] = None,
    ):
        super().__init__()
        self.dataset_path = (
            dataset_path
            / "Buildings-900K"
            / "end-use-load-profiles-for-us-building-stock"
            / "2021"
        )
        self.metadata_path = dataset_path / "metadata"

        self.context_len = context_len
        self.pred_len = pred_len
        self.need_len = context_len + pred_len

        self.building_type_and_year = [
            "comstock_tmy3_release_1",
            "resstock_tmy3_release_1",
            "comstock_amy2018_release_1",
            "resstock_amy2018_release_1",
        ]
        self.census_regions = [
            "by_puma_midwest",
            "by_puma_south",
            "by_puma_northeast",
            "by_puma_west",
        ]

        self.index_file = self.metadata_path / index_file
        self.index_fp = None
        self._read_index_file()

        self.time_transform = TimestampTransform()
        self.spatial_transform = LatLonTransform()

        self.apply_scaler_transform = apply_scaler_transform
        if apply_scaler_transform == "boxcox":
            self.load_transform = BoxCoxTransform()
            self.load_transform.load(scaler_transform_path)
        elif apply_scaler_transform == "standard":
            self.load_transform = StandardScalerTransform()
            self.load_transform.load(scaler_transform_path)

        self.weather_inputs = weather_inputs
        if weather_inputs:
            self.lookup_df = get_puma_county_lookup_table(self.metadata_path)
            self.weather_transforms = []
            for col in weather_inputs:
                st = StandardScalerTransform()
                st.load(self.metadata_path / "transforms" / "weather" / col)
                self.weather_transforms.append(st)

    # --------------------------------------------------------------------------
    # file-pointer helpers
    # --------------------------------------------------------------------------
    def _read_index_file(self) -> None:
        """Cache #lines (time-series) 以及每行字符长度（chunk_size）。"""
        with open(self.index_file, "rb") as fp:
            self.num_time_series = sum(
                buf.count(b"\n") for buf in iter(lambda: fp.read(1 << 20), b"")
            )
        with open(self.index_file, "rb", buffering=0) as fp:
            self.chunk_size = len(fp.readline())

    def init_fp(self):
        self.index_fp = open(self.index_file, "rb", buffering=0)
        self.index_fp.seek(0)

    def __del__(self):
        if self.index_fp:
            self.index_fp.close()

    # --------------------------------------------------------------------------
    # Dataset API
    # --------------------------------------------------------------------------
    def __len__(self):
        return self.num_time_series

    def __getitem__(self, idx: int):
        """返回一个合法长度的样本，不足则**跳过重取**。"""

        max_retry = 16  # 最多尝试这么多行；极端稀疏时可调大

        for _ in range(max_retry):
            # ── 1. 保证文件指针就绪 ───────────────────────────────────────────
            if not self.index_fp:
                self.init_fp()

            assert self.index_fp is not None, "Index file pointer is not initialized."

            self.index_fp.seek(idx * self.chunk_size, 0)
            ts_idx = (
                self.index_fp.read(self.chunk_size).decode().strip("\n").split("\t")
            )

            # ── 2. 解析索引 ────────────────────────────────────────────────
            ts_type, ts_region, ts_puma, bldg_id, seq_ptr_raw = (
                int(ts_idx[0]),
                int(ts_idx[1]),
                ts_idx[2],
                ts_idx[3].lstrip("0"),
                ts_idx[4],
            )
            seq_ptr = int(seq_ptr_raw.lstrip("0") or "0")

            # ── 3. 读 parquet 数据 ────────────────────────────────────────
            parquet_path = (
                self.dataset_path
                / self.building_type_and_year[ts_type]
                / "timeseries_individual_buildings"
                / self.census_regions[ts_region]
                / "upgrade=0"
                / f"puma={ts_puma}"
            )
            df = (
                pq.read_table(parquet_path, columns=["timestamp", bldg_id])
                .to_pandas()
                .sort_values("timestamp")
            )

            # ── 4. 边界检查 ────────────────────────────────────────────────
            if seq_ptr >= self.context_len and seq_ptr + self.pred_len <= len(df):
                # 合法，准备切片
                break
            else:
                # 不合法 → 换下一行
                idx = (idx + 1) % self.num_time_series
        else:
            raise RuntimeError("No valid slice found after multiple retries.")

        # ──────────────────────────────────────────────────────────────────
        # ✅ 此时肯定有足够行数，可以直接切
        # ──────────────────────────────────────────────────────────────────
        start, end = seq_ptr - self.context_len, seq_ptr + self.pred_len

        # ① load
        load_slice = (
            df[bldg_id].iloc[start:end].astype(np.float32).fillna(0.0).to_numpy()
        )
        load_slice = np.clip(load_slice, 0.0, None)
        if self.apply_scaler_transform:
            load_slice = self.load_transform.transform(load_slice)

        # ② time
        time_slice = self.time_transform.transform(df["timestamp"].iloc[start:end])

        # ③ lat / lon & building-type
        latlon = self.spatial_transform.transform(ts_puma).repeat(self.need_len, axis=0)
        btype = np.ones((self.need_len, 1), np.int32) * int(ts_type % 2 == 0)

        sample = {
            "latitude": latlon[:, 0][..., None],
            "longitude": latlon[:, 1][..., None],
            "day_of_year": time_slice[:, 0][..., None],
            "day_of_week": time_slice[:, 1][..., None],
            "hour_of_day": time_slice[:, 2][..., None],
            "building_type": btype,
            "load": load_slice[..., None],
        }

        # ── ④ weather（可选） ────────────────────────────────────────────
        if self.weather_inputs:
            county = self.lookup_df.loc[ts_puma, "nhgis_2010_county_gisjoin"]
            weather_df = pd.read_csv(
                self.dataset_path
                / self.building_type_and_year[ts_type]
                / "weather"
                / f"{county}.csv"
            )
            weather_df.columns = ["timestamp"] + [
                "temperature",
                "humidity",
                "wind_speed",
                "wind_direction",
                "global_horizontal_radiation",
                "direct_normal_radiation",
                "diffuse_horizontal_radiation",
            ]
            weather_df = weather_df[["timestamp"] + self.weather_inputs]

            weather_slice = weather_df.iloc[start:end].copy()
            # 再次断言充足（理论上应满足）
            assert len(weather_slice) == self.need_len

            for idx_w, col in enumerate(self.weather_inputs):
                sample[col] = self.weather_transforms[idx_w].transform(
                    weather_slice[col].to_numpy()
                )[0][..., None]

        return sample

    # --------------------------------------------------------------------------
    # collate_fn 保持不变
    # --------------------------------------------------------------------------
    def collate_fn(self):
        def _collate(samples):
            batch = {
                "latitude": torch.stack(
                    [torch.from_numpy(s["latitude"]) for s in samples]
                ).float(),
                "longitude": torch.stack(
                    [torch.from_numpy(s["longitude"]) for s in samples]
                ).float(),
                "building_type": torch.stack(
                    [torch.from_numpy(s["building_type"]) for s in samples]
                ).long(),
                "day_of_year": torch.stack(
                    [torch.from_numpy(s["day_of_year"]) for s in samples]
                ).float(),
                "day_of_week": torch.stack(
                    [torch.from_numpy(s["day_of_week"]) for s in samples]
                ).float(),
                "hour_of_day": torch.stack(
                    [torch.from_numpy(s["hour_of_day"]) for s in samples]
                ).float(),
                "load": torch.stack(
                    [torch.from_numpy(s["load"]) for s in samples]
                ).float(),
            }
            if self.weather_inputs:
                for col in self.weather_inputs:
                    batch[col] = torch.stack(
                        [torch.from_numpy(s[col]) for s in samples]
                    ).float()
            return batch

        return _collate


if __name__ == "__main__":
    import os

    dataset_path = Path(os.environ.get("BUILDINGS_BENCH", ""))
    index_file = "train_weekly.idx"
    context_len = 168
    pred_len = 24
    apply_scaler_transform = "boxcox"
    scaler_transform_path = dataset_path / "metadata" / "transforms"
    weather_inputs = [
        "temperature",
        "humidity",
        # "wind_speed",
        # "wind_direction",
        # "global_horizontal_radiation",
        # "direct_normal_radiation",
        # "diffuse_horizontal_radiation",
    ]
    dataset = Buildings900K(
        dataset_path,
        index_file,
        context_len,
        pred_len,
        apply_scaler_transform,
        scaler_transform_path,
        weather_inputs,
    )

    print(f"Number of time series: {len(dataset)}")
    print(f"First sample: {dataset[0]}")
