from pathlib import Path
import tempfile
from typing import List, Set, Optional, Iterable
import numpy as np
import os, torch
from torch.utils.data import ConcatDataset, random_split, DataLoader, Dataset
from buildings_bench.data import load_pretraining, load_torch_dataset

# -----------------------------------------------
# Constants
# -----------------------------------------------
TRAIN_VAL_RATIO = 0.9  # 8 : 2
RANDOM_SEED = 42  # 保证可复现

# 需要自行拆分的 10 个数据集（保持变量名不变）
custom_registry = [
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
]


# -----------------------------------------------
# Helpers (internal only)
# -----------------------------------------------
def _iter_leaf_datasets(ds: Dataset) -> Iterable[Dataset]:
    """递归遍历 ConcatDataset，yield 所有叶子子集（用于 worker_init_fn）。"""
    if isinstance(ds, ConcatDataset):
        for sub in ds.datasets:
            yield from _iter_leaf_datasets(sub)
    else:
        yield ds


def _merge_oov_files(oov_paths: List[Path]) -> Path:
    """把多份 oov_xx.txt merge 成一个临时文件，返回其路径"""
    all_ids: Set[str] = set()
    for p in oov_paths:
        if p is None or not p.exists():
            continue
        all_ids.update(x.strip() for x in p.read_text().splitlines() if x.strip())
    tf = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix="_union_oov.txt")
    tf.write("\n".join(sorted(all_ids)))
    tf.close()
    return Path(tf.name)


# -----------------------------------------------
# Public API (KEEP SIGNATURES UNCHANGED)
# -----------------------------------------------
def concat_all(
    registry,
    context_len=168,
    pred_len=24,
    apply_scaler_transform="boxcox",
    scaler_transform_path=Path(os.environ["BUILDINGS_BENCH"]) / "metadata/transforms",
    weather_inputs=None,
    split: str = "",  # '', 'train', 'test'
    oov_path: Path | None = None,  # Path to oov.txt
):
    """把 registry 里的所有楼栋 dataset 拼成一个 ConcatDataset"""
    ds_list = []
    for name in registry:
        gen = load_torch_dataset(
            name,
            context_len=context_len,
            pred_len=pred_len,
            apply_scaler_transform=apply_scaler_transform,
            scaler_transform_path=scaler_transform_path,
            weather_inputs=weather_inputs,
            split=split,
            oov_path=oov_path,
        )
        for _, bldg_ds in gen:
            ds_list.append(bldg_ds)
    return ConcatDataset(ds_list)


def split_dataset(dataset, train_ratio, seed):
    """对任意 Dataset 做随机拆分（保持顺序无关）"""
    n_total = len(dataset)
    n_train = int(n_total * train_ratio)
    n_val = n_total - n_train
    g = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_val], generator=g)


def build_datasets(
    context_len=168,
    pred_len=24,
    apply_scaler_transform="boxcox",
    scaler_transform_path=Path(os.environ["BUILDINGS_BENCH"]) / "metadata/transforms",
    weather_inputs=None,
    split: str = "",  # '', 'train', 'test', 'val'
    oov_path: Path | None = None,
    oov_val: Path | None = None,
    oov_test: Path | None = None,
    val_samples_per_dataset: int = 10,  # 每个数据集在验证集中的样本数
    total_val_samples: int = None,  # 验证集总样本数（优先使用）
):
    # 900K train 全部用于训练
    pretrain_train = load_pretraining(
        "buildings-900k-train",
        context_len=context_len,
        pred_len=pred_len,
        apply_scaler_transform=apply_scaler_transform,
        scaler_transform_path=scaler_transform_path,
        weather_inputs=weather_inputs,
    )

    # 900K val - 全部加载，稍后分割
    pretrain_val_full = load_pretraining(
        "buildings-900k-val",
        context_len=context_len,
        pred_len=pred_len,
        apply_scaler_transform=apply_scaler_transform,
        scaler_transform_path=scaler_transform_path,
        weather_inputs=weather_inputs,
    )

    if split == "val":
        oov_path = (
            _merge_oov_files([oov_val, oov_test]) if oov_val and oov_test else None
        )
        custom_full_train = concat_all(
            custom_registry,
            context_len=context_len,
            pred_len=pred_len,
            apply_scaler_transform=apply_scaler_transform,
            scaler_transform_path=scaler_transform_path,
            weather_inputs=weather_inputs,
            split="train",
            oov_path=oov_path,
        )
        custom_full_val = concat_all(
            custom_registry,
            context_len=context_len,
            pred_len=pred_len,
            apply_scaler_transform=apply_scaler_transform,
            scaler_transform_path=scaler_transform_path,
            weather_inputs=weather_inputs,
            split="test",
            oov_path=oov_val,
        )
        custom_full_test = concat_all(
            custom_registry,
            context_len=context_len,
            pred_len=pred_len,
            apply_scaler_transform=apply_scaler_transform,
            scaler_transform_path=scaler_transform_path,
            weather_inputs=weather_inputs,
            split="test",
            oov_path=oov_test,
        )
        train_dataset = ConcatDataset([pretrain_train, custom_full_train])
        val_dataset = ConcatDataset([custom_full_val])
        test_dataset = ConcatDataset([pretrain_val_full, custom_full_test])

        print(f"最终训练样本：{len(train_dataset):,}")
        print(f"最终验证样本：{len(val_dataset):,}")
        print(f"最终测试样本：{len(test_dataset):,}")
        return train_dataset, val_dataset, test_dataset

    # ---------------- split == "train"：保证每个数据集在验证集中都有固定数量样本 ----------------
    if split == "train":
        # 计算每个数据集的配额
        # 总共有：pretrain_val + len(custom_registry) 个数据集需要分割
        n_datasets = 1 + len(custom_registry)  # 1个pretrain_val + 11个自定义数据集

        if total_val_samples is not None:
            # 使用总验证样本数分配
            base_quota = total_val_samples // n_datasets
            remainder = total_val_samples % n_datasets

            # 分配配额：第一个给pretrain_val，其余给自定义数据集
            quotas = [base_quota] * n_datasets
            for i in range(remainder):
                quotas[i] += 1
        else:
            # 使用固定每个数据集的样本数
            quotas = [val_samples_per_dataset] * n_datasets

        all_train_datasets = []
        all_val_datasets = []
        total_val_actual = 0

        # 1. 处理 pretrain_val
        pretrain_val_quota = quotas[0]
        n_pretrain_val = len(pretrain_val_full)

        if pretrain_val_quota > 0 and n_pretrain_val > pretrain_val_quota:
            # 从 pretrain_val 中抽取配额数量的样本作为验证集
            indices = np.arange(n_pretrain_val)

            # 均匀间隔抽样，确保覆盖整个数据集
            step = max(1, n_pretrain_val // pretrain_val_quota)
            start = np.random.randint(0, step)  # 随机起始点
            val_indices = indices[start::step][:pretrain_val_quota]
            train_indices = np.setdiff1d(indices, val_indices)

            # 创建子集
            pretrain_val_train = torch.utils.data.Subset(
                pretrain_val_full, train_indices
            )
            pretrain_val_val = torch.utils.data.Subset(pretrain_val_full, val_indices)

            all_train_datasets.append(pretrain_val_train)
            all_val_datasets.append(pretrain_val_val)

            total_val_actual += len(pretrain_val_val)
            print(
                f"pretrain_val: 总样本={n_pretrain_val}, 训练={len(pretrain_val_train)}, 验证={len(pretrain_val_val)}"
            )
        else:
            # 如果样本太少，全部用于验证集
            all_val_datasets.append(pretrain_val_full)
            total_val_actual += n_pretrain_val
            print(f"pretrain_val: 总样本={n_pretrain_val}, 全部用于验证集")

        # 2. 处理自定义数据集
        for idx, dataset_name in enumerate(custom_registry):
            quota = quotas[1 + idx]  # 跳过第一个配额（pretrain_val）

            # 加载整个数据集
            full_dataset = concat_all(
                [dataset_name],
                context_len=context_len,
                pred_len=pred_len,
                apply_scaler_transform=apply_scaler_transform,
                scaler_transform_path=scaler_transform_path,
                weather_inputs=weather_inputs,
            )

            # 获取数据集大小
            n_total = len(full_dataset)

            # 确定验证集样本数（不超过总样本数）
            n_val = min(quota, n_total)

            if n_val > 0 and n_total > n_val:
                # 使用系统性的方法抽取样本
                indices = np.arange(n_total)

                # 方法1：均匀间隔抽样
                step = max(1, n_total // n_val)
                start = np.random.randint(0, step)  # 随机起始点
                val_indices = indices[start::step][:n_val]
                train_indices = np.setdiff1d(indices, val_indices)

                # 创建子集
                train_subset = torch.utils.data.Subset(full_dataset, train_indices)
                val_subset = torch.utils.data.Subset(full_dataset, val_indices)

                all_train_datasets.append(train_subset)
                all_val_datasets.append(val_subset)

                total_val_actual += n_val
                print(
                    f"数据集 {dataset_name}: 总样本={n_total}, 训练={len(train_subset)}, 验证={n_val}"
                )
            else:
                # 如果样本太少，全部用于验证集
                all_val_datasets.append(full_dataset)
                total_val_actual += n_total
                print(f"数据集 {dataset_name}: 总样本={n_total}, 全部用于验证集")

        # 3. 添加 pretrain_train 到训练集
        all_train_datasets.append(pretrain_train)

        # 4. 合并所有数据集的训练和验证部分
        train_dataset = ConcatDataset(all_train_datasets)
        val_dataset = ConcatDataset(all_val_datasets)

        print(f"验证集总样本数：{total_val_actual}")
        print(f"最终训练样本：{len(train_dataset):,}")
        print(f"最终验证样本：{len(val_dataset):,}")
        return train_dataset, val_dataset

    # 其他 split 情况：保持原逻辑
    custom_full = concat_all(
        custom_registry,
        context_len=context_len,
        pred_len=pred_len,
        apply_scaler_transform=apply_scaler_transform,
        scaler_transform_path=scaler_transform_path,
        weather_inputs=weather_inputs,
        split=split,
        oov_path=oov_path,
    )

    # 对于 split 为 '' 的情况，pretrain_val_full 全部用于验证
    train_dataset = ConcatDataset([pretrain_train, custom_full])
    val_dataset = ConcatDataset([pretrain_val_full])

    print(f"最终训练样本：{len(train_dataset):,}")
    print(f"最终验证样本：{len(val_dataset):,}")
    return train_dataset, val_dataset


# -----------------------------------------------
# distributed_train.py (sanity check only)
# 启动方式（例如 2 张 GPU）:
# torchrun --nproc_per_node=2 distributed_train.py
# -----------------------------------------------
import torch.distributed as dist
from torch.utils.data import DistributedSampler


def worker_init_fn(worker_id):
    """
    只对 Buildings-900K 子集调用 init_fp()
    - 兼容 ConcatDataset 可能嵌套的情况
    """
    from buildings_bench.data.buildings900K_new import Buildings900K

    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return

    np.random.seed(np.random.get_state()[1][0] + worker_id)

    for sub_ds in _iter_leaf_datasets(worker_info.dataset):
        if isinstance(sub_ds, Buildings900K):
            sub_ds.init_fp()


def main():
    # ① 初始化进程组
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    # ② 构造数据集（调用方式保持不变）
    train_ds, val_ds = build_datasets()  # 返回 ConcatDataset

    # ③ 为训练集创建 DistributedSampler（val 这里不测也行）
    train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=False)

    # ④ DataLoader（与你原来一致：default_collate）
    train_loader = DataLoader(
        train_ds,
        batch_size=32,
        sampler=train_sampler,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=torch.utils.data.default_collate,
        worker_init_fn=worker_init_fn,
    )

    # ⑤ sanity check：只取前 2 个 batch，验证能正常迭代 & keys 正常
    train_sampler.set_epoch(0)
    for i, batch in enumerate(train_loader):
        print(f"[rank={dist.get_rank()}] batch#{i} keys={list(batch.keys())}")
        if i >= 1:
            break

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
