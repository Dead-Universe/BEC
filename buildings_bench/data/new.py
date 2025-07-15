from pathlib import Path
import numpy as np
import os, torch
from torch.utils.data import ConcatDataset, random_split, DataLoader
from buildings_bench.data import load_pretraining, load_torch_dataset
from buildings_bench.data.buildings900K import Buildings900K

# ----------------- 公共参数 -----------------
common_kwargs = dict(
    context_len=168,
    pred_len=24,
    apply_scaler_transform="boxcox",
    scaler_transform_path=Path(os.environ["BUILDINGS_BENCH"]) / "metadata/transforms",
    weather_inputs=None,
)

TRAIN_VAL_RATIO = 0.8  # 8 : 2
RANDOM_SEED = 42  # 保证可复现

# 需要自行拆分的 10 个数据集
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


def concat_all(registry):
    """把 registry 里的所有楼栋 dataset 拼成一个 ConcatDataset"""
    ds_list = []
    for name in registry:
        gen = load_torch_dataset(name, **common_kwargs)
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


def build_datasets():
    # def collate(batch):
    #     out = {}
    #     for k in batch[0]:
    #         out[k] = torch.stack([s[k] for s in batch])
    #     return out

    # def worker_init_fn(_):
    #     # 针对 Buildings-900K 索引文件的多进程安全打开
    #     for sub_ds in _.dataset.datasets:
    #         if isinstance(sub_ds, Buildings900K):
    #             sub_ds.init_fp()

    # 900K 直接按官方划分
    pretrain_train = load_pretraining("buildings-900k-train", **common_kwargs)
    pretrain_val = load_pretraining("buildings-900k-val", **common_kwargs)

    custom_full = concat_all(custom_registry)

    custom_train, custom_val = split_dataset(custom_full, TRAIN_VAL_RATIO, RANDOM_SEED)

    # ------------- 最终训练 / 验证合集 -----------------
    train_dataset = ConcatDataset([pretrain_train, custom_train])
    val_dataset = ConcatDataset([pretrain_val, custom_val])

    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=32,
    #     shuffle=True,
    #     num_workers=8,
    #     collate_fn=collate,
    #     worker_init_fn=worker_init_fn,
    #     pin_memory=True,
    # )
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=32,
    #     shuffle=False,
    #     num_workers=8,
    #     collate_fn=collate,
    #     worker_init_fn=worker_init_fn,
    #     pin_memory=True,
    # )

    print(f"最终训练样本：{len(train_dataset):,}")
    print(f"最终验证样本：{len(val_dataset):,}")

    return train_dataset, val_dataset


# -----------------------------------------------
#  distributed_train.py
#  启动方式（例如 2 张 GPU）:
#  torchrun --nproc_per_node=2 distributed_train.py
# -----------------------------------------------
import os, torch, torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler


def worker_init_fn(worker_id):
    # 只对 Buildings-900K 子集调用 init_fp()
    from buildings_bench.data.buildings900K import Buildings900K

    np.random.seed(np.random.get_state()[1][0] + worker_id)

    worker_info = torch.utils.data.get_worker_info()
    for sub_ds in worker_info.dataset.datasets:
        if isinstance(sub_ds, Buildings900K):
            sub_ds.init_fp()


def main():
    # ① 初始化进程组
    dist.init_process_group(backend="nccl")  # 单机多卡用 nccl；CPU-only 用 gloo
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # ② 构造数据集
    train_ds, val_ds = build_datasets()  # 返回 ConcatDataset

    # ③ 为每个数据集创建 DistributedSampler
    train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=False)
    val_sampler = DistributedSampler(val_ds, shuffle=False, drop_last=False)

    def collate(batch):
        out = {}
        for k in batch[0]:
            elems = [s[k] for s in batch]
            if isinstance(elems[0], torch.Tensor):
                out[k] = torch.stack(elems)
            else:  # ndarray / 标量
                out[k] = torch.stack([torch.as_tensor(e) for e in elems])
        return out

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
    # val_loader = DataLoader(
    #     val_ds,
    #     batch_size=32,
    #     sampler=val_sampler,
    #     shuffle=False,
    #     num_workers=8,
    #     pin_memory=True,
    #     collate_fn=collate,
    #     worker_init_fn=worker_init_fn,
    # )

    # ⑤ 训练循环里每个 epoch 调一次 sampler.set_epoch()
    for epoch in range(100):
        train_sampler.set_epoch(epoch)
        for batch in train_loader:
            # forward / backward / optim.step() …
            print(
                f"Rank {dist.get_rank()} - Epoch {epoch}: Processed batch with keys {list(batch.keys())}"
            )
            pass
        # 验证同理
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
