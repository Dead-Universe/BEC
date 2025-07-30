# utils_mixed.py  (新文件，或放到 utils.py 里)

import random, torch

# 允许的上下限
CTX_MIN, CTX_MAX = 168, 336
PRED_MIN, PRED_MAX = 24, 168


def mixed_len_collate(batch):
    """对每个 mini‑batch 统一随机采样 (L,H)，并做 padding + mask。"""
    # 1) 随机采样
    if random.random() < 0.8:
        ctx_len, pred_len = random.choice([(168, 24), (224, 48), (280, 96), (336, 168)])
    else:
        ctx_len = random.randint(CTX_MIN, CTX_MAX)
        pred_len = random.randint(PRED_MIN, PRED_MAX)

    seq_len = ctx_len + pred_len

    # 2) 逐样本裁剪 / 截断
    for sample in batch:
        for k, v in sample.items():
            # 时间序列类字段 → [T, ...]
            if v.ndim >= 2 and v.size(0) >= seq_len:  # Tensor
                sample[k] = v[:seq_len]
            elif isinstance(v, torch.Tensor) and v.ndim == 1 and v.size(0) >= seq_len:
                sample[k] = v[:seq_len]

    # 3) 拼 batch（先 pad 到同一长度 = seq_len）
    def _stack(key, dtype):
        ts_list = [s[key] for s in batch]
        return torch.stack(ts_list).to(dtype)  # 已保证同长

    out = {
        "latitude": _stack("latitude", torch.float32),
        "longitude": _stack("longitude", torch.float32),
        "day_of_year": _stack("day_of_year", torch.float32),
        "day_of_week": _stack("day_of_week", torch.float32),
        "hour_of_day": _stack("hour_of_day", torch.float32),
        "building_type": _stack("building_type", torch.long),
        "load": _stack("load", torch.float32),
    }

    # 4) 生成 attention mask（1=有效；0=padding）
    #    因为 batch 内已同长，这里只要告诉模型 **ctx_len** & **pred_len**
    out["context_len"] = ctx_len  # int
    out["pred_len"] = pred_len

    return out
