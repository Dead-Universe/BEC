# ──────────────────────────────────────────────
# utils_autobatch.py  （放到脚本同目录或 utils 下）
# ──────────────────────────────────────────────
import gc, torch
import torch.amp


def dry_run_model(model, seq_len, batch_size, device):
    """构造一个最小 dummy batch 做前/反向，验证是否 OOM。"""
    dummy = {
        # 这里只保留与模型 forward 里真正用到的字段
        "latitude": torch.zeros(batch_size, seq_len, 1, device=device),
        "longitude": torch.zeros(batch_size, seq_len, 1, device=device),
        "day_of_year": torch.zeros(batch_size, seq_len, 1, device=device),
        "day_of_week": torch.zeros(batch_size, seq_len, 1, device=device),
        "hour_of_day": torch.zeros(batch_size, seq_len, 1, device=device),
        "building_type": torch.zeros(
            batch_size, seq_len, 1, dtype=torch.int32, device=device
        ),
        "load": torch.zeros(batch_size, seq_len, 1, device=device),
    }
    ctx_len = seq_len // 2  # 随便给个 <= seq_len 的值
    try:
        with torch.amp.autocast(device.type):
            model = model.to(device).train()
            out = model(dummy, context_len=ctx_len, pred_len=seq_len - ctx_len)
            loss = (out * 0).sum()
        loss.backward()
        return True  # 通过
    finally:
        # 释放显存
        del dummy, out, loss
        torch.cuda.empty_cache()
        gc.collect()


def autotune_batch_size(model, ctx_lens, pred_lens, init_bs, device):
    """返回一个保证所有长度组合都能跑通的 batch_size。
    若无解则抛 RuntimeError"""
    bs = init_bs
    combos = sorted({c + p for c, p in zip(ctx_lens, pred_lens)}, reverse=True)
    while bs > 0:
        success = True
        for seq in combos:  # 从最长到最短
            try:
                torch.cuda.empty_cache()
                if not dry_run_model(model, seq, bs, device):
                    success = False
                    break
            except RuntimeError as e:
                if "out of memory" in str(e):
                    success = False
                    break
                else:
                    raise  # 其他错误直接抛
        if success:
            return bs
        bs //= 2  # OOM → batch_size 减半重试
    raise RuntimeError("Unable to fit any batch even with batch_size=1")
