import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import tomli
import torch
from buildings_bench import benchmark_registry, load_torch_dataset, utils
from buildings_bench.evaluation import aggregate, scoring_rule_factory
from buildings_bench.evaluation.managers import BuildingTypes, DatasetMetricsManager
from buildings_bench.models import model_factory
from buildings_bench.tokenizer import LoadQuantizer
from buildings_bench.transforms import BoxCoxTransform
from statsmodels.tsa.seasonal import STL

SCRIPT_PATH = Path(os.path.realpath(__file__)).parent

# 建议在训练脚本入口处限制底层 BLAS 线程，避免和批内并行抢核
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
import hashlib
from typing import Optional

try:
    import lmdb

    _HAS_LMDB = True
except Exception:
    _HAS_LMDB = False

try:
    import zstandard as _zstd

    _ZSTD_COMP = _zstd.ZstdCompressor(level=3)
    _ZSTD_DECOMP = _zstd.ZstdDecompressor()
    _USE_ZSTD = True
except Exception:
    import zlib as _zlib

    _USE_ZSTD = False

_STL_CACHE_ENV = None


def _get_lmdb_env(cache_dir: Path, map_size_gb: int = 64):
    global _STL_CACHE_ENV
    if _STL_CACHE_ENV is None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        _STL_CACHE_ENV = lmdb.open(
            str(cache_dir),
            map_size=map_size_gb * (1 << 30),
            subdir=True,
            lock=True,
            max_dbs=1,
            max_readers=4096,
            readahead=False,
            writemap=False,
            metasync=False,
            sync=True,
        )
    return _STL_CACHE_ENV


def _compress_bytes(b: bytes) -> bytes:
    if _USE_ZSTD:
        return _ZSTD_COMP.compress(b)
    else:
        return _zlib.compress(b, 3)


def _decompress_bytes(b: bytes) -> bytes:
    if _USE_ZSTD:
        return _ZSTD_DECOMP.decompress(b)
    else:
        return _zlib.decompress(b)


# 复用的线程池（每个 DataLoader worker 会各自持有一份）
_STL_POOL = None


def _get_pool(n_workers: int):
    global _STL_POOL
    if _STL_POOL is None:
        _STL_POOL = ThreadPoolExecutor(max_workers=n_workers)
    return _STL_POOL


def _stl_one_series(x_ctx_fp64: np.ndarray, period: int = 24, robust: bool = False):
    """
    在 Box-Cox 尺度上对一条 context 段做 STL。
    返回 (S_ctx, T_ctx, R_ctx)，都是 float32，长度 = len(x_ctx_fp64)。
    失败时回退到简易移动平均趋势。
    """
    try:
        stl = STL(x_ctx_fp64, period=period, robust=robust)  # 可按需加 seasonal/trend
        res = stl.fit()
        S = res.seasonal.astype(np.float32)
        T = res.trend.astype(np.float32)
    except Exception:
        # 回退：奇数窗口的移动平均作为趋势；季节取残差
        win = max(3, (period // 2) * 2 + 1)  # 约等于 period/2，保证奇数
        kern = np.ones(win, dtype=np.float64) / win
        T = np.convolve(x_ctx_fp64, kern, mode="same").astype(np.float32)
        S = x_ctx_fp64.astype(np.float32) - T
    R = x_ctx_fp64.astype(np.float32) - S - T
    return S, T, R


STL_ENABLED = False


def eval_collate_with_stl(
    batch,
    *,
    ctx_len: int,
    pred_len: int,
    stl_boxcox,  # BoxCoxTransform：仅供 STL 用的尺度（独立于模型用的 transform）
    stl_period: int = 24,
    stl_robust: bool = True,
    stl_workers: int | None = None,
    # 缓存相关
    stl_cache_dir: Optional[Path] = None,
    stl_cache_map_gb: int = 64,
    stl_cache_disable: bool = False,
):
    """
    - 输入样本是“实量纲” load；
    - 保持 merged['load'] 为实量纲；
    - 仅将 context 段在 Box-Cox 尺度做 STL，得到 stl_S/T/R（预测段填 0）；
    - 命中缓存直接回填，miss 再计算并写缓存。
    """
    seq_len = ctx_len + pred_len

    # ---------------- 1) 固定窗口裁剪 ----------------
    for s in batch:
        for k, v in s.items():
            if isinstance(v, (torch.Tensor, np.ndarray)) and getattr(v, "ndim", 0) >= 1:
                if v.shape[0] < seq_len:
                    raise ValueError(f"val 样本过短: {v.shape[0]} < {seq_len}")
                s[k] = v[:seq_len]

    def _stack(key, dtype=None):
        ts = [torch.as_tensor(s[key]) for s in batch]
        out = torch.stack(ts)
        return out if dtype is None else out.to(dtype)

    # ---------------- 2) 构建 merged（这就是你问的 merged） ----------------
    merged = {
        "latitude": _stack("latitude", torch.float32),
        "longitude": _stack("longitude", torch.float32),
        "day_of_year": _stack("day_of_year", torch.float32),
        "day_of_week": _stack("day_of_week", torch.float32),
        "hour_of_day": _stack("hour_of_day", torch.float32),
        "building_type": _stack("building_type", torch.long),
        "load": _stack("load", torch.float32),  # 实量纲
    }
    # 把样本里其它张量型键也并进来（若存在）
    for k in batch[0]:
        if k not in merged and isinstance(batch[0][k], (torch.Tensor, np.ndarray)):
            merged[k] = _stack(k, torch.float32)
    if STL_ENABLED:
        # ---------------- 3) STL：在 Box-Cox 尺度，仅 context 段 ----------------
        load_real = merged["load"]  # (B, L, 1)
        if load_real.is_cuda:  # STL 在 CPU 上跑
            load_real = load_real.cpu()
            merged["load"] = load_real

        # 仅供 STL 的 Box-Cox（与模型训练/评测的缩放无关）
        load_bc_any = stl_boxcox.transform(load_real)
        if isinstance(load_bc_any, torch.Tensor):
            load_bc = load_bc_any.detach().cpu().numpy().astype(np.float64, copy=False)
        else:
            load_bc = np.asarray(load_bc_any, dtype=np.float64)
        if load_bc.ndim == 2:  # 兜底补维
            load_bc = load_bc[..., None]

        B, L, _ = load_bc.shape
        S = torch.zeros(B, L, 1, dtype=torch.float32)
        T = torch.zeros(B, L, 1, dtype=torch.float32)
        R = torch.zeros(B, L, 1, dtype=torch.float32)

        # ---------------- 4) 缓存：LMDB（多读单写） ----------------
        use_cache = (
            (not stl_cache_disable) and _HAS_LMDB and (stl_cache_dir is not None)
        )
        if use_cache:
            env = _get_lmdb_env(Path(stl_cache_dir), stl_cache_map_gb)
            meta = f"v1|p{stl_period}|r{int(stl_robust)}|ctx{ctx_len}|scale:boxcox".encode()
        else:
            env, meta = None, None

        to_compute = []
        cached_ctx = {}

        # 4.1 读缓存
        if env is not None:
            with env.begin(write=False) as rtxn:
                for i in range(B):
                    x_ctx_f32 = load_bc[i, :ctx_len, 0].astype(np.float32, copy=False)
                    h = (
                        hashlib.blake2b(x_ctx_f32.tobytes(), digest_size=16)
                        .hexdigest()
                        .encode()
                    )
                    key = meta + b"|" + h
                    val = rtxn.get(key)
                    if val is not None:
                        buf = _decompress_bytes(val)
                        arr = (
                            np.frombuffer(buf, dtype=np.float32)
                            .copy()
                            .reshape(ctx_len, 3)
                        )
                        cached_ctx[i] = (arr[:, 0], arr[:, 1], arr[:, 2])
                    else:
                        to_compute.append(i)
        else:
            to_compute = list(range(B))

        # 4.2 miss 的样本并行做 STL
        n_workers = stl_workers or min(os.cpu_count() or 1, len(to_compute) or 1)
        pool = _get_pool(n_workers)
        futs = [
            (
                i,
                pool.submit(
                    _stl_one_series, load_bc[i, :ctx_len, 0], stl_period, stl_robust
                ),
            )
            for i in to_compute
        ]

        # 4.3 命中回填
        for i, (s_ctx, t_ctx, r_ctx) in cached_ctx.items():
            S[i, :ctx_len, 0] = torch.from_numpy(s_ctx)
            T[i, :ctx_len, 0] = torch.from_numpy(t_ctx)
            R[i, :ctx_len, 0] = torch.from_numpy(r_ctx)

        # 4.4 新计算的回填 + 写缓存
        if futs:
            if env is not None:
                with env.begin(write=True) as wtxn:
                    for i, f in futs:
                        s_ctx, t_ctx, r_ctx = f.result()
                        S[i, :ctx_len, 0] = torch.from_numpy(s_ctx)
                        T[i, :ctx_len, 0] = torch.from_numpy(t_ctx)
                        R[i, :ctx_len, 0] = torch.from_numpy(r_ctx)
                        x_ctx_f32 = load_bc[i, :ctx_len, 0].astype(
                            np.float32, copy=False
                        )
                        h = (
                            hashlib.blake2b(x_ctx_f32.tobytes(), digest_size=16)
                            .hexdigest()
                            .encode()
                        )
                        key = meta + b"|" + h
                        val_arr = np.stack([s_ctx, t_ctx, r_ctx], axis=1).astype(
                            np.float32, copy=False
                        )
                        _ = wtxn.put(
                            key, _compress_bytes(val_arr.tobytes()), overwrite=False
                        )
            else:
                for i, f in futs:
                    s_ctx, t_ctx, r_ctx = f.result()
                    S[i, :ctx_len, 0] = torch.from_numpy(s_ctx)
                    T[i, :ctx_len, 0] = torch.from_numpy(t_ctx)
                    R[i, :ctx_len, 0] = torch.from_numpy(r_ctx)

        # ---------------- 5) 输出 ----------------
        merged["stl_S"] = S  # Box-Cox 尺度
        merged["stl_T"] = T
        merged["stl_R"] = R
    merged["context_len"] = torch.tensor(ctx_len)
    merged["pred_len"] = torch.tensor(pred_len)
    return merged


FM_MODELS = {
    "timesfm": "timesfm",
    "moment": "moment",
    "timemoe": "timemoe",
    "chronos": "chronos",
}


@torch.no_grad()
def zero_shot_learning(args, model_args, results_path: Path):
    device = args.device

    global STL_ENABLED
    STL_ENABLED = args.stl_enabled

    is_fm = any(args.model.startswith(p) for p in FM_MODELS)

    transform_path = (
        Path(os.environ.get("BUILDINGS_BENCH", "")) / "metadata" / "transforms"
    )

    context_len = getattr(args, "context_len", 168)  # Default context length is 168
    forecast_horizon = getattr(args, "forecast_horizon", 24)  # Default horizon is 24

    if not is_fm:
        model, _, predict = model_factory(args.model, model_args)
        model = model.to(device)
        kind = None

        if not model.continuous_loads:
            load_transform = LoadQuantizer(
                with_merge=(not args.tokenizer_without_merge),
                num_centroids=model.vocab_size,
                device="cuda:0" if "cuda" in device else "cpu",
            )
            load_transform.load(transform_path)

        # Load from ckpts
        if args.checkpoint != "":
            model.load_from_checkpoint(args.checkpoint)

        if device.startswith("cuda") and torch.cuda.device_count() > 1:
            print(f"Use DataParallel on GPUs: {list(range(torch.cuda.device_count()))}")
            model = torch.nn.DataParallel(
                model, device_ids=list(range(torch.cuda.device_count()))
            )
        model.eval()
    else:

        kind = next(v for k, v in FM_MODELS.items() if args.model.startswith(k))
        if kind == "timesfm":
            import timesfm

            fm = timesfm.TimesFm(
                hparams=timesfm.TimesFmHparams(
                    backend="gpu" if "cuda" in device else "cpu",
                    # per_core_batch_size=32,
                    horizon_len=forecast_horizon,
                    num_layers=50,
                    use_positional_embedding=False,
                    # context_len=context_len,
                ),
                checkpoint=timesfm.TimesFmCheckpoint(
                    huggingface_repo_id="google/timesfm-2.0-500m-pytorch"
                ),
            )
        elif kind == "moment":
            ### ← 新增：加载 MOMENTPipeline
            from momentfm import MOMENTPipeline

            fm = MOMENTPipeline.from_pretrained(
                (
                    args.model_name  # 例如 "AutonLab/MOMENT-1-large"
                    if hasattr(args, "model_name") and args.model_name != ""
                    else "AutonLab/MOMENT-1-large"
                ),
                model_kwargs={
                    "task_name": "forecasting",
                    "forecast_horizon": forecast_horizon,
                    "seq_len": context_len,
                },
            )
            fm.init()  # 必不可少，内部会构建推理 head
            fm.to(device)
            fm.eval()

            pass
        elif kind == "timemoe":
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                device_map="cuda" if "cuda" in device else "cpu",
                trust_remote_code=True,
            )
        elif kind == "chronos":
            from chronos import BaseChronosPipeline

            # 从 HuggingFace Hub 下载预训练模型，并根据 device 选择推理设备
            pipeline = BaseChronosPipeline.from_pretrained(
                args.model_name,  # e.g. "amazon/chronos-t5-small"
                device_map="cuda" if "cuda" in device else "cpu",
                torch_dtype=torch.bfloat16,
            )
        else:
            fm = None

    if args.benchmark[0] == "all":
        args.benchmark = benchmark_registry
    elif args.benchmark[0] == "real":
        y = [x for x in benchmark_registry if x != "buildings-900k-test"]
        args.benchmark = y
    elif args.benchmark[0] == "bdg-2":
        args.benchmark = [x for x in benchmark_registry if "bdg-2:" in x]
    elif args.benchmark[0] == "test":
        args.benchmark = ["university", "cofactor"]

    if args.ignore_scoring_rules:
        metrics_manager = DatasetMetricsManager()
    elif is_fm or (
        model.module.continuous_loads
        if isinstance(model, torch.nn.DataParallel)
        else model.continuous_loads
    ):
        metrics_manager = DatasetMetricsManager(
            scoring_rule=(
                scoring_rule_factory("crps")
                if (
                    not is_fm
                    and (
                        model.module.continuous_head == "gaussian_nll"
                        if isinstance(model, torch.nn.DataParallel)
                        else model.continuous_head == "gaussian_nll"
                    )
                )
                else None
            )
        )
    else:
        metrics_manager = DatasetMetricsManager(
            scoring_rule=scoring_rule_factory("rps")
        )

    print(f"Evaluating model on test datasets {args.benchmark}...")

    # Iterate over each dataset in the benchmark
    for dataset_name in args.benchmark:
        # Load the dataset generator
        buildings_datasets_generator = load_torch_dataset(
            dataset_name,
            # apply_scaler_transform=args.apply_scaler_transform,
            scaler_transform_path=transform_path,
            include_outliers=args.include_outliers,
            weather_inputs=model_args["weather_inputs"],
            context_len=context_len,
            pred_len=forecast_horizon,
            split="test",
            oov_path=args.oov_path,
        )

        num_of_buildings = len(buildings_datasets_generator)
        print(f"dataset {dataset_name}: {num_of_buildings} buildings")
        # For each building
        for count, (building_name, building_dataset) in enumerate(
            buildings_datasets_generator, start=1
        ):
            print(
                f"dataset {dataset_name} {count}/{num_of_buildings} building-year {building_name} "
                f"day-ahead forecasts {len(building_dataset)}"
            )

            start_time = time.time()

            metrics_manager.add_building_to_dataset_if_missing(
                dataset_name,
                building_name,
            )

            if not is_fm and not (
                model.module.continuous_loads
                if isinstance(model, torch.nn.DataParallel)
                else model.continuous_loads
            ):  # Quantized loads
                transform = load_transform.transform
                inverse_transform = load_transform.undo_transform
            elif (
                not is_fm and args.apply_scaler_transform != ""
            ):  # Scaling continuous values
                # transform = lambda x: x

                # if isinstance(building_dataset, torch.utils.data.ConcatDataset):
                #     load_transform = building_dataset.datasets[0].load_transform
                #     inverse_transform = load_transform.undo_transform
                # else:
                #     load_transform = building_dataset.load_transform
                #     inverse_transform = load_transform.undo_transform
                tpath = (
                    Path(os.getenv("BUILDINGS_BENCH", "")) / "metadata" / "transforms"
                )
                if args.apply_scaler_transform == "boxcox":
                    scaler = BoxCoxTransform()
                    scaler.load(tpath)
                    transform = scaler.transform
                    inverse_transform = scaler.undo_transform
                    # stl_boxcox = BoxCoxTransform()
                    # stl_boxcox.load(transform_path)

            else:  # Continuous unscaled values
                scaler = None
                transform = lambda x: x
                inverse_transform = lambda x: x

            # create a dataloader for the building
            building_dataloader = torch.utils.data.DataLoader(
                building_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=("cuda" in device),
                collate_fn=partial(
                    eval_collate_with_stl,
                    ctx_len=context_len,
                    pred_len=forecast_horizon,
                    stl_boxcox=scaler,
                    stl_period=args.stl_period,
                    stl_robust=args.stl_robust,
                    stl_workers=(args.stl_workers if args.stl_workers > 0 else None),
                    stl_cache_dir=(
                        None if args.stl_cache_disable else args.stl_cache_dir
                    ),
                    stl_cache_map_gb=(
                        args.stl_cache_map_gb
                        if hasattr(args, "stl_cache_map_gb")
                        else 64
                    ),
                    stl_cache_disable=getattr(args, "stl_cache_disable", False),
                ),
            )
            for batch in building_dataloader:

                if kind == "timesfm":
                    full_len = context_len + forecast_horizon
                    # A) pull out your 168+24 window
                    arr = batch["load"][..., 0].cpu().numpy()  # (B, 192)
                    arr = inverse_transform(arr)
                    X_ctx = arr[:, :context_len]  # (B,168)
                    Y_true = torch.tensor(
                        arr[:, context_len:full_len],
                        device=device,
                    )  # (B,24)

                    # B) prepare covariates over [0:168+24)

                    static_num = {
                        "latitude": batch["latitude"][:, 0, 0].cpu().numpy().tolist(),
                        "longitude": batch["longitude"][:, 0, 0].cpu().numpy().tolist(),
                    }
                    static_cat = {
                        "building_type": batch["building_type"][:, 0, 0]
                        .cpu()
                        .numpy()
                        .astype(int)
                        .tolist(),
                    }
                    dynamic_cat = {
                        "day_of_year": batch["day_of_year"][:, :full_len, 0]
                        .cpu()
                        .numpy()
                        .astype(int)
                        .tolist(),
                        "day_of_week": batch["day_of_week"][:, :full_len, 0]
                        .cpu()
                        .numpy()
                        .astype(int)
                        .tolist(),
                        "hour_of_day": batch["hour_of_day"][:, :full_len, 0]
                        .cpu()
                        .numpy()
                        .astype(int)
                        .tolist(),
                    }
                    # if you have weather columns, build dynamic_numerical_covariates similarly...
                    dynamic_num = None  # or a dict of lists of length full_len

                    # C) Forecast — TimesFM will pad 168→192 in multiples of 32 automatically,
                    #    no need to compute patch_len or pad yourself.
                    # raw_forecast, _ = fm.forecast_with_covariates(
                    #     inputs=X_ctx.tolist(),
                    #     dynamic_numerical_covariates=dynamic_num,
                    #     dynamic_categorical_covariates=dynamic_cat,
                    #     static_numerical_covariates=static_num,
                    #     static_categorical_covariates=static_cat,
                    #     freq=[0] * X_ctx.shape[0],
                    #     # explicitly tell it your window is 168+24,
                    #     # but let it pad/truncate internally to the model’s patch multiple.
                    #     # forecast_context_len=context_len,
                    #     # window_size=full_len,
                    #     xreg_mode="xreg + timesfm",
                    #     normalize_xreg_target_per_input=True,
                    #     ridge=0.0,
                    #     force_on_cpu=("cpu" in device),
                    # )

                    raw_forecast, _ = fm.forecast(
                        inputs=X_ctx.tolist(),
                        freq=[0] * X_ctx.shape[0],
                    )

                    # D) take the last 24 steps as your zero-shot prediction
                    preds_np = (
                        np.stack(raw_forecast)
                        if isinstance(raw_forecast, list)
                        else raw_forecast
                    )  # (B, 24)
                    # we know padded_len ≥ 168+24, so slice out the 24 after the first 168:
                    Y_pred = torch.tensor(preds_np, device=device)

                    # E) send into your metrics_manager
                    mask = (
                        batch["building_type"][:, 0, 0] == BuildingTypes.COMMERCIAL_INT
                    )
                    metrics_manager(
                        dataset_name,
                        building_name,
                        Y_true,
                        Y_pred,
                        mask,
                        y_categories=None,
                        y_distribution_params=None,
                        centroids=None,
                    )
                elif kind == "moment":
                    # ① 拿出 (B, context+hor) 的原始负荷
                    arr = batch["load"][..., 0].cpu().numpy()
                    arr = inverse_transform(arr)

                    # —————  上下文 & 真实值  —————
                    X_ctx = arr[:, :context_len]  # (B, 168)
                    Y_true = torch.tensor(
                        arr[:, context_len : context_len + forecast_horizon],
                        device=device,
                    )  # (B, 24)

                    # ② 调整维度：(B, 168) ➜ (B, 1, 168)
                    x_enc = torch.from_numpy(X_ctx).float().unsqueeze(1).to(device)
                    input_mask = torch.ones_like(x_enc[:, 0, :], dtype=torch.long)

                    # ③ Zero-shot 预测
                    with torch.inference_mode():
                        out = fm.forecast(x_enc=x_enc, input_mask=input_mask)
                    Y_pred = out.forecast.squeeze(1)  # (B, 24)

                    # ③ 入指标管理器
                    mask = (
                        batch["building_type"][:, 0, 0] == BuildingTypes.COMMERCIAL_INT
                    ).to(device)
                    metrics_manager(
                        dataset_name,
                        building_name,
                        Y_true,
                        Y_pred,
                        mask,
                        y_categories=None,
                        y_distribution_params=None,
                        centroids=None,
                    )
                elif kind == "timemoe":
                    arr = batch["load"][..., 0].cpu().numpy()
                    arr = inverse_transform(arr)
                    X_ctx = arr[:, :context_len]  # (B,168)
                    Y_true = torch.tensor(
                        arr[:, context_len : context_len + forecast_horizon],
                        device=device,
                    )  # (B,24)
                    # X_ctx 转为 tensor
                    context_tensor = torch.tensor(X_ctx, device=device)

                    # nnormalize context_tensor
                    mean, std = context_tensor.mean(
                        dim=-1, keepdim=True
                    ), context_tensor.std(dim=-1, keepdim=True).clamp_min(1e-6)
                    normed_seqs = (context_tensor - mean) / std

                    output = model.generate(
                        normed_seqs,
                        max_new_tokens=forecast_horizon,
                    )  # (B, 192)

                    normed_predictions = output[:, -forecast_horizon:]  # (B, 24)
                    # 反归一化
                    predictions = normed_predictions * std + mean
                    predictions = predictions.to(device)
                    # 送入 metrics_manager
                    mask = (
                        batch["building_type"][:, 0, 0] == BuildingTypes.COMMERCIAL_INT
                    ).to(device)
                    metrics_manager(
                        dataset_name,
                        building_name,
                        Y_true,
                        predictions,
                        mask,
                        y_categories=None,
                        y_distribution_params=None,
                        centroids=None,
                    )

                    pass
                elif kind == "chronos":
                    # A) 从 batch 中取出原始负荷序列 (B, T, 1) → (B, T)
                    arr = batch["load"][..., 0].cpu().numpy()
                    arr = inverse_transform(arr)

                    # 新增：定义子批次大小 (根据内存调整)
                    sub_batch_size = 48  # 可调参数，建议设为2的幂次
                    batch_size = arr.shape[0]

                    # 新增：初始化结果容器
                    all_Y_pred = []

                    # B) 循环处理子批次
                    for start_idx in range(0, batch_size, sub_batch_size):
                        end_idx = min(start_idx + sub_batch_size, batch_size)
                        sub_arr = arr[start_idx:end_idx]

                        # 切分上下文和真实值
                        X_ctx = sub_arr[:, :context_len]  # (sub_batch, context_len)

                        # C) 组装上下文张量，调用 Chronos 预测分位数
                        context_tensor = torch.tensor(X_ctx)
                        _, mean = pipeline.predict_quantiles(
                            context=context_tensor,
                            prediction_length=forecast_horizon,
                        )

                        # D) 获取预测结果
                        Y_pred = (
                            mean.clone().detach().to(device)
                        )  # (sub_batch, horizon)

                        # 新增：存储子批次结果
                        all_Y_pred.append(Y_pred)

                    # E) 合并所有子批次结果
                    Y_pred_full = torch.cat(all_Y_pred, dim=0)
                    # 一次性获取完整真实值和掩码
                    Y_true_full = torch.tensor(
                        arr[:, context_len : context_len + forecast_horizon],
                        device=device,
                    )  # (B, horizon)
                    mask_full = (
                        batch["building_type"][:, 0, 0] == BuildingTypes.COMMERCIAL_INT
                    ).to(
                        device
                    )  # (B,)

                    # F) 送入 metrics_manager
                    metrics_manager(
                        dataset_name,
                        building_name,
                        Y_true_full,
                        Y_pred_full,
                        mask_full,
                        y_categories=None,
                        y_distribution_params=None,
                        centroids=None,
                    )
                else:
                    building_types_mask = (
                        batch["building_type"][:, 0, 0] == BuildingTypes.COMMERCIAL_INT
                    )

                    for k, v in batch.items():
                        batch[k] = v.to(device)

                    continuous_load = batch["load"].clone()
                    continuous_targets = continuous_load[
                        :,
                        args.context_len :,
                    ]

                    # Transform if needed
                    # 放到cpu
                    batch["load"] = batch["load"].cpu()
                    batch["load"] = transform(batch["load"])
                    # 转为 tensor
                    batch["load"] = torch.tensor(
                        batch["load"], device=device, dtype=torch.float32
                    )

                    # These could be tokens or continuous
                    targets = batch["load"][
                        :,
                        args.context_len :,
                    ]

                    if args.device == "cuda":
                        with torch.amp.autocast("cuda"):
                            if isinstance(model, torch.nn.DataParallel):
                                # DataParallel.wrap 之后，真正的子模块在 model.module
                                predictions, distribution_params = model(batch)
                            else:
                                predictions, distribution_params = predict(
                                    batch,
                                    context_len=context_len,
                                    pred_len=forecast_horizon,
                                )
                    else:
                        predictions, distribution_params = predict(batch)

                    if (
                        isinstance(model, torch.nn.DataParallel)
                        and model.module.continuous_head != "gaussian_nll"
                    ) or (
                        not isinstance(model, torch.nn.DataParallel)
                        and model.continuous_head != "gaussian_nll"
                    ):
                        distribution_params = None

                    predictions = predictions.cpu()
                    predictions = inverse_transform(predictions)
                    if isinstance(predictions, np.ndarray):
                        # 如果 predictions 是 numpy 数组，转换为 tensor
                        predictions = torch.tensor(
                            predictions, device=device, dtype=torch.float32
                        )
                    elif isinstance(predictions, torch.Tensor):
                        # 如果 predictions 已经是 tensor，确保其类型正确
                        predictions = predictions.to(device, dtype=torch.float32)

                    if args.apply_scaler_transform != "":

                        # continuous_targets = inverse_transform(continuous_targets)

                        # print(
                        #     "continuous_targets after inverse_transform:",
                        #     continuous_targets,
                        # )

                        # invert for crps
                        targets = inverse_transform(targets)
                        if (
                            args.apply_scaler_transform == "standard"
                            and distribution_params is not None
                        ):
                            mu = inverse_transform(distribution_params[:, :, 0])
                            sigma = load_transform.undo_transform_std(
                                distribution_params[:, :, 1]
                            )
                            distribution_params = torch.cat(
                                [mu.unsqueeze(-1), sigma.unsqueeze(-1)], -1
                            )

                        elif (
                            args.apply_scaler_transform == "boxcox"
                            and distribution_params is not None
                        ):
                            ######## backproject approximate Gaussian in unscaled space ########
                            mu = inverse_transform(distribution_params[:, :, 0])
                            muplussigma = inverse_transform(
                                torch.sum(distribution_params, -1)
                            )
                            sigma = muplussigma - mu
                            muminussigma = inverse_transform(
                                distribution_params[:, :, 0]
                                - distribution_params[:, :, 1]
                            )
                            sigma = (sigma + (mu - muminussigma)) / 2
                            distribution_params = torch.cat(
                                [mu.unsqueeze(-1), sigma.unsqueeze(-1)], -1
                            )

                    if not (
                        model.module.continuous_loads
                        if isinstance(model, torch.nn.DataParallel)
                        else model.continuous_loads
                    ):
                        centroids = (
                            load_transform.kmeans.centroids.squeeze()
                            if args.tokenizer_without_merge
                            else load_transform.merged_centroids
                        )
                    else:
                        centroids = None

                    metrics_manager(
                        dataset_name,
                        building_name,
                        continuous_targets,
                        predictions,
                        building_types_mask,
                        y_categories=targets,
                        y_distribution_params=distribution_params,
                        centroids=centroids,
                    )
            # if count == 10:
            #     break
            elapsed_time = time.time() - start_time
            print(f"Building {building_name} processed in {elapsed_time:.2f} seconds.")
    print("Generating summaries...")
    variant_name = f":{args.variant_name}" if args.variant_name != "" else ""
    metrics_file = results_path / f"metrics_{args.model}{variant_name}.csv"
    scoring_rule_file = results_path / f"scoring_rule_{args.model}{variant_name}.csv"

    if not args.ignore_scoring_rules and (
        model.module.continuous_head == "gaussian_nll"
        if isinstance(model, torch.nn.DataParallel)
        else model.continuous_head == "gaussian_nll"
    ):
        metrics_df, scoring_rule_df = metrics_manager.summary()

        metrics_df.to_csv(metrics_file, index=False)

        scoring_rule_df.to_csv(scoring_rule_file, index=False)
    else:
        metrics_df = metrics_manager.summary()
        metrics_df.to_csv(metrics_file, index=False)

    # Compute and display aggregate statistics
    with open(
        Path(os.environ.get("BUILDINGS_BENCH", "")) / "metadata" / "oov.txt", "r"
    ) as f:
        oov_bldgs = []
        for line in f:
            oov_bldgs += [line.strip().split(" ")[1]]

        metric_names = [m.name for m in metrics_manager.metrics_list]
        if metrics_manager.scoring_rule:
            metric_names += [metrics_manager.scoring_rule.name]

        if len(args.benchmark) > 1 or (
            len(args.benchmark) == 1 and args.benchmark[0] != "buildings-900k-test"
        ):
            # Returns a dictionary with the median of the nrmse (cv-rmse)
            # and crps metrics for the model with boostrapped 95% confidence intervals
            print("BuildingsBench (real)")
            print()
            real_med = aggregate.return_aggregate(
                model_list=[f"{args.model}{variant_name}"],
                results_dir=str(results_path),
                experiment="zero_shot",
                metrics=metric_names,
                aggregate="median",  # ← 新增参数
                exclude_simulated=True,
                # oov_list=oov_bldgs,
            )
            aggregate.pretty_print(real_med, aggregate="median")
            print()

            # ------- （可选）同时算“均值” -------
            real_mean = aggregate.return_aggregate(
                model_list=[f"{args.model}{variant_name}"],
                results_dir=str(results_path),
                experiment="zero_shot",
                metrics=metric_names,
                aggregate="mean",  # ← 换成 mean
                exclude_simulated=True,
                # oov_list=oov_bldgs,
            )
            aggregate.pretty_print(real_mean, aggregate="mean")

            print("BuildingsBench (real) - only test buildings")
            print()
            real_test_med = aggregate.return_aggregate(
                model_list=[f"{args.model}{variant_name}"],
                results_dir=str(results_path),
                experiment="zero_shot",
                metrics=metric_names,
                aggregate="median",
                exclude_simulated=True,
                oov_file=str(args.oov_path) if args.oov_path else None,
            )
            aggregate.pretty_print(real_test_med, aggregate="median")
            print()
            # （可选）均值
            real_test_mean = aggregate.return_aggregate(
                model_list=[f"{args.model}{variant_name}"],
                results_dir=str(results_path),
                experiment="zero_shot",
                metrics=metric_names,
                aggregate="mean",
                exclude_simulated=True,
                oov_file=str(args.oov_path) if args.oov_path else None,
            )
            aggregate.pretty_print(real_test_mean, aggregate="mean")

        if "buildings-900k-test" in args.benchmark:
            print("Buildings-900K-test (synth)")
            synth_med = aggregate.return_aggregate(
                model_list=[f"{args.model}{variant_name}"],
                results_dir=str(results_path),
                experiment="zero_shot",
                metrics=metric_names,
                aggregate="median",
                exclude_simulated=False,
                only_simulated=True,
                # oov_list=oov_bldgs,
            )
            aggregate.pretty_print(synth_med, aggregate="median")

            # （可选）均值
            synth_mean = aggregate.return_aggregate(
                model_list=[f"{args.model}{variant_name}"],
                results_dir=str(results_path),
                experiment="zero_shot",
                metrics=metric_names,
                aggregate="mean",
                exclude_simulated=False,
                only_simulated=True,
                # oov_list=oov_bldgs,
            )
            aggregate.pretty_print(synth_mean, aggregate="mean")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--results_path", type=str, default="results/")
    parser.add_argument(
        "--model",
        type=str,
        default="",
        required=True,
        help="Name of your model. Should match the config"
        " filename without .toml extension."
        ' Example: "TransformerWithTokenizer-S"',
    )
    parser.add_argument(
        "--benchmark",
        nargs="+",
        type=str,
        default=["all"],
        help='Which datasets in the benchmark to run. Default is ["all."] '
        "See the dataset registry in buildings_bench.data.__init__.py for options.",
    )
    parser.add_argument(
        "--include_outliers",
        action="store_true",
        help="Eval with a filtered variant with certain outliers removed",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=360)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--ignore_scoring_rules",
        action="store_true",
        help="Do not compute a scoring rule for this model.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Path to a checkpoint to load. Optional. "
        " One can also load a checkpoint from Wandb by specifying the run_id.",
    )
    parser.add_argument(
        "--variant_name",
        type=str,
        default="",
        help="Name of the variant. Optional. Used for results files.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use. Default is cuda if available else cpu.",
    )
    parser.add_argument(
        "--tokenizer_without_merge",
        action="store_true",
        default=False,
        help="Use the tokenizer without merge. Default is False.",
    )
    parser.add_argument(
        "--apply_scaler_transform",
        type=str,
        default="",
        choices=["", "standard", "boxcox"],
        help="Apply a scaler transform to the load values.",
    )
    parser.add_argument(
        "--context_len",
        type=int,
        default=168,
        help="Length of the context window. Default is 168.",
    )
    parser.add_argument(
        "--forecast_horizon",
        type=int,
        default=24,
        help="Length of the forecast horizon. Default is 24.",
    )
    parser.add_argument(
        "--oov_path",
        type=str,
        default="",
        help="Path to the out-of-vocabulary (OOV) file. "
        "If not provided, will use the default OOV file.",
    )
    # —— STL 相关参数 ——
    parser.add_argument(
        "--stl_enabled", action="store_true", help="Enable STL decomposition"
    )
    parser.add_argument(
        "--stl_period", type=int, default=24, help="STL 的季节周期（小时）"
    )
    parser.add_argument("--stl_robust", action="store_true", help="使用鲁棒 STL")
    parser.add_argument(
        "--stl_workers", type=int, default=0, help="STL 并行 worker 数（0=自动）"
    )
    parser.add_argument(
        "--stl_cache_dir",
        type=str,
        default="",
        help="STL 分解结果缓存目录（LMDB）。为空则不启用。",
    )
    parser.add_argument(
        "--stl_cache_map_gb",
        type=int,
        default=64,
        help="LMDB map size（GB）。默认 64GB。",
    )
    parser.add_argument(
        "--stl_cache_disable",
        action="store_true",
        help="禁用 STL 缓存（即使目录存在也不读写）。",
    )

    args = parser.parse_args()
    utils.set_seed(args.seed)

    config_path = SCRIPT_PATH / ".." / "buildings_bench" / "configs"
    if (config_path / f"{args.model}.toml").exists():
        toml_args = tomli.load((config_path / f"{args.model}.toml").open("rb"))
        model_args = toml_args["model"]
        if "zero_shot" in toml_args:
            for k, v in toml_args["zero_shot"].items():
                if k != "weather":
                    setattr(args, k, v)
                elif v != "False":
                    setattr(args, k, True)
        if not model_args["continuous_loads"] or "apply_scaler_transform" not in args:
            setattr(args, "apply_scaler_transform", "")
        if "weather_inputs" not in model_args:
            model_args["weather_inputs"] = None
    else:
        raise ValueError(f"Config {args.model}.toml not found.")

    results_path = Path(args.results_path)
    if args.include_outliers:
        results_path = results_path / "buildingsbench_with_outliers"

    results_path.mkdir(parents=True, exist_ok=True)

    # 打印所有参数
    print("Arguments:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")

    zero_shot_learning(args, model_args, results_path)
