import time
import numpy as np
import torch
from pathlib import Path
import argparse
import os
import tomli

from buildings_bench import load_torch_dataset, benchmark_registry
from buildings_bench import utils
from buildings_bench.tokenizer import LoadQuantizer
from buildings_bench.evaluation.managers import BuildingTypes, DatasetMetricsManager
from buildings_bench.evaluation import aggregate
from buildings_bench.models import model_factory
from buildings_bench.evaluation import scoring_rule_factory

SCRIPT_PATH = Path(os.path.realpath(__file__)).parent


FM_MODELS = {
    "timesfm": "timesfm",
    "MOMENT": "moment",
    "TimeMoE": "timemoe",
    "chronos": "chronos",
}


@torch.no_grad()
def zero_shot_learning(args, model_args, results_path: Path):
    device = args.device

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
        model.eval()
    else:

        freq = getattr(args, "freq", "H")
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
            # fm = MOMENTPipeline.from_pretrained(
            #     args.model,
            #     model_kwargs={
            #         "task_name": "forecasting",
            #         "forecast_horizon": forecast_horizon,
            #     },
            # )
            # fm.init()
            pass
        elif kind == "timemoe":
            # fm = TimeMoEForecaster(model_path=args.model)
            pass
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

    if args.ignore_scoring_rules:
        metrics_manager = DatasetMetricsManager()
    elif is_fm or model.continuous_loads:
        metrics_manager = DatasetMetricsManager(
            scoring_rule=scoring_rule_factory("crps") if not is_fm else None
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
            apply_scaler_transform=args.apply_scaler_transform,
            scaler_transform_path=transform_path,
            include_outliers=args.include_outliers,
            weather_inputs=model_args["weather_inputs"],
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

            if not is_fm and not model.continuous_loads:  # Quantized loads
                transform = load_transform.transform
                inverse_transform = load_transform.undo_transform
            elif args.apply_scaler_transform != "":  # Scaling continuous values
                transform = lambda x: x

                if isinstance(building_dataset, torch.utils.data.ConcatDataset):
                    load_transform = building_dataset.datasets[0].load_transform
                    inverse_transform = load_transform.undo_transform
                else:
                    load_transform = building_dataset.load_transform
                    inverse_transform = load_transform.undo_transform
            else:  # Continuous unscaled values
                transform = lambda x: x
                inverse_transform = lambda x: x

            # create a dataloader for the building
            building_dataloader = torch.utils.data.DataLoader(
                building_dataset, batch_size=args.batch_size, shuffle=False
            )
            for batch in building_dataloader:

                if kind == "timesfm":
                    full_len = context_len + forecast_horizon
                    # A) pull out your 168+24 window
                    arr = batch["load"][..., 0].cpu().numpy()  # (B, 192)
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
                    raw_forecast, _ = fm.forecast_with_covariates(
                        inputs=X_ctx.tolist(),
                        dynamic_numerical_covariates=dynamic_num,
                        dynamic_categorical_covariates=dynamic_cat,
                        static_numerical_covariates=static_num,
                        static_categorical_covariates=static_cat,
                        freq=[0] * X_ctx.shape[0],
                        # explicitly tell it your window is 168+24,
                        # but let it pad/truncate internally to the model’s patch multiple.
                        # forecast_context_len=context_len,
                        # window_size=full_len,
                        xreg_mode="xreg + timesfm",
                        normalize_xreg_target_per_input=True,
                        ridge=0.0,
                        force_on_cpu=("cpu" in device),
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
                    pass
                elif kind == "timemoe":
                    pass
                elif kind == "chronos":
                    # A) 从 batch 中取出原始负荷序列 (B, T, 1) → (B, T)
                    arr = batch["load"][..., 0].cpu().numpy()

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
                    continuous_targets = continuous_load[:, model.context_len :]

                    # Transform if needed
                    batch["load"] = transform(batch["load"])
                    # These could be tokens or continuous
                    targets = batch["load"][:, model.context_len :]

                    if args.device == "cuda":
                        with torch.amp.autocast("cuda"):
                            predictions, distribution_params = predict(batch)
                    else:
                        predictions, distribution_params = predict(batch)

                    predictions = inverse_transform(predictions)

                    if args.apply_scaler_transform != "":
                        continuous_targets = inverse_transform(continuous_targets)
                        # invert for crps
                        targets = inverse_transform(targets)
                        if args.apply_scaler_transform == "standard":
                            mu = inverse_transform(distribution_params[:, :, 0])
                            sigma = load_transform.undo_transform_std(
                                distribution_params[:, :, 1]
                            )
                            distribution_params = torch.cat(
                                [mu.unsqueeze(-1), sigma.unsqueeze(-1)], -1
                            )

                        elif args.apply_scaler_transform == "boxcox":
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

                    if not model.continuous_loads:
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

    if not args.ignore_scoring_rules:
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
            results_dict = aggregate.return_aggregate_median(
                model_list=[f"{args.model}{variant_name}"],
                results_dir=str(results_path),
                experiment="zero_shot",
                metrics=metric_names,
                exclude_simulated=True,
                oov_list=oov_bldgs,
            )
            aggregate.pretty_print_aggregates(results_dict)

        if "buildings-900k-test" in args.benchmark:
            print("Buildings-900K-test (synth)")
            results_dict = aggregate.return_aggregate_median(
                model_list=[f"{args.model}{variant_name}"],
                results_dir=str(results_path),
                experiment="zero_shot",
                metrics=metric_names,
                exclude_simulated=False,
                only_simulated=True,
                oov_list=oov_bldgs,
            )
            aggregate.pretty_print_aggregates(results_dict)


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

    zero_shot_learning(args, model_args, results_path)
