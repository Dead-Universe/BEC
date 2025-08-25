import datetime
import torch
import torch.nn as nn
import torch.distributed
import torch.utils.data.distributed
import torch.amp
from pathlib import Path
import transformers
import argparse
import wandb
import swanlab
import os
import tomli
from timeit import default_timer as timer
from socket import gethostname
from buildings_bench import utils
from buildings_bench import BuildingTypes
from buildings_bench import load_pretraining
from buildings_bench.tokenizer import LoadQuantizer
from buildings_bench.evaluation.managers import MetricsManager
from buildings_bench.models import model_factory
from buildings_bench.evaluation.metrics import MetricType
from buildings_bench.evaluation import metrics_factory
from buildings_bench.evaluation import scoring_rule_factory
from buildings_bench.data.new import build_datasets, worker_init_fn
from buildings_bench.transforms import BoxCoxTransform, StandardScalerTransform

DATASET_FULL = True

SCRIPT_PATH = Path(os.path.realpath(__file__)).parent


@torch.no_grad()
def validation(
    model,
    val_dataloader,
    args,
    loss,
    load_transform,
    transform,
    inverse_transform,
    predict,
):
    model.eval()
    step = 0

    if args.ignore_scoring_rules:
        metrics_manager = MetricsManager(
            metrics=metrics_factory(
                "nrmse", types=[MetricType.SCALAR, MetricType.HOUR_OF_DAY]
            )
            + metrics_factory("nmae", types=[MetricType.SCALAR, MetricType.HOUR_OF_DAY])
        )
    elif model.module.continuous_head == "mse":
        metrics_manager = MetricsManager(
            metrics=metrics_factory(
                "nrmse", types=[MetricType.SCALAR, MetricType.HOUR_OF_DAY]
            )
            + metrics_factory(
                "nmae", types=[MetricType.SCALAR, MetricType.HOUR_OF_DAY]
            ),
        )
    elif model.module.continuous_loads:
        metrics_manager = MetricsManager(
            metrics=metrics_factory(
                "nrmse", types=[MetricType.SCALAR, MetricType.HOUR_OF_DAY]
            )
            + metrics_factory(
                "nmae", types=[MetricType.SCALAR, MetricType.HOUR_OF_DAY]
            ),
            scoring_rule=scoring_rule_factory("crps"),
        )
    else:
        metrics_manager = MetricsManager(
            metrics=metrics_factory(
                "nrmse", types=[MetricType.SCALAR, MetricType.HOUR_OF_DAY]
            )
            + metrics_factory(
                "nmae", types=[MetricType.SCALAR, MetricType.HOUR_OF_DAY]
            ),
            scoring_rule=scoring_rule_factory("rps"),
        )
    for batch in val_dataloader:
        building_types_mask = batch["building_type"][:, 0, 0] == 1

        for k, v in batch.items():
            batch[k] = v.to(model.device)

        continuous_load = batch["load"].clone()
        continuous_targets = continuous_load[:, model.module.context_len :]

        # Transform if needed
        batch["load"] = transform(batch["load"])
        targets = batch["load"][:, model.module.context_len :]

        with torch.amp.autocast("cuda"):
            preds = model(batch)
            batch_loss = loss(preds, targets)
            predictions, distribution_params = predict(batch)
            if model.module.continuous_head == "mse":
                distribution_params = None

        predictions = inverse_transform(predictions)

        if args.apply_scaler_transform != "":
            continuous_targets = inverse_transform(continuous_targets)
            # unscale for crps
            targets = inverse_transform(targets)
            if (
                args.apply_scaler_transform == "standard"
                and distribution_params is not None
            ):
                mu = inverse_transform(distribution_params[:, :, 0])
                sigma = load_transform.undo_transform_std(distribution_params[:, :, 1])
                distribution_params = torch.cat(
                    [mu.unsqueeze(-1), sigma.unsqueeze(-1)], -1
                )

            elif (
                args.apply_scaler_transform == "boxcox"
                and distribution_params is not None
            ):
                ######## approximate Gaussian in unscaled space ########
                mu = inverse_transform(distribution_params[:, :, 0])
                muplussigma = inverse_transform(torch.sum(distribution_params, -1))
                sigma = muplussigma - mu
                muminussigma = inverse_transform(
                    distribution_params[:, :, 0] - distribution_params[:, :, 1]
                )
                sigma = (sigma + (mu - muminussigma)) / 2
                distribution_params = torch.cat(
                    [mu.unsqueeze(-1), sigma.unsqueeze(-1)], -1
                )

        if not model.module.continuous_loads:
            centroids = (
                load_transform.kmeans.centroids.squeeze()
                if args.tokenizer_without_merge
                else load_transform.merged_centroids
            )
        else:
            centroids = None

        metrics_manager(
            continuous_targets,
            predictions,
            building_types_mask,
            loss=batch_loss,
            y_categories=targets,
            y_distribution_params=distribution_params,
            centroids=centroids,
        )

        step += 1
        # don't run for too long
        if step == 500:
            break

    model.train()
    summary = metrics_manager.summary(with_loss=True, with_ppl=True)

    return summary["loss"], summary["ppl"], summary


def main(args, model_args):
    """Main training loop

    Args:
        args (argparse.Namespace): Command line arguments
        model_args (dict): Model arguments
    """

    utils.set_seed(args.random_seed)

    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    # Optimize for fixed input sizes
    torch.backends.cudnn.benchmark = False

    ######################### DDP setup  #########################
    # SLURM_LOCALID: gpu local rank (=0 as the first gpu of the node)
    # SLURM_PROCID: gpu global rank (=4 as the fifth gpu among the 8)
    # MASTER_ADDR and MASTER_PORT env variables should be set when calling this script
    gpus_per_node = torch.cuda.device_count()
    args.world_size = int(os.environ["WORLD_SIZE"])
    if args.disable_slurm:
        local_rank = int(os.environ["LOCAL_RANK"])
        args.rank = local_rank
    else:
        args.rank = int(os.environ["SLURM_PROCID"])
        print(
            f"Hello from rank {args.rank} of {args.world_size} on {gethostname()} where there are"
            f" {gpus_per_node} allocated GPUs per node.",
            flush=True,
        )

        local_rank = args.rank - gpus_per_node * (args.rank // gpus_per_node)

    print(
        f"About to call init_process_group on rank {args.rank} with local rank {local_rank}",
        flush=True,
    )
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        timeout=datetime.timedelta(seconds=3600),
        rank=args.rank,
    )
    if args.rank == 0:
        print(f"Group initialized? {torch.distributed.is_initialized()}", flush=True)
    torch.cuda.set_device(local_rank)

    print(
        f"rank {args.rank} torch cuda available = ",
        torch.cuda.is_available(),
        flush=True,
    )
    print(
        f"rank {args.rank} torch cuda device count = ",
        torch.cuda.device_count(),
        flush=True,
    )
    print(
        f"rank {args.rank} torch cuda current device = ",
        torch.cuda.current_device(),
        flush=True,
    )
    print(
        f"rank {args.rank} torch cuda get_device_name = ",
        torch.cuda.get_device_name(0),
        flush=True,
    )
    print(f"rank {args.rank} torch threads = ", torch.get_num_threads(), flush=True)

    print(f'dataset path = {os.environ.get("BUILDINGS_BENCH", "")}')

    checkpoint_dir = SCRIPT_PATH / ".." / "checkpoints"
    checkpoint_name = f"{experiment_args.model}"
    if args.note != "":
        checkpoint_name += f"_{args.note}"

    transform_path = (
        Path(os.environ.get("BUILDINGS_BENCH", "")) / "metadata" / "transforms"
    )

    if args.rank == 0:
        if not checkpoint_dir.exists():
            os.makedirs(checkpoint_dir)

        wandb_project = os.environ.get("WANDB_PROJECT", "")
        if wandb_project == "":
            print("WANDB_PROJECT environment variable not set, disabling wandb")
            args.disable_wandb = True
        config_dict = vars(args).copy()
        config_dict["model_args"] = model_args
        if args.disable_wandb:
            run = wandb.init(project=wandb_project, mode="disabled", config=args)
        elif args.resume_from_checkpoint != "":
            # run = wandb.init(
            #     id=args.wandb_run_id,
            #     project=wandb_project,
            #     notes=args.note,
            #     resume="allow",
            #     config=args,
            # )
            run_swanlab = swanlab.init(
                project=wandb_project,
                experiment_name=args.name,
                tags=[args.model, "pretraining"],
                config=config_dict,
            )
            pass
        else:
            # run = wandb.init(project=wandb_project, notes=args.note, config=args)
            run_swanlab = swanlab.init(
                project=wandb_project,
                experiment_name=args.name,
                tags=[args.model, "pretraining"],
                config=config_dict,
            )

    global_batch_size = args.world_size * args.batch_size

    #################### Model setup ####################

    model, loss, predict = model_factory(args.model, model_args)
    model = model.to(local_rank)
    print(
        f"rank {args.rank} number of trainable parameters is "
        f"= {sum(p.numel() for p in model.parameters())}",
        flush=True,
    )

    #################### Dataset setup ####################

    if DATASET_FULL == True:
        train_dataset, val_dataset = build_datasets()  # 返回 ConcatDataset

        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset=train_dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True,
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            drop_last=False,
            worker_init_fn=worker_init_fn,
            collate_fn=torch.utils.data.default_collate,
            shuffle=(train_sampler is None),
            num_workers=args.num_workers,
            pin_memory=True,
        )

        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            # sampler=val_sampler,
            drop_last=False,
            worker_init_fn=worker_init_fn,
            collate_fn=torch.utils.data.default_collate,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        if args.apply_scaler_transform == "boxcox":
            load_transform = BoxCoxTransform()
        elif args.apply_scaler_transform == "standard":
            load_transform = StandardScalerTransform()

        transform_path = (
            Path(os.environ.get("BUILDINGS_BENCH", "")) / "metadata" / "transforms"
        )

        load_transform.load(transform_path)

        if not model.continuous_loads:
            load_transform = LoadQuantizer(
                with_merge=(not args.tokenizer_without_merge),
                num_centroids=model.vocab_size,
                device=f"cuda:{local_rank}",
            )
            load_transform.load(transform_path)
        else:
            load_transform = load_transform
    else:

        train_dataset = load_pretraining(
            "buildings-900k-train",
            args.num_buildings,
            args.apply_scaler_transform,
            transform_path,
            weather_inputs=model_args["weather_inputs"],
            custom_idx_filename=args.train_idx_filename,
        )

        val_dataset = load_pretraining(
            "buildings-900k-val",
            args.num_buildings,
            args.apply_scaler_transform,
            transform_path,
            weather_inputs=model_args["weather_inputs"],
            custom_idx_filename=args.val_idx_filename,
        )

        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset=train_dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True,
        )
        # val_sampler = torch.utils.data.distributed.DistributedSampler(
        #                                  dataset=val_dataset,
        #                                  num_replicas=args.world_size,
        #                                  rank=args.rank, shuffle=False)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            drop_last=False,
            worker_init_fn=utils.worker_init_fn_eulp,
            collate_fn=train_dataset.collate_fn(),
            shuffle=(train_sampler is None),
            num_workers=args.num_workers,
            pin_memory=True,
        )

        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            # sampler=val_sampler,
            drop_last=False,
            worker_init_fn=utils.worker_init_fn_eulp,
            collate_fn=val_dataset.collate_fn(),
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        if not model.continuous_loads:
            load_transform = LoadQuantizer(
                with_merge=(not args.tokenizer_without_merge),
                num_centroids=model.vocab_size,
                device=f"cuda:{local_rank}",
            )
            load_transform.load(transform_path)
        else:
            load_transform = train_dataset.load_transform

    if not model.continuous_loads:
        transform = load_transform.transform
        inverse_transform = load_transform.undo_transform
    elif args.apply_scaler_transform != "":
        transform = lambda x: x
        inverse_transform = load_transform.undo_transform
    else:  # Continuous unscaled values
        transform = lambda x: x
        inverse_transform = lambda x: x

    #################### Optimizer setup ##########################

    # wrap model with DistributedDataParallel
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,  # ← 关键开关
    )

    print(f"rank {args.rank} wrapped model in DDP", flush=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01
    )
    train_steps = args.train_tokens // (global_batch_size * model.module.pred_len)

    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.02 * train_steps),
        num_training_steps=train_steps,
    )

    scaler = torch.amp.GradScaler()

    best_val_loss = float("inf")
    best_train_loss = float("inf")

    #################### Resume from checkpoint ####################

    if args.resume_from_checkpoint != "":
        model, optimizer, scheduler, step, best_val_loss = utils.load_model_checkpoint(
            checkpoint_dir / args.resume_from_checkpoint,
            model,
            optimizer,
            scheduler,
            local_rank,
        )
        seen_tokens = step * global_batch_size * model.module.pred_len
    else:
        step = 0
        seen_tokens = 0
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=args.init_scale)

    #################### Training loop ##############################

    print(f"rank {args.rank} step {step} train_steps = {train_steps}", flush=True)

    # fix sampling seed such that each gpu gets different part of dataset
    train_sampler.set_epoch(0)
    # val_sampler.set_epoch(0)
    model.train()
    start_time = timer()

    for batch in train_dataloader:
        start_time = timer()
        optimizer.zero_grad()

        for k, v in batch.items():
            batch[k] = v.to(model.device)

        # Apply transform to load if needed
        batch["load"] = transform(batch["load"])

        # backwards is called in here
        with torch.amp.autocast("cuda"):
            preds = model(batch)
            targets = batch["load"][:, model.module.context_len :]
            # preds are [bsz_sz, pred_len, vocab_size] if logits
            # preds are [bsz_sz, pred_len, 2] if Gaussian
            # preds are [bsz_sz, pred_len, 1] if MSE
            # targets is [bsz_sz, pred_len, 1]
            batch_loss = loss(preds, targets)

        # Scale Gradients
        scaler.scale(batch_loss).backward()

        # Update Optimizer
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        end_time = timer()

        secs_per_step = end_time - start_time
        # world_size * batch_size = global batch size with DDP training
        seen_tokens += global_batch_size * model.module.pred_len
        step += 1

        ppl = torch.exp(batch_loss.detach())

        if args.rank == 0:
            if best_train_loss > batch_loss.item():
                best_train_loss = batch_loss.item()
                model_name = checkpoint_name + "_best_train.pt"
                utils.save_model_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    step,
                    best_train_loss,
                    checkpoint_dir / model_name,
                )
                print(
                    f"rank {args.rank} step {step} saved best train model to {model_name}",
                    flush=True,
                )

        if args.rank == 0 and step % 500 == 0:
            # wandb.log(
            #     {
            #         "train/loss": batch_loss,
            #         "train/batch_ppl": ppl,
            #         "train/seen_tokens (M)": seen_tokens / 1000000,
            #         "train/secs_per_step": secs_per_step,
            #         "train/lr": optimizer.param_groups[0]["lr"],
            #     },
            #     step=step,
            # )
            swanlab.log(
                {
                    "train/loss": batch_loss.item(),
                    "train/batch_ppl": ppl.item(),
                    "train/seen_tokens (M)": seen_tokens / 1000000,
                    "train/secs_per_step": secs_per_step,
                    "train/lr": optimizer.param_groups[0]["lr"],
                },
                step=step,
            )
            if args.model.startswith("_TransformerWithGaussianAndMoEs"):
                # wandb.log(
                #     {
                #         "train/aux_loss": model.module.encoder.layers[
                #             0
                #         ].moe_ffn.loss_coef
                #     },
                #     step=step,
                # )
                # swanlab.log(
                #     {
                #         "train/aux_loss": model.module.encoder.layers[
                #             0
                #         ].moe_ffn.loss_coef
                #     },
                #     step=step,
                # )
                # router_freq_list = (
                #     model.module.encoder.layers[0]
                #     .moe_ffn.router_freq.detach()
                #     .cpu()
                #     .tolist()
                # )

                # # 构造一个字典，每个 value 都是 float
                # metrics = {
                #     f"train/router_freq_{i}": float(freq)
                #     for i, freq in enumerate(router_freq_list)
                # }

                # # 一次性记录所有子指标
                # swanlab.log(metrics, step=step)
                group_freq_list = (
                    model.module.encoder.layers[0]
                    .moe_ffn.group_freq.detach()
                    .cpu()
                    .tolist()
                )

                # 构造一个字典，每个 value 都是 float
                metrics = {
                    f"train/group_freq_{i}": float(freq)
                    for i, freq in enumerate(group_freq_list)
                }

                # 一次性记录所有子指标
                swanlab.log(metrics, step=step)

                expert_freq_list = (
                    model.module.encoder.layers[0]
                    .moe_ffn.expert_freq.detach()
                    .cpu()
                    .tolist()
                )

                # 构造一个字典，每个 value 都是 float
                metrics = {
                    f"train/expert_freq_{i}": float(freq)
                    for i, freq in enumerate(expert_freq_list)
                }

                # 一次性记录所有子指标
                swanlab.log(metrics, step=step)

        # if args.rank == 0 and step % 1000 == 0:
        #     print(
        #         f"rank {args.rank} step {step} loss {batch_loss:.5f} "
        #         f"ppl {ppl:.5f} seen_tokens {seen_tokens / 1000000:.2f}M "
        #         f"in {(secs_per_step):.2f} seconds",
        #         flush=True,
        #     )

        # if args.rank == 0 and step % 4000 == 0:
        #     start_time = timer()
        #     print(f"started validation at step {step}...")

        #     val_loss, val_ppl, val_metrics = validation(
        #         model,
        #         val_dataloader,
        #         args,
        #         loss,
        #         load_transform,
        #         transform,
        #         inverse_transform,
        #         predict,
        #     )

        #     end_time = timer()
        #     secs_per_step = end_time - start_time
        #     print(
        #         f"finished validation at step {step} with loss {val_loss:.5f} "
        #         f"and ppl {val_ppl:.5f} in {(secs_per_step/60):.2f} minutes",
        #         flush=True,
        #     )
        #     # only rank 0 needs to save model
        #     if val_loss < best_val_loss:

        #         best_val_loss = val_loss

        #         model_name = checkpoint_name + "_best.pt"

        #         # delete previous
        #         if (checkpoint_dir / model_name).exists():
        #             Path(checkpoint_dir / model_name).unlink()
        #         # save best
        #         utils.save_model_checkpoint(
        #             model,
        #             optimizer,
        #             scheduler,
        #             step,
        #             best_val_loss,
        #             checkpoint_dir / model_name,
        #         )

        #     # we always save the last val model
        #     last_model_name = checkpoint_name + "_last.pt"
        #     utils.save_model_checkpoint(
        #         model,
        #         optimizer,
        #         scheduler,
        #         step,
        #         best_val_loss,
        #         checkpoint_dir / last_model_name,
        #     )

        #     for building_type in [BuildingTypes.RESIDENTIAL, BuildingTypes.COMMERCIAL]:
        #         for metric_name, metric_result in val_metrics[building_type].items():
        #             if metric_result.type == MetricType.SCALAR:
        #                 # wandb.log(
        #                 #     {f"val/{building_type}/{metric_name}": metric_result.value},
        #                 #     step=step,
        #                 # )
        #                 swanlab.log(
        #                     {f"val/{building_type}/{metric_name}": metric_result.value},
        #                     step=step,
        #                 )
        #             else:
        #                 multi_hour_value = metric_result.value
        #                 # Create a wandb.Table for each hour of day metric then plot a line plot
        #                 # table = wandb.Table(columns=["time (hour)", metric_name])

        #                 # for row_idx in range(multi_hour_value.shape[0]):
        #                 #     table.add_data(row_idx, multi_hour_value[row_idx].item())
        #                 # wandb.log(
        #                 #     {
        #                 #         f"val/{building_type}/{metric_name}": wandb.plot.line(
        #                 #             table,
        #                 #             "time (hour)",
        #                 #             metric_name,
        #                 #             title=f"Time vs {metric_name}",
        #                 #         )
        #                 #     },
        #                 #     step=step,
        #                 # )

        #                 hours = list(
        #                     range(multi_hour_value.shape[0])
        #                 )  # 生成小时序列 [0, 1, ..., N-1]
        #                 values = [x.item() for x in multi_hour_value]  # 提取指标值列表

        #                 from pyecharts import options as opts

        #                 # 创建 pyecharts 折线图对象
        #                 line = swanlab.echarts.Line()
        #                 line.add_xaxis(hours)  # X轴：时间（小时）
        #                 line.add_yaxis(metric_name, values)  # Y轴：指标值

        #                 # 设置图表标题和坐标轴标签
        #                 line.set_global_opts(
        #                     title_opts=opts.TitleOpts(title=f"Time vs {metric_name}"),
        #                     xaxis_opts=opts.AxisOpts(name="time (hour)"),
        #                     yaxis_opts=opts.AxisOpts(name=metric_name),
        #                 )

        #                 # 记录到 SwanLab（指定分类和步骤）
        #                 swanlab.log(
        #                     {f"val/{building_type}/{metric_name}": line}, step=step
        #                 )

        #     # wandb.log(
        #     #     {
        #     #         "val/loss": val_loss,
        #     #         "val/ppl": val_ppl,
        #     #     },
        #     #     step=step,
        #     # )
        #     swanlab.log(
        #         {
        #             "val/loss": val_loss,
        #             "val/ppl": val_ppl,
        #         },
        #         step=step,
        #     )
        #     print(f"finished validation with loss {val_loss:.5f} at step {step}...")

        if step >= train_steps:
            # stop training after this many steps/train_tokens
            break

    # save final checkpoint
    if args.rank == 0:
        last_model_name = checkpoint_name + "_last.pt"
        utils.save_model_checkpoint(
            model,
            optimizer,
            scheduler,
            step,
            best_val_loss,
            checkpoint_dir / last_model_name,
        )
        # run.finish()
        run_swanlab.finish()

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Experiment args. If provided in config file, these will be overridden.
    # Use arg `hyper_opt` to avoid overriding the argparse args with the config file.
    parser.add_argument(
        "--model",
        type=str,
        default="",
        required=True,
        help="Name of your model. Should match the config"
        " filename without .toml extension."
        ' Example: "TransformerWithTokenizer-S"',
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.00006)
    parser.add_argument("--warmup_steps", type=int, default=10000)
    parser.add_argument(
        "--train_tokens",
        type=int,
        default=1000000000,  # 1B
        help="This script pretrains a model for as many steps"
        " as it takes to see `train_tokens`.",
    )
    parser.add_argument("--random_seed", type=int, default=99)
    parser.add_argument(
        "--ignore_scoring_rules",
        action="store_true",
        help="Do not compute a scoring rule for this model.",
    )
    parser.add_argument("--resume_from_checkpoint", type=str, default="")
    parser.add_argument(
        "--hyper_opt",
        nargs="*",
        default=[],
        help="Tells this script to not override the argparse values for"
        " these hyperparams with values in the config file."
        " Expects the hyperparameter value to be set via argparse "
        " from the CLI. Example: --hyper_opt batch_size lr",
    )

    # Wandb
    parser.add_argument("--disable_wandb", action="store_true")
    parser.add_argument(
        "--note",
        type=str,
        default="",
        help="Note to append to model checkpoint name. " "Also used for wandb notes.",
    )
    parser.add_argument("--wandb_run_id", type=str, default="")

    # DDP
    parser.add_argument("--disable_slurm", action="store_true")
    parser.add_argument(
        "--world-size",
        default=-1,
        type=int,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--rank", default=-1, type=int, help="node rank for distributed training"
    )
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument("--num_workers", type=int, default=8)

    # Variants
    parser.add_argument(
        "--num_buildings",
        type=int,
        default=-1,
        help="Number of buildings to use for training. "
        "Default is -1 which uses all buildings. "
        "Options {1000, 10000, 100000}.",
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
        "--train_idx_filename",
        type=str,
        default="",
        help="Name of index files for training",
    )
    parser.add_argument(
        "--val_idx_filename",
        type=str,
        default="",
        help="Name of index files for validation",
    )

    experiment_args = parser.parse_args()

    # validate hyperopt args, if any
    for arg in experiment_args.hyper_opt:
        if not hasattr(experiment_args, arg):
            raise ValueError(f"Hyperopt arg {arg} not found in argparse args.")

    config_path = SCRIPT_PATH / ".." / "buildings_bench" / "configs"

    if (config_path / f"{experiment_args.model}.toml").exists():
        toml_args = tomli.load(
            (config_path / f"{experiment_args.model}.toml").open("rb")
        )
        model_args = toml_args["model"]
        if "pretrain" in toml_args:
            for k, v in toml_args["pretrain"].items():
                if not k in experiment_args.hyper_opt:
                    if hasattr(experiment_args, k):
                        print(f"Overriding argparse default for {k} with {v}")
                    # Just set the argparse value to the value in the config file
                    # even if there is no default
                    setattr(experiment_args, k, v)
        if (
            not model_args["continuous_loads"]
            or "apply_scaler_transform" not in experiment_args
        ):
            setattr(experiment_args, "apply_scaler_transform", "")
        if "weather_inputs" not in model_args:
            model_args["weather_inputs"] = None
    else:
        raise ValueError(f"Config {experiment_args.model}.toml not found.")

    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available for pretraining!")

    main(experiment_args, model_args)
