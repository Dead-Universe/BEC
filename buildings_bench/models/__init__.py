# buildings_bench.models
from typing import Callable, Dict, Tuple

import torch
from buildings_bench.models.autoformer import AutoformerUnified, AutoformerBB
from buildings_bench.models.chronos import ChronosAsLoadForecastAdapter
from buildings_bench.models.deep_rnn import DeepAutoregressiveRNN

# Import models here
from buildings_bench.models.dlinear_regression import DLinearRegression, DLinearUnified
from buildings_bench.models.linear_regression import LinearRegression
from buildings_bench.models.moment import MomentAsLoadForecastAdapter
from buildings_bench.models.nliner import NLinearUnified, NLinearRegression
from buildings_bench.models.patchtst import LoadForecastingPatchTST_A, PatchTSTBB
from buildings_bench.models.persistence import (
    AveragePersistence,
    CopyLastDayPersistence,
    CopyLastWeekPersistence,
)
from buildings_bench.models.self_transformers import LoadForecastingTransformer
from buildings_bench.models.sub_models import TimeSeriesTransformer
from buildings_bench.models.timemoe import TimeMoeForecasting
from buildings_bench.models.transformer_moes_update_20 import (
    LoadForecastingTransformerMoE,
)
from buildings_bench.models.buildmoe import BuildMoE
from buildings_bench.models.buildmoe_new import NewBuildMoE
from buildings_bench.models.buildmoe_ar import BuildMoE as BuildMoEAR
from buildings_bench.models.buildmoe_patch import BuildMoEBatch
from buildings_bench.models.seasonal_naive import SeasonalNaive24, SeasonalNaive168
from buildings_bench.models.nhits import NHITSBB
from buildings_bench.models.time_mixer import TimeMixerBB, LoadForecastingTimeMixer_A

model_registry = {
    "TransformerWithTokenizer-L": LoadForecastingTransformer,
    "TransformerWithTokenizer-M": LoadForecastingTransformer,
    "TransformerWithTokenizer-S": LoadForecastingTransformer,
    "TransformerWithTokenizer-L-ignore-spatial": LoadForecastingTransformer,
    "TransformerWithTokenizer-L-8192": LoadForecastingTransformer,
    "TransformerWithTokenizer-L-474": LoadForecastingTransformer,
    "TransformerWithMSE": LoadForecastingTransformer,
    "TransformerWithGaussian-L": LoadForecastingTransformer,
    "TransformerWithGaussian-M": LoadForecastingTransformer,
    "TransformerWithGaussian-S": LoadForecastingTransformer,
    "TransformerWithGaussian-weather-S": LoadForecastingTransformer,
    "TransformerWithGaussian-weather-M": LoadForecastingTransformer,
    "TransformerWithGaussian-weather-L": LoadForecastingTransformer,
    "TransformerWithGaussian-th-S": LoadForecastingTransformer,
    "TransformerWithGaussian-th-M": LoadForecastingTransformer,
    "TransformerWithGaussian-th-L": LoadForecastingTransformer,
    "TransformerWithGaussian-t-L": LoadForecastingTransformer,
    "TransformerWithGaussian-t-M": LoadForecastingTransformer,
    "TransformerWithGaussian-t-S": LoadForecastingTransformer,
    "AveragePersistence": AveragePersistence,
    "CopyLastDayPersistence": CopyLastDayPersistence,
    "CopyLastWeekPersistence": CopyLastWeekPersistence,
    "LinearRegression": LinearRegression,
    "DLinearRegression": DLinearRegression,
    "DLinearUnified": DLinearUnified,
    "DeepAutoregressiveRNN": DeepAutoregressiveRNN,
    # Register your model here
    "TransformerWithGaussianAndMoEs-M": LoadForecastingTransformerMoE,
    "TransformerWithGaussianAndMoEs-L": LoadForecastingTransformerMoE,
    "TransformerWithGaussianAndMoEs-S": LoadForecastingTransformerMoE,
    "Decoder-L": LoadForecastingTransformerMoE,
    "Encoder-L": LoadForecastingTransformerMoE,
    "Transformer-L": LoadForecastingTransformerMoE,
    "BuildMoE": LoadForecastingTransformerMoE,
    "BuildMoE-top-k-1": BuildMoE,
    "BuildMoE-top-k-2": BuildMoE,
    "BuildMoE-top-k-1-without-shared-export": BuildMoE,
    "BuildMoE-top-k-2-not-use-loopback": BuildMoE,
    "BuildMoE-top-k-2-without-shared-export": BuildMoE,
    "BuildMoE-top-k-4-without-shared-export": BuildMoE,
    "BuildMoE-top-k-2-decoder-not-use-loopback": BuildMoE,
    "BuildMoE-top-k-2-without-shared-export-expert-4": BuildMoE,
    "BuildMoE-top-k-2-without-shared-export-expert-12": BuildMoE,
    "BuildMoE-top-k-2-without-shared-export-expert-12-not-use-loopback": BuildMoE,
    "BuildMoE-top-k-4": BuildMoE,
    "BuildMoE-top-k-2-decoder": BuildMoE,
    "BuildMoE-top-k-2-decoder-headwise-gate": BuildMoE,
    "BuildMoE-top-k-2-decoder-elementwise-gate": BuildMoE,
    "BuildMoE-top-k-2-decoder-elementwise-gate-revin": BuildMoE,
    "BuildMoE-top-k-2-decoder-update": NewBuildMoE,
    "BuildMoE-ar": BuildMoEAR,
    "BuildMoE-batch": BuildMoEBatch,
    "BuildMoE-dense": BuildMoE,
    "TimeSeriesTransformer-S": TimeSeriesTransformer,
    "PatchTST-S": LoadForecastingPatchTST_A,
    "PatchTST-L": LoadForecastingPatchTST_A,
    "TimeMixer-L": LoadForecastingTimeMixer_A,
    "PatchTSTBB": PatchTSTBB,
    "TimeMoE-S": TimeMoeForecasting,
    "TimeMoE-L": TimeMoeForecasting,
    "TimeMoE-200M": TimeMoeForecasting,
    "Chronos-L": ChronosAsLoadForecastAdapter,
    "Chronos-Base": ChronosAsLoadForecastAdapter,
    "Moment-L": MomentAsLoadForecastAdapter,
    "Autoformer-L": AutoformerUnified,
    "AutoformerBB": AutoformerBB,
    "NLinear": NLinearUnified,
    "NLinearRegression": NLinearRegression,
    "SeasonalNaive24": SeasonalNaive24,
    "SeasonalNaive168": SeasonalNaive168,
    "NHITSBB": NHITSBB,
    "TimeMixerBB": TimeMixerBB,
}


def model_factory(
    model_name: str, model_args: Dict
) -> Tuple[torch.nn.Module, Callable, Callable]:
    """Instantiate and returns a model for the benchmark.

    Returns the model itself,
    the loss function to use, and the predict function.

    The predict function should return a tuple of two tensors:
    (point predictions, prediction distribution parameters) where
    the distribution parameters may be, e.g., logits, or mean and variance.

    Args:
        model_name (str): Name of the model.
        model_args (Dict): The keyword arguments for the model.
    Returns:
        model (torch.nn.Module): the instantiated model
        loss (Callable): loss function
        predict (Callable): predict function
    """
    assert (
        model_name in model_registry.keys()
    ), f"Model {model_name} not in registry: {model_registry.keys()}"

    model = model_registry[model_name](**model_args)
    loss = model.loss
    predict = model.predict
    return model, loss, predict
