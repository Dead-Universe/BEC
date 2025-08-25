# buildings_bench.models
import torch
from typing import Callable, Tuple, Dict

# Import models here
from buildings_bench.models.dlinear_regression import DLinearRegression
from buildings_bench.models.linear_regression import LinearRegression
from buildings_bench.models.self_transformers import LoadForecastingTransformer
from buildings_bench.models.persistence import (
    AveragePersistence,
    CopyLastDayPersistence,
    CopyLastWeekPersistence,
)
from buildings_bench.models.deep_rnn import DeepAutoregressiveRNN
from buildings_bench.models.transformer_moes_update_20 import (
    LoadForecastingTransformerMoE,
)
from buildings_bench.models.sub_models import TimeSeriesTransformer
from buildings_bench.models.patchtst import LoadForecastingPatchTST_A
from buildings_bench.models.timemoe import TimeMoeForecasting

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
    "DeepAutoregressiveRNN": DeepAutoregressiveRNN,
    # Register your model here
    "TransformerWithGaussianAndMoEs-M": LoadForecastingTransformerMoE,
    "TransformerWithGaussianAndMoEs-L": LoadForecastingTransformerMoE,
    "TransformerWithGaussianAndMoEs-S": LoadForecastingTransformerMoE,
    "TimeSeriesTransformer-S": TimeSeriesTransformer,
    "PatchTST-S": LoadForecastingPatchTST_A,
    "TimeMoE-S": TimeMoeForecasting,
    "TimeMoE-L": TimeMoeForecasting,
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
