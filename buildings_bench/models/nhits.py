# Cell
import numpy as np

import torch as t
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Dict, Optional
from functools import partial

import torch

from buildings_bench.models.base_model import BaseModel


class RepeatVector(nn.Module):
    """
    Receives x input of dim [N,C], and repeats the vector
    to create tensor of shape [N, C, K]
    : repeats: int, the number of repetitions for the vector.
    """

    def __init__(self, repeats):
        super(RepeatVector, self).__init__()
        self.repeats = repeats

    def forward(self, x):
        x = x.unsqueeze(-1).repeat(1, 1, self.repeats)  # <------------ Mejorar?
        return x


# Cell
class _StaticFeaturesEncoder(nn.Module):
    def __init__(self, in_features, out_features):
        super(_StaticFeaturesEncoder, self).__init__()
        layers = [
            nn.Dropout(p=0.5),
            nn.Linear(in_features=in_features, out_features=out_features),
            nn.ReLU(),
        ]
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)
        return x


class _sEncoder(nn.Module):
    def __init__(self, in_features, out_features, n_time_in):
        super(_sEncoder, self).__init__()
        layers = [
            nn.Dropout(p=0.5),
            nn.Linear(in_features=in_features, out_features=out_features),
            nn.ReLU(),
        ]
        self.encoder = nn.Sequential(*layers)
        self.repeat = RepeatVector(repeats=n_time_in)

    def forward(self, x):
        # Encode and repeat values to match time
        x = self.encoder(x)
        x = self.repeat(x)  # [N,S_out] -> [N,S_out,T]
        return x


# Cell
class IdentityBasis(nn.Module):
    def __init__(self, backcast_size: int, forecast_size: int, interpolation_mode: str):
        super().__init__()
        assert (interpolation_mode in ["linear", "nearest"]) or (
            "cubic" in interpolation_mode
        )
        self.forecast_size = forecast_size
        self.backcast_size = backcast_size
        self.interpolation_mode = interpolation_mode

    def forward(
        self, theta: t.Tensor, insample_x_t: t.Tensor, outsample_x_t: t.Tensor
    ) -> Tuple[t.Tensor, t.Tensor]:

        backcast = theta[:, : self.backcast_size]
        knots = theta[:, self.backcast_size :]

        if self.interpolation_mode == "nearest":
            knots = knots[:, None, :]
            forecast = F.interpolate(
                knots, size=self.forecast_size, mode=self.interpolation_mode
            )
            forecast = forecast[:, 0, :]
        elif self.interpolation_mode == "linear":
            knots = knots[:, None, :]
            forecast = F.interpolate(
                knots, size=self.forecast_size, mode=self.interpolation_mode
            )  # , align_corners=True)
            forecast = forecast[:, 0, :]
        elif "cubic" in self.interpolation_mode:
            batch_size = int(self.interpolation_mode.split("-")[-1])
            knots = knots[:, None, None, :]
            forecast = t.zeros((len(knots), self.forecast_size)).to(knots.device)
            n_batches = int(np.ceil(len(knots) / batch_size))
            for i in range(n_batches):
                forecast_i = F.interpolate(
                    knots[i * batch_size : (i + 1) * batch_size],
                    size=self.forecast_size,
                    mode="bicubic",
                )  # , align_corners=True)
                forecast[i * batch_size : (i + 1) * batch_size] += forecast_i[
                    :, 0, 0, :
                ]

        return backcast, forecast


# Cell
def init_weights(module, initialization):
    if type(module) == t.nn.Linear:
        if initialization == "orthogonal":
            t.nn.init.orthogonal_(module.weight)
        elif initialization == "he_uniform":
            t.nn.init.kaiming_uniform_(module.weight)
        elif initialization == "he_normal":
            t.nn.init.kaiming_normal_(module.weight)
        elif initialization == "glorot_uniform":
            t.nn.init.xavier_uniform_(module.weight)
        elif initialization == "glorot_normal":
            t.nn.init.xavier_normal_(module.weight)
        elif initialization == "lecun_normal":
            pass  # t.nn.init.normal_(module.weight, 0.0, std=1/np.sqrt(module.weight.numel()))
        else:
            assert 1 < 0, f"Initialization {initialization} not found"


# Cell
ACTIVATIONS = ["ReLU", "Softplus", "Tanh", "SELU", "LeakyReLU", "PReLU", "Sigmoid"]


class _NHITSBlock(nn.Module):
    """
    N-HiTS block which takes a basis function as an argument.
    """

    def __init__(
        self,
        n_time_in: int,
        n_time_out: int,
        n_x: int,
        n_s: int,
        n_s_hidden: int,
        n_theta: int,
        n_theta_hidden: list,
        n_pool_kernel_size: int,
        pooling_mode: str,
        basis: nn.Module,
        n_layers: int,
        batch_normalization: bool,
        dropout_prob: float,
        activation: str,
    ):
        """ """
        super().__init__()

        assert pooling_mode in ["max", "average"]

        n_time_in_pooled = int(np.ceil(n_time_in / n_pool_kernel_size))

        if n_s == 0:
            n_s_hidden = 0
        n_theta_hidden = [
            n_time_in_pooled + (n_time_in + n_time_out) * n_x + n_s_hidden
        ] + n_theta_hidden

        self.n_time_in = n_time_in
        self.n_time_out = n_time_out
        self.n_s = n_s
        self.n_s_hidden = n_s_hidden
        self.n_x = n_x
        self.n_pool_kernel_size = n_pool_kernel_size
        self.batch_normalization = batch_normalization
        self.dropout_prob = dropout_prob

        assert activation in ACTIVATIONS, f"{activation} is not in {ACTIVATIONS}"
        activ = getattr(nn, activation)()

        if pooling_mode == "max":
            self.pooling_layer = nn.MaxPool1d(
                kernel_size=self.n_pool_kernel_size,
                stride=self.n_pool_kernel_size,
                ceil_mode=True,
            )
        elif pooling_mode == "average":
            self.pooling_layer = nn.AvgPool1d(
                kernel_size=self.n_pool_kernel_size,
                stride=self.n_pool_kernel_size,
                ceil_mode=True,
            )

        hidden_layers = []
        for i in range(n_layers):
            hidden_layers.append(
                nn.Linear(
                    in_features=n_theta_hidden[i], out_features=n_theta_hidden[i + 1]
                )
            )
            hidden_layers.append(activ)

            if self.batch_normalization:
                hidden_layers.append(nn.BatchNorm1d(num_features=n_theta_hidden[i + 1]))

            if self.dropout_prob > 0:
                hidden_layers.append(nn.Dropout(p=self.dropout_prob))

        output_layer = [nn.Linear(in_features=n_theta_hidden[-1], out_features=n_theta)]
        layers = hidden_layers + output_layer

        # n_s is computed with data, n_s_hidden is provided by user, if 0 no statics are used
        if (self.n_s > 0) and (self.n_s_hidden > 0):
            self.static_encoder = _StaticFeaturesEncoder(
                in_features=n_s, out_features=n_s_hidden
            )
        self.layers = nn.Sequential(*layers)
        self.basis = basis

    def forward(
        self,
        insample_y: t.Tensor,
        insample_x_t: t.Tensor,
        outsample_x_t: t.Tensor,
        x_s: t.Tensor,
    ) -> Tuple[t.Tensor, t.Tensor]:

        insample_y = insample_y.unsqueeze(1)
        # Pooling layer to downsample input
        insample_y = self.pooling_layer(insample_y)
        insample_y = insample_y.squeeze(1)

        batch_size = len(insample_y)
        if self.n_x > 0:
            insample_y = t.cat((insample_y, insample_x_t.reshape(batch_size, -1)), 1)
            insample_y = t.cat((insample_y, outsample_x_t.reshape(batch_size, -1)), 1)

        # Static exogenous
        if (self.n_s > 0) and (self.n_s_hidden > 0):
            x_s = self.static_encoder(x_s)
            insample_y = t.cat((insample_y, x_s), 1)

        # Compute local projection weights and projection
        theta = self.layers(insample_y)
        backcast, forecast = self.basis(theta, insample_x_t, outsample_x_t)

        return backcast, forecast


# Cell
class _NHITS(nn.Module):
    """
    N-HiTS Model.
    """

    def __init__(
        self,
        n_time_in,
        n_time_out,
        n_s,
        n_x,
        n_s_hidden,
        n_x_hidden,
        stack_types: list,
        n_blocks: list,
        n_layers: list,
        n_theta_hidden: list,
        n_pool_kernel_size: list,
        n_freq_downsample: list,
        pooling_mode,
        interpolation_mode,
        dropout_prob_theta,
        activation,
        initialization,
        batch_normalization,
        shared_weights,
    ):
        super().__init__()

        self.n_time_out = n_time_out

        blocks = self.create_stack(
            stack_types=stack_types,
            n_blocks=n_blocks,
            n_time_in=n_time_in,
            n_time_out=n_time_out,
            n_x=n_x,
            n_x_hidden=n_x_hidden,
            n_s=n_s,
            n_s_hidden=n_s_hidden,
            n_layers=n_layers,
            n_theta_hidden=n_theta_hidden,
            n_pool_kernel_size=n_pool_kernel_size,
            n_freq_downsample=n_freq_downsample,
            pooling_mode=pooling_mode,
            interpolation_mode=interpolation_mode,
            batch_normalization=batch_normalization,
            dropout_prob_theta=dropout_prob_theta,
            activation=activation,
            shared_weights=shared_weights,
            initialization=initialization,
        )
        self.blocks = t.nn.ModuleList(blocks)

    def create_stack(
        self,
        stack_types,
        n_blocks,
        n_time_in,
        n_time_out,
        n_x,
        n_x_hidden,
        n_s,
        n_s_hidden,
        n_layers,
        n_theta_hidden,
        n_pool_kernel_size,
        n_freq_downsample,
        pooling_mode,
        interpolation_mode,
        batch_normalization,
        dropout_prob_theta,
        activation,
        shared_weights,
        initialization,
    ):

        block_list = []
        for i in range(len(stack_types)):
            # print(f'| --  Stack {stack_types[i]} (#{i})')
            for block_id in range(n_blocks[i]):

                # Batch norm only on first block
                if (len(block_list) == 0) and (batch_normalization):
                    batch_normalization_block = True
                else:
                    batch_normalization_block = False

                # Shared weights
                if shared_weights and block_id > 0:
                    nbeats_block = block_list[-1]
                else:
                    if stack_types[i] == "identity":
                        n_theta = n_time_in + max(n_time_out // n_freq_downsample[i], 1)
                        basis = IdentityBasis(
                            backcast_size=n_time_in,
                            forecast_size=n_time_out,
                            interpolation_mode=interpolation_mode,
                        )

                    else:
                        assert 1 < 0, f"Block type not found!"

                    nbeats_block = _NHITSBlock(
                        n_time_in=n_time_in,
                        n_time_out=n_time_out,
                        n_x=n_x,
                        n_s=n_s,
                        n_s_hidden=n_s_hidden,
                        n_theta=n_theta,
                        n_theta_hidden=n_theta_hidden[i],
                        n_pool_kernel_size=n_pool_kernel_size[i],
                        pooling_mode=pooling_mode,
                        basis=basis,
                        n_layers=n_layers[i],
                        batch_normalization=batch_normalization_block,
                        dropout_prob=dropout_prob_theta,
                        activation=activation,
                    )

                # Select type of evaluation and apply it to all layers of block
                init_function = partial(init_weights, initialization=initialization)
                nbeats_block.layers.apply(init_function)
                # print(f'     | -- {nbeats_block}')
                block_list.append(nbeats_block)
        return block_list

    def forward(
        self,
        S: t.Tensor,
        Y: t.Tensor,
        X: t.Tensor,
        insample_mask: t.Tensor,
        outsample_mask: t.Tensor,
        return_decomposition: bool = False,
    ):

        # insample
        insample_y = Y[:, : -self.n_time_out]
        insample_x_t = X[:, :, : -self.n_time_out]
        insample_mask = insample_mask[:, : -self.n_time_out]

        # outsample
        outsample_y = Y[:, -self.n_time_out :]
        outsample_x_t = X[:, :, -self.n_time_out :]
        outsample_mask = outsample_mask[:, -self.n_time_out :]

        if return_decomposition:
            forecast, block_forecasts = self.forecast_decomposition(
                insample_y=insample_y,
                insample_x_t=insample_x_t,
                insample_mask=insample_mask,
                outsample_x_t=outsample_x_t,
                x_s=S,
            )
            return outsample_y, forecast, block_forecasts, outsample_mask

        else:
            forecast = self.forecast(
                insample_y=insample_y,
                insample_x_t=insample_x_t,
                insample_mask=insample_mask,
                outsample_x_t=outsample_x_t,
                x_s=S,
            )
            return outsample_y, forecast, outsample_mask

    def forecast(
        self,
        insample_y: t.Tensor,
        insample_x_t: t.Tensor,
        insample_mask: t.Tensor,
        outsample_x_t: t.Tensor,
        x_s: t.Tensor,
    ):

        residuals = insample_y.flip(dims=(-1,))
        insample_x_t = insample_x_t.flip(dims=(-1,))
        insample_mask = insample_mask.flip(dims=(-1,))

        forecast = insample_y[:, -1:]  # Level with Naive1
        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(
                insample_y=residuals,
                insample_x_t=insample_x_t,
                outsample_x_t=outsample_x_t,
                x_s=x_s,
            )
            residuals = (residuals - backcast) * insample_mask
            forecast = forecast + block_forecast

        return forecast

    def forecast_decomposition(
        self,
        insample_y: t.Tensor,
        insample_x_t: t.Tensor,
        insample_mask: t.Tensor,
        outsample_x_t: t.Tensor,
        x_s: t.Tensor,
    ):

        residuals = insample_y.flip(dims=(-1,))
        insample_x_t = insample_x_t.flip(dims=(-1,))
        insample_mask = insample_mask.flip(dims=(-1,))

        n_batch, n_channels, n_t = (
            outsample_x_t.size(0),
            outsample_x_t.size(1),
            outsample_x_t.size(2),
        )

        level = insample_y[:, -1:]  # Level with Naive1
        block_forecasts = [level.repeat(1, n_t)]

        forecast = level
        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(
                insample_y=residuals,
                insample_x_t=insample_x_t,
                outsample_x_t=outsample_x_t,
                x_s=x_s,
            )
            residuals = (residuals - backcast) * insample_mask
            forecast = forecast + block_forecast
            block_forecasts.append(block_forecast)

        # (n_batch, n_blocks, n_t)
        block_forecasts = t.stack(block_forecasts)
        block_forecasts = block_forecasts.permute(1, 0, 2)

        return forecast, block_forecasts


class NHITSBB(BaseModel):
    """
    Wrapper for NHITS, adapted to BuildingsBench BaseModel interface
    """

    def __init__(
        self,
        context_len: int = 168,
        pred_len: int = 24,
        continuous_loads: bool = True,
        **kwargs,
    ):
        super().__init__(context_len, pred_len, continuous_loads=continuous_loads)

        # instantiate your original NHITS core
        self.model = _NHITS(
            n_time_in=context_len,
            n_time_out=pred_len,
            n_s=0,
            n_x=0,
            n_s_hidden=0,
            n_x_hidden=0,
            stack_types=kwargs.get("stack_types", ["identity"]),
            n_blocks=kwargs.get("n_blocks", [2]),
            n_layers=kwargs.get("n_layers", [2]),
            n_theta_hidden=kwargs.get("n_theta_hidden", [[512, 512]]),
            n_pool_kernel_size=kwargs.get("n_pool_kernel_size", [2]),
            n_freq_downsample=kwargs.get("n_freq_downsample", [1]),
            pooling_mode=kwargs.get("pooling_mode", "max"),
            interpolation_mode=kwargs.get("interpolation_mode", "linear"),
            dropout_prob_theta=kwargs.get("dropout_prob_theta", 0.0),
            activation=kwargs.get("activation", "ReLU"),
            initialization=kwargs.get("initialization", "glorot_uniform"),
            batch_normalization=kwargs.get("batch_normalization", False),
            shared_weights=kwargs.get("shared_weights", False),
        )

    def forward(
        self,
        x: Dict[str, torch.Tensor],
        context_len: Optional[int] = None,
        pred_len: Optional[int] = None,
    ):
        """
        x['load']: (B, L, 1)
        """
        if context_len is None:
            context_len = self.context_len
        if pred_len is None:
            pred_len = self.pred_len

        load = x["load"][:, : context_len + pred_len, :]  # (B,L,1)

        # Build fake S,X,masks because _NHITS expects them
        B, L, C = load.shape
        S = torch.zeros(B, 0, device=load.device)
        X = torch.zeros(B, 0, L, device=load.device)
        sample_mask = torch.ones(B, pred_len, device=load.device)
        available_mask = torch.ones(B, L, device=load.device)

        # The NHITS forward expects dict like training_step
        Y = load.squeeze(-1)  # (B,L)

        outsample_y, forecast, _ = self.model(
            S=S,
            Y=Y,
            X=X,
            insample_mask=available_mask,
            outsample_mask=sample_mask,
            return_decomposition=False,
        )

        return forecast.unsqueeze(-1)  # (B,pred_len,1)

    def loss(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Huber or MSE
        return F.mse_loss(pred, y)

    @torch.no_grad()
    def predict(self, x, context_len=None, pred_len=None):
        preds = self.forward(x, context_len, pred_len)
        return preds, None

    def unfreeze_and_get_parameters_for_finetuning(self):
        for p in self.parameters():
            p.requires_grad = True
        return self.parameters()

    def load_from_checkpoint(self, checkpoint_path: str):
        state = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        state = {k.replace("module.", ""): v for k, v in state.items()}
        self.load_state_dict(state, strict=False)
