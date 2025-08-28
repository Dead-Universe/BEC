from __future__ import annotations

from typing import List

import torch
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union


import copy
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch.nn as nn
from transformers.models.t5.modeling_t5 import (
    ACT2FN,
    T5Config,
    T5LayerNorm,
    T5PreTrainedModel,
    T5Stack,
)
from transformers.utils import ModelOutput

if TYPE_CHECKING:
    from transformers import PreTrainedModel

logger = logging.getLogger(__file__)


def left_pad_and_stack_1D(tensors: List[torch.Tensor]) -> torch.Tensor:
    max_len = max(len(c) for c in tensors)
    padded = []
    for c in tensors:
        assert isinstance(c, torch.Tensor)
        assert c.ndim == 1
        padding = torch.full(
            size=(max_len - len(c),), fill_value=torch.nan, device=c.device
        )
        padded.append(torch.concat((padding, c), dim=-1))
    return torch.stack(padded)


class ForecastType(Enum):
    SAMPLES = "samples"
    QUANTILES = "quantiles"


class PipelineRegistry(type):
    REGISTRY: Dict[str, "PipelineRegistry"] = {}

    def __new__(cls, name, bases, attrs):
        """See, https://github.com/faif/python-patterns."""
        new_cls = type.__new__(cls, name, bases, attrs)
        if name is not None:
            cls.REGISTRY[name] = new_cls

        return new_cls


class BaseChronosPipeline(metaclass=PipelineRegistry):
    forecast_type: ForecastType
    dtypes = {"bfloat16": torch.bfloat16, "float32": torch.float32}

    def __init__(self, inner_model: "PreTrainedModel"):
        """
        Parameters
        ----------
        inner_model : PreTrainedModel
            A hugging-face transformers PreTrainedModel, e.g., T5ForConditionalGeneration
        """
        # for easy access to the inner HF-style model
        self.inner_model = inner_model

    def _prepare_and_validate_context(
        self, context: Union[torch.Tensor, List[torch.Tensor]]
    ):
        if isinstance(context, list):
            context = left_pad_and_stack_1D(context)
        assert isinstance(context, torch.Tensor)
        if context.ndim == 1:
            context = context.unsqueeze(0)
        assert context.ndim == 2

        return context

    def predict(
        self,
        context: Union[torch.Tensor, List[torch.Tensor]],
        prediction_length: Optional[int] = None,
        **kwargs,
    ):
        """
        Get forecasts for the given time series. Predictions will be
        returned in fp32 on the cpu.

        Parameters
        ----------
        context
            Input series. This is either a 1D tensor, or a list
            of 1D tensors, or a 2D tensor whose first dimension
            is batch. In the latter case, use left-padding with
            ``torch.nan`` to align series of different lengths.
        prediction_length
            Time steps to predict. Defaults to a model-dependent
            value if not given.

        Returns
        -------
        forecasts
            Tensor containing forecasts. The layout and meaning
            of the forecasts values depends on ``self.forecast_type``.
        """
        raise NotImplementedError()

    def predict_quantiles(
        self,
        context: Union[torch.Tensor, List[torch.Tensor]],
        prediction_length: Optional[int] = None,
        quantile_levels: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get quantile and mean forecasts for given time series.
        Predictions will be returned in fp32 on the cpu.

        Parameters
        ----------
        context : Union[torch.Tensor, List[torch.Tensor]]
            Input series. This is either a 1D tensor, or a list
            of 1D tensors, or a 2D tensor whose first dimension
            is batch. In the latter case, use left-padding with
            ``torch.nan`` to align series of different lengths.
        prediction_length : Optional[int], optional
            Time steps to predict. Defaults to a model-dependent
            value if not given.
        quantile_levels : List[float], optional
            Quantile levels to compute, by default [0.1, 0.2, ..., 0.9]

        Returns
        -------
        quantiles
            Tensor containing quantile forecasts. Shape
            (batch_size, prediction_length, num_quantiles)
        mean
            Tensor containing mean (point) forecasts. Shape
            (batch_size, prediction_length)
        """
        raise NotImplementedError()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path],
        *model_args,
        **kwargs,
    ):
        """
        Load the model, either from a local path or from the HuggingFace Hub.
        Supports the same arguments as ``AutoConfig`` and ``AutoModel``
        from ``transformers``.
        """
        from transformers import AutoConfig

        torch_dtype = kwargs.get("torch_dtype", "auto")
        if torch_dtype != "auto" and isinstance(torch_dtype, str):
            kwargs["torch_dtype"] = cls.dtypes[torch_dtype]

        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        is_valid_config = hasattr(config, "chronos_pipeline_class") or hasattr(
            config, "chronos_config"
        )

        if not is_valid_config:
            raise ValueError("Not a Chronos config file")

        pipeline_class_name = getattr(
            config, "chronos_pipeline_class", "ChronosPipeline"
        )
        class_ = PipelineRegistry.REGISTRY.get(pipeline_class_name)
        if class_ is None:
            raise ValueError(
                f"Trying to load unknown pipeline class: {pipeline_class_name}"
            )

        return class_.from_pretrained(  # type: ignore[attr-defined]
            pretrained_model_name_or_path, *model_args, **kwargs
        )


@dataclass
class ChronosBoltConfig:
    context_length: int
    prediction_length: int
    input_patch_size: int
    input_patch_stride: int
    quantiles: List[float]
    use_reg_token: bool = False


@dataclass
class ChronosBoltOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    quantile_preds: Optional[torch.Tensor] = None
    attentions: Optional[torch.Tensor] = None
    cross_attentions: Optional[torch.Tensor] = None


class Patch(nn.Module):
    def __init__(self, patch_size: int, patch_stride: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[-1]

        if length % self.patch_size != 0:
            padding_size = (
                *x.shape[:-1],
                self.patch_size - (length % self.patch_size),
            )
            padding = torch.full(
                size=padding_size, fill_value=torch.nan, dtype=x.dtype, device=x.device
            )
            x = torch.concat((padding, x), dim=-1)

        x = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)
        return x


class InstanceNorm(nn.Module):
    """
    See, also, RevIN. Apply standardization along the last dimension.
    """

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
        loc_scale: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if loc_scale is None:
            loc = torch.nan_to_num(torch.nanmean(x, dim=-1, keepdim=True), nan=0.0)
            scale = torch.nan_to_num(
                torch.nanmean((x - loc).square(), dim=-1, keepdim=True).sqrt(), nan=1.0
            )
            scale = torch.where(scale == 0, self.eps, scale)
        else:
            loc, scale = loc_scale

        return (x - loc) / scale, (loc, scale)

    def inverse(
        self, x: torch.Tensor, loc_scale: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        loc, scale = loc_scale
        return x * scale + loc


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        out_dim: int,
        act_fn_name: str,
        dropout_p: float = 0.0,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout_p)
        self.hidden_layer = nn.Linear(in_dim, h_dim)
        self.act = ACT2FN[act_fn_name]
        self.output_layer = nn.Linear(h_dim, out_dim)
        self.residual_layer = nn.Linear(in_dim, out_dim)

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = T5LayerNorm(out_dim)

    def forward(self, x: torch.Tensor):
        hid = self.act(self.hidden_layer(x))
        out = self.dropout(self.output_layer(hid))
        res = self.residual_layer(x)

        out = out + res

        if self.use_layer_norm:
            return self.layer_norm(out)
        return out


class ChronosBoltModelForForecasting(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [  # type: ignore
        r"input_patch_embedding\.",
        r"output_patch_embedding\.",
    ]
    _keys_to_ignore_on_load_unexpected = [r"lm_head.weight"]  # type: ignore
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]  # type: ignore

    def __init__(self, config: T5Config):
        assert hasattr(config, "chronos_config"), "Not a Chronos config file"

        super().__init__(config)
        self.model_dim = config.d_model

        self.chronos_config = ChronosBoltConfig(**config.chronos_config)

        # Only decoder_start_id (and optionally REG token)
        if self.chronos_config.use_reg_token:
            config.reg_token_id = 1

        config.vocab_size = 2 if self.chronos_config.use_reg_token else 1
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        # Input patch embedding layer
        self.input_patch_embedding = ResidualBlock(
            in_dim=self.chronos_config.input_patch_size * 2,
            h_dim=config.d_ff,
            out_dim=config.d_model,
            act_fn_name=config.dense_act_fn,
            dropout_p=config.dropout_rate,
        )

        # patching layer
        self.patch = Patch(
            patch_size=self.chronos_config.input_patch_size,
            patch_stride=self.chronos_config.input_patch_stride,
        )

        # instance normalization, also referred to as "scaling" in Chronos and GluonTS
        self.instance_norm = InstanceNorm()

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        self.encoder = T5Stack(encoder_config, self.shared)

        self._init_decoder(config)

        self.num_quantiles = len(self.chronos_config.quantiles)
        quantiles = torch.tensor(self.chronos_config.quantiles, dtype=self.dtype)
        self.register_buffer("quantiles", quantiles, persistent=False)

        self.output_patch_embedding = ResidualBlock(
            in_dim=config.d_model,
            h_dim=config.d_ff,
            out_dim=self.num_quantiles * self.chronos_config.prediction_length,
            act_fn_name=config.dense_act_fn,
            dropout_p=config.dropout_rate,
        )

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def _init_weights(self, module):
        super()._init_weights(module)
        """Initialize the weights"""
        factor = self.config.initializer_factor
        if isinstance(module, (self.__class__)):
            module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, ResidualBlock):
            module.hidden_layer.weight.data.normal_(
                mean=0.0,
                std=factor * ((self.chronos_config.input_patch_size * 2) ** -0.5),
            )
            if (
                hasattr(module.hidden_layer, "bias")
                and module.hidden_layer.bias is not None
            ):
                module.hidden_layer.bias.data.zero_()

            module.residual_layer.weight.data.normal_(
                mean=0.0,
                std=factor * ((self.chronos_config.input_patch_size * 2) ** -0.5),
            )
            if (
                hasattr(module.residual_layer, "bias")
                and module.residual_layer.bias is not None
            ):
                module.residual_layer.bias.data.zero_()

            module.output_layer.weight.data.normal_(
                mean=0.0, std=factor * ((self.config.d_ff) ** -0.5)
            )
            if (
                hasattr(module.output_layer, "bias")
                and module.output_layer.bias is not None
            ):
                module.output_layer.bias.data.zero_()

    def encode(
        self, context: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor
    ]:
        mask = (
            mask.to(context.dtype)
            if mask is not None
            else torch.isnan(context).logical_not().to(context.dtype)
        )

        batch_size, _ = context.shape
        if context.shape[-1] > self.chronos_config.context_length:
            context = context[..., -self.chronos_config.context_length :]
            mask = mask[..., -self.chronos_config.context_length :]

        # scaling
        context, loc_scale = self.instance_norm(context)

        # the scaling op above is done in 32-bit precision,
        # then the context is moved to model's dtype
        context = context.to(self.dtype)
        mask = mask.to(self.dtype)

        # patching
        patched_context = self.patch(context)
        patched_mask = torch.nan_to_num(self.patch(mask), nan=0.0)
        patched_context = torch.where(patched_mask > 0.0, patched_context, 0.0)
        # concat context and mask along patch dim
        patched_context = torch.cat([patched_context, patched_mask], dim=-1)

        # attention_mask = 1 if at least one item in the patch is observed
        attention_mask = (
            patched_mask.sum(dim=-1) > 0
        )  # (batch_size, patched_seq_length)

        input_embeds = self.input_patch_embedding(patched_context)

        if self.chronos_config.use_reg_token:
            # Append [REG]
            reg_input_ids = torch.full(
                (batch_size, 1),
                self.config.reg_token_id,
                device=input_embeds.device,
            )
            reg_embeds = self.shared(reg_input_ids)
            input_embeds = torch.cat([input_embeds, reg_embeds], dim=-2)
            attention_mask = torch.cat(
                [
                    attention_mask.to(self.dtype),
                    torch.ones_like(reg_input_ids).to(self.dtype),
                ],
                dim=-1,
            )

        encoder_outputs = self.encoder(
            attention_mask=attention_mask,
            inputs_embeds=input_embeds,
        )

        return encoder_outputs[0], loc_scale, input_embeds, attention_mask

    def forward(
        self,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
    ) -> ChronosBoltOutput:
        batch_size = context.size(0)

        hidden_states, loc_scale, input_embeds, attention_mask = self.encode(
            context=context, mask=mask
        )
        sequence_output = self.decode(input_embeds, attention_mask, hidden_states)

        quantile_preds_shape = (
            batch_size,
            self.num_quantiles,
            self.chronos_config.prediction_length,
        )
        quantile_preds = self.output_patch_embedding(sequence_output).view(
            *quantile_preds_shape
        )

        loss = None
        if target is not None:
            # normalize target
            target, _ = self.instance_norm(target, loc_scale)
            target = target.unsqueeze(1)  # type: ignore
            assert self.chronos_config.prediction_length >= target.shape[-1]

            target = target.to(quantile_preds.device)
            target_mask = (
                target_mask.unsqueeze(1).to(quantile_preds.device)
                if target_mask is not None
                else ~torch.isnan(target)
            )
            target[~target_mask] = 0.0

            # pad target and target_mask if they are shorter than model's prediction_length
            if self.chronos_config.prediction_length > target.shape[-1]:
                padding_shape = (
                    *target.shape[:-1],
                    self.chronos_config.prediction_length - target.shape[-1],
                )
                target = torch.cat(
                    [target, torch.zeros(padding_shape).to(target)], dim=-1
                )
                target_mask = torch.cat(
                    [target_mask, torch.zeros(padding_shape).to(target_mask)], dim=-1
                )

            loss = (
                2
                * torch.abs(
                    (target - quantile_preds)
                    * (
                        (target <= quantile_preds).float()
                        - self.quantiles.view(1, self.num_quantiles, 1)  # type: ignore
                    )
                )
                * target_mask.float()
            )
            loss = loss.mean(dim=-2)  # Mean over prediction horizon
            loss = loss.sum(dim=-1)  # Sum over quantile levels
            loss = loss.mean()  # Mean over batch

        # Unscale predictions
        quantile_preds = self.instance_norm.inverse(
            quantile_preds.view(batch_size, -1),
            loc_scale,
        ).view(*quantile_preds_shape)

        return ChronosBoltOutput(
            loss=loss,
            quantile_preds=quantile_preds,
        )

    def _init_decoder(self, config):
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

    def decode(
        self,
        input_embeds,
        attention_mask,
        hidden_states,
        output_attentions=False,
    ):
        """
        Parameters
        ----------
        input_embeds: torch.Tensor
            Patched and embedded inputs. Shape (batch_size, patched_context_length, d_model)
        attention_mask: torch.Tensor
            Attention mask for the patched context. Shape (batch_size, patched_context_length), type: torch.int64
        hidden_states: torch.Tensor
            Hidden states returned by the encoder. Shape (batch_size, patched_context_length, d_model)

        Returns
        -------
        last_hidden_state
            Last hidden state returned by the decoder, of shape (batch_size, 1, d_model)
        """
        batch_size = input_embeds.shape[0]
        decoder_input_ids = torch.full(
            (batch_size, 1),
            self.config.decoder_start_token_id,
            device=input_embeds.device,
        )
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            output_attentions=output_attentions,
            return_dict=True,
        )

        return decoder_outputs.last_hidden_state  # sequence_outputs, b x 1 x d_model


from typing import Dict, Optional, Tuple, Literal
import torch
import torch.nn.functional as F
from torch import nn


# === 引用你已有的基类与 Chronos 主体 ===
from buildings_bench.models.base_model import BaseModel


import copy


class ChronosAsLoadForecastAdapter(BaseModel):
    """
    在 __init__ 中直接构造 ChronosBoltModelForForecasting，
    并提供与 LoadForecastingTransformerMoE 一致的接口。
    """

    def __init__(
        self,
        # —— 与 MoE 接口相近的参数 —— #
        max_context_len: int = 336,
        max_pred_len: int = 168,
        context_len: int = 168,
        pred_len: int = 24,
        d_model: int = 256,
        dim_feedforward: int = 1024,
        num_encoder_layers: int = 6,  # T5 的 encoder 层数
        num_decoder_layers: int = 6,  # T5 的 decoder 层数
        activation: str = "relu",  # T5Config.feed_forward_proj
        dropout: float = 0.0,
        num_heads: int = 8,  # T5Config.num_heads
        # —— Chronos 特有（设有合理默认） —— #
        input_patch_size: int = 24,
        input_patch_stride: int = 8,
        quantiles: List[float] = [0.5],
        use_reg_token: bool = False,
        # —— 训练头部定义（与 MoE 对齐） —— #
        continuous_head: Literal["mse", "gaussian_nll", "huber"] = "huber",
        # torch dtype（可选）
        torch_dtype: Optional[torch.dtype] = torch.float32,
        device: Optional[torch.device | str] = "cuda",
        **kwargs,
    ):
        super().__init__(context_len, pred_len, continuous_loads=True)

        # 记录最大长度（Chronos 固定）
        self.max_context_len = max_context_len
        self.max_pred_len = max_pred_len
        self.continuous_head = continuous_head

        # ---- 构造 T5Config + chronos_config ----
        cfg = T5Config(
            d_model=d_model,
            d_ff=dim_feedforward,
            num_heads=num_heads,
            num_layers=num_encoder_layers,  # encoder 层数
            num_decoder_layers=num_decoder_layers,
            dropout_rate=dropout,
            feed_forward_proj=activation,  # 'relu'/'gelu'等
            decoder_start_token_id=0,
            vocab_size=1,
        )
        cfg.chronos_config = {
            "context_length": max_context_len,
            "prediction_length": max_pred_len,
            "input_patch_size": input_patch_size,
            "input_patch_stride": input_patch_stride,
            "quantiles": list(quantiles),
            "use_reg_token": use_reg_token,
        }

        # ---- 实例化 Chronos 主体 ----
        self.chronos = ChronosBoltModelForForecasting(cfg)
        if torch_dtype is not None:
            self.chronos = self.chronos.to(torch_dtype)
        if device is not None:
            self.chronos = self.chronos.to(device)

    # --------------------- forward（训练图） ----------------------
    def forward(
        self,
        x: Dict[str, torch.Tensor],
        context_len: Optional[int] = None,
        pred_len: Optional[int] = None,
    ) -> torch.Tensor:
        """
        输入 x['load']: (B, ctx+pred, 1)
        返回: (B, pred_len, 1) 连续值预测
        """
        if context_len is None:
            context_len = self.max_context_len
        if pred_len is None:
            pred_len = self.max_pred_len

        load = x["load"]  # (B, ctx+pred, 1)
        assert load.dim() == 3 and load.size(-1) == 1, "x['load'] 应为 (B, T, 1)"

        # 取历史作为 Chronos context（Chronos 内部会按 context_length 截断）
        ctx = load[:, :context_len, 0]  # (B, ctx)

        out = self.chronos(
            context=ctx,  # 需要计算图 => 不要 no_grad
            mask=None,
            target=None,
            target_mask=None,
        )
        q_preds = out.quantile_preds  # (B, n_q, max_pred_len)
        assert q_preds is not None, "Chronos 未返回 quantile_preds"

        # 取分位均值（也可改 0.5 分位）
        mean_pred = q_preds.mean(dim=1)  # (B, max_pred_len)
        y_hat = mean_pred[:, :pred_len].unsqueeze(-1)  # (B, pred_len, 1)
        return y_hat

    # --------------------- loss -------------------------
    def loss(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        与 MoE 的连续头保持一致：
        - mse / huber：直接对 (B, L, 1)
        - gaussian_nll：若仅有均值通道，则使用固定 σ²=1 计算（无需第二通道）
        """
        if self.continuous_head == "huber":
            err = F.huber_loss(pred, y, delta=1.0, reduction="none")
            loss = err.mean()
        elif self.continuous_head == "mse":
            loss = F.mse_loss(pred, y, reduction="mean")
        else:  # gaussian_nll（固定 sigma^2=1）
            mu = pred[..., :1]
            sigma_sq = torch.ones_like(mu)
            loss = 0.5 * (torch.log(2 * torch.pi * sigma_sq) + (y - mu) ** 2 / sigma_sq)
            loss = loss.mean()
        return loss

    # --------------------- predict ----------------------
    @torch.no_grad()
    def predict(
        self,
        x: Dict[str, torch.Tensor],
        context_len: int = 168,
        pred_len: int = 24,
    ):
        preds = self.forward(x, context_len=context_len, pred_len=pred_len)
        return preds, preds

    # ----------------- finetune parameters ---------------
    def unfreeze_and_get_parameters_for_finetuning(self):
        return self.chronos.parameters()

    # ----------------- load checkpoint -------------------
    def _strip_prefixes(self, sd: dict) -> dict:
        new_sd = {}
        for k, v in sd.items():
            if k.startswith("module.chronos."):
                new_sd[k[len("module.chronos.") :]] = v
            elif k.startswith("chronos."):
                new_sd[k[len("chronos.") :]] = v
            elif k.startswith("module."):
                # 万一保存的是直接的内层模型但被 DataParallel 包过
                new_sd[k[len("module.") :]] = v
            else:
                new_sd[k] = v
        return new_sd

    def load_from_checkpoint(self, checkpoint_path):
        sd = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]
        if not isinstance(sd, dict):
            raise ValueError(
                "无效的 checkpoint：应为 state_dict 或 {'model': state_dict}"
            )

        sd = self._strip_prefixes(sd)
        missing, unexpected = self.chronos.load_state_dict(sd, strict=True)
        print(
            "[Chronos Adapter] load_state_dict differences:",
            f"missing={missing}, unexpected={unexpected}",
        )


# ======================= Quick self-test =======================
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) 构造“像 MoE 一样”的初始化
    adapter = ChronosAsLoadForecastAdapter(
        max_context_len=672,
        max_pred_len=168,
        context_len=96,  # 本轮实际使用
        pred_len=40,  # 本轮实际使用
        d_model=128,
        dim_feedforward=256,
        num_encoder_layers=2,
        num_decoder_layers=2,
        activation="relu",
        dropout=0.1,
        input_patch_size=16,
        input_patch_stride=8,
        quantiles=[0.1, 0.5, 0.9],
        use_reg_token=False,
        continuous_head="huber",
        device=device,
    ).train()

    # 2) 伪造 batch（键位与 MoE 保持一致）
    ctx, pred = 96, 40
    B = 2
    load = torch.rand(B, ctx + pred, 1, device=device)
    stl_S = torch.randn(B, ctx + pred, 1, device=device)
    stl_T = torch.randn(B, ctx + pred, 1, device=device)
    stl_R = torch.randn(B, ctx + pred, 1, device=device)
    batch = {"load": load, "stl_S": stl_S, "stl_T": stl_T, "stl_R": stl_R}

    # 3) 训练向前 + 反传
    y_hat = adapter.forward(batch, context_len=ctx, pred_len=pred)  # (B, pred, 1)
    target = batch["load"][:, -pred:]  # (B, pred, 1)
    loss = adapter.loss(y_hat, target)
    loss.backward()
    print("[Adapter] Sanity-check OK – loss:", float(loss))

    # 4) 推理
    adapter.eval()
    preds, _ = adapter.predict(batch, context_len=ctx, pred_len=pred)
    print("[Adapter] Inference preds shape:", preds.shape)  # [B, pred, 1]
