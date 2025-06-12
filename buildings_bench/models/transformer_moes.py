import math
import torch
from typing import Tuple, Dict, List
from torch import nn
import torch.nn.functional as F
import numpy as np
from buildings_bench.models.base_model import BaseModel


class TokenEmbedding(nn.Module):
    """Helper Module to convert tensor of input
    indices into corresponding tensor of token embeddings.
    """

    def __init__(self, vocab_size: int, emb_size: int):
        """
        Args:
            vocab_size (int): number of quantized load values in the entire vocabulary.
            emb_size (int): embedding size.
        """
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: torch.Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class PositionalEncoding(nn.Module):
    """Helper Module that adds positional encoding to the token embedding to
    introduce a notion of order within a time-series.
    """

    def __init__(self, emb_size: int, dropout: float, maxlen: int = 500):
        """
        Args:
            emb_size (int): embedding size.
            dropout (float): dropout rate.
            maxlen (int): maximum possible length of the incoming time series.
        """
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding: torch.Tensor) -> torch.Tensor:
        # batch first - use size(1)
        # need to permute token embeddings from [batch_size, seqlen x emb_size] to [seqlen x batch_size, emb_size]
        return self.dropout(
            token_embedding.permute(1, 0, 2)
            + self.pos_embedding[: token_embedding.size(1), :]
        ).permute(1, 0, 2)


class TimeSeriesSinusoidalPeriodicEmbedding(nn.Module):
    """This module produces a sinusoidal periodic embedding for a sequence of values in [-1, +1]."""

    def __init__(self, embedding_dim: int) -> None:
        """
        Args:
            embedding_dim (int): embedding size.
        """
        super().__init__()
        self.linear = nn.Linear(2, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`x` is expected to be [batch_size, seqlen, 1]."""
        with torch.no_grad():
            x = torch.cat([torch.sin(np.pi * x), torch.cos(np.pi * x)], dim=2)
        # [batch_size, seqlen x 2] --> [batch_size, seqlen, embedding_dim]
        return self.linear(x)


class ZeroEmbedding(nn.Module):
    """Outputs zeros of the desired output dim."""

    def __init__(self, embedding_dim: int):
        """
        Args:
            embedding_dim (int): embedding size.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.zeros_embedding = nn.Parameter(
            torch.zeros(1, 1, embedding_dim), requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`x` is expected to be [batch_size, seqlen, 1]."""
        return self.zeros_embedding.repeat(x.shape[0], x.shape[1], 1)


# ---------------------------------------------------------------------------
# 1.  —— Switch / Top-k MoE FFN ——————————
# ---------------------------------------------------------------------------


class SwitchFeedForward(nn.Module):
    """
    Vectorised Top-k Mixture-of-Experts FFN  (单机/单卡友好实现)
    Args
    ----
    model_dim     : 输入/输出维度 (= d_model)
    hidden_dim    : 专家内隐层维度 (= dim_feedforward)
    num_experts   : 专家数 E
    top_k         : 每个 token 选中的专家数  (1 = Switch-Transformer, 2 = GLaM/Time-MoE)
    """

    def __init__(
        self,
        model_dim: int,
        hidden_dim: int,
        num_experts: int = 8,
        top_k: int = 1,
        activation: nn.Module | str = nn.GELU(),
    ):
        super().__init__()
        self.num_experts, self.top_k = num_experts, top_k
        self.activation = (
            activation
            if isinstance(activation, nn.Module)
            else self._get_act(activation)
        )

        # 专家 FFN 列表
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(model_dim, hidden_dim, bias=False),
                    self.activation,
                    nn.Linear(hidden_dim, model_dim, bias=False),
                )
                for _ in range(num_experts)
            ]
        )

        # 门控
        self.gate = nn.Linear(model_dim, num_experts, bias=False)

        # 保存负载均衡辅助损失
        self.register_buffer("loss_coef", torch.tensor(0.0), persistent=False)

    def _get_act(self, name):
        return {"gelu": nn.GELU(), "relu": nn.ReLU()}[name]

    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x  : [B, T, C]
        out: [B, T, C]
        返回 gate logits 仅供调试，可随意删掉
        """
        B, T, C = x.shape
        S = B * T
        flat_x = x.view(S, C)  # [S, C]

        # -------- ① 门控分配 -------- #
        logits = self.gate(flat_x)  # [S, E]
        probs = torch.softmax(logits, dim=-1)  # soft   (用于正则)
        top_val, top_idx = probs.topk(self.top_k, dim=-1)  # [S, k]

        # 为 scatter 做准备，把 token 复制 k 次
        # 每个 token 第 i=0..k-1 个专家
        dispatch_ids = top_idx.flatten()  # [S*k]
        dispatch_weights = top_val.flatten()  # [S*k]
        # token 自身索引
        token_idx = torch.arange(S, device=x.device).repeat_interleave(self.top_k)

        # -------- ② 按专家分组前向 -------- #
        out_flat = torch.zeros_like(flat_x)
        for eid in range(self.num_experts):
            mask = dispatch_ids == eid  # 当前专家负责的样本
            if not mask.any():
                continue
            sel_tokens = token_idx[mask]  # [N_e]
            sel_inputs = flat_x[sel_tokens]  # [N_e, C]
            sel_weights = dispatch_weights[mask].unsqueeze(1)  # [N_e, 1]

            # 前向 & 加权
            sel_outputs = self.experts[eid](sel_inputs) * sel_weights
            # 累加（因为 top-k>1 时同一 token 可能被多专家加权求和）
            out_flat.index_add_(0, sel_tokens, sel_outputs)

        # -------- ③ 负载均衡辅助损失 -------- #
        if self.training:
            # 真实负载：各专家被选 token 占比
            load = torch.bincount(dispatch_ids, minlength=self.num_experts).float()
            load = load / load.sum()  # [E]
            # 概率均值
            prob = probs.mean(0)  # [E]
            self.loss_coef = (load * prob).sum() * self.num_experts

        return out_flat.view(B, T, C), logits


# ---------------------------------------------------------------------------
# 2.  —— MoE-Encoder / Decoder Layer ——————————
# ---------------------------------------------------------------------------


class MoETransformerEncoderLayer(nn.Module):
    """ """

    def __init__(
        self, d_model, nhead, dim_feedforward, dropout, num_experts, top_k, activation
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.moe_ffn = SwitchFeedForward(
            d_model,
            dim_feedforward,
            num_experts=num_experts,
            top_k=top_k,
            activation=activation,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        src,
        src_mask=None,
        src_key_padding_mask=None,
        is_causal: bool = False,  # <-- 新增；保持缺省 False
        **kwargs,  #   兼容未来额外参数
    ):
        # Self-Attention
        sa_out, _ = self.self_attn(
            src,
            src,
            src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,  # 避免额外显存
            is_causal=is_causal,  # 传回去也行，直接忽略也行
        )
        src = self.norm1(src + self.drop(sa_out))
        # MoE-FFN
        ffn_out, _ = self.moe_ffn(src)
        src = self.norm2(src + self.drop(ffn_out))
        return src


class MoETransformerDecoderLayer(nn.Module):

    def __init__(
        self, d_model, nhead, dim_feedforward, dropout, num_experts, top_k, activation
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.moe_ffn = SwitchFeedForward(
            d_model,
            dim_feedforward,
            num_experts=num_experts,
            top_k=top_k,
            activation=activation,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        is_causal: bool = False,  # <-- 同样新增
        **kwargs,
    ):
        tgt2, _ = self.self_attn(
            tgt,
            tgt,
            tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
        )
        tgt = self.norm1(tgt + self.drop(tgt2))

        tgt2, _ = self.cross_attn(
            tgt,
            memory,
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        tgt = self.norm2(tgt + self.drop(tgt2))

        ffn_out, _ = self.moe_ffn(tgt)
        tgt = self.norm3(tgt + self.drop(ffn_out))
        return tgt


class LoadForecastingTransformerMoE(BaseModel):
    """
    An encoder-decoder time series Transformer. Based on PyTorch nn.Transformer.

    - Uses masking in the decoder to prevent the model from peeking into the future
    - Uses N(0, 0.02) for weight initialization
    - Trains with teacher forcing (i.e. the target is used as the input to the decoder)
    - continuous_loads (True) just predict target values
                     (False) categorical over quantized load values
    """

    def __init__(
        self,
        num_experts: int = 8,
        top_k: int = 1,
        aux_loss_weight: float = 0.01,
        context_len: int = 168,
        pred_len: int = 24,
        vocab_size=2274,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
        activation: str = "gelu",
        continuous_loads: bool = False,
        continuous_head: str = "mse",
        ignore_spatial: bool = False,
        weather_inputs: List[str] | None = None,
    ):
        """
        Args:
            context_len (int): length of the input sequence.
            pred_len (int): length of the output sequence.
            vocab_size (int): number of quantized load values in the entire vocabulary.
            num_encoder_layers (int): number of encoder layers.
            num_decoder_layers (int): number of decoder layers.
            d_model (int): number of expected features in the encoder/decoder inputs.
            nhead (int): number of heads in the multi-head attention models.
            dim_feedforward (int): dimension of the feedforward network model.
            dropout (float): dropout value.
            activation (str): the activation function of encoder/decoder intermediate layer, relu or gelu.
            continuous_loads (bool): whether inputs are continuous/to train the model to predict continuous values.
            continuous_head (str): 'mse' or 'gaussian_nll'.
            ignore_spatial (bool): whether to ignore the spatial features.
            weather_inputs (List[str]): list of weather features to use. Default: None.
        """
        super().__init__(context_len, pred_len, continuous_loads)

        self.continuous_head = continuous_head
        self.vocab_size = vocab_size
        self.ignore_spatial = ignore_spatial
        self.aux_loss_weight = aux_loss_weight

        scale = d_model // 256
        self.weather_features = weather_inputs
        if weather_inputs:
            self.weather_embedding = nn.Linear(len(weather_inputs), 64)
            d_model += 64

        # self.transformer = Transformer(
        #     d_model=d_model,
        #     nhead=nhead,
        #     num_encoder_layers=num_encoder_layers,
        #     num_decoder_layers=num_decoder_layers,
        #     dim_feedforward=dim_feedforward,
        #     dropout=dropout,
        #     activation=activation,
        #     batch_first=True,
        # )

        # ———  MoE-Encoder / Decoder  ———
        enc_layer = MoETransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            num_experts=num_experts,
            top_k=top_k,
            activation=activation,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_encoder_layers)

        dec_layer = MoETransformerDecoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            num_experts=num_experts,
            top_k=top_k,
            activation=activation,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_decoder_layers)

        if self.continuous_loads:
            out_dim = 1 if self.continuous_head == "mse" else 2
            self.logits = nn.Linear(d_model, out_dim)
            self.power_embedding = nn.Linear(1, 64 * scale)
        else:
            self.logits = nn.Linear(d_model, self.vocab_size)
            self.power_embedding = TokenEmbedding(self.vocab_size, 64 * scale)
        self.tgt_mask = nn.Transformer.generate_square_subsequent_mask(self.pred_len)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)
        self.building_embedding = nn.Embedding(2, 32 * scale)
        self.lat_embedding = nn.Linear(1, 32 * scale)
        self.lon_embedding = nn.Linear(1, 32 * scale)
        if self.ignore_spatial:
            self.lat_embedding = ZeroEmbedding(32 * scale)
            self.lon_embedding = ZeroEmbedding(32 * scale)
        self.day_of_year_encoding = TimeSeriesSinusoidalPeriodicEmbedding(32 * scale)
        self.day_of_week_encoding = TimeSeriesSinusoidalPeriodicEmbedding(32 * scale)
        self.hour_of_day_encoding = TimeSeriesSinusoidalPeriodicEmbedding(32 * scale)

    def to(self, device):
        self.tgt_mask = self.tgt_mask.to(device)
        return super().to(device)

    def forward(self, x):
        r"""Forward pass of the time series transformer.

        Args:
            x (Dict): dictionary of input tensors.
        Returns:
            logits (torch.Tensor): [batch_size, pred_len, vocab_size] if not continuous_loads,
                                   [batch_size, pred_len, 1] if continuous_loads and continuous_head == 'mse',
                                   [batch_size, pred_len, 2] if continuous_loads and continuous_head == 'gaussian_nll'.
        """
        if self.weather_features:
            time_series_inputs = [
                self.lat_embedding(x["latitude"]),
                self.lon_embedding(x["longitude"]),
                self.building_embedding(x["building_type"]).squeeze(2),
                self.day_of_year_encoding(x["day_of_year"]),
                self.day_of_week_encoding(x["day_of_week"]),
                self.hour_of_day_encoding(x["hour_of_day"]),
                self.weather_embedding(
                    torch.cat([x[ft] for ft in self.weather_features], dim=2)
                ),
                self.power_embedding(x["load"]).squeeze(2),
            ]

        else:
            time_series_inputs = [
                self.lat_embedding(x["latitude"]),
                self.lon_embedding(x["longitude"]),
                self.building_embedding(x["building_type"]).squeeze(2),
                self.day_of_year_encoding(x["day_of_year"]),
                self.day_of_week_encoding(x["day_of_week"]),
                self.hour_of_day_encoding(x["hour_of_day"]),
                self.power_embedding(x["load"]).squeeze(2),
            ]
        time_series_embed = torch.cat(time_series_inputs, dim=2)
        # [batch_size, context_len, d_model]
        src_series_inputs = time_series_embed[:, : self.context_len, :]
        # [batch_size, pred_len, d_model]
        # The last element of the target sequence is not used as input
        # The last element of the source sequence is used as the initial decoder input
        tgt_series_inputs = time_series_embed[:, self.context_len - 1 : -1, :]
        src_series_embed = self.positional_encoding(src_series_inputs)
        tgt_series_embed = self.positional_encoding(tgt_series_inputs)

        # The output of TransformerEncoder is the sequence from the last layer
        # The shape will be [batch_size, context_len, d_model]

        memory = self.encoder(src_series_embed)
        outs = self.decoder(tgt_series_embed, memory, tgt_mask=self.tgt_mask)
        return self.logits(outs)

    def predict(self, x: Dict) -> Tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        return self.generate_sample(x, greedy=True)

    def loss(self, x, y):
        if self.continuous_loads and self.continuous_head == "mse":
            task_loss = F.mse_loss(x, y)
        elif self.continuous_loads and self.continuous_head == "gaussian_nll":
            task_loss = F.gaussian_nll_loss(
                x[:, :, 0].unsqueeze(2), y, F.softplus(x[:, :, 1].unsqueeze(2)) ** 2
            )
        else:
            task_loss = F.cross_entropy(
                x.reshape(-1, self.vocab_size), y.long().reshape(-1)
            )
        aux_list = [layer.moe_ffn.loss_coef for layer in self.encoder.layers]
        #  若 Decoder 也用了 MoE，则加上
        if isinstance(self.decoder.layers[0], MoETransformerDecoderLayer):
            aux_list += [layer.moe_ffn.loss_coef for layer in self.decoder.layers]

        aux_loss = torch.stack(aux_list).mean()  # 标量

        # --- 3) 总损失 ------------------------------------
        total_loss = (
            task_loss + self.aux_loss_weight * aux_loss if self.training else task_loss
        )
        return total_loss

    def unfreeze_and_get_parameters_for_finetuning(self):
        # for p in self.parameters():
        #     p.requires_grad_(False)
        # self.logits.requires_grad_(True)
        # return self.logits.parameters()
        return self.parameters()

    def load_from_checkpoint(self, checkpoint_path):
        stored_ckpt = torch.load(checkpoint_path)
        model_state_dict = stored_ckpt["model"]
        new_state_dict = {}
        for k, v in model_state_dict.items():
            # remove string 'module.' from the key
            if "module." in k:
                new_state_dict[k.replace("module.", "")] = v
            else:
                new_state_dict[k] = v
        self.load_state_dict(new_state_dict)
        # print(f"Loaded model checkpoint from {checkpoint_path}...")

    @torch.no_grad()
    def generate_sample(self, x, temperature=1.0, greedy=False, num_samples=1):
        """Sample from the conditional distribution.

        Use output of decoder at each prediction step as input to the next decoder step.
        Implements greedy decoding and random temperature-controlled sampling.

        Top-k sampling and nucleus sampling are deprecated.

        Args:
            x (Dict): dictionary of input tensors
            temperature (float): temperature for sampling
            greedy (bool): whether to use greedy decoding
            num_samples (int): number of samples to generate

        Returns:
            predictions (torch.Tensor): of shape [batch_size, pred_len, 1] or shape [batch_size, num_samples, pred_len] if num_samples > 1.
            distribution_parameters (torch.Tensor): of shape [batch_size, pred_len, 1]. Not returned if sampling.
        """
        if self.weather_features:
            time_series_inputs = [
                self.lat_embedding(x["latitude"]),
                self.lon_embedding(x["longitude"]),
                self.building_embedding(x["building_type"]).squeeze(2),
                self.day_of_year_encoding(x["day_of_year"]),
                self.day_of_week_encoding(x["day_of_week"]),
                self.hour_of_day_encoding(x["hour_of_day"]),
                self.weather_embedding(
                    torch.cat([x[ft] for ft in self.weather_features], dim=2)
                ),
                self.power_embedding(x["load"]).squeeze(2),
            ]
        else:
            time_series_inputs = [
                self.lat_embedding(x["latitude"]),
                self.lon_embedding(x["longitude"]),
                self.building_embedding(x["building_type"]).squeeze(2),
                self.day_of_year_encoding(x["day_of_year"]),
                self.day_of_week_encoding(x["day_of_week"]),
                self.hour_of_day_encoding(x["hour_of_day"]),
                self.power_embedding(x["load"]).squeeze(2),
            ]
        time_series_embed = torch.cat(time_series_inputs, dim=2)
        # [batch_size, context_len, d_model]
        src_series_inputs = time_series_embed[:, : self.context_len, :]
        tgt_series_inputs = time_series_embed[:, self.context_len - 1 : -1, :]
        src_series_embed = self.positional_encoding(src_series_inputs)

        encoder_output = self.encoder(src_series_embed)
        decoder_input = tgt_series_inputs[:, 0, :].unsqueeze(1)
        if num_samples > 1 and not greedy:
            # [batch_size, 1, emb_size] --> [batch_size * num_sampes, 1, emb_size]
            decoder_input = decoder_input.repeat_interleave(num_samples, dim=0)
            encoder_output = encoder_output.repeat_interleave(num_samples, dim=0)
        all_preds, all_logits = [], []
        for k in range(1, self.pred_len + 1):
            decoder_embed = self.positional_encoding(decoder_input)
            tgt_mask = self.tgt_mask[:k, :k].to(encoder_output.device)
            decoder_output = self.decoder(
                decoder_embed, encoder_output, tgt_mask=tgt_mask
            )
            # [batch_size, 1] if continuous (2 if head is gaussian_nll) or [batch_size, vocab_size] if not continuous_loads
            outputs = self.logits(decoder_output[:, -1, :])
            all_logits += [outputs.unsqueeze(1)]

            if self.continuous_loads:
                if self.continuous_head == "mse":
                    all_preds += [outputs]
                elif self.continuous_head == "gaussian_nll":
                    if greedy:
                        all_preds += [outputs[:, 0].unsqueeze(1)]  # mean only
                        outputs = all_preds[-1]  # [batch_size, 1, 1]
                    else:
                        mean = outputs[:, 0]
                        std = torch.nn.functional.softplus(outputs[:, 1])
                        outputs = (
                            torch.distributions.normal.Normal(mean, std)
                            .sample()
                            .unsqueeze(1)
                        )
                        all_preds += [outputs]

            elif not greedy:
                # Sample from a Categorical distribution with logits outputs
                all_preds += [
                    torch.multinomial(
                        torch.nn.functional.softmax(outputs / temperature, dim=1), 1
                    )
                ]
                # change outputs to the predicted load tokens
                outputs = all_preds[-1]  # [batch_size * num_samples, 1]
            else:
                # outputs are [batch_size, vocab_size]
                # Greedy decoding
                all_preds += [outputs.argmax(dim=1).unsqueeze(1)]
                # change outputs to the predicted load tokens
                outputs = all_preds[-1]

            # [batch_size, d_model]
            if k < self.pred_len:
                # [batch_size, d_model]
                next_decoder_input = tgt_series_inputs[:, k]
                if num_samples > 1 and not greedy:
                    # [batch_size, d_model] --> [batch_size * num_samples, d_model]
                    next_decoder_input = next_decoder_input.repeat_interleave(
                        num_samples, dim=0
                    )
                # Use the embedding predicted load instead of the ground truth load
                embedded_pred = self.power_embedding(outputs)
                if not self.continuous_loads:
                    # [batch_size, 1, 1, 64*scale] --> [batch_size, 64*scale]
                    embedded_pred = embedded_pred.squeeze(2).squeeze(1)
                next_decoder_input = torch.cat(
                    [next_decoder_input[:, : -embedded_pred.shape[-1]], embedded_pred],
                    dim=1,
                )
                # Append the next decoder input to the decoder input
                decoder_input = torch.cat(
                    [decoder_input, next_decoder_input.unsqueeze(1)], dim=1
                )
        if num_samples == 1 or greedy:
            if self.continuous_head == "gaussian_nll":
                # [batch_size, pred_len, 2]
                gaussian_params = torch.stack(all_logits, 1)[:, :, 0, :]
                means = gaussian_params[:, :, 0]
                sigma = torch.nn.functional.softplus(gaussian_params[:, :, 1])
                return torch.stack(all_preds, 1), torch.cat(
                    [means.unsqueeze(2), sigma.unsqueeze(2)], 2
                )
            else:
                return torch.stack(all_preds, 1), torch.stack(all_logits, 1)[:, :, 0, :]
        else:
            # [batch_size, num_samples, pred_len]
            return torch.stack(all_preds, 1).reshape(-1, num_samples, self.pred_len)


# ----------------------------------------------------------------------
# Quick sanity-check
# ----------------------------------------------------------------------
if __name__ == "__main__":

    # ========= 1. 超参 =========
    cfg = dict(
        # —— 主网络 ——                 # ← 与用户给定的保持一致
        context_len=168,
        pred_len=24,
        num_encoder_layers=6,
        num_decoder_layers=3,
        nhead=8,
        dim_feedforward=2048,
        d_model=512,
        dropout=0.0,
        activation="gelu",
        continuous_loads=True,
        continuous_head="gaussian_nll",
        ignore_spatial=False,
        # —— MoE 相关 ——              # ← 采用之前讨论的推荐值
        num_experts=8,
        top_k=2,
        aux_loss_weight=0.01,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    # ========= 2. 构造模型 =========
    model = LoadForecastingTransformerMoE(**cfg).to(device)
    model.train()  # 打开训练模式以计算辅助 loss

    B = 4  # batch size
    T = cfg["context_len"] + cfg["pred_len"]

    # ========= 3. 造假数据 =========
    dummy = {
        "latitude": torch.rand(B, T, 1, device=device) * 2 - 1,  # [-1,1]
        "longitude": torch.rand(B, T, 1, device=device) * 2 - 1,
        "building_type": torch.zeros(B, T, 1, dtype=torch.long, device=device),
        "day_of_year": torch.rand(B, T, 1, device=device) * 2 - 1,
        "day_of_week": torch.rand(B, T, 1, device=device) * 2 - 1,
        "hour_of_day": torch.rand(B, T, 1, device=device) * 2 - 1,
        "load": torch.rand(B, T, 1, device=device),  # 连续功率值
    }
    # 真实目标 —— 仅预测 future 段
    target = dummy["load"][:, -cfg["pred_len"] :, :]

    # ========= 4. 前向 / 计算损失 =========
    out = model(dummy)  # [B, pred_len, 2] (mean, logσ)
    loss = model.loss(out, target)
    print("Forward OK → loss =", float(loss))

    # ========= 5. 反向 =========
    loss.backward()
    n_grad = sum(p.grad is not None for p in model.parameters())
    print(f"Backward OK → grads on {n_grad} tensors")

    pred_out = model.predict(dummy)  # Tuple 或 Tensor
    if isinstance(pred_out, (tuple, list)):
        preds, params = pred_out
        print(
            "Predict  OK → preds shape:", preds.shape, ", params shape:", params.shape
        )
    else:
        print("Predict  OK → preds shape:", pred_out.shape)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = (
        sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    )
    print(
        f"[Param] total = {total_params:.2f} M , trainable = {trainable_params:.2f} M"
    )
