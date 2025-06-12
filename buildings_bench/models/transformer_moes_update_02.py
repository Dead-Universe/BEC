# HierDSFeedForward

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


import torch.distributed as dist


class HierDSFeedForward(nn.Module):
    """
    改进版两级路由 MoE:
      1) Group-Gate via Gumbel-Softmax (hard-ST) for top-1 group selection；
         推理时纯硬路由 (p_group=1)
      2) Expert-Gate 在每个组内做 softmax for top-k experts，避免全量掩码开销
      3) Shared & expert FFN paths with dropout & bias
      4) 学习型 balance biases via buffer (不参与梯度) + clamp 限幅防止溢出
      5) Persistent frequency buffers + DDP 同步
      6) LayerNorm on inputs for 稳定性
      7) 预先合并 expert 权重 & 预计算 group id
      8) 使用 index_add_ 替换 scatter_add_ 提升性能
    """

    def __init__(
        self,
        model_dim: int,
        hidden_dim: int,
        num_groups: int = 2,
        experts_per_gp: int = 4,
        top_k: int = 2,
        gamma_group: float = 1e-3,
        gamma_expert: float = 1e-3,
        gumbel_tau: float = 1.0,
        dropout_rate: float = 0.1,
        bias_clamp: float = 1.0,
    ):
        super().__init__()
        assert top_k <= experts_per_gp, "top_k must be ≤ experts_per_gp"

        # —— 基本配置 ——
        self.model_dim = model_dim
        self.hidden_dim = hidden_dim
        self.num_groups = num_groups
        self.experts_per_gp = experts_per_gp
        self.num_experts = num_groups * experts_per_gp
        self.top_k = top_k
        self.gamma_group = gamma_group
        self.gamma_expert = gamma_expert
        self.tau = gumbel_tau
        self.bias_clamp = bias_clamp

        # —— 输入归一化 ——
        self.input_norm = nn.LayerNorm(model_dim)

        # —— Shared FFN path ——
        self.shared_in = nn.Linear(model_dim, hidden_dim * 2, bias=False)
        self.shared_out = nn.Linear(hidden_dim, model_dim, bias=True)

        # —— Expert-specific FFN path ——
        self.expert_in = nn.Linear(model_dim, hidden_dim * 2, bias=False)
        self.expert_out_weight = nn.Parameter(
            torch.empty(self.num_experts, model_dim, hidden_dim)
        )
        nn.init.xavier_uniform_(self.expert_out_weight)
        # 专家输出 bias
        self.expert_out_bias = nn.Parameter(torch.zeros(self.num_experts, model_dim))

        # —— Dropout ——
        self.dropout = nn.Dropout(dropout_rate)

        # —— Routing gates ——
        self.group_gate = nn.Linear(model_dim, num_groups, bias=False)
        self.expert_gate = nn.Linear(model_dim, self.num_experts, bias=False)
        # balance biases & freq buffers
        self.register_buffer("group_bias", torch.zeros(num_groups))
        self.register_buffer("expert_bias", torch.zeros(self.num_experts))
        self.register_buffer("group_freq", torch.zeros(num_groups))
        self.register_buffer("expert_freq", torch.zeros(self.num_experts))

    @staticmethod
    def _swiglu(x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=-1)
        return F.silu(a) * b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, C] → out: [B, T, C]
        """
        B, T, C = x.shape
        S = B * T
        flat = x.reshape(S, C)
        flat = self.input_norm(flat)

        # —— (0) Shared FFN ——
        h_shared = self._swiglu(self.shared_in(flat))  # [S, hidden_dim]
        h_shared = self.dropout(h_shared)
        out_shared = self.shared_out(h_shared)  # [S, C]

        # —— (1) Group-Gate (top-1) ——
        g_logits = self.group_gate(flat) + self.group_bias  # [S, G]
        if self.training:
            g_mask = F.gumbel_softmax(
                g_logits, tau=self.tau, hard=True, dim=-1
            )  # [S, G]
            g_probs = F.softmax(g_logits, dim=-1)
            p_group = (g_probs * g_mask).sum(dim=-1, keepdim=True)  # [S,1]
        else:
            g_idx = g_logits.argmax(dim=-1)
            g_mask = F.one_hot(g_idx, num_classes=self.num_groups).float()  # [S, G]
            # 推理时纯硬路由，不做概率缩放
            p_group = torch.ones((S, 1), device=flat.device)

        group_idx = g_mask.argmax(dim=-1)  # [S]

        # —— (2) Expert-Gate within selected group ——
        # 重塑 logits 为 [S, G, E_per_gp]
        e_logits = self.expert_gate(flat).view(S, self.num_groups, self.experts_per_gp)
        expert_bias = self.expert_bias.view(self.num_groups, self.experts_per_gp)
        # 按 token 选出对应组的 logits & bias
        e_logits_sel = (
            e_logits[torch.arange(S), group_idx] + expert_bias[group_idx]
        )  # [S, E_per_gp]
        e_probs_sel = F.softmax(e_logits_sel, dim=-1)  # [S, E_per_gp]

        # 合并 group 概率后选 Top-k
        p_expert_sel = e_probs_sel * p_group  # [S, E_per_gp]
        topv, topi = p_expert_sel.topk(self.top_k, dim=-1)  # [S, top_k]

        # 展平成 dispatch
        tok_idx = (
            torch.arange(S, device=flat.device)
            .unsqueeze(1)
            .repeat(1, self.top_k)
            .view(-1)
        )
        disp_ids = (group_idx.unsqueeze(1) * self.experts_per_gp + topi).view(
            -1
        )  # [N_sel]
        disp_w = topv.view(-1)  # [N_sel]

        # —— (3) Expert FFN & vectorized dispatch ——
        h_expert = self._swiglu(self.expert_in(flat))  # [S, hidden_dim]
        h_expert = self.dropout(h_expert)
        h_sel = h_expert[tok_idx]  # [N_sel, hidden_dim]
        W2_sel = self.expert_out_weight[disp_ids]  # [N_sel, C, hidden_dim]
        exp_out = torch.bmm(W2_sel, h_sel.unsqueeze(-1)).squeeze(-1)  # [N_sel, C]
        # 加上专家输出 bias
        b_sel = self.expert_out_bias[disp_ids]  # [N_sel, C]
        exp_out = (exp_out + b_sel) * disp_w.unsqueeze(1)

        # 累加路由结果
        routed_out = torch.zeros_like(out_shared)  # [S, C]
        routed_out.index_add_(0, tok_idx, exp_out)

        # —— (4) 更新 balance biases & freq buffers + DDP 同步 + clamp ——
        if self.training:
            with torch.no_grad():
                g_hist = g_mask.sum(dim=0) / S  # [G]
                e_hist = torch.bincount(disp_ids, minlength=self.num_experts).float()
                e_hist = e_hist / e_hist.sum()  # [E]
                # 更新 biases
                self.group_bias.add_(
                    self.gamma_group * (g_hist - 1.0 / self.num_groups)
                )
                self.expert_bias.add_(
                    self.gamma_expert * (e_hist - 1.0 / self.num_experts)
                )
                # 限幅防止溢出
                self.group_bias.clamp_(-self.bias_clamp, self.bias_clamp)
                self.expert_bias.clamp_(-self.bias_clamp, self.bias_clamp)
                # 更新 freq buffers
                self.group_freq.copy_(g_hist)
                self.expert_freq.copy_(e_hist)
                # DDP 同步
                if dist.is_initialized():
                    dist.all_reduce(self.group_bias, op=dist.ReduceOp.SUM)
                    self.group_bias /= dist.get_world_size()
                    dist.all_reduce(self.expert_bias, op=dist.ReduceOp.SUM)
                    self.expert_bias /= dist.get_world_size()
                    dist.all_reduce(self.group_freq, op=dist.ReduceOp.SUM)
                    self.group_freq /= dist.get_world_size()
                    dist.all_reduce(self.expert_freq, op=dist.ReduceOp.SUM)
                    self.expert_freq /= dist.get_world_size()

        # —— (5) 合并 Shared & Routed 输出 ——
        out = out_shared + routed_out  # [S, C]
        return out.view(B, T, C)


# ---------------------------------------------------------------------------
# 2.  —— MoE-Encoder / Decoder Layer ——————————
# ---------------------------------------------------------------------------


class MoETransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        dropout,
        num_groups,
        experts_per_gp,
        top_k,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.moe_ffn = HierDSFeedForward(
            d_model,
            dim_feedforward,
            num_groups=num_groups,
            experts_per_gp=experts_per_gp,
            top_k=top_k,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self, src, src_mask=None, src_key_padding_mask=None, is_causal=False, **kw
    ):
        sa, _ = self.self_attn(
            src,
            src,
            src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
        )
        src = self.norm1(src + self.drop(sa))
        src = self.norm2(src + self.drop(self.moe_ffn(src)))
        return src


class MoETransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        dropout,
        num_groups,
        experts_per_gp,
        top_k,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.moe_ffn = HierDSFeedForward(
            d_model,
            dim_feedforward,
            num_groups=num_groups,
            experts_per_gp=experts_per_gp,
            top_k=top_k,
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
        is_causal=False,
        **kw
    ):
        t2, _ = self.self_attn(
            tgt,
            tgt,
            tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
        )
        tgt = self.norm1(tgt + self.drop(t2))

        t2, _ = self.cross_attn(
            tgt,
            memory,
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        tgt = self.norm2(tgt + self.drop(t2))

        tgt = self.norm3(tgt + self.drop(self.moe_ffn(tgt)))
        return tgt


# ---------------------------------------------------------------------------
# 3.  —— Transformer with MoE FFN ——————————
# ---------------------------------------------------------------------------


class LoadForecastingTransformerMoE(BaseModel):
    """
    Encoder-Decoder Transformer with DeepSeek-MoE FFN.
    """

    def __init__(
        self,
        num_groups: int = 2,
        experts_per_gp: int = 4,
        top_k: int = 2,
        context_len: int = 168,
        pred_len: int = 24,
        vocab_size: int = 2274,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
        continuous_loads: bool = False,
        continuous_head: str = "mse",
        ignore_spatial: bool = False,
        weather_inputs: List[str] | None = None,
    ):
        super().__init__(context_len, pred_len, continuous_loads)
        self.context_len = context_len
        self.pred_len = pred_len
        self.vocab_size = vocab_size
        self.continuous_loads = continuous_loads
        self.continuous_head = continuous_head
        self.ignore_spatial = ignore_spatial

        scale = d_model // 256
        self.weather_features = weather_inputs
        if weather_inputs:
            self.weather_embedding = nn.Linear(len(weather_inputs), 64)
            d_model += 64

        enc_layer = MoETransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            num_groups=num_groups,
            experts_per_gp=experts_per_gp,
            top_k=top_k,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_encoder_layers)

        dec_layer = MoETransformerDecoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            num_groups=num_groups,
            experts_per_gp=experts_per_gp,
            top_k=top_k,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_decoder_layers)

        # ----  Prediction head ---------------------------------------------
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

    def forward(self, x: Dict[str, torch.Tensor]):

        # ---- 构造时间序列 embedding ----
        ts_inputs = [
            self.lat_embedding(x["latitude"]),
            self.lon_embedding(x["longitude"]),
            self.building_embedding(x["building_type"]).squeeze(2),
            self.day_of_year_encoding(x["day_of_year"]),
            self.day_of_week_encoding(x["day_of_week"]),
            self.hour_of_day_encoding(x["hour_of_day"]),
            self.power_embedding(x["load"]).squeeze(2),
        ]
        if self.weather_features:
            ts_inputs.insert(
                -1,
                self.weather_embedding(
                    torch.cat([x[ft] for ft in self.weather_features], dim=2)
                ),
            )

        ts_embed = torch.cat(ts_inputs, dim=2)  # [B, T, d_model]

        src_series_inputs = ts_embed[:, : self.context_len, :]
        tgt_series_inputs = ts_embed[:, self.context_len - 1 : -1, :]

        src_series_embed = self.positional_encoding(src_series_inputs)
        tgt_series_embed = self.positional_encoding(tgt_series_inputs)

        memory = self.encoder(src_series_embed)
        outs = self.decoder(tgt_series_embed, memory, tgt_mask=self.tgt_mask)
        return self.logits(outs)

    def loss(self, pred, y):
        """
        task loss only (aux loss 已移除)  pred:[B, pred_len, ...]
        """
        if self.continuous_loads and self.continuous_head == "mse":
            return F.mse_loss(pred, y)
        elif self.continuous_loads and self.continuous_head == "gaussian_nll":
            return F.gaussian_nll_loss(
                pred[:, :, 0].unsqueeze(2),
                y,
                F.softplus(pred[:, :, 1].unsqueeze(2)) ** 2,
            )
        else:
            return F.cross_entropy(
                pred.reshape(-1, self.vocab_size), y.long().reshape(-1)
            )

    def predict(self, x: Dict) -> Tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        return self.generate_sample(x, greedy=True)

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
    torch.manual_seed(0)
    # 模型配置
    cfg = dict(
        num_groups=2,
        experts_per_gp=4,
        top_k=2,
        context_len=168,
        pred_len=24,
        num_encoder_layers=2,
        num_decoder_layers=1,
        d_model=256,
        dim_feedforward=512,
        continuous_loads=True,
        continuous_head="mse",
    )

    # 创建模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LoadForecastingTransformerMoE(**cfg).to(device).train()

    # 创建假数据 (B=2, T=192: 168 + 24)
    B = 2
    T = cfg["context_len"] + cfg["pred_len"]

    dummy = {
        "latitude": torch.rand(B, T, 1, device=device) * 2 - 1,
        "longitude": torch.rand(B, T, 1, device=device) * 2 - 1,
        "building_type": torch.zeros(B, T, 1, dtype=torch.long, device=device),
        "day_of_year": torch.rand(B, T, 1, device=device) * 2 - 1,
        "day_of_week": torch.rand(B, T, 1, device=device) * 2 - 1,
        "hour_of_day": torch.rand(B, T, 1, device=device) * 2 - 1,
        "load": torch.rand(B, T, 1, device=device),
    }
    target = dummy["load"][:, -cfg["pred_len"] :, :]

    out = model(dummy)
    loss = model.loss(out, target)
    loss.backward()

    print("sanity-check OK – loss:", float(loss))

    print("Group freq:", model.encoder.layers[0].moe_ffn.group_freq.cpu())
    print("Expert freq:", model.encoder.layers[0].moe_ffn.expert_freq.cpu())

    dummy_input = {
        "latitude": torch.rand(B, T, 1, device=device) * 2 - 1,  # 假设的纬度
        "longitude": torch.rand(B, T, 1, device=device) * 2 - 1,  # 假设的经度
        "building_type": torch.zeros(
            B, T, 1, dtype=torch.long, device=device
        ),  # 假设的建筑类型
        "day_of_year": torch.rand(B, T, 1, device=device) * 2 - 1,  # 假设的日期
        "day_of_week": torch.rand(B, T, 1, device=device) * 2 - 1,  # 假设的星期几
        "hour_of_day": torch.rand(B, T, 1, device=device) * 2 - 1,  # 假设的小时
        "load": torch.rand(B, T, 1, device=device),  # 假设的负荷数据
    }

    # 调用模型的predict方法
    model.eval()  # 设置为评估模式
    predictions, _ = model.predict(dummy_input)

    # 输出结果
    print("Predictions Shape:", predictions.shape)
    # print("Predictions:", predictions)
