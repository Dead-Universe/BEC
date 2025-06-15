# HierDSFeedForward

import math
import torch
from typing import Tuple, Dict, List
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
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


class HierDSFeedForward(nn.Module):
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
        self.expert_out_bias = nn.Parameter(torch.zeros(self.num_experts, model_dim))

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
        B, T, C = x.shape
        S = B * T
        flat = x.reshape(S, C)
        flat = self.input_norm(flat)

        # (0) Shared FFN
        h_shared = self._swiglu(self.shared_in(flat))
        h_shared = self.dropout(h_shared)
        out_shared = self.shared_out(h_shared)

        # (1) Group-Gate
        g_logits = self.group_gate(flat) + self.group_bias  # (S, G)
        if self.training:
            # Gumbel-Softmax 硬化采样得到 one-hot
            g_mask = F.gumbel_softmax(g_logits, tau=self.tau, hard=True, dim=-1)
            # 真正的组概率
            p_group_all = F.softmax(g_logits, dim=-1)  # (S, G)
            # 根据采样后选中的组，gather 出单个标量概率
            g_idx = g_mask.argmax(dim=-1, keepdim=True)  # (S, 1)
            p_group = p_group_all.gather(1, g_idx)  # (S, 1)
        else:
            g_idx = g_logits.argmax(dim=-1)  # (S,)
            g_mask = F.one_hot(g_idx, self.num_groups).float()
            p_group = torch.ones((S, 1), device=flat.device)

        group_idx = g_mask.argmax(dim=-1)  # (S,)

        # (2) Expert-Gate
        # reshape to (S, G, E_g)
        e_logits = self.expert_gate(flat).view(S, self.num_groups, self.experts_per_gp)
        expert_bias = self.expert_bias.view(self.num_groups, self.experts_per_gp)
        # select the logits & bias of the chosen group per token
        e_logits_sel = (
            e_logits[torch.arange(S), group_idx] + expert_bias[group_idx]
        )  # (S, E_g)
        e_probs_sel = F.softmax(e_logits_sel, dim=-1)  # (S, E_g)

        # 混合权重：组概率 * 组内专家概率
        p_expert = p_group * e_probs_sel  # (S, E_g)
        topv, topi = p_expert.topk(self.top_k, dim=-1)  # each (S, K)
        # 归一化 top-k 权重
        topv = topv / (topv.sum(dim=-1, keepdim=True) + 1e-8)

        # 计算 dispatch indices & weights
        tok_idx = (
            torch.arange(S, device=flat.device)
            .unsqueeze(-1)
            .expand(-1, self.top_k)
            .reshape(-1)
        )
        disp_ids = (group_idx.unsqueeze(-1) * self.experts_per_gp + topi).reshape(-1)
        disp_w = topv.reshape(-1)  # (S*K,)

        # (3) Expert FFN
        h_expert = self._swiglu(self.expert_in(flat))
        h_expert = self.dropout(h_expert)
        h_sel = h_expert[tok_idx]  # (S*K, D)
        W2_sel = self.expert_out_weight[disp_ids]  # (S*K, C, D)
        b_sel = self.expert_out_bias[disp_ids]  # (S*K, C)

        # 确保梯度流过 expert path
        exp_out = torch.bmm(W2_sel, h_sel.unsqueeze(-1)).squeeze(-1)  # (S*K, C)
        weighted_out = exp_out * disp_w.unsqueeze(-1) + b_sel  # (S*K, C)

        # 聚合回 routed_out
        routed_out = torch.zeros_like(out_shared)  # (S, C)
        routed_out.index_add_(0, tok_idx, weighted_out)  # preserve grad

        # (4) 负载均衡统计 & bias 更新
        if self.training:
            with torch.no_grad():
                # per-process 统计
                g_count = g_mask.sum(dim=0)  # (G,)
                e_flat_idx = disp_ids
                e_count = torch.bincount(e_flat_idx, minlength=self.num_experts).float()
                # 全局同步
                if dist.is_initialized():
                    buf = torch.cat([g_count, e_count], dim=0).to(flat.device)
                    dist.all_reduce(buf, op=dist.ReduceOp.SUM)
                    g_count, e_count = buf.split([self.num_groups, self.num_experts])
                    world_size = dist.get_world_size()
                    total_tokens = S * world_size
                else:
                    total_tokens = S
                # 归一化频率
                g_global = g_count / total_tokens
                e_global = e_count / (total_tokens * self.top_k)

                # 更新 bias，并 clamp
                self.group_bias += self.gamma_group * (g_global - 1.0 / self.num_groups)
                self.expert_bias += self.gamma_expert * (
                    e_global - 1.0 / self.num_experts
                )
                self.group_bias.clamp_(-self.bias_clamp, self.bias_clamp)
                self.expert_bias.clamp_(-self.bias_clamp, self.bias_clamp)

                # 更新 buffer
                self.group_freq.copy_(g_global)
                self.expert_freq.copy_(e_global)

        # (5) 合并输出
        out = out_shared + routed_out
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
