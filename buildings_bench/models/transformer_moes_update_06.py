# RoPE位置编码

import math
import torch
from typing import Tuple, Dict, List
from torch import nn
import torch.nn.functional as F
import numpy as np
from buildings_bench.models.base_model import BaseModel


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Args:
        dim (int): Dimension of the input tensor.
        eps (float): Epsilon value for numerical stability. Defaults to 1e-6.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        """
        Forward pass for RMSNorm.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor with the same shape as input.
        """
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)


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


class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE).
    将序列位置编码直接融入 Q/K，使得注意力天然具有位置感知能力。
    """

    def __init__(self, dim: int, base: int = 10000):
        """
        Args:
            dim: 每个 head 上的隐藏维度 d_head。
            base: 频率基底，通常设为 10000。
        """
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len: int, device: torch.device):
        """
        生成 sin/cos 张量，形状均为 [1, 1, seq_len, dim].
        """
        positions = torch.arange(seq_len, device=device).float()
        sinusoid_inp = torch.einsum("i,j->ij", positions, self.inv_freq)
        sin = sinusoid_inp.sin().unsqueeze(0).unsqueeze(0)  # [1,1,seq_len,dim/2]
        cos = sinusoid_inp.cos().unsqueeze(0).unsqueeze(0)
        # 之后在每个 head 上重复使用
        return sin, cos

    @staticmethod
    def apply_rotary(
        x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor
    ) -> torch.Tensor:
        """
        对 x 应用旋转：x[..., ::2] 和 x[..., 1::2] 交替成对旋转。
        x: [B, num_heads, seq_len, d_head]
        sin, cos: [1,1,seq_len,d_head/2]
        返回同样形状的张量。
        """
        # 拆成奇偶两份
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        # 旋转变换
        x_rot = torch.stack(
            [x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1
        )  # [B, h, T, d_head/2, 2]
        # 把最后两维展平回 d_head
        return x_rot.flatten(-2)


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
    """
    Two-level router:  ➜ Group-Gate (top-1)  ➜ Expert-Gate (top-k in group)
    """

    def __init__(
        self,
        model_dim: int,
        hidden_dim: int,
        num_groups: int = 2,
        experts_per_gp: int = 4,
        top_k: int = 2,
        gamma_group: float = 1e-3,
        gamma_exp: float = 1e-3,
    ):
        super().__init__()
        self.num_groups = num_groups
        self.experts_per_gp = experts_per_gp
        self.num_experts = num_groups * experts_per_gp
        self.top_k = top_k
        self.gamma_group = gamma_group
        self.gamma_exp = gamma_exp

        # —— Shared & Private FFN ------------------------------------------------
        self.shared_in = nn.Linear(model_dim, hidden_dim * 2, bias=False)
        self.shared_out = nn.Linear(hidden_dim, model_dim, bias=False)

        # W1 共享，W2 私有
        self.w1_shared = nn.Linear(model_dim, hidden_dim * 2, bias=False)
        self.w2_expert = nn.ModuleList(
            [
                nn.Linear(hidden_dim, model_dim, bias=False)
                for _ in range(self.num_experts)
            ]
        )

        # —— Router -------------------------------------------------------------
        self.group_gate = nn.Linear(model_dim, num_groups, bias=False)
        self.expert_gate = nn.Linear(model_dim, self.num_experts, bias=False)

        # bias 和频率跟踪不使用 .data 更新
        self.group_bias = nn.Parameter(torch.zeros(num_groups))
        self.expert_bias = nn.Parameter(torch.zeros(self.num_experts))
        self.register_buffer("group_freq", torch.zeros(num_groups), persistent=False)
        self.register_buffer(
            "expert_freq", torch.zeros(self.num_experts), persistent=False
        )

    @staticmethod
    def _swiglu(x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, -1)
        return F.silu(a) * b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, C] → out: [B, T, C]
        """
        B, T, C = x.shape
        S = B * T
        flat = x.view(S, C)  # [S, C]

        # —— (0) Shared expert path ——
        hid = self._swiglu(self.shared_in(flat))  # [S, 2H]
        shared_out = self.shared_out(hid)  # [S, C]

        # —— (1) Group Gate (top-1) ——
        g_logits = self.group_gate(flat) + self.group_bias  # [S, G]
        g_probs = F.softmax(g_logits, dim=-1)  # P(g)  [S, G]
        g_idx = torch.argmax(g_probs, dim=-1)  # [S]

        # —— (2) Expert Gate (within selected group) ——
        e_logits = self.expert_gate(flat) + self.expert_bias  # [S, E]
        device = flat.device

        # 构造 mask，只保留每个 token 所在组内的 experts
        # offsets: [S,1] × experts_per_gp → [S, experts_per_gp] 专家在全局的 index
        offsets = g_idx.unsqueeze(1) * self.experts_per_gp
        idx_in_group = offsets + torch.arange(
            self.experts_per_gp, device=device
        )  # [S, E_gp]

        # scatter 构造 masked_logits
        neg_inf = float("-inf")
        masked_logits = torch.full_like(e_logits, neg_inf)
        masked_logits = masked_logits.scatter(
            dim=1, index=idx_in_group, src=e_logits.gather(1, idx_in_group)
        )  # [S, E], 组外 logits 都是 -inf

        # 条件概率 P(e | g)
        e_probs = F.softmax(masked_logits, dim=-1)  # [S, E]

        # 提取对应 token 的 P(g)
        g_prob = g_probs.gather(1, g_idx.unsqueeze(1))  # [S, 1]

        # 联合概率 P(e) = P(g) * P(e|g)
        p_expert = e_probs * g_prob  # [S, E]

        # top-k 专家及其联合概率
        topv, topi = p_expert.topk(self.top_k, dim=-1)  # both [S, top_k]

        # 路由权重 = P(e) * γ  （如果用 sigmoid，可在此做归一化）
        weights = topv * 1.0  # route_scale 可乘在这里

        # ---- (3) Expert FFN & 合并 ----
        disp_ids = topi.flatten()  # [S*top_k]
        disp_w = weights.flatten()  # [S*top_k]
        tok_idx = torch.arange(S, device=device).repeat_interleave(self.top_k)

        routed_out = torch.zeros_like(shared_out)
        for eid in range(self.num_experts):
            m = disp_ids == eid
            if not m.any():
                continue
            sel_tok = tok_idx[m]
            sel_in = flat[sel_tok]
            # 共享的第一层
            h = self._swiglu(self.w1_shared(sel_in))
            # 专家的第二层
            exp_out = self.w2_expert[eid](h) * disp_w[m].unsqueeze(1)
            routed_out.index_add_(0, sel_tok, exp_out)

        # —— (4) 跟踪频率，仅在 training 时累积 EMA，不使用 .data 操作 ——
        if self.training:
            # 组频率
            g_hist = torch.bincount(g_idx, minlength=self.num_groups).float()
            g_hist = g_hist / g_hist.sum()
            self.group_bias.grad = None  # 清除旧梯度
            self.group_bias.data += self.gamma_group * (g_hist - 1.0 / self.num_groups)
            self.group_freq.copy_(g_hist.detach())

            # 专家频率
            disp_ids_all = disp_ids
            e_hist = torch.bincount(disp_ids_all, minlength=self.num_experts).float()
            e_hist = e_hist / e_hist.sum()
            self.expert_bias.grad = None
            self.expert_bias.data += self.gamma_exp * (e_hist - 1.0 / self.num_experts)
            self.expert_freq.copy_(e_hist.detach())

        # 合并 shared 与 routed
        out = shared_out + routed_out  # [S, C]
        return out.view(B, T, C)


# ---------------------------------------------------------------------------
# 2.  —— Latent Attention ——————————
# ---------------------------------------------------------------------------


class LatentAttention(nn.Module):
    """
    Multi-Head Latent Attention  (DeepSeek-V3 / Flash-V3 核心)
    - 把 K,V 先投影到 d_c 维潜空间，再重建到 dh 维
    - 支持 is_causal / attn_mask / key_padding_mask
    """

    def __init__(
        self, d_model: int, nhead: int, d_compress: int = None, dropout: float = 0.0
    ):
        super().__init__()
        # 新增：告诉 Transformer 这里是 batch-first 模式
        self.batch_first = True

        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.d_c = d_compress or (d_model // 4)

        # Q 投影保持不变
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        # K,V 先进低维，再重建
        self.Wk_c = nn.Linear(d_model, self.d_c, bias=False)
        self.Wv_c = nn.Linear(d_model, self.d_c, bias=False)
        self.Wk = nn.Linear(self.d_c, d_model, bias=False)
        self.Wv = nn.Linear(self.d_c, d_model, bias=False)

        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        # RoPE
        self.rotary_emb = RotaryEmbedding(self.d_head)

    # ------------------------------
    def _shape(self, x: torch.Tensor, seq_len: int, bsz: int):
        # [B, T, d] -> [B, h, T, dh]
        return x.view(bsz, seq_len, self.nhead, self.d_head).transpose(1, 2)

    # ------------------------------
    def forward(
        self,
        q,
        k,
        v,
        attn_mask=None,
        key_padding_mask=None,
        is_causal: bool = False,
        need_weights=False,
    ):
        B, Tq, _ = q.shape
        Tk = k.size(1)

        q = self._shape(self.Wq(q), Tq, B)  # [B,h,Tq,dh]
        # ① 压缩
        k_c = self.Wk_c(k)  # [B,Tk,d_c]
        v_c = self.Wv_c(v)
        # ② 重建到 d
        k = self._shape(self.Wk(k_c), Tk, B)  # [B,h,Tk,dh]
        v = self._shape(self.Wv(v_c), Tk, B)

        sin_q, cos_q = self.rotary_emb(Tq, q.device)
        sin_k, cos_k = self.rotary_emb(Tk, k.device)
        q = self.rotary_emb.apply_rotary(q, sin_q, cos_q)
        k = self.rotary_emb.apply_rotary(k, sin_k, cos_k)

        # scaled dot-product
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
            self.d_head
        )  # [B,h,Tq,Tk]

        # masks
        if attn_mask is not None:
            attn += attn_mask.unsqueeze(0)
        if key_padding_mask is not None:
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )
        if is_causal and q.size(-2) != k.size(-2):
            raise ValueError("decoder cross-attn shouldn't be causal")
        if is_causal and Tq == Tk:  # square causal
            causal_mask = torch.triu(torch.ones(Tq, Tk, device=attn.device), 1)
            attn = attn.masked_fill(causal_mask == 1, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # [B,h,Tq,dh]
        out = out.transpose(1, 2).contiguous().view(B, Tq, self.d_model)
        out = self.out(out)
        if need_weights:
            return out, attn.mean(1)
        return out, None


# ---------------------------------------------------------------------------
# 3.  —— MoE-Encoder / Decoder Layer ——————————
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
        d_compress_ratio: int = 4,
    ):
        super().__init__()
        self.self_attn = LatentAttention(
            d_model,
            nhead,
            d_compress=d_model // d_compress_ratio,
            dropout=dropout,
        )
        self.moe_ffn = HierDSFeedForward(
            d_model,
            dim_feedforward,
            num_groups=num_groups,
            experts_per_gp=experts_per_gp,
            top_k=top_k,
        )
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
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
        d_compress_ratio: int = 4,
    ):
        super().__init__()
        self.self_attn = LatentAttention(
            d_model,
            nhead,
            d_compress=d_model // d_compress_ratio,
            dropout=dropout,
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
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.norm3 = RMSNorm(d_model)
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
# 4.  —— Transformer with MoE FFN ——————————
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
        d_compress_ratio: int = 4,  # 压缩比例，默认 4
        # ===== MTP 超参 =====
        mtp_k: int = 1,  # 同时预测未来 k 步
        mtp_lambda: float = 0.7,  # 主 loss 权重,0.7–0.9 常为好值。
    ):
        super().__init__(context_len, pred_len, continuous_loads)
        self.context_len = context_len
        self.pred_len = pred_len
        self.vocab_size = vocab_size
        self.continuous_loads = continuous_loads
        self.continuous_head = continuous_head
        self.ignore_spatial = ignore_spatial

        self.mtp_k = mtp_k
        self.mtp_lambda = mtp_lambda

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
            d_compress_ratio=d_compress_ratio,
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
            d_compress_ratio=d_compress_ratio,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_decoder_layers)

        # —— Prediction heads —— 主 head + MTP 辅助 head
        if self.continuous_loads:
            out_dim = 1 if continuous_head == "mse" else 2
            self.logits = nn.Linear(d_model, out_dim)
            self.logits_mtp = nn.Linear(d_model, out_dim)
            self.power_embedding = nn.Linear(1, 64 * scale)
        else:
            self.logits = nn.Linear(d_model, vocab_size)
            self.logits_mtp = nn.Linear(d_model, vocab_size)
            self.power_embedding = TokenEmbedding(vocab_size, 64 * scale)

        self.tgt_mask = nn.Transformer.generate_square_subsequent_mask(pred_len)
        self.building_embedding = nn.Embedding(2, 32 * scale)
        self.lat_embedding = nn.Linear(1, 32 * scale)
        self.lon_embedding = nn.Linear(1, 32 * scale)
        if ignore_spatial:
            self.lat_embedding = ZeroEmbedding(32 * scale)
            self.lon_embedding = ZeroEmbedding(32 * scale)

        self.day_of_year_encoding = TimeSeriesSinusoidalPeriodicEmbedding(32 * scale)
        self.day_of_week_encoding = TimeSeriesSinusoidalPeriodicEmbedding(32 * scale)
        self.hour_of_day_encoding = TimeSeriesSinusoidalPeriodicEmbedding(32 * scale)

    def to(self, device):
        self.tgt_mask = self.tgt_mask.to(device)
        return super().to(device)

    def forward(self, x: Dict[str, torch.Tensor]):

        # —— 构造时间序列 embedding ——
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

        src = ts_embed[:, : self.context_len, :]
        tgt = ts_embed[:, self.context_len - 1 : -1, :]

        memory = self.encoder(src)
        outs = self.decoder(
            tgt, memory, tgt_mask=self.tgt_mask
        )  # [B, pred_len, d_model]

        # —— 主 logits ——
        logits = self.logits(outs)  # [B, pred_len, out_dim]

        # —— MTP 辅助 logits ——
        if self.training and self.mtp_k > 0:
            aux_hidden = outs[:, : -self.mtp_k, :]  # [B, pred_len-mtp_k, d_model]
            self._aux_logits = self.logits_mtp(
                aux_hidden
            )  # [B, pred_len-mtp_k, out_dim]

        return logits

    def loss(self, pred: torch.Tensor, y: torch.Tensor):
        """
        计算主 loss 和 MTP 辅助 loss，再按 mtp_lambda 混合返回
        pred: [B, pred_len, out_dim]
        y:    [B, pred_len, ...]
        """
        # — 主 Loss —
        if self.continuous_loads and self.continuous_head == "mse":
            loss_main = F.mse_loss(pred, y)
        elif self.continuous_loads and self.continuous_head == "gaussian_nll":
            loss_main = F.gaussian_nll_loss(
                pred[:, :, 0].unsqueeze(2),
                y,
                F.softplus(pred[:, :, 1].unsqueeze(2)) ** 2,
            )
        else:
            loss_main = F.cross_entropy(
                pred.reshape(-1, self.vocab_size), y.long().reshape(-1)
            )

        # — 辅助 MTP Loss —
        if self.mtp_k > 0:
            aux = self._aux_logits  # [B, pred_len-mtp_k, out_dim]
            target2 = y[:, self.mtp_k :]  # 真实 t+2...t+pred_len
            if self.continuous_loads and self.continuous_head == "mse":
                loss_aux = F.mse_loss(aux, target2)
            elif self.continuous_loads and self.continuous_head == "gaussian_nll":
                loss_aux = F.gaussian_nll_loss(
                    aux[:, :, 0].unsqueeze(2),
                    target2,
                    F.softplus(aux[:, :, 1].unsqueeze(2)) ** 2,
                )
            else:
                loss_aux = F.cross_entropy(
                    aux.reshape(-1, self.vocab_size), target2.long().reshape(-1)
                )
            # 混合
            return self.mtp_lambda * loss_main + (1.0 - self.mtp_lambda) * loss_aux

        return loss_main

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

        encoder_output = self.encoder(src_series_inputs)
        decoder_input = tgt_series_inputs[:, 0, :].unsqueeze(1)
        if num_samples > 1 and not greedy:
            # [batch_size, 1, emb_size] --> [batch_size * num_sampes, 1, emb_size]
            decoder_input = decoder_input.repeat_interleave(num_samples, dim=0)
            encoder_output = encoder_output.repeat_interleave(num_samples, dim=0)
        all_preds, all_logits = [], []
        for k in range(1, self.pred_len + 1):

            tgt_mask = self.tgt_mask[:k, :k].to(encoder_output.device)
            decoder_output = self.decoder(
                decoder_input, encoder_output, tgt_mask=tgt_mask
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
