from typing import Dict, Literal, Optional, Tuple
import numpy as np
from pydantic import BaseModel as PyBaseModel, ConfigDict
from buildings_bench.models.base_model import BaseModel
import torch
from torch import nn
import torch.nn.functional as F

import math

from buildings_bench.models.model_kernel import apply_init
from buildings_bench.models.rope import RoPEMultiheadAttention


class ModelArgs(PyBaseModel):
    """
    模型超参数配置类，集中管理所有网络级别的可配置参数。

    Attributes:
        max_seq_len (int): 最大序列长度
        dim (int): 模型隐藏层维度
        inter_dim (int): MLP 中间层维度
        moe_inter_dim (int): MoE 中间层维度
        n_encoder_layers (int): 编码器层数量
        n_decoder_layers (int): 解码器层数量
        n_dense_layers (int): Dense 层数量
        n_heads (int): 注意力头数
        n_routed_experts (int): MoE 路由专家数量
        n_shared_experts (int): MoE 共享专家数量
        n_activated_experts (int): MoE 激活专家数量
        n_expert_groups (int): 专家分组数
        n_limited_groups (int): 路由限制组数
        score_func (Literal["softmax", "sigmoid"]): 路由评分函数
        route_scale (float): 路由评分缩放因子
        init_method (Literal[...]): 权重初始化方法
        init_gain (float): 权重初始化增益
        building_type (int): 建筑类型数量（用于嵌入）
        context_len (int): 上下文长度（用于时间序列预测）
        pred_len (int): 预测长度（用于时间序列预测）
    """

    max_seq_len: int = 168 * 4
    dim: int = 256
    inter_dim: int = 1024
    moe_inter_dim: int = 512
    n_encoder_layers: int = 8
    n_decoder_layers: int = 8
    n_dense_layers: int = 1
    n_heads: int = 16
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    n_activated_experts: int = 4
    n_expert_groups: int = 2
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.0
    init_method: Literal[
        "kaiming_uniform",
        "kaiming_normal",
        "xavier_uniform",
        "xavier_normal",
        "normal",
        "zeros",
    ] = "kaiming_uniform"
    init_gain: float = 1.0
    building_type: int = 2
    context_len: int = 168
    pred_len: int = 24

    model_config = ConfigDict(
        extra="ignore",  # 多余键直接忽略；改成 "forbid" 可强制报错
        validate_assignment=True,  # 运行时修改字段也会做校验/转换
    )


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


class MoEPositionalEncoding(nn.Module):
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
        super(MoEPositionalEncoding, self).__init__()
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


class RMSNorm(nn.Module):
    """
    RMSNorm —— 根均方归一化层

    Args:
        dim (int): 特征维度
        eps (float): 数值稳定性 epsilon
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        # 缩放参数初始化为 1
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        """
        调用 PyTorch 内置 rms_norm 完成归一化
        """
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)


class Gate(nn.Module):
    """
    MoE 路由 Gate

    Args:
        args (ModelArgs): 全局超参数配置
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.n_groups = args.n_expert_groups
        self.topk_groups = args.n_limited_groups
        self.score_func = args.score_func
        self.route_scale = args.route_scale

        # 用于生成路由得分的线性权重和可选 bias
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        self.bias = (
            nn.Parameter(torch.zeros(args.n_routed_experts))
            if self.dim == 7168
            else None
        )

        # 内部的投影层
        self.router = nn.Linear(
            in_features=args.dim,
            out_features=args.n_routed_experts,
            bias=False,
        )

        # 初始化 Gate 自身的 weight
        self.reset_parameters()

    def reset_parameters(self):
        """初始化 Gate 权重"""
        apply_init(self.weight, self.args.init_method, self.args.init_gain)
        # bias 已在构造时置零

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回 routing weights 和 experts indices
        """
        scores = self.router(x)
        # 评分函数
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = scores.sigmoid()
        original_scores = scores.clone()

        # 加 bias
        if self.bias is not None:
            scores = scores + self.bias

        # 多组路由逻辑
        if self.n_groups > 1:
            scores = scores.reshape(x.size(0), self.n_groups, -1)
            if self.bias is None:
                group_scores = scores.amax(dim=-1)
            else:
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = scores.new_ones(x.size(0), self.n_groups, dtype=bool).scatter_(
                1, indices, False
            )
            scores = scores.masked_fill(mask.unsqueeze(-1), float("-inf")).flatten(1)

        # 最终 top-k 选专家
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices).clone()

        # sigmoid 下归一化
        if self.score_func == "sigmoid":
            weights = weights / weights.sum(dim=-1, keepdim=True)

        weights = weights * self.route_scale
        return weights.type_as(x), indices


class MLP(nn.Module):
    """
    MLP 模块，包含两个线性层和一个激活函数

    Args:
        dim (int): 输入/输出维度
        inter_dim (int): 隐藏维度
    """

    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        # 三个线性子模块，共享同一份 model_args
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        gate: w1 + silu, transform: w3, 最终融合并 w2 投影回输出空间
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Expert(MLP):
    """
    Mixture-of-Experts 中的 Expert 层

    Args:
        dim (int): 输入/输出维度
        inter_dim (int): 隐藏维度
    """


class MoE(nn.Module):
    """
    Mixture-of-Experts (MoE) module.

    Attributes:
        dim (int): Dimensionality of input features.
        n_routed_experts (int): Total number of experts in the model.
        n_local_experts (int): Number of experts handled locally in distributed systems.
        n_activated_experts (int): Number of experts activated for each input.
        gate (nn.Module): Gating mechanism to route inputs to experts.
        experts (nn.ModuleList): List of expert modules.
        shared_experts (nn.Module): Shared experts applied to all inputs.
    """

    def __init__(self, args: ModelArgs):
        """
        Initializes the MoE module.

        Args:
            args (ModelArgs): Model arguments containing MoE parameters.
        """
        super().__init__()
        self.dim = args.dim
        self.n_routed_experts = args.n_routed_experts
        self.gate = Gate(args)
        self.experts = nn.ModuleList(
            [Expert(args.dim, args.moe_inter_dim) for _ in range(self.n_routed_experts)]
        )
        self.shared_experts = Expert(
            args.dim, args.n_shared_experts * args.moe_inter_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MoE module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert routing and computation.
        """
        shape = x.size()
        x = x.reshape(-1, self.dim)
        weights, indices = self.gate(x)
        y = torch.zeros_like(x)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts)
        for i in range(self.n_routed_experts):
            if counts[i] == 0:
                continue
            idx, top = torch.where(indices == i)
            expert = self.experts[i]
            y[idx] += expert(x[idx]) * weights[idx, top, None]
        z = self.shared_experts(x)
        return (y + z).reshape(shape)


class Encoder(nn.Module):
    """
    Transformer Encoder Block

    Attributes:
        attn (nn.Module): Attention layer (MLA).
        ffn (nn.Module): Feed-forward network (MLP or MoE).
        attn_norm (nn.Module): Layer normalization for attention.
        ffn_norm (nn.Module): Layer normalization for feed-forward network.
    """

    def __init__(self, layer_id: int, args: ModelArgs):
        """
        Initializes the Transformer block.

        Args:
            layer_id (int): Layer index in the transformer.
            args (ModelArgs): Model arguments containing block parameters.
        """
        super().__init__()
        self.self_attn = RoPEMultiheadAttention(
            embed_dim=args.dim, num_heads=args.n_heads, dropout=0.0
        )
        self.ffn = (
            MLP(args.dim, args.inter_dim)
            if layer_id < args.n_dense_layers
            else MoE(args)
        )

        self.attn_norm = RMSNorm(args.dim)
        self.ffn_norm = RMSNorm(args.dim)
        self.ans_norm = RMSNorm(args.dim)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass for the Transformer block.

        Args:
            src (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor after block computation.
        """
        q = k = v = self.attn_norm(src)
        h, _ = self.self_attn(
            q,
            k,
            v,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
        )
        y = src + h
        z = y + self.ffn(self.ffn_norm(y))
        return self.ans_norm(z)  # [B, S, D] 输出


class Decoder(nn.Module):
    """


    Attributes:

    """

    def __init__(self, layer_id: int, args: ModelArgs):
        """
        Initializes the Transformer block.

        Args:
            layer_id (int): Layer index in the transformer.
            args (ModelArgs): Model arguments containing block parameters.
        """
        super().__init__()
        self.norm1 = RMSNorm(args.dim)
        self.self_attn = RoPEMultiheadAttention(
            embed_dim=args.dim, num_heads=args.n_heads, dropout=0.0
        )
        self.ffn = (
            MLP(args.dim, args.inter_dim)
            if layer_id < args.n_dense_layers
            else MoE(args)
        )
        self.norm2 = RMSNorm(args.dim)
        self.multihead_attn = RoPEMultiheadAttention(
            embed_dim=args.dim, num_heads=args.n_heads, dropout=0.0
        )
        self.ffn_norm = RMSNorm(args.dim)
        self.ans_norm = RMSNorm(args.dim)

    def forward(
        self,
        tgt: torch.Tensor,  # [B, S_tgt, D]
        memory: torch.Tensor,  # [B, S_src, D] — 编码器输出
        tgt_mask: Optional[torch.Tensor] = None,  # causal mask
        memory_mask: Optional[torch.Tensor] = None,  # padding mask (可选)
        tgt_key_padding_mask: Optional[
            torch.Tensor
        ] = None,  # padding mask for tgt (可选)
        memory_key_padding_mask: Optional[
            torch.Tensor
        ] = None,  # padding mask for memory (可选
        tgt_is_causal: bool = True,  # 是否使用因果注意力
        memory_is_causal: bool = False,  # 是否使用因果注意力
    ) -> torch.Tensor:
        """
        Forward pass for the Transformer block.

        Args:
            tgt (torch.Tensor): Target tensor of shape [B, S_tgt, D].
            memory (torch.Tensor): Memory tensor of shape [B, S_src, D].
            tgt_mask (Optional[torch.Tensor]): Mask for target tensor.
            memory_mask (Optional[torch.Tensor]): Mask for memory tensor.

        Returns:
            torch.Tensor: Output tensor after block computation.
        """
        q = k = v = self.norm1(tgt)
        attn_out, _ = self.self_attn(
            q,
            k,
            v,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False,
            is_causal=tgt_is_causal,
        )
        y = tgt + attn_out
        q = self.norm2(y)
        k = v = memory
        cross_out, _ = self.multihead_attn(
            q,
            k,
            v,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
            is_causal=memory_is_causal,
        )
        z = y + cross_out
        out = z + self.ffn(self.ffn_norm(z))
        return self.ans_norm(out)  # [B, S_tgt, D] 输出


# ──────────────────────────────────────────────
# 1.  通用位置编码
# ──────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len)[:, None]
        rate = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10_000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * rate)
        pe[:, 1::2] = torch.cos(pos * rate)
        self.register_buffer("pe", pe)  # [max_len, D]

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """
        Args:
            x         : [B, T, D]
            start_pos : 绝对位置偏移，默认 0
        """
        return x + self.pe[start_pos : start_pos + x.size(1)]


# ──────────────────────────────────────────────
# 2.  单字段 → 向量 的小工具
# ──────────────────────────────────────────────
def make_linear(in_dim, out_dim):  # 简写
    return nn.Sequential(nn.Linear(in_dim, out_dim), nn.GELU())


class PeriodicEmbed(nn.Module):
    """把 [-1,1] 周期特征 → SinCos → Linear"""

    def __init__(self, out_dim):
        super().__init__()
        self.proj = nn.Linear(2, out_dim)

    def forward(self, x):  # x: [B,T,1]
        x = torch.cat([torch.sin(math.pi * x), torch.cos(math.pi * x)], dim=-1)
        return self.proj(x)
