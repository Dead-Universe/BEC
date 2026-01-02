from __future__ import annotations
import math
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed
import torch.nn.functional as F
from buildings_bench.models.rope import RoPEMultiheadAttention
from pydantic import BaseModel as PyBaseModel
from pydantic import ConfigDict
from torch import nn


def _global_mean_nograd(t: torch.Tensor) -> torch.Tensor:
    """
    同步并返回全局平均值；若 torch.distributed 未初始化则原样返回。
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        with torch.no_grad():
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
            t /= torch.distributed.get_world_size()
    return t


def load_balancing_loss_func(
    *,
    gate_logits: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]],
    top_k: int,
    num_experts: int,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    DDP 友好的 MoE load-balancing loss：
      • 统计量在所有 rank 间做 all-reduce 平均
      • stop-gradient trick（数值=全局，梯度=本地）
    """
    if not gate_logits:
        return torch.tensor(0.0, dtype=torch.float32, device="cpu")

    if not gate_logits or gate_logits[0] is None:
        return torch.tensor(0.0, dtype=torch.float32, device=gate_logits[0].device)

    device = gate_logits[0].device
    logits = torch.cat([g.float().to(device) for g in gate_logits], dim=0)  # [T,E]
    probs = F.softmax(logits, dim=-1)  # [T,E]

    # -------- route top-k ----------
    _, top_idx = torch.topk(probs, top_k, dim=-1)  # [T,k]
    expert_mask = F.one_hot(top_idx, num_experts).float()  # [T,k,E]

    # -------- optional pad ----------
    if attention_mask is not None:
        B, L = attention_mask.shape
        n_layers = logits.shape[0] // (B * L)

        pad = (
            attention_mask[None, :, :, None, None]  # [1,B,L,1,1]
            .expand(n_layers, B, L, top_k, num_experts)  # [n,B,L,k,E]
            .reshape_as(expert_mask)  # [T,k,E]
        )

        expert_mask *= pad
        valid_mask = pad[:, 0, :]  # [T,E]
        probs *= valid_mask

        denom = valid_mask.sum(0).clamp_min(1)  # [E]  有效 token 数
        tokens_per_exp = expert_mask.sum(0) / denom  # [k,E] (no grad)
        router_prob_exp = probs.sum(0) / denom  # [E]   (grad)
    else:
        tokens_per_exp = expert_mask.mean(0)  # [k,E]
        router_prob_exp = probs.mean(0)  # [E]

    # -------- DDP: 全局同步 ----------
    tokens_per_exp = _global_mean_nograd(tokens_per_exp)  # detached
    router_prob_sync = _global_mean_nograd(
        router_prob_exp.detach().clone()
    )  # detached copy → no grad

    # stop-gradient：数值用全局、梯度留本地
    router_prob_exp = router_prob_sync + (router_prob_exp - router_prob_exp.detach())

    # -------- final aux loss ----------
    aux = (tokens_per_exp * router_prob_exp).sum() * num_experts

    return aux


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
    use_dense: bool = False
    arch_mode: Literal["encdec", "encoder", "decoder"] = "encdec"
    use_headwise_gate: bool = False
    use_elementwise_gate: bool = False

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
    def __init__(
        self, args: ModelArgs, temperature: float = 1.0, noisy_std: float = 0.0
    ):
        super().__init__()
        self.args = args
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.n_groups = args.n_expert_groups
        self.topk_groups = args.n_limited_groups
        self.score_func = args.score_func
        self.route_scale = args.route_scale

        self.bias = nn.Parameter(torch.zeros(args.n_routed_experts))

        self.router = nn.Linear(args.dim, args.n_routed_experts, bias=False)

        self.temperature = temperature
        self.noisy_std = noisy_std

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.router.weight)  # 关键：全 0
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        T = x.size(0)
        E = self.args.n_routed_experts
        k = self.topk
        G = self.n_groups
        g = self.topk_groups

        # 1) logits (+noise) & cache
        logits = self.router(x) + self.bias  # [T,E]
        if self.noisy_std > 0 and self.training:
            logits = logits + self.noisy_std * torch.randn_like(logits)
        self.last_logits = logits  # for debug/other uses

        # 2) scores before grouping (作为“原始概率”)
        if self.score_func == "softmax":
            scores = F.softmax(logits / max(self.temperature, 1e-6), dim=-1)  # [T,E]
        else:  # "sigmoid"
            scores = torch.sigmoid(logits)  # [T,E]
        original_scores = scores

        # 3) grouping / limited groups → 仅用于选 idx 的“可选集合”
        #    我们构造一个 allow mask (float, 0/1)，后面也会用于 lb_p_mean
        if G > 1:
            assert E % G == 0, "E must be divisible by number of groups"
            group_size = E // G
            scores_g = scores.view(T, G, group_size)  # [T,G,E/G]
            if self.bias is None:
                group_rep = scores_g.amax(dim=-1)  # [T,G]
            else:
                top2 = scores_g.topk(2, dim=-1).values.sum(dim=-1)  # [T,G]
                group_rep = top2
            grp_idx = group_rep.topk(g, dim=-1).indices  # [T,g]

            # bool mask: True 表示被屏蔽（不可选）
            mask = torch.ones(T, G, dtype=torch.bool, device=scores.device)
            mask.scatter_(1, grp_idx, False)
            # 生成 allow mask 到专家粒度 [T,E]，True=允许
            allow = (~mask).unsqueeze(-1).expand(T, G, group_size).reshape(T, E)
            # 为了 topk 方便，把不允许位置设为 -inf（在 scores 上）
            scores_g = scores_g.masked_fill(mask.unsqueeze(-1), float("-inf"))
            scores_for_topk = scores_g.flatten(1)  # [T,E]
        else:
            allow = torch.ones(T, E, dtype=torch.bool, device=scores.device)
            scores_for_topk = scores  # [T,E]

        # 4) 选 k 个专家（用于真正路由）
        idx = scores_for_topk.topk(k, dim=-1).indices  # [T,k]
        w = original_scores.gather(1, idx).clone()  # [T,k] 来自“原始概率”
        w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-9)  # 归一化
        if self.route_scale != 1.0:
            w = w * self.route_scale

        # ====== 下面是新增：给 loss 的轻量缓存 ======
        with torch.no_grad():
            # 使用 one-hot 统计使用频率：对 token 与 k 两个维度平均 → [E]
            counts = torch.bincount(idx.view(-1), minlength=E).float()  # [E]
            usage_frac = counts / (T * k)
            self.lb_usage_frac = usage_frac  # no grad

            # 构造与“可选集合”一致的有效概率分布 p_eff，并做专家维度归一化
            allow_f = allow.float()
            p_eff = original_scores * allow_f  # [T,E]
            p_eff = p_eff / p_eff.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            # 平均到专家维度 → [E] （保留梯度的版本在下方计算）
            # 注意：为了 stop-grad 技巧，我们既需要 no-grad，也需要带梯度的版本
        # 带梯度的 p_eff_mean（用 original_scores 保留梯度；allow_f 不需要梯度）
        p_eff_with_grad = original_scores * allow.float()
        p_eff_with_grad = p_eff_with_grad / p_eff_with_grad.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-9)
        self.lb_p_mean = p_eff_with_grad.mean(dim=0)  # [E], with grad

        # 额外记录 token / expert / k
        self.lb_T = int(T)
        self.lb_E = int(E)
        self.lb_k = int(k)

        return w.type_as(x), idx


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
    def __init__(self, args: ModelArgs):
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
        更高效的 MoE 前向：
        - 显存：不构建 [T*k, D]，只多了若干 [T*k] 的一维辅助张量（极小）
        - 速度：避免对 indices 做 E 次全表扫描
        """
        shape = x.size()  # [B, ..., D]
        x = x.reshape(-1, self.dim)  # [T, D]
        T, D = x.shape
        device = x.device
        dtype = x.dtype

        # gate: 每个 token 选 k 个 expert
        weights, indices = self.gate(x)  # [T, k], [T, k]
        T, k = indices.shape
        E = self.n_routed_experts

        # 累积输出
        y = torch.zeros(T, D, device=device, dtype=dtype)

        # ------- 关键：一次性 flatten，避免 E 次 where(indices == i) --------
        indices_flat = indices.reshape(-1)  # [T*k]
        weights_flat = weights.reshape(-1)  # [T*k]
        # token_id_flat[t*k + s] = t
        token_id_flat = (
            torch.arange(T, device=device).unsqueeze(1).expand(T, k).reshape(-1)
        )

        # 按 expert id 排序，把同一 expert 的 (token,slot) 挤在一起
        sorted_exp, order = torch.sort(indices_flat)  # [T*k]
        sorted_tokens = token_id_flat[order]  # [T*k]
        sorted_w = weights_flat[order]  # [T*k]

        # searchsorted 找出每个 expert 在 sorted_exp 中的段 [start, end)
        boundaries = torch.searchsorted(
            sorted_exp, torch.arange(E + 1, device=device, dtype=sorted_exp.dtype)
        )  # [E+1]

        # ------- 逐 expert 处理自己这一段，显存只保留 x_e / out_e 子集 --------
        for e in range(E):
            start = int(boundaries[e].item())
            end = int(boundaries[e + 1].item())
            if start == end:
                continue  # 这个 expert 没有 token

            token_ids_e = sorted_tokens[start:end]  # [N_e]
            w_e = sorted_w[start:end].unsqueeze(1)  # [N_e, 1]

            x_e = x[token_ids_e]  # [N_e, D]
            out_e = self.experts[e](x_e)  # [N_e, D]

            # 累加回对应 token
            y.index_add_(0, token_ids_e, out_e * w_e)

        # 共享 expert：对所有 token 作用一遍
        z = self.shared_experts(x)  # [T, D]

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
            embed_dim=args.dim,
            num_heads=args.n_heads,
            dropout=0.0,
            use_headwise_gate=args.use_headwise_gate,
            use_elementwise_gate=args.use_elementwise_gate,
        )
        self.ffn = (
            MLP(args.dim, args.inter_dim)
            if layer_id < args.n_dense_layers
            else MoE(args)
        )
        if args.use_dense:
            self.ffn = MLP(
                args.dim,
                (args.n_activated_experts + args.n_shared_experts) * args.moe_inter_dim,
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
            embed_dim=args.dim,
            num_heads=args.n_heads,
            dropout=0.0,
            use_headwise_gate=args.use_headwise_gate,
            use_elementwise_gate=args.use_elementwise_gate,
        )
        self.ffn = (
            MLP(args.dim, args.inter_dim)
            if layer_id < args.n_dense_layers
            else MoE(args)
        )
        if args.use_dense:
            self.ffn = MLP(
                args.dim,
                (args.n_activated_experts + args.n_shared_experts) * args.moe_inter_dim,
            )
        if args.arch_mode == "encdec":
            self.norm2 = RMSNorm(args.dim)
            self.multihead_attn = RoPEMultiheadAttention(
                embed_dim=args.dim,
                num_heads=args.n_heads,
                dropout=0.0,
                use_headwise_gate=args.use_headwise_gate,
                use_elementwise_gate=args.use_elementwise_gate,
            )
        self.ffn_norm = RMSNorm(args.dim)
        self.ans_norm = RMSNorm(args.dim)

    def forward(
        self,
        tgt: torch.Tensor,  # [B, S_tgt, D]
        memory: Optional[torch.Tensor] = None,  # [B, S_src, D] — 编码器输出
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

        # ===== 关键改动：memory 为空则跳过 cross-attention =====
        if memory is not None:
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
        else:
            # z = self.norm2(y)  # 与 enc-dec 情况保持相近的归一化路径
            z = y

        out = z + self.ffn(self.ffn_norm(z))
        return self.ans_norm(out)  # [B, S_tgt, D] 输出


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
