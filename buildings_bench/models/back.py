from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple
from pydantic import BaseModel as PyBaseModel, ConfigDict
from buildings_bench.models.base_model import BaseModel
import torch
from torch import nn
import torch.nn.functional as F

import math

from buildings_bench.models.model_kernel import apply_init


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
            scores = scores.view(x.size(0), self.n_groups, -1)
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
        x = x.view(-1, self.dim)
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
        return (y + z).view(shape)


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
        self.attn = nn.MultiheadAttention(  # ← PyTorch 自带
            embed_dim=args.dim, num_heads=args.n_heads, batch_first=True
        )
        self.ffn = (
            MLP(args.dim, args.inter_dim)
            if layer_id < args.n_dense_layers
            else MoE(args)
        )

        self.attn_norm = RMSNorm(args.dim)
        self.ffn_norm = RMSNorm(args.dim)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor after block computation.
        """
        x = self.attn_norm(x)
        h, _ = self.attn(
            x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )
        y = x + h
        z = y + self.ffn(self.ffn_norm(y))
        return z


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
        self.causal_attn = nn.MultiheadAttention(
            embed_dim=args.dim, num_heads=args.n_heads, batch_first=True
        )
        self.ffn = (
            MLP(args.dim, args.inter_dim)
            if layer_id < args.n_dense_layers
            else MoE(args)
        )
        self.norm2 = RMSNorm(args.dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=args.dim, num_heads=args.n_heads, batch_first=True
        )
        self.ffn_norm = RMSNorm(args.dim)

    def forward(
        self,
        tgt: torch.Tensor,  # [B, S_tgt, D]
        memory: torch.Tensor,  # [B, S_src, D] — 编码器输出
        tgt_mask: Optional[torch.Tensor] = None,  # causal mask
        memory_mask: Optional[torch.Tensor] = None,  # padding mask (可选)
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
        attn_out, _ = self.causal_attn(q, k, v, attn_mask=tgt_mask)
        y = tgt + attn_out
        q = self.norm2(y)
        k = v = memory
        cross_out, _ = self.cross_attn(q, k, v, attn_mask=memory_mask)
        z = y + cross_out
        out = z + self.ffn(self.ffn_norm(z))
        return out


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


# ──────────────────────────────────────────────
# 3.  主模型
# ──────────────────────────────────────────────
class TimeSeriesTransformer(BaseModel):
    def __init__(self, **kwargs):
        args = ModelArgs(**kwargs)
        super().__init__(args.context_len, args.pred_len, True)
        D = args.dim

        # --- 各字段嵌入 ---
        width = D // 8  # 256 → 32
        self.lat = make_linear(1, width)
        self.lon = make_linear(1, width)
        self.btype = nn.Embedding(args.building_type, width)  # 假设 2 类建筑
        self.doy = PeriodicEmbed(width)
        self.dow = PeriodicEmbed(width)
        self.hod = PeriodicEmbed(width)
        self.load = make_linear(1, width * 2)  # 给负荷多分一点维

        self.pos = PositionalEncoding(D, args.max_seq_len)

        # --- Encoder / Decoder 堆叠 ---
        self.enc_layers = nn.ModuleList(
            Encoder(i, args) for i in range(args.n_encoder_layers)
        )
        self.dec_layers = nn.ModuleList(
            Decoder(i, args) for i in range(args.n_decoder_layers)
        )

        self.head = nn.Linear(D, 1)  # 连续回归

        # 生成 causal mask 备用
        self.register_buffer(
            "tgt_mask",
            torch.full((args.pred_len, args.pred_len), float("-inf")).triu_(1),
        )

    # ---------- helper ----------
    def _build_embed(self, feats: dict) -> torch.Tensor:
        """把七个原始字段 → [B,T,D]"""
        parts = [
            self.lat(feats["latitude"]),
            self.lon(feats["longitude"]),
            self.btype(feats["building_type"].long().squeeze(-1)),
            self.doy(feats["day_of_year"]),
            self.dow(feats["day_of_week"]),
            self.hod(feats["hour_of_day"]),
            self.load(feats["load"]),
        ]
        return torch.cat(parts, dim=-1)  # [B,T,D]

    def loss(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """固定为 MSE."""
        return F.mse_loss(pred, y)

    # ---------- forward ----------
    def forward(self, feats: dict):
        """
        feats : dict[field] = [B, T, 1]  (building_type 可以是 [B,T,1] int)
        """
        _, T, _ = feats["load"].shape
        assert T == self.context_len + self.pred_len

        x = self._build_embed(feats)  # [B,T,D]

        src = x[:, : self.context_len]  # 编码器输入
        tgt = x[:, self.context_len - 1 : -1]  # 预测步左移一位 (teacher forcing)

        # 位置编码
        src = self.pos(src)
        tgt = self.pos(tgt)

        # --- 编码器 ---
        for layer in self.enc_layers:
            src = layer(src, None, None)  # 无 mask, padding 可加

        memory = src  # [B, ctx, D]

        # --- 解码器 ---
        tgt_mask = self.tgt_mask.to(tgt.device)
        for layer in self.dec_layers:
            tgt = layer(tgt, memory, tgt_mask)

        logits = self.head(tgt)  # [B, pred_len, 1]
        return logits

    def unfreeze_and_get_parameters_for_finetuning(self):
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

    @torch.inference_mode()
    def predict(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        自回归逐步预测 24 步。返回 [B, pred_len, 1]
        要求 feats['load'] 未来 24 步可填 0 占位。
        """
        device = next(self.parameters()).device
        feats = {k: v.to(device).clone() for k, v in feats.items()}
        B = feats["load"].shape[0]

        # ---------- Encoder ----------
        src_feats = {k: v[:, : self.context_len] for k, v in feats.items()}
        src_embed = self.pos(self._build_embed(src_feats))
        for layer in self.enc_layers:
            src_embed = layer(src_embed, None, None)
        memory = src_embed  # [B,ctx,D]

        # ---------- Decoder (autoregressive) ----------
        preds = torch.zeros(B, self.pred_len, 1, device=device)
        tgt_embed = self.pos(
            self._build_embed(
                {
                    k: v[:, self.context_len - 1 : self.context_len]
                    for k, v in feats.items()
                }
            )
        )

        for t in range(self.pred_len):
            out = tgt_embed
            for layer in self.dec_layers:
                out = layer(out, memory, tgt_mask=self.tgt_mask[: t + 1, : t + 1])

            next_load = self.head(out[:, -1:])  # [B,1]
            # ① 记录输出（保持 3-D）
            preds[:, t : t + 1] = next_load

            # ② 写回 feats['load'] —— 需挤掉两维得到 [B]
            feats["load"][:, self.context_len + t, 0] = next_load.squeeze(-1).squeeze(
                -1
            )

            if t == self.pred_len - 1:  # 最后一轮不再扩充输入
                break

            # ③ 生成下一步嵌入并追加
            nxt_embed = self._build_embed(
                {
                    k: v[:, self.context_len + t : self.context_len + t + 1]
                    for k, v in feats.items()
                }
            )
            nxt_embed = self.pos(nxt_embed, tgt_embed.size(1))
            tgt_embed = torch.cat([tgt_embed, nxt_embed], dim=1)

        return preds  # [B, pred, 1]


if __name__ == "__main__":
    # ---------- 0. 设备 ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using", device)

    # ---------- 1. 模型 ----------
    torch.manual_seed(0)
    args = ModelArgs()
    model = TimeSeriesTransformer(**args.model_dump()).to(device)
    model.train()  # 打开梯度

    # ---------- 2. 假数据 ----------
    B = 4
    T = args.context_len + args.pred_len
    fake = {
        "latitude": (torch.rand(B, T, 1) * 2 - 1).to(device),
        "longitude": (torch.rand(B, T, 1) * 2 - 1).to(device),
        "building_type": torch.randint(0, 1, (B, T, 1), device=device),
        "day_of_year": (torch.rand(B, T, 1) * 2 - 1).to(device),
        "day_of_week": (torch.rand(B, T, 1) * 2 - 1).to(device),
        "hour_of_day": (torch.rand(B, T, 1) * 2 - 1).to(device),
        "load": torch.rand(B, T, 1, device=device),
    }

    # 打开 anomaly 检测，若仍有 in-place 毛病会直接指示报错行
    torch.autograd.set_detect_anomaly(True)

    # ---------- 3. 正向 ----------
    preds = model(fake)  # teacher-forcing
    target = torch.rand_like(preds)  # 随机目标
    loss = model.loss(preds, target)
    print("forward OK, loss =", loss.item())

    # ---------- 4. 反向 ----------
    loss.backward()  # 若有 in-place 改写会报 RuntimeError
    print("backward OK, grads computed")

    # ---------- 5. 简单一步优化验证 ----------
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    opt.step()
    print("optimizer step OK")
