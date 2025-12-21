# ► rope.py  （放到你的项目 utils 目录或同文件上方均可）
from typing import Optional, Tuple
import torch, torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
# 1.  RotaryEmbedding 及辅助函数
# ──────────────────────────────────────────────
class RotaryEmbedding(nn.Module):
    """生成 RoPE 正余弦基，支持任意序列长度（不依赖 max_len）"""

    def __init__(self, dim: int):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)  # [d/2]

    def forward(self, seq_len: int, device=None):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)  # [S]
        freqs = torch.einsum("s , d -> s d", t, self.inv_freq)  # [S, d/2]
        return torch.stack((freqs.sin(), freqs.cos()), dim=-1)  # [S, d/2, 2]


def apply_rope(x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    """
    x:    [B, H, S, d]   (d 偶数)
    rope: [S, d/2, 2]    (sin, cos)
    """
    sin, cos = rope[..., 0], rope[..., 1]  # [S, d/2]

    # broadcast 到四维
    sin = sin.unsqueeze(0).unsqueeze(0)  # [1,1,S,d/2]
    cos = cos.unsqueeze(0).unsqueeze(0)

    x_even = x[..., 0::2]  # [B,H,S,d/2]
    x_odd = x[..., 1::2]

    x[..., 0::2] = x_even * cos - x_odd * sin
    x[..., 1::2] = x_even * sin + x_odd * cos
    return x


class RoPEMultiheadAttention(nn.Module):
    """
    API 与 torch.nn.MultiheadAttention 完全相同（默认 batch_first=True）

    支持两种 Qwen3 风格的 Gate：
      - use_headwise_gate=True:
          q_proj 输出: [B, Sq, D + H]
          其中额外的 H 维作为每个 head 的 gate_score（标量 gate）
      - use_elementwise_gate=True:
          q_proj 输出: [B, Sq, 2D]
          其中后一半 D 维作为每个 head 每个维度的 gate_score（elementwise gate）

    二者互斥：不能同时为 True。
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        use_headwise_gate: bool = False,  # ★ 原有 headwise gate
        use_elementwise_gate: bool = False,  # ★ NEW: elementwise gate
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        # ★ gate 模式互斥
        if use_headwise_gate and use_elementwise_gate:
            raise ValueError(
                "use_headwise_gate 与 use_elementwise_gate 不能同时为 True"
            )

        self.use_headwise_gate = use_headwise_gate
        self.use_elementwise_gate = use_elementwise_gate

        # -------- q / k / v 投影 --------
        # Qwen3 的做法：
        #   - headwise: q_out_dim = D + H
        #   - elementwise: q_out_dim = 2D
        #   - 无 gate: q_out_dim = D
        if self.use_headwise_gate:
            q_out_dim = embed_dim + num_heads
        elif self.use_elementwise_gate:
            q_out_dim = embed_dim * 2
        else:
            q_out_dim = embed_dim

        self.q_proj = nn.Linear(embed_dim, q_out_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

        self.rope_emb = RotaryEmbedding(self.head_dim)

        self.batch_first = True
        self._rope_cache: dict[tuple[int, torch.device], torch.Tensor] = {}

    # --------------------------------------------------
    # RoPE cache
    # --------------------------------------------------
    def _get_rope(self, seq_len: int, device: torch.device) -> torch.Tensor:
        key = (seq_len, device)
        rope = self._rope_cache.get(key)
        if rope is None:
            rope = self.rope_emb(seq_len, device=device)
            self._rope_cache[key] = rope
        return rope

    # --------------------------------------------------
    # forward
    # --------------------------------------------------
    def forward(
        self,
        query: torch.Tensor,  # [B, Sq, D]
        key: Optional[torch.Tensor] = None,  # [B, Sk, D]
        value: Optional[torch.Tensor] = None,  # [B, Sk, D]
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if key is None:
            key = query
        if value is None:
            value = key

        B, Sq, _ = query.shape
        Sk = key.shape[1]

        # ---------------- Q 投影 + Gate 逻辑 ----------------
        # q_raw:
        #   - headwise: [B, Sq, D + H]
        #   - elementwise: [B, Sq, 2D]
        #   - 无 gate: [B, Sq, D]
        q_raw = self.q_proj(query)

        gate_score = None

        if self.use_headwise_gate:
            # headwise gate: 每个 head 一个标量 gate
            # q_raw: [B, Sq, D + H] -> [B, Sq, H, head_dim + 1]
            q_raw = q_raw.view(B, Sq, self.num_heads, self.head_dim + 1)
            # 切分出真正的 q 和 gate_score
            # q_part: [B, Sq, H, head_dim]
            # gate_score: [B, Sq, H, 1]
            q_part, gate_score = torch.split(
                q_raw,
                [self.head_dim, 1],
                dim=-1,
            )
            # 变到 [B, H, Sq, d] / [B, H, Sq, 1]
            q = q_part.permute(0, 2, 1, 3).contiguous()  # [B,H,Sq,d]
            gate_score = gate_score.permute(0, 2, 1, 3).contiguous()  # [B,H,Sq,1]

        elif self.use_elementwise_gate:
            # ★ NEW: elementwise gate（和 Qwen3 的 elementwise_attn_output_gate 一致）
            # q_raw: [B, Sq, 2D] -> [B, Sq, H, 2*head_dim]
            q_raw = q_raw.view(B, Sq, self.num_heads, 2 * self.head_dim)
            # q_part: [B, Sq, H, head_dim]
            # gate_score: [B, Sq, H, head_dim]
            q_part, gate_score = torch.split(
                q_raw,
                [self.head_dim, self.head_dim],
                dim=-1,
            )
            # 变到 [B, H, Sq, d]
            q = q_part.permute(0, 2, 1, 3).contiguous()  # [B,H,Sq,d]
            gate_score = gate_score.permute(0, 2, 1, 3).contiguous()  # [B,H,Sq,d]

        else:
            # 原始无 gate 路径
            # q_raw: [B, Sq, D] -> [B, Sq, H, d] -> [B,H,Sq,d]
            q = q_raw.view(B, Sq, self.num_heads, self.head_dim)
            q = q.permute(0, 2, 1, 3).contiguous()  # [B,H,Sq,d]

        # ---------------- K / V 投影 ----------------
        k = self.k_proj(key).view(B, Sk, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(B, Sk, self.num_heads, self.head_dim)

        k = k.permute(0, 2, 1, 3).contiguous()  # [B,H,Sk,d]
        v = v.permute(0, 2, 1, 3).contiguous()  # [B,H,Sk,d]

        # ---------------- RoPE 只作用于 q/k ----------------
        q = apply_rope(q, self._get_rope(Sq, query.device))
        k = apply_rope(k, self._get_rope(Sk, key.device))

        # ---------------- 组合 mask，调用 SDPA ----------------
        merged_mask = attn_mask
        if key_padding_mask is not None:
            # key_padding_mask: [B, Sk] (一般 True/False 或 0/1)
            pad = key_padding_mask[:, None, None, :]  # [B,1,1,Sk]
            if merged_mask is None:
                merged_mask = pad
            else:
                if merged_mask.dtype == torch.bool:
                    merged_mask = merged_mask | pad
                else:
                    merged_mask = merged_mask + pad  # 0 / -inf 叠加

        attn_out = F.scaled_dot_product_attention(
            q,  # [B,H,Sq,d]
            k,  # [B,H,Sk,d]
            v,  # [B,H,Sk,d]
            attn_mask=merged_mask,  # [B,1,Sq,Sk] 或 None
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=is_causal,
        )  # [B,H,Sq,d]

        # ---------------- 应用 Gate ----------------
        if gate_score is not None:
            # headwise: gate_score [B,H,Sq,1] → 自动 broadcast 到 d
            # elementwise: gate_score [B,H,Sq,d] → 与 attn_out 按元素对应
            attn_out = attn_out * torch.sigmoid(gate_score)

        # ---------------- heads 合并回 [B, Sq, D] ----------------
        out = attn_out.transpose(1, 2).reshape(B, Sq, self.embed_dim)
        out = self.out_proj(out)

        if need_weights:
            # 这里暂时不返回注意力权重
            return out, None
        return out, None
