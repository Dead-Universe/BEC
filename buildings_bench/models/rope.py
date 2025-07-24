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
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        # 独立的 q / k / v 投影
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

        self.rope_emb = RotaryEmbedding(self.head_dim)

        # 关键！让 TransformerEncoder 识别 batch_first
        self.batch_first = True

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

        # 1) 线性映射 → [B, S, H, d]
        q = self.q_proj(query).view(B, Sq, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(B, Sk, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(B, Sk, self.num_heads, self.head_dim)

        # 2) 交换维度 → [B, H, S, d]
        q, k, v = [t.permute(0, 2, 1, 3) for t in (q, k, v)]

        # 3) RoPE 只作用于 q/k
        q = apply_rope(q, self.rope_emb(Sq, query.device))
        k = apply_rope(k, self.rope_emb(Sk, key.device))

        # 4) scaled dot‑product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, Sq, Sk]

        # padding & attn mask
        if attn_mask is not None:
            attn = attn + attn_mask
        if key_padding_mask is not None:
            attn = attn.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        if is_causal:
            causal_mask = torch.triu(
                torch.full((Sq, Sk), float("-inf"), device=attn.device),
                diagonal=1,
            )
            attn = attn + causal_mask

        attn = F.softmax(attn, dim=-1, dtype=torch.float32)
        attn = self.dropout(attn)

        # 5) 计算输出，reshape 回 [B, Sq, D]
        out = (attn @ v).transpose(1, 2).reshape(B, Sq, self.embed_dim)
        out = self.out_proj(out)

        if need_weights:
            # 返回平均到 head 的权重，保持与官方接口一致
            return out, attn.mean(dim=1)
        return out, None
