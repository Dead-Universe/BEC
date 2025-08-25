import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ===================== Utils =====================


def count_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def _get_alibi_slopes(n_heads: int) -> torch.Tensor:
    def get_slopes_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if math.log2(n_heads).is_integer():
        slopes = get_slopes_power_of_2(n_heads)
    else:
        cp2 = 2 ** math.floor(math.log2(n_heads))
        slopes = get_slopes_power_of_2(cp2)
        extra = get_slopes_power_of_2(2 * cp2)[0::2][: n_heads - cp2]
        slopes += extra
    return torch.tensor(slopes).float()


def _safe_pad1d(
    x: torch.Tensor, pad_left: int, pad_right: int, prefer="reflect"
) -> torch.Tensor:
    """Robust 1D padding: try reflect if尺寸允许，否则退化为 replicate。"""
    if pad_left == 0 and pad_right == 0:
        return x
    L = x.size(-1)
    mode = prefer
    # reflect 需要 pad <= L-1 且 L>=2
    if mode == "reflect" and (L < 2 or pad_left > L - 1 or pad_right > L - 1):
        mode = "replicate"
    return F.pad(x, (pad_left, pad_right), mode=mode)


# ===================== Periodic Bias (learnable λ) =====================


class PeriodicBias(nn.Module):
    """
    Learnable cosine periodic biases for encoder/decoder self-attn and cross-attn.
    periods: hours (e.g., 24, 168, 8760)
    """

    def __init__(self, periods=(24, 168, 8760), init_lambdas=(0.25, 0.20, 0.10)):
        super().__init__()
        P = len(periods)
        self.periods = tuple(float(p) for p in periods)
        # separate λ for enc-self, dec-self, cross
        self.lam_enc = nn.Parameter(
            torch.tensor(init_lambdas, dtype=torch.float32)
        )  # [P]
        self.lam_dec = nn.Parameter(
            torch.tensor(init_lambdas, dtype=torch.float32)
        )  # [P]
        self.lam_cross = nn.Parameter(
            torch.tensor(init_lambdas, dtype=torch.float32)
        )  # [P]

    def _sum_cos(self, rel: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        # rel: [Nq,Nk] hours, lam: [P]
        bias = torch.zeros_like(rel)
        for i, p in enumerate(self.periods):
            bias = bias + lam[i] * torch.cos(2 * math.pi * rel / p)
        return bias

    @torch.no_grad()
    def clamp_lambdas_(self, minv=-1.0, maxv=1.0):
        self.lam_enc.clamp_(minv, maxv)
        self.lam_dec.clamp_(minv, maxv)
        self.lam_cross.clamp_(minv, maxv)

    def enc_self(self, N: int, stride: int, device=None) -> torch.Tensor:
        # token i/j 距离 -> 小时差 = (i-j)*stride
        i = torch.arange(N, device=device).view(N, 1)
        j = torch.arange(N, device=device).view(1, N)
        rel = (i - j).float() * float(stride)  # [N,N]
        return self._sum_cos(rel, self.lam_enc)

    def dec_self(self, H: int, device=None) -> torch.Tensor:
        # 查询步差（单位：小时）：0..H-1
        i = torch.arange(H, device=device).view(H, 1)
        j = torch.arange(H, device=device).view(1, H)
        rel = (i - j).float()  # [H,H]
        return self._sum_cos(rel, self.lam_dec)

    def cross(
        self, H: int, N: int, stride_k: int, patch_k: int, device=None
    ) -> torch.Tensor:
        # 未来 h=0..H-1（+h 小时），历史 token j 的中心时刻按负时间轴：
        # centers_j = -((N-1 - j)*stride_k + (patch_k-1)/2)
        h = torch.arange(H, device=device).float().view(H, 1)  # [H,1]
        j = torch.arange(N, device=device).float()  # [N]
        centers = -((N - 1 - j) * float(stride_k) + (float(patch_k) - 1.0) / 2.0)  # [N]
        rel = h - centers.view(1, N)  # [H,N]
        return self._sum_cos(rel, self.lam_cross)


# ===================== Tokenizer =====================


class PatchTokenizer1D(nn.Module):
    def __init__(self, d_model: int, patch_size: int = 16, stride: int = 8):
        super().__init__()
        self.patch_size = int(patch_size)
        self.stride = int(stride)
        self.proj = nn.Conv1d(
            1, d_model, kernel_size=self.patch_size, stride=self.stride, bias=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        # x: [B,1,L]
        B, C, L = x.shape
        assert C == 1
        if L < self.patch_size:
            # pad 到 patch_size，确保至少 1 个 token
            x = _safe_pad1d(x, 0, self.patch_size - L, prefer="reflect")
        tok = self.proj(x)  # [B,d,N]
        tok = tok.transpose(1, 2)  # [B,N,d]
        tok = self.norm(tok)
        return tok, self.stride, self.patch_size


# ===================== Decomposition =====================


class Decomp1D(nn.Module):
    def __init__(self, d_model: int, kernel_sizes=(7, 25, 49), causal: bool = False):
        super().__init__()
        self.causal = causal
        self.ks = tuple(int(k) for k in kernel_sizes)
        self.alpha = nn.Parameter(torch.zeros(len(self.ks)))
        self.convs = nn.ModuleList()
        for k in self.ks:
            conv = nn.Conv1d(
                d_model, d_model, kernel_size=k, groups=d_model, bias=False
            )
            with torch.no_grad():
                conv.weight.zero_()
                conv.weight.add_(1.0 / k)
            self.convs.append(conv)

    def forward(self, x: torch.Tensor):
        # x: [B,N,d]  —— 在 token 轴做平滑
        xt = x.transpose(1, 2)  # [B,d,N]
        outs = []
        N = xt.size(-1)
        for conv, k in zip(self.convs, self.ks):
            if self.causal:
                pad_l, pad_r = (k - 1), 0
            else:
                pad_l, pad_r = (k // 2), (k // 2)
            y = conv(_safe_pad1d(xt, pad_l, pad_r, prefer="reflect"))
            outs.append(y)
        T = torch.stack(outs, dim=-1)  # [B,d,N,S]
        w = torch.softmax(self.alpha, dim=0).view(1, 1, 1, -1)
        T = (T * w).sum(-1).transpose(1, 2)  # [B,N,d]
        S = x - T
        return S, T


# ===================== Attention (ALiBi + periodic bias) =====================


class BiasMultiheadAttention(nn.Module):
    def __init__(
        self, d_model: int, nhead: int, dropout: float = 0.0, bias: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert self.head_dim * nhead == d_model

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)
        self.drop = nn.Dropout(dropout)

        slopes = _get_alibi_slopes(nhead)
        self.register_buffer("alibi_slopes", slopes, persistent=False)

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        is_self_attn: bool = False,
        causal: bool = False,
    ):
        B, Nq, _ = x_q.shape
        Nk = x_kv.size(1)
        q = self.q_proj(x_q).view(B, Nq, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_kv).view(B, Nk, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_kv).view(B, Nk, self.nhead, self.head_dim).transpose(1, 2)

        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
            self.head_dim
        )  # [B,H,Nq,Nk]

        # ALiBi 仅用于自注意
        if is_self_attn and Nq == Nk:
            i = torch.arange(Nq, device=x_q.device).view(Nq, 1)
            j = torch.arange(Nk, device=x_q.device).view(1, Nk)
            dist = (j - i).clamp(min=0).float()
            alibi = -self.alibi_slopes.view(1, -1, 1, 1) * dist.view(1, 1, Nq, Nk)
            logits = logits + alibi

        if attn_bias is not None:
            logits = logits + (
                attn_bias.view(1, 1, Nq, Nk) if attn_bias.dim() == 2 else attn_bias
            )

        if causal and is_self_attn and Nq == Nk:
            mask = torch.triu(torch.ones(Nq, Nk, device=x_q.device), diagonal=1).bool()
            logits = logits.masked_fill(mask.view(1, 1, Nq, Nk), float("-inf"))

        attn = F.softmax(logits, dim=-1)
        attn = self.drop(attn)
        out = (
            torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, Nq, self.d_model)
        )
        return self.o_proj(out)


# ===================== FFN (Dense & MoE) =====================


class DenseFFN(nn.Module):
    def __init__(self, d_model: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x), x.new_zeros(())


class MoeFFN(nn.Module):
    """
    Top-2 gating MoE-FFN (single machine, no capacity limit).
    Returns (y, aux_loss) where aux_loss promotes load balance.
    """

    def __init__(
        self,
        d_model: int,
        hidden: int,
        dropout: float,
        num_experts: int = 4,
        top_k: int = 2,
        router_jitter: float = 0.01,
    ):
        super().__init__()
        assert top_k in (1, 2)
        self.top_k = top_k
        self.num_experts = num_experts
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, d_model),
                    nn.Dropout(dropout),
                )
                for _ in range(num_experts)
            ]
        )
        self.router_jitter = router_jitter

    def forward(self, x: torch.Tensor):
        B, N, D = x.shape
        T = B * N
        xf = x.reshape(T, D)
        logits = self.router(xf)
        if self.training and self.router_jitter > 0:
            logits = logits + (torch.rand_like(logits) - 0.5) * (2 * self.router_jitter)
        gates = F.softmax(logits, dim=-1)  # [T,E]
        topv, topi = torch.topk(gates, k=self.top_k, dim=-1)  # [T,k]
        # 重归一（质量守恒）
        denom = topv.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        topv = topv / denom

        weights = torch.zeros_like(gates).scatter(1, topi, topv)

        # 负载均衡（简化版）
        prob_per = gates.mean(0)  # [E] 期望门值
        frac_per = (weights > 0).float().mean(0)  # [E] 实际分配占比
        aux = (self.num_experts * torch.sum(prob_per * frac_per)).to(x.dtype)

        out = torch.zeros_like(xf)
        for e in range(self.num_experts):
            sel = weights[:, e] > 0
            if not torch.any(sel):
                continue
            o = self.experts[e](xf[sel])
            out.index_add_(
                0,
                sel.nonzero(as_tuple=False).squeeze(-1),
                o * weights[sel, e].unsqueeze(-1),
            )
        return out.view(B, N, D), aux


# ===================== Encoder / Decoder =====================


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        ffn_hidden_dense: int,
        dropout: float,
        decomp_kernels=(7, 25, 49),
        use_moe: bool = False,
        moe_hidden: int = 2048,
        num_experts: int = 4,
        top_k: int = 2,
        router_jitter: float = 0.01,
    ):
        super().__init__()
        self.decomp = Decomp1D(d_model, decomp_kernels, causal=False)
        self.attn = BiasMultiheadAttention(d_model, nhead, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        if use_moe:
            self.ffn = MoeFFN(
                d_model,
                moe_hidden,
                dropout,
                num_experts=num_experts,
                top_k=top_k,
                router_jitter=router_jitter,
            )
        else:
            self.ffn = DenseFFN(d_model, ffn_hidden_dense, dropout)
        self.trend_conv = nn.Conv1d(
            d_model, d_model, kernel_size=5, padding=2, groups=d_model, bias=True
        )
        self.use_moe = use_moe

    def forward(self, x: torch.Tensor, periodic_bias: Optional[torch.Tensor] = None):
        # x: [B,N,d]
        S, T = self.decomp(x)
        y = self.attn(
            self.norm1(S),
            self.norm1(S),
            attn_bias=periodic_bias,
            is_self_attn=True,
            causal=False,
        )
        x_s = S + y
        ffn_out, aux = self.ffn(self.norm2(x_s))
        x_s = x_s + ffn_out
        t_out = self.trend_conv(T.transpose(1, 2)).transpose(1, 2)
        out = x_s + t_out
        return out, aux


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, ffn_hidden: int, dropout: float):
        super().__init__()
        self.self_attn = BiasMultiheadAttention(d_model, nhead, dropout)
        self.cross_attn = BiasMultiheadAttention(d_model, nhead, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = DenseFFN(d_model, ffn_hidden, dropout)

    def forward(
        self,
        q: torch.Tensor,
        mem: torch.Tensor,
        self_bias: Optional[torch.Tensor] = None,
        cross_bias: Optional[torch.Tensor] = None,
    ):
        x = q + self.self_attn(
            self.norm1(q),
            self.norm1(q),
            attn_bias=self_bias,
            is_self_attn=True,
            causal=False,
        )
        x = x + self.cross_attn(
            self.norm2(x),
            self.norm2(mem),
            attn_bias=cross_bias,
            is_self_attn=False,
            causal=False,
        )
        ffn_out, _ = self.ffn(self.norm3(x))
        return x + ffn_out


# ===================== Query Generator =====================


class QueryGenerator(nn.Module):
    def __init__(self, d_model: int, periods=(24, 168, 8760)):
        super().__init__()
        self.periods = periods
        in_dim = 1 + 2 * len(periods)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )

    def forward(self, H: int, device=None):
        # 用 0..H-1，和周期偏置保持一致
        h = torch.arange(0, H, device=device).float()
        base = (h.clamp_min(1.0) / float(max(H, 1))).unsqueeze(
            -1
        )  # 避免 H=0 的边界（实际不会传 H=0）
        feats = [base]
        for p in self.periods:
            ang = 2 * math.pi * h / p
            feats += [torch.sin(ang).unsqueeze(-1), torch.cos(ang).unsqueeze(-1)]
        return self.mlp(torch.cat(feats, dim=-1))  # [H,d]


# ===================== Model =====================


@dataclass
class ModelConfig:
    d_model: int = 1024
    nhead: int = 16
    enc_layers: int = 14
    dec_layers: int = 1  # 1 层解码器，控制总参 < 200M
    ffn_hidden_enc_dense: int = 2816  # 非 MoE 层 FFN 隐层
    ffn_hidden_dec: int = 1536
    dropout: float = 0.1
    patch_size: int = 16
    stride: int = 8
    decomp_kernels: Tuple[int, ...] = (7, 25, 49)
    periods: Tuple[int, ...] = (24, 168, 8760)
    # MoE 配置：4 层 MoE、4 专家、top-2
    moe_layers: Tuple[int, ...] = (2, 5, 8, 11)
    moe_hidden: int = 2048
    moe_num_experts: int = 4
    moe_top_k: int = 2
    moe_router_jitter: float = 0.01


class BuildingEnergyForecastingModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = PatchTokenizer1D(cfg.d_model, cfg.patch_size, cfg.stride)
        self.periodic = PeriodicBias(periods=cfg.periods)

        enc = []
        for i in range(cfg.enc_layers):
            enc.append(
                EncoderLayer(
                    cfg.d_model,
                    cfg.nhead,
                    cfg.ffn_hidden_enc_dense,
                    cfg.dropout,
                    decomp_kernels=cfg.decomp_kernels,
                    use_moe=(i in cfg.moe_layers),
                    moe_hidden=cfg.moe_hidden,
                    num_experts=cfg.moe_num_experts,
                    top_k=cfg.moe_top_k,
                    router_jitter=cfg.moe_router_jitter,
                )
            )
        self.enc_layers = nn.ModuleList(enc)

        self.query_gen = QueryGenerator(cfg.d_model, cfg.periods)
        self.dec_layers = nn.ModuleList(
            [
                DecoderLayer(cfg.d_model, cfg.nhead, cfg.ffn_hidden_dec, cfg.dropout)
                for _ in range(cfg.dec_layers)
            ]
        )
        self.out_head = nn.Linear(cfg.d_model, 1)

    @staticmethod
    def _instance_norm(x: torch.Tensor, eps: float = 1e-5):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True).clamp_min(eps)
        return (x - mean) / std, mean, std

    def forward(self, x: torch.Tensor, H: int, return_aux: bool = False):
        # x: [B,1,L] -> y: [B,H]
        B, C, L = x.shape
        assert C == 1
        xn, mean, std = self._instance_norm(x)

        tok, stride, patch = self.tokenizer(xn)  # [B,N,d]
        B, N, _ = tok.shape

        # Encoder 自注意周期偏置
        per_enc = self.periodic.enc_self(N, stride, device=tok.device)  # [N,N]
        aux_total = tok.new_zeros(())
        for layer in self.enc_layers:
            tok, aux = layer(tok, periodic_bias=per_enc)
            aux_total = aux_total + aux

        # Decoder queries（H 可变）
        q = self.query_gen(H, device=x.device).unsqueeze(0).expand(B, -1, -1)  # [B,H,d]
        per_dec = self.periodic.dec_self(H, device=x.device)  # [H,H]
        per_cross = self.periodic.cross(H, N, stride, patch, device=x.device)  # [H,N]

        dec = q
        for layer in self.dec_layers:
            dec = layer(dec, tok, self_bias=per_dec, cross_bias=per_cross)

        y = self.out_head(dec).squeeze(-1)  # [B,H]
        y = y * std.squeeze(1) + mean.squeeze(1)  # 反归一化
        return (y, aux_total) if return_aux else y


# ===================== Quick check =====================
if __name__ == "__main__":
    cfg = ModelConfig()
    model = BuildingEnergyForecastingModel(cfg)
    print("Params(M):", round(count_parameters(model) / 1e6, 3))
    B, L, H = 2, 336, 168
    x = torch.randn(B, 1, L)
    with torch.no_grad():
        y, aux = model(x, H, return_aux=True)
    print("Shapes:", y.shape, "aux_loss:", float(aux))
