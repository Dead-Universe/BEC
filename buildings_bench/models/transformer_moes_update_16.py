from typing import Dict, Optional, Tuple, Literal, List
import math
import torch
import torch.nn.functional as F
from torch import nn

from buildings_bench.models.base_model import BaseModel
from buildings_bench.models.sub_models_update_02 import (
    Encoder,
    Decoder,
    Gate,
    ModelArgs,
    load_balancing_loss_func,
)


# ==============================================================
# 1.   Multi-Period Fourier  (替换 AdaptiveFourier)
# ==============================================================


class MultiPeriodFourier(nn.Module):
    """
    多基期显式傅里叶基（仅用幅值，无额外相位网络）
    ------------------------------------------------
    channels : 负荷维数（单变量 =1）
    periods  : 主周期列表，如 [24, 168]  (小时)
    order    : 每个周期使用 1…order 个谐波   →  基数 = 2*order
    """

    def __init__(self, channels: int, periods: List[int], order: int = 3):
        super().__init__()
        self.C, self.periods, self.F = channels, periods, order
        # 幅值参数：   amp[:, p, 0:F]   → sin   ,  amp[:, p, F:2F] → cos
        self.amp = nn.Parameter(torch.zeros(channels, len(periods), 2 * order))
        # 每个周期一个可学习相位（单位: 时间步）
        self.phi = nn.Parameter(torch.zeros(len(periods)))  # shape (P,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T, C)   —— 只用 T 构造基底
        return seasonal : (B, T, C)
        """
        B, T, C = x.shape
        assert C == self.C
        device = x.device

        t = torch.arange(T, device=device).view(1, 1, 1, T)  # (1,1,1,T)
        seasonal = torch.zeros(B, T, C, device=device)

        with torch.no_grad():
            # 用当前窗口 load 的标准差作为尺度因子，防止不同建筑/季节量级差太大
            scale = x.std(dim=1, keepdim=True).clamp_min(1e-3)  # (B,1,C)

        for p_idx, P in enumerate(self.periods):
            f = torch.arange(1, self.F + 1, device=device).view(
                1, 1, self.F, 1
            )  # (1,1,F,1)
            theta = 2 * math.pi * f * (t + self.phi[p_idx]) / P
            sin_basis = torch.sin(theta)  # (1,1,F,T)
            cos_basis = torch.cos(theta)  # (1,1,F,T)

            # 幅值取出并 reshape 便于广播
            amp_sin = self.amp[:, p_idx, : self.F].unsqueeze(-1)  # (C,F,1)
            amp_cos = self.amp[:, p_idx, self.F :].unsqueeze(-1)  # (C,F,1)

            # (C,F,1)*(1,1,F,T)  →  (C,F,T)  →  sum(F) → (C,T)
            term = (amp_sin * sin_basis + amp_cos * cos_basis).sum(2)  # (C,T)
            term = term.squeeze(0)
            # seasonal += term.t().unsqueeze(0).expand(B, -1, -1)  # (B,T,C) 累加
            seasonal += term.t().unsqueeze(0).expand(B, -1, -1) * scale

        return seasonal


# ==============================================================
# 2.   Multi-Scale STL 分解层（改 1、2）
# ==============================================================


class MultiScaleSTL(nn.Module):
    """
    * Trend     : 对称 softmax 核，**长度截断后自适应混合权**（建议 2）
    * Seasonal  : Multi-Period Fourier (24h + 168h 默认)
    * Residual  : 守恒差
    """

    def __init__(
        self,
        channels: int = 1,
        trend_kernels: List[int] = (25, 169, 1025),
        periods: List[int] = (24, 168),
        fourier_order: int = 3,
        local_ks: int | None = None,
        padding_mode: str = "circular",
    ):
        super().__init__()
        self.C = channels
        self.trend_kernels = list(trend_kernels)
        self.periods = list(periods)
        self.F = fourier_order
        self.padding_mode = padding_mode

        # ---------- 1) 多尺度对称 softmax 核 ----------
        self.trend_logits = nn.ParameterList()
        for ks in self.trend_kernels:
            if ks % 2 == 0:
                raise ValueError("trend kernel must be odd")
            half_len = ks // 2 + 1
            self.trend_logits.append(nn.Parameter(torch.zeros(channels, half_len)))
        # 混合 logits
        self.mix_logits = nn.Parameter(torch.zeros(len(self.trend_kernels)))

        # ---------- 2) 多周期 Fourier ----------
        self.afourier = MultiPeriodFourier(
            channels=channels, periods=self.periods, order=fourier_order
        )

        # ---------- 3) 局部季节 “零均值” ----------
        local_ks = local_ks or max(3, self.trend_kernels[0] // 5)
        if local_ks % 2 == 0:
            local_ks += 1
        self.local_ks = local_ks
        self.local_logits = nn.Parameter(torch.zeros(channels, local_ks // 2 + 1))

    # --------- 工具 ---------
    @staticmethod
    def _sym_kernel(half_w: torch.Tensor) -> torch.Tensor:
        w = F.softmax(half_w, dim=-1)
        return torch.cat([w, torch.flip(w[..., :-1], [-1])], dim=-1)

    def _safe_sym_kernel(self, half_logit: torch.Tensor, T: int) -> torch.Tensor:
        full_len = 2 * half_logit.size(-1) - 1
        if full_len > 2 * T - 1:  # reflect padding 上限
            need_full = T | 1  # 最近奇数 (≤T)
            half_logit = half_logit[..., : (need_full // 2 + 1)]
        return self._sym_kernel(half_logit)

    def _pad(self, x: torch.Tensor, ks: int):
        p = ks // 2
        if ks > 1:
            return F.pad(x, (p, p), mode=self.padding_mode)  # ← 直接循环
        return x

    # ---------- 前向 ----------
    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x : (B, T, C)
        """
        if x.ndim != 3:
            raise ValueError("x must be (B,T,C)")
        B, T, C = x.shape
        if C != self.C:
            raise ValueError(f"init channels={self.C}, but input={C}")

        x_c = x.permute(0, 2, 1)  # (B,C,T)

        # ===== 1) Trend =====
        trends, eff_lens = [], []
        for logit in self.trend_logits:
            w = self._safe_sym_kernel(logit, T)  # (C,K)
            eff_lens.append(w.size(-1))
            w = w.view(C, 1, -1)
            trend_i = F.conv1d(self._pad(x_c, w.size(-1)), w, groups=C)
            trends.append(trend_i)  # (B,C,T)
        trend_stack = torch.stack(trends, dim=-1)  # (B,C,T,S)

        # ------ 2) 自适应混合 ------
        eff_len = torch.tensor(eff_lens, device=x.device, dtype=x.dtype)
        scale = 1.0 / torch.sqrt(eff_len)  # 1/√L
        mix_w = F.softmax(self.mix_logits * scale, dim=0)  # (S,)
        trend = (trend_stack * mix_w).sum(-1).permute(0, 2, 1)  # (B,T,C)

        # ===== 2) Seasonal – Global multi-period =====
        seasonal_global = self.afourier(x)  # (B,T,C)

        # ===== 3) 局部季节 =====
        raw_local = x - trend - seasonal_global  # (B,T,C)
        raw_local_c = raw_local.permute(0, 2, 1)  # (B,C,T)
        w_loc = self._safe_sym_kernel(self.local_logits, T).view(C, 1, -1)
        smooth_local = F.conv1d(self._pad(raw_local_c, w_loc.size(-1)), w_loc, groups=C)
        smooth_local = smooth_local.permute(0, 2, 1)
        seasonal_local = raw_local - smooth_local
        seasonal = seasonal_global + seasonal_local
        seasonal = seasonal - seasonal.mean(
            dim=1, keepdim=True
        )  # 强行让季节项“零均值”以免吃掉趋势

        residual = x - trend - seasonal
        return seasonal, trend, residual

    # ---------- 正则 ----------
    def regularization_terms(self, T: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Trend kernel二阶差分
        regs = []
        for logit in self.trend_logits:
            k = self._safe_sym_kernel(logit, T)  # (C,K)
            regs.append(((k[:, 2:] - 2 * k[:, 1:-1] + k[:, :-2]) ** 2).mean())
        trend_reg = torch.stack(regs).mean()

        # Fourier 幅值稀疏
        fourier_reg = self.afourier.amp.abs().mean()
        return trend_reg, fourier_reg


# ==============================================================
# 3.   主模型：只改 (a) 分解层实例  (b) 正则 λ
# ==============================================================


class LoadForecastingTransformerMoE(BaseModel):
    """
    输入仍只有 load；实现 1-3 建议
    ----------------------------------------
    * periods 固定 [24,168]  & order=3
    * trend_mix-scale & 正则 λ 动态
    """

    def __init__(
        self,
        max_context_len: int = 336,
        max_pred_len: int = 168,
        context_len: int = 168,
        pred_len: int = 24,
        vocab_size: int = 2274,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 8,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.0,
        activation: str = "gelu",
        continuous_loads: bool = True,
        continuous_head: Literal["mse", "gaussian_nll", "huber"] = "mse",
        ignore_spatial: bool = False,
        weather_inputs: list | None = None,
        trend_kernels: tuple = (25, 169, 1025),
        fourier_order: int = 3,
        periods: tuple = (24, 168),
    ):
        super().__init__(context_len, pred_len, continuous_loads=continuous_loads)
        self.max_context_len = max_context_len
        self.max_pred_len = max_pred_len
        self.continuous_head = continuous_head

        # -------- 分解层 (新的) --------
        self.decomp = MultiScaleSTL(
            channels=1,
            trend_kernels=list(trend_kernels),
            periods=list(periods),
            fourier_order=fourier_order,
        )

        # -------- 其余模块与原实现相同 --------
        self.seasonal_embedding = nn.Linear(1, d_model)
        self.trend_embedding = nn.Linear(1, d_model)
        self.residual_embedding = nn.Linear(1, d_model)

        self.cfg = ModelArgs(
            max_seq_len=max_context_len + max_pred_len,
            dim=d_model,
            inter_dim=dim_feedforward,
            moe_inter_dim=d_model // 2,
            n_encoder_layers=num_encoder_layers,
            n_decoder_layers=num_decoder_layers,
            n_heads=nhead,
            n_routed_experts=num_experts,
            n_shared_experts=2,
            n_activated_experts=top_k,
            n_expert_groups=1,
            n_limited_groups=1,
            score_func="softmax",
            route_scale=1.0,
        )
        enc_layer = Encoder(self.cfg.n_dense_layers, self.cfg)
        self.seasonal_encoder = nn.TransformerEncoder(
            enc_layer, num_encoder_layers // 2, enable_nested_tensor=False
        )
        self.trend_encoder = nn.TransformerEncoder(
            enc_layer, 1, enable_nested_tensor=False
        )
        self.residual_encoder = nn.TransformerEncoder(
            enc_layer, num_encoder_layers // 2, enable_nested_tensor=False
        )

        dec_layer = Decoder(self.cfg.n_dense_layers, self.cfg)
        self.seasonal_decoder = nn.TransformerDecoder(
            dec_layer, num_decoder_layers // 2
        )
        self.trend_decoder = nn.TransformerDecoder(dec_layer, 1)
        self.residual_decoder = nn.TransformerDecoder(
            dec_layer, num_encoder_layers // 2
        )

        out_dim = 1 if continuous_head in ("mse", "huber") else 2
        self.seasonal_head = nn.Linear(d_model, out_dim)
        self.trend_head = nn.Linear(d_model, out_dim)
        self.residual_head = nn.Linear(d_model, out_dim)

        self._gate_logits: List[torch.Tensor] = []

        def _save_logits(module: Gate, _inp, _out):
            self._gate_logits.append(module.last_logits)

        for m in self.modules():
            if isinstance(m, Gate):
                m.register_forward_hook(_save_logits, with_kwargs=False)

    # --------------------- forward ----------------------
    def forward(
        self,
        x: Dict[str, torch.Tensor],
        context_len: Optional[int] = None,
        pred_len: Optional[int] = None,
    ):
        self._gate_logits.clear()
        if context_len is None:
            context_len = self.max_context_len
        if pred_len is None:
            pred_len = self.max_pred_len

        load = x["load"]
        assert load.size(1) == context_len + pred_len

        seasonal, trend, residual = self.decomp(load[:, :context_len])
        self._cached_reg = self.decomp.regularization_terms(context_len)
        self._cached_context_len = context_len  # 供 loss 动态 λ
        s_ctx, t_ctx, r_ctx = seasonal, trend, residual

        s_mem = self.seasonal_encoder(self.seasonal_embedding(s_ctx))
        t_mem = self.trend_encoder(self.trend_embedding(t_ctx))
        r_mem = self.residual_encoder(self.residual_embedding(r_ctx))

        B = load.size(0)
        query = torch.zeros(
            B, pred_len, self.cfg.dim, device=load.device, dtype=load.dtype
        )

        s_out = self.seasonal_decoder(query, s_mem)
        t_out = self.trend_decoder(query, t_mem)
        r_out = self.residual_decoder(query, r_mem)

        y_hat = (
            self.seasonal_head(s_out)
            + self.trend_head(t_out)
            + self.residual_head(r_out)
        )
        return y_hat

    # --------------------- loss with 动态 λ ------------------------
    def loss(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        trend_reg, fourier_reg = self._cached_reg
        ctx_len = self._cached_context_len
        # ------------------ λ 动态 -----------------
        λ_trend = 0.05 * (ctx_len / 168)
        λ_fourier = 1e-3 * (24 / max(self.decomp.periods))

        λ_aux = 0.01

        aux_loss = load_balancing_loss_func(
            gate_logits=tuple(self._gate_logits),
            top_k=self.cfg.n_activated_experts,
            num_experts=self.cfg.n_routed_experts,
            attention_mask=None,
        )
        self._gate_logits.clear()

        if self.continuous_loads:
            if self.continuous_head == "huber":
                err = F.huber_loss(pred, y, delta=1.0, reduction="none")
            elif self.continuous_head == "mse":
                err = F.mse_loss(pred, y, reduction="none")
            else:  # gaussian_nll
                mu, sigma_sq = pred[..., :1], F.softplus(pred[..., 1:]) ** 2
                err = 0.5 * (
                    torch.log(2 * torch.pi * sigma_sq) + (y - mu) ** 2 / sigma_sq
                )
            loss = err.mean()
        else:
            B, L, _ = pred.shape
            loss = F.cross_entropy(pred.reshape(B * L, -1), y.long().reshape(B * L))

        return loss + λ_trend * trend_reg + λ_fourier * fourier_reg + λ_aux * aux_loss

    # -------------- 其余接口不变 ------------------------------
    @torch.no_grad()
    def predict(
        self, x: Dict[str, torch.Tensor], context_len: int = 168, pred_len: int = 24
    ):
        preds = self.forward(x, context_len, pred_len)
        if self.continuous_head in ("mse", "huber"):
            return preds, preds
        else:
            return preds[..., :1], preds

    def unfreeze_and_get_parameters_for_finetuning(self):
        return self.parameters()

    def load_from_checkpoint(self, checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu")["model"]
        state = {k.replace("module.", ""): v for k, v in state.items()}
        self.load_state_dict(state)


# ==============================================================
# Quick self-test (与原来一致)
# ==============================================================

if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = (
        LoadForecastingTransformerMoE(
            max_context_len=672,
            max_pred_len=168,
            continuous_head="huber",
            fourier_order=3,
            periods=(24, 168),
        )
        .to(device)
        .train()
    )

    ctx, pred = 96, 40
    B = 2
    dummy = {"load": torch.rand(B, ctx + pred, 1, device=device)}
    target = dummy["load"][:, -pred:]

    out = model(dummy, context_len=ctx, pred_len=pred)
    loss = model.loss(out, target)
    loss.backward()
    print("Sanity-check OK – loss:", float(loss))

    model.eval()
    preds, _ = model.predict(dummy, context_len=ctx, pred_len=pred)
    print("Inference preds shape:", preds.shape)  # [B, pred, 1]
