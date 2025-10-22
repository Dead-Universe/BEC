import math
from typing import Dict, List, Literal, Optional

from buildings_bench.models.moe import MoEActivationHook
import torch
import torch.nn.functional as F
from buildings_bench.models.base_model import BaseModel
from buildings_bench.models.sub_models_update_02 import (
    Decoder,
    Encoder,
    Gate,
    ModelArgs,
    load_balancing_loss_func,
)
from torch import nn


class LoadForecastingTransformerMoE(BaseModel):

    def __init__(
        self,
        max_context_len: int = 336,
        max_pred_len: int = 168,
        context_len: int = 168,
        pred_len: int = 24,
        vocab_size: int = 2274,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 8,
        d_model: int = 768,
        nhead: int = 12,
        dim_feedforward: int = 2048,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.0,
        activation: str = "gelu",
        continuous_loads: bool = True,
        continuous_head: Literal["mse", "gaussian_nll", "huber"] = "mse",
        ignore_spatial: bool = False,
        weather_inputs: list | None = None,
        use_dense: bool = False,
        arch_mode: Literal["encdec", "encoder", "decoder"] = "encdec",
    ):
        super().__init__(context_len, pred_len, continuous_loads=continuous_loads)
        self.max_context_len = max_context_len
        self.max_pred_len = max_pred_len
        self.continuous_head = continuous_head
        self.arch_mode = arch_mode
        self.use_dense = use_dense

        # ------- 三路编码/解码（结构保持不变） -------
        self.embedding = nn.Linear(1, d_model)

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
            use_dense=use_dense,
            arch_mode=arch_mode,
        )

        if self.arch_mode == "encoder" or self.arch_mode == "encdec":
            enc_layer = Encoder(self.cfg.n_dense_layers, self.cfg)
            self.encoder = nn.TransformerEncoder(
                enc_layer, num_encoder_layers, enable_nested_tensor=False
            )
        if self.arch_mode == "decoder" or self.arch_mode == "encdec":
            dec_layer = Decoder(self.cfg.n_dense_layers, self.cfg)
            self.decoder = nn.TransformerDecoder(dec_layer, num_decoder_layers)

        out_dim = 1 if continuous_head in ("mse", "huber") else 2
        self.head = nn.Linear(self.cfg.dim, out_dim)

        # 收集门控 logits 用于辅助均衡损失
        self._gate_logits: List[torch.Tensor] = []

        def _save_logits(module: Gate, _inp, _out):
            self._gate_logits.append(module.last_logits)

        for m in self.modules():
            if isinstance(m, Gate):
                m.register_forward_hook(_save_logits, with_kwargs=False)

        self._sched_cfg = dict(
            tau_hi=2.5,
            tau_mid=1.3,
            tau_lo=1.0,
            sigma_hi=0.05,
            sigma_mid=0.01,
            sigma_lo=0.0,
            explore_frac=0.10,
            settle_frac=0.60,  # 三阶段分界
            entropy_target_frac=0.85,  # 目标归一化熵 ρ
            k_tau=0.20,
            k_sigma=0.10,  # 闭环比例增益
            ema_alpha=0.10,  # EMA 平滑
            pulse_tau=0.6,
            pulse_sigma=0.03,  # 防塌缩脉冲幅度
            pulse_decay=0.95,  # 脉冲指数衰减
            tau_bounds=(1.0, 3.0),
            sigma_bounds=(0.0, 0.1),
        )
        # —— 运行时状态 ——
        self._sched_state = dict(
            ema_Hn=None,
            ema_maxshare=None,
            pulse_t=0.0,  # 脉冲强度（逐步衰减）
            collapsed_streak=0,
        )

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

        load = x["load"]  # (B, ctx+pred, 1)，与 stl_* 同尺度（Box-Cox）
        assert load.size(1) == context_len + pred_len, "load 长度与 ctx+pred 不一致"
        B, _, _ = load.shape
        ctx = load[:, :context_len]  # [B, ctx, 1]

        if self.arch_mode == "encdec":
            # ===== 原始 Encoder-Decoder 路径=====

            # 编码器部分
            mem = self.encoder(self.embedding(ctx))  # 编码器双向可用
            # 解码器部分
            query = torch.zeros(
                B, self.max_pred_len, self.cfg.dim, device=load.device, dtype=load.dtype
            )
            out = self.decoder(query, mem)
            y_hat = self.head(out)
            return y_hat[:, :pred_len, :]  # 仅取预测段

        elif self.arch_mode == "encoder":
            # ===== 纯 Encoder 路径：上下文 + 预测占位零，启用因果掩码 =====
            zeros_pred = torch.zeros(
                B, self.max_pred_len, 1, device=load.device, dtype=load.dtype
            )
            inp = torch.cat([ctx, zeros_pred], dim=1)  # [B, ctx+pred, 1]
            h = self.encoder(self.embedding(inp), is_causal=True)  # 关键：因果
            out = h[:, -self.max_pred_len :, :]  # 仅取预测段
            return self.head(out)[:, :pred_len, :]

        else:  # self.arch_mode == "decoder"
            zeros_pred = torch.zeros(
                B, self.max_pred_len, 1, device=load.device, dtype=load.dtype
            )
            tgt_vals = torch.cat([ctx, zeros_pred], dim=1)  # [B, ctx+pred, 1]

            tgt = self.embedding(tgt_vals)  # [B, ctx+pred, D]
            # 关键：启用因果掩码，保证预测段位置只看见上下文与自己左边的预测占位
            h = self.decoder(tgt, memory=None, tgt_is_causal=True)

            out = h[:, -self.max_pred_len :, :]  # 仅取预测段
            return self.head(out)[:, :pred_len, :]  # [B, pred, out_dim]

    @torch.no_grad()
    def router_feedback(self, stats: dict | None):
        """训练时把 inspect_router 的返回 dict 传进来（可为 None）"""
        if not stats:
            return
        st = self._sched_state
        cfg = self._sched_cfg
        a = cfg["ema_alpha"]
        Hn = float(stats.get("entropy_norm", 0.0))
        ms = float(stats.get("max_share", 1.0))
        # EMA
        st["ema_Hn"] = Hn if st["ema_Hn"] is None else (1 - a) * st["ema_Hn"] + a * Hn
        st["ema_maxshare"] = (
            ms if st["ema_maxshare"] is None else (1 - a) * st["ema_maxshare"] + a * ms
        )
        # 塌缩计数
        if bool(stats.get("collapsed", False)):
            st["collapsed_streak"] += 1
        else:
            st["collapsed_streak"] = 0

    @torch.no_grad()
    def _apply_gate_params(
        self, tau: float, sigma: float, extra_tau: float = 0.0, extra_sigma: float = 0.0
    ):
        cfg = self._sched_cfg
        lo, hi = cfg["tau_bounds"]
        tau = min(max(tau + extra_tau, lo), hi)
        lo, hi = cfg["sigma_bounds"]
        sigma = min(max(sigma + extra_sigma, lo), hi)
        for m in self.modules():
            if isinstance(m, Gate):
                m.temperature = float(tau)
                m.noisy_std = float(sigma)

    @torch.no_grad()
    def _update_gate_schedule(self, progress: float):
        """三阶段 + 闭环 + 脉冲；进度∈[0,1]，需先调用 router_feedback"""
        p = float(min(max(progress, 0.0), 1.0))
        cfg, st = self._sched_cfg, self._sched_state
        ph1, ph2 = cfg["explore_frac"], cfg["settle_frac"]

        # —— 基线：分段余弦 ——
        def cos_interp(x, a, b):  # x∈[0,1]
            return b + 0.5 * (a - b) * (1 + math.cos(math.pi * x))

        if p <= ph1:
            tau = cfg["tau_hi"]
            sigma = cfg["sigma_hi"]
        elif p <= ph2:
            t = (p - ph1) / max(1e-6, (ph2 - ph1))
            tau = cos_interp(t, cfg["tau_hi"], cfg["tau_mid"])
            sigma = cos_interp(t, cfg["sigma_hi"], cfg["sigma_mid"])
        else:
            # 先落到低位
            tau, sigma = cfg["tau_lo"], cfg["sigma_lo"]

        # —— 闭环：根据 EMA(Hn / max_share) 细调 ——
        if st["ema_Hn"] is not None and st["ema_maxshare"] is not None:
            target = cfg["entropy_target_frac"]  # ρ
            err = target - float(st["ema_Hn"])  # 想要更均匀 => err>0 => 升温/加噪
            tau += cfg["k_tau"] * err
            sigma += cfg["k_sigma"] * err
            # 额外抑制 max_share 偏大
            over = max(
                0.0,
                st["ema_maxshare"]
                - (
                    1.0
                    / max(1, getattr(getattr(self, "cfg", None), "n_routed_experts", 1))
                ),
            )
            if over > 0:
                tau += 0.5 * cfg["k_tau"] * over
                sigma += 0.5 * cfg["k_sigma"] * over

        # —— 防塌缩：脉冲注入 + 衰减 ——
        if st["collapsed_streak"] >= 3:  # 连续3次判定塌缩就触发
            st["pulse_t"] = max(st["pulse_t"], 1.0)
            st["collapsed_streak"] = 0
        extra_tau = cfg["pulse_tau"] * st["pulse_t"]
        extra_sigma = cfg["pulse_sigma"] * st["pulse_t"]
        st["pulse_t"] *= cfg["pulse_decay"]

        self._apply_gate_params(tau, sigma, extra_tau, extra_sigma)

    @torch.no_grad()
    def inspect_router(
        self,
        *,
        # 触发塌缩的判据（至少满足其中两个才判为 COLLAPSED）
        max_share_thr: float = 0.50,  # 单一专家占比 ≥ 50%
        h_norm_thr: float = 0.60,  # 归一化熵 ≤ 0.60
        neff_frac_thr: float = 0.50,  # 有效专家数 N_eff ≤ 0.5 * E
        min_nonzero_frac: float = 0.50,  # 被使用过的专家占比 ≤ 50%
        #
        topm: tuple = (1, 2, 4),  # 额外汇报前 m 个专家的累计占比
        ascii_bar: bool = True,  # 打印 ASCII 条形图
        clear: bool = True,  # 是否清空 model._gate_logits（建议 True）
    ):
        """
        汇总所有 Gate 的 logits（按你当前的 hook 方式）来评估专家利用率，
        并打印“是否路由塌缩”的结论与触发原因。若没有收集到 logits 会直接提示并返回 None。
        """
        gl = getattr(self, "_gate_logits", None)
        if not gl:
            print("[router] no gate logits collected.")
            return None

        # 兼容：通常 gl 是 List[Tensor]；若有人改成 List[(name, Tensor)] 也能处理
        tensors = []
        for item in gl:
            if item is None:
                continue
            if (
                isinstance(item, (tuple, list))
                and len(item) == 2
                and torch.is_tensor(item[1])
            ):
                tensors.append(item[1])
            elif torch.is_tensor(item):
                tensors.append(item)
        if not tensors:
            print("[router] empty gate logits.")
            if clear:
                self._gate_logits.clear()
            return None

        logits = torch.cat([t.float().to("cpu") for t in tensors], dim=0)  # [T, E]
        E = logits.size(-1)
        if E == 0:
            print("[router] invalid expert dimension E=0.")
            if clear:
                self._gate_logits.clear()
            return None

        probs = F.softmax(logits, dim=-1)
        # 取每个 token 的 top-k 专家，统计使用频率
        topk = int(getattr(getattr(self, "cfg", None), "n_activated_experts", 1))
        top_idx = probs.topk(topk, dim=-1).indices.reshape(-1)  # [T * topk]
        usage = torch.bincount(top_idx, minlength=E).float()  # [E]
        total_tokens = int(usage.sum().item())
        usage = usage / usage.sum().clamp_min(1)

        # 指标
        eps = 1e-12
        H = float((-(usage + eps) * (usage + eps).log()).sum().item())  # 香农熵
        H_max = math.log(E)
        H_norm = float(H / H_max) if H_max > 0 else 0.0  # 归一化熵
        N_eff_exp = float(math.exp(H))  # 依据熵的有效专家数
        N_eff_hhi = float(1.0 / (usage.pow(2).sum().item() + eps))  # HHI 的倒数
        max_share = float(usage.max().item())
        nonzero = int((usage > 0).sum().item())
        nonzero_frac = float(nonzero / E)

        # Top-m 累积占比
        sorted_usage, _ = torch.sort(usage, descending=True)
        top_shares = {m: float(sorted_usage[: min(m, E)].sum().item()) for m in topm}

        # 与均匀分布的发散度（可当“负载不均衡度”参考）
        # KL(usage || uniform) = log(E) - H
        kl_to_uniform = float(H_max - H)

        # 组合判据：满足至少两条视为塌缩
        reasons = []
        if max_share >= max_share_thr:
            reasons.append(f"max_share≥{max_share_thr:.2f}")
        if H_norm <= h_norm_thr:
            reasons.append(f"H_norm≤{h_norm_thr:.2f}")
        if N_eff_exp <= neff_frac_thr * E:
            reasons.append(f"N_eff(exp)≤{neff_frac_thr:.2f}·E")
        if nonzero_frac <= min_nonzero_frac:
            reasons.append(f"active_frac≤{min_nonzero_frac:.2f}")
        collapsed = len(reasons) >= 2

        # 打印
        print(f"[router] E={E}  topk={topk}  tokens={total_tokens}")
        print(
            "[router] "
            f"entropy={H:.3f} (norm={H_norm:.3f})  "
            f"N_eff(exp)={N_eff_exp:.2f}  N_eff(HHI)={N_eff_hhi:.2f}  "
            f"KL_to_uniform={kl_to_uniform:.3f}"
        )
        print(
            "[router] "
            f"max_share={max_share:.3f}  active={nonzero}/{E} ({nonzero_frac:.2f})  "
            + " ".join([f"top{m}={top_shares[m]:.3f}" for m in topm if m <= E])
        )
        if ascii_bar:
            width = 20  # 每个专家条形图的宽度
            lines = []
            for i, u in enumerate(usage.tolist()):
                n = int(round(u * width))
                bar = "|" * n if n > 0 else ""
                lines.append(f"  {i:02d} {bar:<20} {u:.3f}")
            print("[router] usage per expert:\n" + "\n".join(lines))

        status = "COLLAPSED" if collapsed else "ok"
        print(
            f"[router] status: {status}"
            + (f" | triggers: {', '.join(reasons)}" if reasons else "")
        )

        if clear:
            self._gate_logits.clear()

        return {
            "E": E,
            "topk": topk,
            "tokens": total_tokens,
            "usage": usage.tolist(),
            "entropy": H,
            "entropy_norm": H_norm,
            "N_eff_exp": N_eff_exp,
            "N_eff_hhi": N_eff_hhi,
            "max_share": max_share,
            "active": nonzero,
            "active_frac": nonzero_frac,
            "top_shares": top_shares,
            "KL_to_uniform": kl_to_uniform,
            "collapsed": collapsed,
            "reasons": reasons,
            "thresholds": {
                "max_share_thr": max_share_thr,
                "h_norm_thr": h_norm_thr,
                "neff_frac_thr": neff_frac_thr,
                "min_nonzero_frac": min_nonzero_frac,
            },
        }

    # --------------------- loss（已移除分解正则） ---------------------
    def loss(
        self, pred: torch.Tensor, y: torch.Tensor, progress: Optional[float] = None
    ) -> torch.Tensor:
        # --- 新增：根据进度调度门控温度/噪声 ---
        if progress is not None and not self.use_dense:
            # 将进度转换为整数百分比 (0-100)
            current_percent = int(progress * 100)

            # 检查是否到达新的百分比点
            if not hasattr(self, "_last_percent"):
                self._last_percent = -1  # 初始化

            if current_percent > self._last_percent:
                stats = self.inspect_router(clear=False)
                self.router_feedback(stats)
                self._last_percent = current_percent

            self._update_gate_schedule(progress)
        if self.use_dense:
            aux_loss = torch.tensor(0.0, device=pred.device)
        else:
            # 门控均衡辅助项
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
            else:
                raise NotImplementedError()
            loss = err.mean()
        else:
            raise NotImplementedError()

        # 可按需调整 λ_aux，这里保持和你之前一致的 0.01
        λ_aux = 0.01
        return loss + λ_aux * aux_loss

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


def moe_static_params(model: torch.nn.Module):
    """
    一次性静态统计 MoE 参数量（无需前向）：
      activated_params_total = shared_params + Σ_layer (top_k[layer] * per_expert_params[layer])

    约定/假设（与你的实现匹配）：
      - 一个 MoE 层满足：模块具有 .experts (nn.ModuleList) （你的 MoE 正是如此）
      - "路由专家"指 MoE.experts；shared_experts 视为常驻参数（始终参与计算）
      - 每层专家结构一致；若不一致则以该层“专家平均参数量”作为 per_expert_params[layer]
    """
    import torch.nn as nn

    # 1) 找出所有 MoE 层
    moe_layers = []  # [(mod_name, mod)]
    for name, mod in model.named_modules():
        if hasattr(mod, "experts") and isinstance(
            getattr(mod, "experts"), nn.ModuleList
        ):
            moe_layers.append((name or "root", mod))

    # 2) 逐层统计“每个专家的参数量”（per_expert）、专家数、top-k
    per_expert_params_per_layer = []  # [int]
    experts_per_layer = []  # [int]
    top_k_per_layer = []  # [int or None]
    routed_expert_params_total = (
        0  # 所有层所有专家总和（仅路由专家，不含 shared_experts）
    )

    for layer_name, mod in moe_layers:
        experts: nn.ModuleList = mod.experts
        n_exp = len(experts)
        experts_per_layer.append(n_exp)

        if n_exp == 0:
            per_expert_params_per_layer.append(0)
        else:
            # 计算该层每个专家参数量；若不完全一致则取平均
            sizes = []
            for e in experts:
                sizes.append(sum(p.numel() for p in e.parameters() if p.requires_grad))
            if len(set(sizes)) == 1:
                per_exp = sizes[0]
            else:
                per_exp = int(sum(sizes) // n_exp)  # 取平均做近似
            per_expert_params_per_layer.append(per_exp)
            routed_expert_params_total += per_exp * n_exp

        tk = None
        if hasattr(mod, "gate") and hasattr(mod.gate, "topk"):
            try:
                tk = int(mod.gate.topk)
            except Exception:
                tk = None
        top_k_per_layer.append(tk if tk is not None else 0)

    # 3) 总参数量 & 常驻参数量（总 - 路由专家）
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    shared_params = (
        total_params - routed_expert_params_total
    )  # 含 embedding/attention/router/shared_experts 等

    # 4) 激活专家参数量（跨所有 MoE 层求和）
    activated_expert_params_sum = 0
    for per_exp, tk in zip(per_expert_params_per_layer, top_k_per_layer):
        activated_expert_params_sum += per_exp * int(tk)

    activated_params_total = shared_params + activated_expert_params_sum

    # 5) 汇总结果
    out = {
        "total_params": int(total_params),
        "shared_params": int(shared_params),
        "routed_expert_params_total": int(routed_expert_params_total),
        "num_moe_layers": len(moe_layers),
        "experts_per_layer": experts_per_layer,
        "per_expert_params_per_layer": per_expert_params_per_layer,
        "top_k_per_layer": top_k_per_layer,
        "activated_expert_params_sum": int(
            activated_expert_params_sum
        ),  # Σ_layer (topk * per_expert)
        "activated_params_total": int(activated_params_total),  # shared + 上面这一项
        "activation_rate": float(activated_params_total / max(1, total_params)),
    }
    return out


# ======================= Quick self-test =======================
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = (
        LoadForecastingTransformerMoE(
            max_context_len=336,
            max_pred_len=168,
            continuous_head="huber",
            use_dense=False,
            arch_mode="decoder",
            num_decoder_layers=16,
            num_encoder_layers=1,
        )
        .to(device)
        .train()
    )

    moe_static = moe_static_params(model)

    print("Model params:", moe_static)

    ctx, pred = 96, 40
    B = 2
    # 伪造 batch：注意提供 stl_S/T/R
    load = torch.rand(B, ctx + pred, 1, device=device)

    dummy = {"load": load}
    target = dummy["load"][:, -pred:]

    out = model(dummy, context_len=ctx, pred_len=pred)
    loss = model.loss(out, target)
    loss.backward()
    print("Sanity-check OK – loss:", float(loss))

    model.eval()
    preds, _ = model.predict(dummy, context_len=ctx, pred_len=pred)
    print("Inference preds shape:", preds.shape)  # [B, pred, 1]

    import os
    from pathlib import Path
    import pandas as pd
    import torch

    # ───────────────────────────────────────────
    # Ⅰ. 读 CSV（最后 168+24=192 行）
    # ───────────────────────────────────────────
    # CSV_PATH = "/home/hadoop/bec/buildings-bench/v2.0.0/BuildingsBench/cofactor/building6397_clean=2018.csv"
    CSV_PATH = "/home/hadoop/bec/buildings-bench/v2.0.0/BuildingsBench/Borealis/home2_clean=2011.csv"
    df = (
        pd.read_csv(CSV_PATH, parse_dates=["timestamp"])
        .sort_values("timestamp")
        .iloc[-192:]
    )

    y_raw = df["power"].values.astype("float32")  # (192,)

    # ───────────────────────────────────────────
    # Ⅱ. Box‑Cox 归一化
    # ───────────────────────────────────────────
    from buildings_bench.transforms import BoxCoxTransform

    boxcox = BoxCoxTransform()
    transform_dir = (
        Path(
            os.environ.get(
                "BUILDINGS_BENCH",
                "/home/hadoop/bec/buildings-bench/v2.0.0/BuildingsBench",
            )
        )
        / "metadata"
        / "transforms"
    )
    boxcox.load(transform_dir)

    y_norm = torch.from_numpy(boxcox.transform(y_raw))  # tensor (192,)

    # quick sanity…往返误差应极小
    rt_diff = (torch.from_numpy(y_raw) - boxcox.undo_transform(y_norm)).abs().max()
    print(f"↔ Box‑Cox round‑trip max|diff|: {rt_diff:.3e}")

    # ───────────────────────────────────────────
    # Ⅲ. 打包模型输入 [B=1, T=192, 1]
    # ───────────────────────────────────────────
    sample = {"load": y_norm.unsqueeze(0).unsqueeze(-1)}  # (1,192,1)
    y_true_raw = y_raw[-24:]

    # ───────────────────────────────────────────
    # Ⅳ. 载入模型与权重
    # ───────────────────────────────────────────

    cfg = dict(
        context_len=168,
        pred_len=24,
        num_encoder_layers=8,
        num_decoder_layers=10,
        d_model=768,
        dim_feedforward=2048,
        num_experts=8,
        top_k=2,
        nhead=12,
        dropout=0.0,
        continuous_loads=True,
        continuous_head="huber",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LoadForecastingTransformerMoE(**cfg).to(device).eval()

    moe_hook = MoEActivationHook(model, sync_ddp=True)

    CKPT = "/home/hadoop/bec/BuildingsBench/checkpoints/TransformerWithGaussianAndMoEs-L_Update-20-huber_best_val.pt"
    if not os.path.exists(CKPT):
        raise FileNotFoundError(f"Checkpoint not found: {CKPT}")
    model.load_from_checkpoint(CKPT)
    print(f"✅ Loaded checkpoint: {CKPT}")

    # ───────────────────────────────────────────
    # Ⅴ. 推断 & 反归一化
    # ───────────────────────────────────────────
    with torch.no_grad():
        pred_norm, _ = model.predict(
            {k: v.to(device) for k, v in sample.items()}
        )  # (1,24,1)

    pred_raw = boxcox.undo_transform(pred_norm.cpu()).squeeze()  # (24,)

    # ───────────────────────────────────────────
    # Ⅵ. 打印结果
    # ───────────────────────────────────────────
    print("\n=== 24‑hour Forecast vs Truth (kW) ===")
    print(" idx |  predict |   actual")
    print("-----+----------+----------")
    for i, (p_pred, p_true) in enumerate(zip(pred_raw, y_true_raw), 1):
        # p_pred 可能是 0‑维 tensor；p_true 一定是 float
        p_val = float(p_pred) if isinstance(p_pred, torch.Tensor) else p_pred
        print(f" t+{i:02d} | {p_val:8.5f} | {p_true:8.5f}")

    fig = moe_hook.plot(normalize=True)

    fig.savefig("moe_activation_heatmap.png")
