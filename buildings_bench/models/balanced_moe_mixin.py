import math
from typing import Dict, List, Tuple, Optional, Any
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed
import torch.utils.hooks


class BalancedMoEMixin(nn.Module):
    """
    MoE 路由均衡与调度的混入类（需配合带统计缓存的 Gate 使用）。

    依赖 Gate.forward 设置以下缓存：
      - gate.lb_usage_frac : [E]  无梯度，专家使用频率（按 token 与 top-k 平均）
      - gate.lb_p_mean     : [E]  有梯度，在“可选专家集合”归一化后的概率均值
      - gate.lb_T          : int  本层本步 token 数
      - gate.lb_E          : int  专家数
      - gate.lb_k          : int  top-k

    用法：
      - 模型类继承 BalancedMoEMixin
      - 在 __init__() 末尾调用 self._init_balancer()
      - 训练时：
            self.moe_step(progress)                 # 统计+调度
            aux_loss = self.moe_auxiliary_loss()    # LBL 辅助损失
    """

    # ---------------------- 初始化 ----------------------
    def _init_balancer(self, use_moe_balancer: bool = True) -> None:
        self.use_moe_balancer = use_moe_balancer
        # 1) 收集 Gate
        self._moe_gates: List[Tuple[str, nn.Module]] = [
            (name, m)
            for name, m in self.named_modules()
            if m.__class__.__name__ == "Gate"
        ]
        if not self._moe_gates:
            # raise RuntimeError("BalancedMoEMixin: 未找到 Gate 模块。")
            print("⚠️  BalancedMoEMixin: 未找到 Gate 模块。")
            self.use_moe_balancer = False
            return

        # 2) 注册一个轻量 hook（仅用于标记该步被前向过）
        self._moe_handles: List[torch.utils.hooks.RemovableHandle] = []
        self._gate_seen: Dict[str, int] = {}
        for name, gate in self._moe_gates:
            h = gate.register_forward_hook(self._make_gate_hook(name))
            self._moe_handles.append(h)

        # 3) 默认配置
        self._moe_cfg: Dict[str, Any] = dict(
            rho_target=0.85,  # 目标归一化熵
            tau_hi=2.5,  # 初始温度（探索）
            tau_lo=1.0,  # 最终温度（收敛）
            sigma_hi=0.03,  # 初始噪声
            sigma_lo=0.0,  # 最终噪声
            explore_frac=0.15,  # 前 15% 探索
            settle_frac=0.65,  # 65% 后收敛
            k_fb=0.2,  # 闭环反馈增益
            pulse_gain=0.6,  # pulse 强度
            pulse_decay=0.95,  # pulse 衰减
            collapse_trig=dict(  # 判定 collapse 的阈值（满足≥2条）
                max_share=0.40, Hn=0.65, neff_frac=0.5, active_frac=0.5
            ),
            revive_usage_thr=5e-4,  # 死专家阈值（使用频率）
            revive_bias=0.02,  # 死专家 bias 提升
            lambda_global=0.0,  # 全局 LBL 权重（默认关闭）
            lambda_layer=1.0,  # 逐层 LBL 权重
            tau_bounds=(1.0, 3.0),  # 温度范围
            sigma_bounds=(0.0, 0.1),  # 噪声范围
            # 新增：消融相关开关
            enable_stage_schedule=True,  # 是否启用阶段式 τ/σ 调度
            enable_revive=True,  # 是否启用 revive_dead_experts
        )

        # 4) 运行态
        self._moe_state: Dict[str, Any] = dict(
            global_ema=dict(entropy_norm=None, max_share=None),  # 全局 EMA（用于调度）
            layers={
                name: dict(
                    ema_entropy_norm=None,
                    ema_max_share=None,
                    collapsed_streak=0,
                    pulse=0.0,
                )
                for name, _ in self._moe_gates
            },
            last_percent=-1,
        )

    # ---------------------- Hook ----------------------
    def _make_gate_hook(self, name: str):
        def hook(module: nn.Module, _inp: Tuple[Any], _out: Any) -> None:
            # 仅标记该 gate 在本步发生过前向（用于统计/可视化）
            self._gate_seen[name] = self._gate_seen.get(name, 0) + 1

        return hook

    def remove_hooks(self) -> None:
        for h in self._moe_handles:
            h.remove()
        self._moe_handles.clear()

    # ---------------------- 工具函数 ----------------------
    @staticmethod
    def _ddp_mean_(t: torch.Tensor) -> torch.Tensor:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            with torch.no_grad():
                torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
                t /= torch.distributed.get_world_size()
        return t

    @staticmethod
    def _entropy_metrics_from_usage(u: torch.Tensor) -> Dict[str, float]:
        """
        u: [E], sum(u)=1
        """
        eps = 1e-12
        H = float((-(u + eps) * (u + eps).log()).sum().item())
        E = u.numel()
        Hn = H / max(1e-12, math.log(E))
        ms = float(u.max().item())
        active_frac = float((u > 0).float().mean().item())
        neff_exp = float(math.exp(H))
        return dict(
            entropy_norm=Hn,
            max_share=ms,
            active_frac=active_frac,
            neff_exp=neff_exp,
            E=E,
        )

    # ---------------------- 统计 ----------------------
    @torch.no_grad()
    def _layerwise_stats(self, clear: bool = False) -> Dict[str, Any]:
        """
        仅依赖 Gate 缓存，计算逐层与全局统计（全局=按 token 数加权的层均值）。
        返回:
          {
            "per_layer": {
               layer_name: { usage:[E], tokens:int, entropy_norm:..., max_share:..., ... }
            },
            "global_": { usage:[E], tokens:int, entropy_norm:..., max_share:..., ... } or None
          }
        """
        per: Dict[str, Any] = {}
        device = None
        total_tokens = 0
        sum_usage = None
        E_ref = None

        for name, gate in self._moe_gates:
            if not hasattr(gate, "lb_p_mean") or getattr(gate, "lb_T", 0) == 0:
                continue

            u = gate.lb_usage_frac.detach()  # [E], no grad
            E = int(gate.lb_E)
            T = int(gate.lb_T)
            k = int(getattr(gate, "lb_k", 1))
            tokens = T * k

            if device is None:
                device = u.device
            per[name] = dict(
                usage=u.cpu(), tokens=tokens, **self._entropy_metrics_from_usage(u)
            )

            # 全局（按 token 加权平均每层 usage；要求各层 E 一致——你的模型是固定 E）
            if E_ref is None:
                E_ref = E
            if E != E_ref:
                # 若某层 E 不一致，跳过全局 usage 聚合（仍返回 per_layer）
                sum_usage = None
            else:
                weighted = u * float(tokens)
                sum_usage = weighted if sum_usage is None else sum_usage + weighted
                total_tokens += tokens

        if sum_usage is None or total_tokens == 0:
            gstats = None
        else:
            g_usage = (sum_usage / float(total_tokens)).cpu()
            gstats = dict(
                usage=g_usage,
                tokens=total_tokens,
                **self._entropy_metrics_from_usage(g_usage),
            )

        # clear 钩子时代码原先用于清 logits；此版本无需要
        return dict(per_layer=per, global_=gstats)

    # ---------------------- 调度入口 ----------------------
    @torch.no_grad()
    def moe_step(self, progress: Optional[float] = None, revive: bool = True) -> None:
        """
        一步闭环：统计 → 反馈 → 调度 → 可选复活。
        progress ∈ [0,1]
        """
        if progress is None:
            progress = 0.0
        stats = self._layerwise_stats(clear=False)
        if self.use_moe_balancer:
            self._feedback_from_stats(stats)
            self._schedule_and_apply(progress)
            if revive and self._moe_cfg.get("enable_revive", True):
                self.revive_dead_experts()
        # 按一定频率打印
        pct = int(progress * 100)
        if pct != self._moe_state["last_percent"] and pct % 10 == 0:
            self._moe_state["last_percent"] = pct
            print(f"\n=== MoE Balancer Step @ {pct:3d}% ===")
            self.print_usage_bars(stats)

    # ---------------------- 辅助损失（逐层，按 token 加权） ----------------------
    def moe_auxiliary_loss(self) -> torch.Tensor:
        """
        逐层 MoE 负载均衡辅助损失（E * sum_i f_i * P_i），
        其中 f_i=lb_usage_frac（DDP 同步），P_i=lb_p_mean（数值=全局同步、梯度=本地），
        最终按 token 数加权到单一标量。
        """
        # 选择设备
        device = None
        for _, gate in self._moe_gates:
            if hasattr(gate, "lb_p_mean"):
                device = gate.lb_p_mean.device
                break
        if device is None:
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device("cpu")

        total_loss = torch.tensor(0.0, device=device)
        total_tokens = 0

        for name, gate in self._moe_gates:
            if not hasattr(gate, "lb_p_mean") or getattr(gate, "lb_T", 0) == 0:
                continue

            E = int(gate.lb_E)
            T_l = int(gate.lb_T)

            # f_i：专家使用频率（跨进程均值，无梯度）
            tpe = gate.lb_usage_frac.to(device)
            tpe_sync = self._ddp_mean_(tpe.detach())

            # P_i：概率均值（数值=同步，梯度=本地）
            rp_local = gate.lb_p_mean.to(device)
            rp_sync = self._ddp_mean_(rp_local.detach().clone())
            rp = rp_sync + (rp_local - rp_local.detach())

            loss_l = (tpe_sync * rp).sum() * E
            total_loss = total_loss + loss_l * T_l
            total_tokens += T_l

        if total_tokens == 0:
            return torch.tensor(0.0, device=device)

        lam_l = float(self._moe_cfg.get("lambda_layer", 1.0))
        return lam_l * (total_loss / total_tokens)

    # ---------------------- 反馈、调度、复活 ----------------------
    @torch.no_grad()
    def _feedback_from_stats(self, stats: Dict[str, Any]) -> None:
        """更新 EMA 与 collapse 计数（仅用 usage 口径）。"""
        if not stats:
            return
        cfg, st = self._moe_cfg, self._moe_state
        alpha = 0.1

        # 全局 EMA（若存在）
        if stats.get("global_") is not None:
            Hn = float(stats["global_"]["entropy_norm"])
            ms = float(stats["global_"]["max_share"])
            ge = st["global_ema"]
            ge["entropy_norm"] = (
                Hn
                if ge["entropy_norm"] is None
                else (1 - alpha) * ge["entropy_norm"] + alpha * Hn
            )
            ge["max_share"] = (
                ms
                if ge["max_share"] is None
                else (1 - alpha) * ge["max_share"] + alpha * ms
            )

        # 分层 EMA + collapse 计数
        thr = cfg["collapse_trig"]
        for name, rec in st["layers"].items():
            if name not in stats["per_layer"]:
                continue
            layer = stats["per_layer"][name]
            Hn_l = float(layer["entropy_norm"])
            ms_l = float(layer["max_share"])
            act_l = float(layer["active_frac"])
            neff_l = float(layer["neff_exp"])
            E_l = int(layer["E"])

            rec["ema_entropy_norm"] = (
                Hn_l
                if rec["ema_entropy_norm"] is None
                else (1 - alpha) * rec["ema_entropy_norm"] + alpha * Hn_l
            )
            rec["ema_max_share"] = (
                ms_l
                if rec["ema_max_share"] is None
                else (1 - alpha) * rec["ema_max_share"] + alpha * ms_l
            )

            reasons = 0
            if ms_l >= thr["max_share"]:
                reasons += 1
            if Hn_l <= thr["Hn"]:
                reasons += 1
            if neff_l <= thr["neff_frac"] * E_l:
                reasons += 1
            if act_l <= thr["active_frac"]:
                reasons += 1
            rec["collapsed_streak"] = rec["collapsed_streak"] + 1 if reasons >= 2 else 0

    @torch.no_grad()
    def _schedule_and_apply(self, progress: float) -> None:
        """根据全局与分层反馈，更新每个 Gate 的 tau/sigma。"""
        cfg, st = self._moe_cfg, self._moe_state
        p = float(min(max(progress, 0.0), 1.0))

        # ========== (1) τ/σ 基线：可选阶段式调度 ==========
        if not cfg.get("enable_stage_schedule", True):
            # 消融：固定使用收敛段的 τ_lo / σ_lo，且不再用进度 p 做阶段插值
            tau = cfg["tau_lo"]
            sigma = cfg["sigma_lo"]
        else:
            # 原来的三阶段 cos 调度
            ph1, ph2 = cfg["explore_frac"], cfg["settle_frac"]

            def cos_interp(x: float, a: float, b: float) -> float:
                return b + 0.5 * (a - b) * (1 + math.cos(math.pi * x))

            if p <= ph1:
                tau, sigma = cfg["tau_hi"], cfg["sigma_hi"]
            elif p <= ph2:
                t = (p - ph1) / max(1e-6, (ph2 - ph1))
                tau = cos_interp(t, cfg["tau_hi"], cfg["tau_lo"])
                sigma = cos_interp(t, cfg["sigma_hi"], cfg["sigma_lo"])
            else:
                tau, sigma = cfg["tau_lo"], cfg["sigma_lo"]

            # 全局反馈（若有）
            ge = st["global_ema"]
            if ge["entropy_norm"] is not None:
                err = cfg["rho_target"] - float(ge["entropy_norm"])
                tau += cfg["k_fb"] * err
                sigma += 0.5 * cfg["k_fb"] * err

        # 分层反馈 + pulse
        for name, g in self._moe_gates:
            rec = st["layers"][name]
            tau_i, sigma_i = tau, sigma

            if rec["ema_entropy_norm"] is not None:
                err_l = cfg["rho_target"] - float(rec["ema_entropy_norm"])
                tau_i += 0.5 * cfg["k_fb"] * err_l
                sigma_i += 0.25 * cfg["k_fb"] * err_l

            if rec["collapsed_streak"] >= 3:
                rec["pulse"] = max(rec["pulse"], 1.0)
                rec["collapsed_streak"] = 0
            if rec["pulse"] > 0.0:
                tau_i += cfg["pulse_gain"] * rec["pulse"]
                sigma_i += 0.3 * cfg["pulse_gain"] * rec["pulse"]
                rec["pulse"] *= cfg["pulse_decay"]

            lo, hi = cfg["tau_bounds"]
            tau_i = float(min(max(tau_i, lo), hi))
            lo, hi = cfg["sigma_bounds"]
            sigma_i = float(min(max(sigma_i, lo), hi))

            if hasattr(g, "temperature"):
                g.temperature = tau_i
            if hasattr(g, "noisy_std"):
                g.noisy_std = sigma_i

    @torch.no_grad()
    def revive_dead_experts(self) -> None:
        """给长期冷门专家加 bias 激励（口径与损失一致，基于 usage_frac）。"""
        cfg = self._moe_cfg
        thr, boost = cfg["revive_usage_thr"], cfg["revive_bias"]

        for name, g in self._moe_gates:
            if not hasattr(g, "lb_usage_frac") or getattr(g, "lb_T", 0) == 0:
                continue
            usage = g.lb_usage_frac.detach()  # [E]
            dead = usage < thr
            if dead.any() and hasattr(g, "bias"):
                g.bias.data[dead] += boost

    # ---------------------- 可视化 ----------------------
    @torch.no_grad()
    def print_usage_bars(
        self, stats: Optional[Dict[str, Any]] = None, width: int = 20
    ) -> None:
        """
        打印专家 usage 的 ASCII 条形图（口径与损失一致）。
        如果不传 stats，就会自动统计一次。
        """
        if stats is None:
            stats = self._layerwise_stats(clear=False)

        def _bar_line(u: torch.Tensor, wid: int = 20) -> List[str]:
            lines = []
            for i, v in enumerate(u.tolist()):
                n = int(round(v * wid))
                bar = "|" * n if n > 0 else ""
                lines.append(f"  {i:02d} {bar:<{wid}} {v:.3f}")
            return lines

        g = stats.get("global_")
        if g is not None:
            print(
                f"\n[Global] E={g['E']} tokens={g['tokens']} "
                f"Hn={g['entropy_norm']:.3f} max_share={g['max_share']:.3f}"
            )
            print("\n".join(_bar_line(g["usage"], width)))

        for lname, rec in stats["per_layer"].items():
            print(
                f"\n[{lname}] E={rec['E']} tokens={rec['tokens']} "
                f"Hn={rec['entropy_norm']:.3f} max_share={rec['max_share']:.3f}"
            )
            print("\n".join(_bar_line(rec["usage"], width)))

    # ---------------------- 外部参数注入接口 ----------------------
    @torch.no_grad()
    def update_clrs_config(
        self,
        *,
        rho_star=None,
        k_fb=None,
        c_pulse=None,
        tau_hi=None,
        sigma_hi=None,
        phi1=None,
        phi2=None,
    ):
        """
        从外部（LHS / 微调脚本）注入 CLRS 超参。
        只更新传入的字段，不覆盖未传字段。
        参数含义（论文对照）:
          rho_star     ≡ ρ*      (目标路由熵 / 均衡程度)
          k_fb         ≡ k_fb    (闭环反馈增益)
          c_pulse      ≡ c_pulse (脉冲恢复强度)
          tau_hi       ≡ τ_hi    (探索期温度上界)
          sigma_hi     ≡ σ_hi    (探索期噪声上界)
          phi1         ≡ φ₁      (探索阶段比例)
          phi2         ≡ φ₂      (收敛阶段开始点)
          lambda_aux   ≡ λ_aux   (MoE 负载均衡辅助损失权重)
        """
        cfg = self._moe_cfg

        if rho_star is not None:
            cfg["rho_target"] = float(rho_star)
        if k_fb is not None:
            cfg["k_fb"] = float(k_fb)
        if c_pulse is not None:
            cfg["pulse_gain"] = float(c_pulse)

        # τ, σ 动态调度基线
        if tau_hi is not None:
            cfg["tau_hi"] = float(tau_hi)
        if sigma_hi is not None:
            cfg["sigma_hi"] = float(sigma_hi)

        # 三阶段进度区间
        if phi1 is not None:
            cfg["explore_frac"] = float(phi1)
        if phi2 is not None:
            cfg["settle_frac"] = float(phi2)

        print(f"[CLRS] Updated config:")
        for k, v in cfg.items():
            if k in (
                "rho_target",
                "k_fb",
                "tau_hi",
                "sigma_hi",
                "explore_frac",
                "settle_frac",
                "pulse_gain",
            ):
                print(f"   {k} = {v}")
        print(f"   aux_loss_weight = {self.aux_loss_weight}")

    # ---------------------- 消融配置助手 ----------------------
    @torch.no_grad()
    def configure_clrs_ablation(
        self,
        *,
        use_clrs: Optional[bool] = None,
        disable_stage_schedule: bool = False,
        disable_pulse: bool = False,
        disable_revive: bool = False,
        disable_lbl: bool = False,
    ) -> None:
        """
        训练前/微调前调用，用于一键设置常用消融配置。

        参数含义：
          use_clrs:
              - None  : 不改动（沿用当前 self.use_moe_balancer）
              - False : 完全关闭 CLRS（仅保留普通 MoE + 可选 LBL）
          disable_stage_schedule:
              - True  : 关闭阶段式 τ/σ 调度（固定 τ_lo / σ_lo）
          disable_pulse:
              - True  : 关闭 collapse pulse（相当于 pulse_gain=0）
          disable_revive:
              - True  : 关闭 revive_dead_experts（不再对冷门专家加 bias）
          disable_lbl:
              - True  : 关闭 LBL 辅助损失（lambda_layer = lambda_global = 0）
        """
        # 1) CLRS 总开关
        if use_clrs is not None:
            self.use_moe_balancer = bool(use_clrs)

        cfg = self._moe_cfg

        # 2) 阶段式调度消融
        if disable_stage_schedule:
            cfg["enable_stage_schedule"] = False

        # 3) pulse 消融
        if disable_pulse:
            cfg["pulse_gain"] = 0.0

        # 4) revive 消融（通过全局开关控制；moe_step 已经支持）
        if disable_revive:
            cfg["enable_revive"] = False

        # 5) LBL 消融（把权重置 0 即可）
        if disable_lbl:
            cfg["lambda_layer"] = 0.0
            cfg["lambda_global"] = 0.0

        # 打印当前关键配置，方便 sanity check
        print("[CLRS Ablation] use_clrs          =", self.use_moe_balancer)
        print(
            "[CLRS Ablation] stage_schedule_on =",
            cfg.get("enable_stage_schedule", True),
        )
        print("[CLRS Ablation] pulse_gain        =", cfg["pulse_gain"])
        print("[CLRS Ablation] revive_enabled    =", cfg.get("enable_revive", True))
        print("[CLRS Ablation] lambda_layer      =", cfg["lambda_layer"])
        print("[CLRS Ablation] lambda_global     =", cfg["lambda_global"])


if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from buildings_bench.models.sub_models_update_02 import (
        Decoder,
        Encoder,
        ModelArgs,
    )

    # ====== 一个简单的 MoE Transformer ======
    class MoETransformer(BalancedMoEMixin):
        def __init__(self, cfg: ModelArgs):
            super().__init__()
            self.cfg = cfg

            # 一个简化的 Encoder+Decoder
            self.encoder = nn.TransformerEncoder(
                Encoder(cfg.n_dense_layers, cfg),
                num_layers=cfg.n_encoder_layers,
                enable_nested_tensor=False,
            )
            self.decoder = nn.TransformerDecoder(
                Decoder(cfg.n_dense_layers, cfg),
                num_layers=cfg.n_decoder_layers,
            )

            self.embedding = nn.Linear(1, cfg.dim)
            self.head = nn.Linear(cfg.dim, 1)

            # 初始化 mixin（收集 Gate）
            self._init_balancer()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: [B, T, 1]
            h = self.embedding(x)
            mem = self.encoder(h)
            out = self.decoder(h, mem)
            return self.head(out)

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = ModelArgs(
        max_seq_len=192,
        dim=64,
        inter_dim=128,
        moe_inter_dim=32,
        n_encoder_layers=2,
        n_decoder_layers=2,
        n_heads=4,
        n_routed_experts=4,
        n_shared_experts=1,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        score_func="softmax",
        route_scale=1.0,
        use_dense=False,
        arch_mode="encdec",
    )

    model = MoETransformer(cfg).to(device).train()
    B, T = 8, 32
    x = torch.randn(B, T, 1, device=device)
    target = torch.randn(B, T, 1, device=device)

    # ---- 记录指标 ----
    history = dict(step=[], aux_loss=[], entropy_norm=[], max_share=[])

    for step in range(1000):
        out = model(x)
        main_loss = F.mse_loss(out, target)
        aux_loss = model.moe_auxiliary_loss()
        loss = main_loss + 0.01 * aux_loss

        loss.backward()
        model.zero_grad()

        progress = step / 1000
        model.moe_step(progress=progress)

        if step % 10 == 0:
            stats = model._layerwise_stats(clear=True)
            model.print_usage_bars(stats, width=30)
            g = stats["global_"]
            history["step"].append(step)
            history["aux_loss"].append(float(aux_loss))
            history["entropy_norm"].append(g["entropy_norm"])
            history["max_share"].append(g["max_share"])

    # ---- 绘图 ----
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    axes[0].plot(history["step"], history["aux_loss"], label="Aux loss")
    axes[0].set_ylabel("Aux loss")
    axes[0].legend()

    axes[1].plot(history["step"], history["entropy_norm"], label="Entropy norm (Hn)")
    axes[1].set_ylabel("Entropy norm")
    axes[1].legend()

    axes[2].plot(history["step"], history["max_share"], label="Max share")
    axes[2].set_ylabel("Max share")
    axes[2].set_xlabel("Step")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("moe_balancing_curves.png")
    print("✅ Saved plot: moe_balancing_curves.png")
