from typing import Dict, Literal, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from buildings_bench.models.base_model import BaseModel
from buildings_bench.models.balanced_moe_mixin import BalancedMoEMixin
from buildings_bench.models.sub_models_update_02 import (
    Encoder,  # ✅ 用 Encoder + is_causal=True 实现 decoder-only（更稳更快）
    ModelArgs,
)


class BuildMoE(BaseModel, BalancedMoEMixin):
    """
    Decoder-only BuildMoE with TimeMoe-style multi-horizon heads.

    - forward(): only uses history ctx, causal stack (decoder-only)
    - inference output: [B, pred_len, out_dim]
    - training loss: multi-head AR loss over all t (sliding windows) + MoE aux loss
    - minimal cache: only caches hidden_states h from last forward for speed
    """

    def __init__(
        self,
        max_context_len: int = 336,
        max_pred_len: int = 168,
        context_len: int = 168,
        pred_len: int = 24,
        num_decoder_layers: int = 10,
        d_model: int = 768,
        nhead: int = 12,
        dim_feedforward: int = 2048,
        num_experts: int = 8,
        top_k: int = 2,
        continuous_loads: bool = True,
        continuous_head: Literal["huber"] = "huber",
        use_dense: bool = False,
        aux_loss_weight: float = 0.01,
        n_shared_experts: int = 0,
        use_moe_balancer: bool = True,
        use_moe_loss: bool = True,
        use_headwise_gate: bool = False,
        use_elementwise_gate: bool = False,
        # TimeMoe-style multi-head horizons
        horizon_lengths: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__(context_len, pred_len, continuous_loads=continuous_loads)

        if continuous_head != "huber":
            raise NotImplementedError(
                "This simplified version only supports huber head."
            )

        self.max_context_len = int(max_context_len)
        self.max_pred_len = int(max_pred_len)
        self.continuous_head = continuous_head

        self.use_dense = bool(use_dense)
        self.use_moe_loss = bool(use_moe_loss)
        self.aux_loss_weight = float(aux_loss_weight)

        # ------- embedding -------
        self.embedding = nn.Linear(1, d_model)

        # ------- config (decoder-only) -------
        self.cfg = ModelArgs(
            max_seq_len=self.max_context_len + self.max_pred_len,
            dim=d_model,
            inter_dim=dim_feedforward,
            moe_inter_dim=d_model // 2,
            n_encoder_layers=0,
            n_decoder_layers=num_decoder_layers,  # for record only
            n_heads=nhead,
            n_routed_experts=num_experts,
            n_shared_experts=n_shared_experts,
            n_activated_experts=top_k,
            n_expert_groups=1,
            n_limited_groups=1,
            score_func="softmax",
            route_scale=1.0,
            use_dense=self.use_dense,
            arch_mode="decoder",
            use_headwise_gate=use_headwise_gate,
            use_elementwise_gate=use_elementwise_gate,
        )

        # 打印配置信息
        print("BuildMoE Config:", self.cfg)

        # ✅ decoder-only：用 TransformerEncoder 堆叠 + is_causal=True
        #    这样不需要 memory（也不会出现 TransformerDecoder memory=None 的坑）
        dec_layer = Encoder(self.cfg.n_dense_layers, self.cfg)
        self.decoder = nn.TransformerEncoder(
            dec_layer,
            num_layers=num_decoder_layers,
            enable_nested_tensor=False,
        )

        # output dim
        self.out_dim = 1

        # ------- multi-horizon heads (TimeMoe style) -------
        if horizon_lengths is None:
            horizon_lengths = [pred_len, max_pred_len]

        # 规范化 + 去重 + 过滤非法
        hs = []
        for h in horizon_lengths:
            try:
                h = int(h)
            except Exception:
                continue
            if h > 0:
                hs.append(h)

        # ✅ 强制包含默认 pred_len，避免你常用 pred_len 时发生 “选更大 horizon 再 slice”
        if int(pred_len) not in hs:
            hs.append(int(pred_len))

        # 可选：也强制包含 max_pred_len（用于长预测）
        if int(max_pred_len) not in hs:
            hs.append(int(max_pred_len))

        self.horizon_lengths = sorted(set(hs))
        if len(self.horizon_lengths) == 0:
            raise ValueError(
                "horizon_lengths must contain at least one positive integer."
            )

        self.horizon_length_map = {h: i for i, h in enumerate(self.horizon_lengths)}
        self.ar_heads = nn.ModuleList(
            [nn.Linear(self.cfg.dim, self.out_dim * h) for h in self.horizon_lengths]
        )

        # ------- minimal cache (for speed) -------
        # 只缓存 hidden_states；不缓存 multi，不缓存 windows
        self._cached_hidden: Optional[torch.Tensor] = None  # [B, ctx, D]
        self._cached_ctx_len: Optional[int] = None

        # ------- MoE balancer init -------
        self._init_balancer(use_moe_balancer)

    # ----------------- horizon selection -----------------
    def _select_horizon_for_predlen(self, pred_len: int) -> int:
        """
        Prefer exact pred_len (no slicing).
        Fallback to smallest horizon >= pred_len (then slicing is unavoidable).
        """
        pred_len = int(pred_len)
        if pred_len in self.horizon_length_map:
            return pred_len
        ge = [h for h in self.horizon_lengths if h >= pred_len]
        if ge:
            return min(ge)
        raise ValueError(
            f"pred_len={pred_len} is larger than all horizon_lengths={self.horizon_lengths}. "
            f"Please include a horizon >= pred_len (or include pred_len itself)."
        )

    # --------------------- forward (decoder-only) ----------------------
    def forward(
        self,
        x: Dict[str, torch.Tensor],
        context_len: Optional[int] = None,
        pred_len: Optional[int] = None,
    ) -> torch.Tensor:
        if context_len is None:
            context_len = self.max_context_len
        if pred_len is None:
            pred_len = self.max_pred_len

        load = x["load"]  # [B, L, 1]
        B, L, _ = load.shape

        # only history
        ctx = load[:, :context_len, :]  # [B, ctx, 1]
        ctx_embed = self.embedding(ctx)  # [B, ctx, D]

        # causal stack
        h = self.decoder(ctx_embed, is_causal=True)  # [B, ctx, D]

        # minimal cache for loss speed
        self._cached_hidden = h
        self._cached_ctx_len = h.shape[1]

        # inference: last token -> predict pred_len horizon (prefer exact, no slice)
        H_sel = self._select_horizon_for_predlen(pred_len)
        head = self.ar_heads[self.horizon_length_map[H_sel]]

        h_last = h[:, -1, :]  # [B, D]
        out = head(h_last).view(B, H_sel, self.out_dim)  # [B, H_sel, 1]

        # ✅ 若 H_sel == pred_len：完全不切片
        if H_sel == int(pred_len):
            return out
        # 否则只能切（和 TimeMoe 一样的 unavoidable fallback）
        return out[:, : int(pred_len), :]

    # --------------------- loss (TimeMoe-style multi-head AR over all t) ---------------------
    def loss(
        self,
        pred: torch.Tensor,
        y: torch.Tensor,  # either [B, pred_len, 1] (your current pipeline) or [B, ctx+pred, 1] (full)
        progress: Optional[float] = None,
        *,
        x: Optional[
            Dict[str, torch.Tensor]
        ] = None,  # ✅ 可选：用来拼 y_full（避免缓存 ctx）
    ) -> torch.Tensor:
        """
        TimeMoe-style AR loss:
        - For each horizon H:
            For each t in [0..ctx-1], predict next H steps from h[t]
            Compare with y_full[t : t+H] (sliding windows)
        """
        # MoE schedule
        if progress is not None and not self.use_dense:
            self.moe_step(progress=progress)

        # MoE aux loss
        aux_loss = torch.tensor(0.0, device=y.device)
        if self.use_moe_loss and not self.use_dense:
            aux_loss = self.moe_auxiliary_loss().to(y.device)

        # hidden cache required for "all t" loss without recomputing forward
        h = getattr(self, "_cached_hidden", None)
        if h is None or h.dim() != 3:
            # fallback: point loss on pred vs y (truncate)
            y_cut = y[:, : pred.shape[1], :]
            err = F.huber_loss(pred, y_cut, delta=1.0, reduction="none")
            return err.mean() + self.aux_loss_weight * aux_loss

        B, T_ctx, D = h.shape

        # ------------------ build y_full (no heavy cache) ------------------
        # If user passes only future segment [B, pred_len, 1], we need history to build y_full.
        # Prefer using x["load"] to grab ctx history (no cache).
        if y.dim() == 2:
            y = y.unsqueeze(-1)

        y_full = y
        if y_full.size(1) < T_ctx + 1:
            # 你当前管线常见：y = load[:, -pred:]
            # 用 x["load"] 的历史段拼 full
            if x is None:
                raise ValueError(
                    "y is too short for TimeMoe-style sliding-window loss, and x is None. "
                    "Please pass x=sample into loss(), or pass full y (ctx+future)."
                )
            load = x["load"]
            if load.dim() == 2:
                load = load.unsqueeze(-1)
            if load.size(1) < T_ctx:
                raise ValueError(
                    f"x['load'] length ({load.size(1)}) < ctx_len ({T_ctx}), cannot build y_full."
                )
            hist = load[:, :T_ctx, :]  # [B, ctx, 1]
            y_full = torch.cat([hist, y_full], dim=1)  # [B, ctx+future, 1]

        L_y = y_full.size(1)

        # flatten for heads
        h_flat = h.reshape(B * T_ctx, D)

        ar_loss = 0.0
        n_used = 0

        # reuse labels base
        labels_base = y_full.transpose(-1, -2)  # [B, 1, L_y]

        # multi-head AR: each t predicts next H steps; compare with sliding windows of y_full
        for head, H in zip(self.ar_heads, self.horizon_lengths):
            H = int(H)
            if L_y < T_ctx + H:
                # not enough labels to supervise this horizon
                continue

            preds = head(h_flat).view(B, T_ctx, H, self.out_dim)  # [B, ctx, H, 1]

            # windows: start at t=0..ctx-1, each window length H
            windows = labels_base.unfold(
                dimension=-1, size=H, step=1
            )  # [B, 1, L_y-H+1, H]
            windows = windows[:, :, :T_ctx, :]  # [B, 1, ctx, H]
            targets = windows.permute(0, 2, 3, 1)  # [B, ctx, H, 1]

            err = F.huber_loss(preds, targets, delta=1.0, reduction="none")
            ar_loss = ar_loss + err.mean()
            n_used += 1

        if n_used == 0:
            # fallback to point loss if no horizon can be used
            y_cut = y_full[
                :, T_ctx : T_ctx + pred.shape[1], :
            ]  # align to future segment if possible
            if y_cut.size(1) != pred.size(1):
                y_cut = y_full[:, : pred.size(1), :]
            err = F.huber_loss(pred, y_cut, delta=1.0, reduction="none")
            main_loss = err.mean()
        else:
            main_loss = ar_loss / n_used

        return main_loss + self.aux_loss_weight * aux_loss

    # --------------------- predict ----------------------
    @torch.no_grad()
    def predict(
        self, x: Dict[str, torch.Tensor], context_len: int = 168, pred_len: int = 24
    ):
        out = self.forward(
            x, context_len=context_len, pred_len=pred_len
        )  # [B, pred_len, 1]
        return out, out

    # --------------------- finetune param groups ----------------------
    def unfreeze_and_get_parameters_for_finetuning(
        self,
        strategy: Literal[
            "head",
            "head_last_block",
            "experts_only",
            "router_experts",
            "norms_biases",
            "full",
        ] = "router_experts",
        *,
        base_lr: float = 3e-5,
        head_lr: float = 1e-4,
        router_lr: Optional[float] = None,  # None -> base_lr
        wd: float = 0.01,
    ):
        """
        Return optimizer parameter groups for fine-tuning.
        - "head": only train multi-horizon heads (ar_heads)
        - "head_last_block": train heads + last decoder block
        - "experts_only": train only MoE experts (+ heads)
        - "router_experts": train MoE experts + router/gate (+ heads)
        - "norms_biases": train all norms & biases (+ heads)
        - "full": train everything (head_lr for heads, base_lr for body)

        Note: Norm/bias uses no weight decay.
        """
        # 1) freeze all
        for p in self.parameters():
            p.requires_grad = False

        def is_norm_or_bias(n: str, p: torch.nn.Parameter) -> bool:
            if p.ndim == 1 and (
                "bias" in n
                or "bn" in n
                or "norm" in n
                or "ln" in n
                or "layernorm" in n.lower()
            ):
                return True
            if (
                "norm" in n.lower() or "ln" in n.lower() or "layer_norm" in n.lower()
            ) and p.ndim == 1:
                return True
            return False

        def add_params(named_params, lr, weight_decay, train=True, bucket=None):
            decay, no_decay = [], []
            for n, p in named_params:
                if p is None or (not p.dtype.is_floating_point):
                    continue
                if train:
                    p.requires_grad = True
                if is_norm_or_bias(n, p):
                    no_decay.append(p)
                else:
                    decay.append(p)
            groups = []
            if decay:
                groups.append({"params": decay, "lr": lr, "weight_decay": weight_decay})
            if no_decay:
                groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})
            if bucket is not None:
                bucket.extend(groups)
            return groups

        param_groups = []

        # helper: pick MoE submodules inside decoder
        def pick_moe_submodules(module: nn.Module, train_router: bool):
            r_lr = router_lr if (router_lr is not None) else base_lr
            for _, mod in module.named_modules():
                if hasattr(mod, "experts") and isinstance(
                    getattr(mod, "experts"), nn.ModuleList
                ):
                    add_params(
                        mod.experts.named_parameters(), base_lr, wd, bucket=param_groups
                    )
                    if train_router and hasattr(mod, "gate"):
                        add_params(
                            mod.gate.named_parameters(), r_lr, wd, bucket=param_groups
                        )

        if strategy == "head":
            add_params(
                self.ar_heads.named_parameters(), head_lr, wd, bucket=param_groups
            )

        elif strategy == "head_last_block":
            add_params(
                self.ar_heads.named_parameters(), head_lr, wd, bucket=param_groups
            )
            if hasattr(self.decoder, "layers") and len(self.decoder.layers) > 0:
                last_block = self.decoder.layers[-1]
                add_params(
                    last_block.named_parameters(), base_lr, wd, bucket=param_groups
                )

        elif strategy in ("experts_only", "router_experts"):
            train_router = strategy == "router_experts"
            pick_moe_submodules(self.decoder, train_router=train_router)
            add_params(
                self.ar_heads.named_parameters(), head_lr, wd, bucket=param_groups
            )

        elif strategy == "norms_biases":
            for n, p in self.named_parameters():
                if is_norm_or_bias(n, p):
                    p.requires_grad = True
            add_params(
                self.ar_heads.named_parameters(), head_lr, wd, bucket=param_groups
            )

            in_group = {id(p) for g in param_groups for p in g["params"]}
            others = [
                (n, p)
                for n, p in self.named_parameters()
                if p.requires_grad and id(p) not in in_group
            ]
            add_params(others, base_lr, wd, train=False, bucket=param_groups)

        elif strategy == "full":
            add_params(
                self.ar_heads.named_parameters(), head_lr, wd, bucket=param_groups
            )
            head_param_set = {id(p) for g in param_groups for p in g["params"]}
            body = [
                (n, p)
                for n, p in self.named_parameters()
                if id(p) not in head_param_set
            ]
            add_params(body, base_lr, wd, bucket=param_groups)

        else:
            raise ValueError(f"Unknown finetune strategy: {strategy}")

        if not param_groups:
            add_params(
                self.ar_heads.named_parameters(), head_lr, wd, bucket=param_groups
            )

        return param_groups

    def load_from_checkpoint(self, checkpoint_path: str):
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
    import torch

    # 固定随机种子，方便复现
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ==== 初始化模型 ====
    model = BuildMoE(arch_mode="decoder").to(device)

    # ==== 统计 MoE 参数量 ====
    stats = moe_static_params(model)
    for k, v in stats.items():
        print(f"{k}: {v}")

    # ==== 伪造输入数据 ====
    B = 4
    ctx, pred = 96, 22
    load = torch.randn(B, ctx + pred, 1, device=device)  # (B, 120, 1)
    sample = {"load": load}
    target = load[:, -pred:]  # 预测的 ground truth

    # ==== 前向推理 ====
    out = model(sample, context_len=ctx, pred_len=pred)  # (B, pred, 1)
    print("Forward out:", out.shape)

    # ==== 损失函数 ====
    progress = 0.1  # 模拟训练进度 (10%)
    loss = model.loss(out, target, progress=progress, x=sample)
    print("Loss (with aux):", float(loss))

    # ==== 反向传播 ====
    loss.backward()
    print("Backward pass OK")
