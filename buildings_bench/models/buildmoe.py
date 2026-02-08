# -*- coding: utf-8 -*-
from typing import Dict, Literal, Optional, Tuple

from buildings_bench.models.balanced_moe_mixin import BalancedMoEMixin
from buildings_bench.models.base_model import BaseModel
from buildings_bench.models.sub_models_update_02 import (
    Decoder,
    Encoder,
    ModelArgs,
)

import torch
import torch.nn.functional as F
from torch import nn


class QueryMixin(nn.Module):
    """
    Query 生成混入类：
      - query_mode="zero" / "learned" / "multi_expert"
      - 输出 queries: [B, pred_len, D]

    用法：
      - 模型类继承 QueryMixin
      - 在 __init__ 末尾调用 self._init_queries(...)
      - forward 中调用: query = self.build_queries(ctx_embed, pred_len)
    """

    # ---------------------- 初始化 ----------------------
    def _init_queries(
        self,
        *,
        query_mode: Literal["zero", "learned", "multi_expert"] = "zero",
        query_top_k: Optional[int] = None,
        d_model: int,
        max_pred_len: int,
        num_experts: int,
        # multi_expert 专用超参
        num_latents: int = 8,
        init_std: float = 0.02,
    ) -> None:
        self.query_mode = query_mode
        self.query_top_k = query_top_k

        self._q_d_model = int(d_model)
        self._q_max_pred_len = int(max_pred_len)

        # --- learned ---
        if self.query_mode == "learned":
            self.learned_queries = nn.Parameter(
                torch.randn(max_pred_len, d_model) * init_std
            )
        else:
            self.learned_queries = None

        # --- multi_expert ---
        if self.query_mode == "multi_expert":
            self.num_query_experts = int(num_experts)

            # [E, max_pred, D]
            self.query_experts = nn.Parameter(
                torch.randn(self.num_query_experts, max_pred_len, d_model) * init_std
            )

            # step embedding
            self.query_pos = nn.Parameter(torch.randn(max_pred_len, d_model) * init_std)
            self.pred_len_emb = nn.Embedding(max_pred_len + 1, d_model)

            # latent tokens
            self.num_latents = int(num_latents)
            self.latents = nn.Parameter(
                torch.randn(self.num_latents, d_model) * init_std
            )

            # stage1: latents attend ctx
            self.lat_q = nn.Linear(d_model, d_model, bias=False)
            self.ctx_k = nn.Linear(d_model, d_model, bias=False)
            self.ctx_v = nn.Linear(d_model, d_model, bias=False)
            self.lat_out = nn.Linear(d_model, d_model, bias=False)

            # stage2: step attend latents
            self.step_q = nn.Linear(d_model, d_model, bias=False)
            self.lat_k = nn.Linear(d_model, d_model, bias=False)
            self.lat_v = nn.Linear(d_model, d_model, bias=False)
            self.step_out = nn.Linear(d_model, d_model, bias=False)

            # gate mlp
            self.query_gate_mlp = nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, self.num_query_experts),
            )

        else:
            # zero/learned：不创建 multi_expert 相关参数
            self.num_query_experts = 0
            self.query_experts = None
            self.query_pos = None
            self.pred_len_emb = None
            self.num_latents = 0
            self.latents = None
            self.lat_q = None
            self.ctx_k = None
            self.ctx_v = None
            self.lat_out = None
            self.step_q = None
            self.lat_k = None
            self.lat_v = None
            self.step_out = None
            self.query_gate_mlp = None

    # ---------------------- 公共入口 ----------------------
    def build_queries(self, ctx_embed: torch.Tensor, pred_len: int) -> torch.Tensor:
        """
        ctx_embed: [B, ctx, D]
        return:   [B, pred, D]
        """
        if self.query_mode == "zero":
            return self._build_queries_zero(ctx_embed, pred_len)

        if self.query_mode == "learned":
            return self._build_queries_learned(ctx_embed, pred_len)

        if self.query_mode == "multi_expert":
            return self._build_queries_multi_expert(ctx_embed, pred_len)

        raise ValueError(f"Unknown query_mode: {self.query_mode}")

    # ---------------------- 具体实现 ----------------------
    def _build_queries_zero(
        self, ctx_embed: torch.Tensor, pred_len: int
    ) -> torch.Tensor:
        B, _, D = ctx_embed.shape
        return torch.zeros(
            B, pred_len, D, device=ctx_embed.device, dtype=ctx_embed.dtype
        )

    def _build_queries_learned(
        self, ctx_embed: torch.Tensor, pred_len: int
    ) -> torch.Tensor:
        assert (
            self.learned_queries is not None
        ), "query_mode!=learned but learned_queries is None"
        B, _, D = ctx_embed.shape
        q = self.learned_queries[:pred_len, :].to(
            device=ctx_embed.device, dtype=ctx_embed.dtype
        )
        return q.unsqueeze(0).expand(B, pred_len, D)

    @staticmethod
    def _sdpa_1head(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        D = q.size(-1)
        attn = torch.matmul(q, k.transpose(-2, -1)) / (D**0.5)
        attn = torch.softmax(attn, dim=-1)
        return torch.matmul(attn, v)

    def _build_queries_multi_expert(
        self, ctx_embed: torch.Tensor, pred_len: int
    ) -> torch.Tensor:
        # --- sanity ---
        assert self.query_experts is not None
        assert self.query_pos is not None
        assert self.pred_len_emb is not None
        assert self.latents is not None
        assert self.query_gate_mlp is not None
        assert self.lat_q is not None
        assert self.ctx_k is not None
        assert self.ctx_v is not None
        assert self.lat_out is not None
        assert self.step_q is not None
        assert self.lat_k is not None
        assert self.lat_v is not None
        assert self.step_out is not None

        B, T, D = ctx_embed.shape
        device, dtype = ctx_embed.device, ctx_embed.dtype

        # ----- step base embedding (pos + pred_len emb) -----
        pos = (
            self.query_pos[:pred_len].unsqueeze(0).to(device=device, dtype=dtype)
        )  # [1,pred,D]
        len_vec = self.pred_len_emb(torch.tensor([pred_len], device=device)).to(
            dtype=dtype
        )  # [1,D]
        len_vec = len_vec.unsqueeze(1).expand(B, pred_len, D)  # [B,pred,D]
        step_base = pos.expand(B, pred_len, D) + len_vec  # [B,pred,D]

        # ===== Stage 1: latents attend ctx =====
        lat = (
            self.latents.unsqueeze(0).expand(B, -1, -1).to(device=device, dtype=dtype)
        )  # [B,M,D]
        q1 = self.lat_q(lat)
        k1 = self.ctx_k(ctx_embed)
        v1 = self.ctx_v(ctx_embed)
        lat_ctx = self._sdpa_1head(q1, k1, v1)
        lat_ctx = self.lat_out(lat_ctx)

        # ===== Stage 2: step tokens attend latents =====
        q2 = self.step_q(step_base)
        k2 = self.lat_k(lat_ctx)
        v2 = self.lat_v(lat_ctx)
        step_ctx = self._sdpa_1head(q2, k2, v2)
        step_ctx = self.step_out(step_ctx)

        # ===== Gate over experts =====
        gate_inp = torch.cat([step_base, step_ctx], dim=-1)  # [B,pred,2D]
        logits = self.query_gate_mlp(gate_inp)  # [B,pred,E]

        if (
            self.query_top_k is not None
            and 0 < self.query_top_k < self.num_query_experts
        ):
            topv, topi = torch.topk(logits, k=self.query_top_k, dim=-1)  # [B,pred,k]
            masked = torch.full_like(logits, float("-inf"))
            masked.scatter_(-1, topi, topv)
            weights = torch.softmax(masked, dim=-1)
        else:
            weights = torch.softmax(logits, dim=-1)

        # experts templates: [E,pred,D] -> [1,pred,E,D]
        exp_q = self.query_experts[:, :pred_len, :].to(
            device=device, dtype=dtype
        )  # [E,pred,D]
        exp_q = exp_q.permute(1, 0, 2).unsqueeze(0)  # [1,pred,E,D]

        # weighted sum -> [B,pred,D]
        queries = (weights.unsqueeze(-1) * exp_q).sum(dim=2)
        return queries


class RevIN(nn.Module):
    """
    Reversible Instance Normalization for time series.
    - 只做 per-sample 的 (x - mu) / std
    - mu/std 必须来自 context (避免泄漏)
    """

    def __init__(self, num_features: int = 1, eps: float = 1e-5, affine: bool = False):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(1, 1, num_features))
            self.beta = nn.Parameter(torch.zeros(1, 1, num_features))

    @torch.no_grad()
    def _get_stats(self, x_ctx: torch.Tensor):
        # x_ctx: [B, T_ctx, C]
        mu = x_ctx.mean(dim=1, keepdim=True)  # [B,1,C]
        var = x_ctx.var(dim=1, keepdim=True, unbiased=False)  # [B,1,C]
        std = torch.sqrt(var + self.eps)  # [B,1,C]
        return mu, std

    def norm(self, x: torch.Tensor, mu: torch.Tensor, std: torch.Tensor):
        # x: [B,T,C]
        x = (x - mu) / std
        if self.affine:
            x = x * self.gamma + self.beta
        return x

    def denorm(self, x: torch.Tensor, mu: torch.Tensor, std: torch.Tensor):
        if self.affine:
            x = (x - self.beta) / (self.gamma + 1e-12)
        x = x * std + mu
        return x


class RevINMixin(nn.Module):
    """
    RevIN 归一化/反归一化的混入类（no leakage: stats ONLY from ctx）。

    用法：
      - 模型类继承 RevINMixin
      - 在 __init__() 末尾调用 self._init_revin(...)
      - forward 里：
            ctx_in, mu, std = self.revin_norm_ctx(ctx)
            ...
            y_hat = self.revin_denorm(y_hat, mu, std)
      - 对于 encoder-only: 需要 future 占位输入（norm 后为 0），用：
            future_in = self.revin_future_zeros(mu, std, B, pred_len, device, dtype)
    """

    # ---------------------- 初始化 ----------------------
    def _init_revin(
        self,
        *,
        use_revin: bool = False,
        revin_eps: float = 1e-5,
        revin_affine: bool = False,
        num_features: int = 1,
    ) -> None:
        self.use_revin = bool(use_revin)
        self.revin = (
            RevIN(num_features=num_features, eps=revin_eps, affine=revin_affine)
            if self.use_revin
            else None
        )

    # ---------------------- 核心接口 ----------------------
    def revin_norm_ctx(
        self, ctx: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        ctx: [B, ctx, C]  (你这里 C=1)
        return:
          ctx_in: [B, ctx, C]
          mu, std: [B, 1, C]
        """
        if self.use_revin and (self.revin is not None):
            mu, std = self.revin._get_stats(ctx)  # stats ONLY from ctx
            ctx_in = self.revin.norm(ctx, mu, std)
            return ctx_in, mu, std
        return ctx, None, None

    def revin_denorm(
        self,
        y_hat: torch.Tensor,
        mu: Optional[torch.Tensor],
        std: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        y_hat: [B, ..., C]
        """
        if (
            self.use_revin
            and (self.revin is not None)
            and (mu is not None)
            and (std is not None)
        ):
            return self.revin.denorm(y_hat, mu, std)
        return y_hat

    def revin_future_zeros(
        self,
        mu: Optional[torch.Tensor],
        std: Optional[torch.Tensor],
        *,
        B: int,
        pred_len: int,
        device: torch.device,
        dtype: torch.dtype,
        channels: int = 1,
    ) -> torch.Tensor:
        """
        构造 future 占位输入，使得在 RevIN norm 后 future token == 0（严格 no-leakage）
        - 若启用 RevIN 且 mu/std 可用：future_raw = mu expand，然后 norm -> 0
        - 否则：直接返回 0

        return: [B, pred_len, C]
        """
        if (
            self.use_revin
            and (self.revin is not None)
            and (mu is not None)
            and (std is not None)
        ):
            # mu: [B,1,C] -> [B,pred,C]
            future_raw = mu.expand(B, pred_len, channels).to(device=device, dtype=dtype)
            future_in = self.revin.norm(future_raw, mu, std)  # == 0
            return future_in
        return torch.zeros(B, pred_len, channels, device=device, dtype=dtype)


class BuildMoE(BaseModel, BalancedMoEMixin, RevINMixin, QueryMixin):

    def __init__(
        self,
        max_context_len: int = 336,
        max_pred_len: int = 168,
        context_len: int = 168,
        pred_len: int = 24,
        num_encoder_layers: int = 8,
        num_decoder_layers: int = 10,
        d_model: int = 768,
        nhead: int = 12,
        dim_feedforward: int = 2048,
        num_experts: int = 8,
        top_k: int = 2,
        continuous_loads: bool = True,
        continuous_head: Literal["huber"] = "huber",
        use_dense: bool = False,
        arch_mode: Literal["encdec", "encoder", "decoder"] = "encdec",
        aux_loss_weight: float = 0.01,
        n_shared_experts: int = 0,
        use_moe_balancer: bool = True,
        use_moe_loss: bool = True,
        use_headwise_gate: bool = False,
        use_elementwise_gate: bool = False,
        use_revin: bool = False,
        revin_affine: bool = False,
        revin_eps: float = 1e-5,
        query_mode: Literal["zero", "learned", "multi_expert"] = "zero",
        query_top_k: Optional[int] = None,
        use_causal: bool = True,
        **kwargs,
    ):
        super().__init__(context_len, pred_len, continuous_loads=continuous_loads)
        self.max_context_len = max_context_len
        self.max_pred_len = max_pred_len
        self.continuous_head = continuous_head
        self.arch_mode = arch_mode
        self.use_dense = use_dense
        self.aux_loss_weight = aux_loss_weight
        self.use_moe_loss = use_moe_loss
        self.use_causal = use_causal

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
            n_shared_experts=n_shared_experts,
            n_activated_experts=top_k,
            n_expert_groups=1,
            n_limited_groups=1,
            score_func="softmax",
            route_scale=1.0,
            use_dense=use_dense,
            arch_mode=arch_mode,
            use_headwise_gate=use_headwise_gate,
            use_elementwise_gate=use_elementwise_gate,
        )

        if self.arch_mode == "encoder" or self.arch_mode == "encdec":
            enc_layer = Encoder(self.cfg.n_dense_layers, self.cfg)
            self.encoder = nn.TransformerEncoder(
                enc_layer, num_encoder_layers, enable_nested_tensor=False
            )
        if self.arch_mode == "decoder" or self.arch_mode == "encdec":
            dec_layer = Decoder(self.cfg.n_dense_layers, self.cfg)
            self.decoder = nn.TransformerDecoder(dec_layer, num_decoder_layers)

        self._init_revin(
            use_revin=use_revin,
            revin_eps=revin_eps,
            revin_affine=revin_affine,
            num_features=1,
        )

        self._init_queries(
            query_mode=query_mode,
            query_top_k=query_top_k,
            d_model=d_model,
            max_pred_len=max_pred_len,
            num_experts=num_experts,
            num_latents=8,
            init_std=0.02,
        )

        out_dim = 1 if continuous_head in ("mse", "huber") else 2
        self.head = nn.Linear(self.cfg.dim, out_dim)

        self._init_balancer(use_moe_balancer)

    # --------------------- forward ----------------------
    def forward(self, x, context_len=None, pred_len=None):
        context_len = self.max_context_len if context_len is None else context_len
        pred_len = self.max_pred_len if pred_len is None else pred_len

        load = x["load"]
        assert load.size(1) == context_len + pred_len
        B = load.size(0)
        ctx = load[:, :context_len]  # [B, ctx, 1]

        # 1) RevIN (no leakage)
        ctx_in, mu, std = self.revin_norm_ctx(ctx)

        # 2) shared ctx_embed + query (ALL modes)
        ctx_embed = self.embedding(ctx_in)
        query = self.build_queries(ctx_embed, pred_len)

        # 3) architectures
        if self.arch_mode == "encdec":
            mem = self.encoder(ctx_embed)
            out = self.decoder(query, mem, tgt_is_causal=self.use_causal)
            y_hat = self.head(out)
            return self.revin_denorm(y_hat, mu, std)

        if self.arch_mode == "decoder":
            inp = torch.cat([ctx_embed, query], dim=1)
            h = self.decoder(tgt=inp, memory=None, tgt_is_causal=True)
            pred_h = h[:, -pred_len:, :]
            y_hat = self.head(pred_h)
            return self.revin_denorm(y_hat, mu, std)

        if self.arch_mode == "encoder":
            inp = torch.cat([ctx_embed, query], dim=1)
            h = self.encoder(inp, is_causal=False)
            pred_h = h[:, -pred_len:, :]
            y_hat = self.head(pred_h)
            return self.revin_denorm(y_hat, mu, std)

        raise ValueError(f"Unknown arch_mode: {self.arch_mode}")

    # --------------------- loss（已移除分解正则） ---------------------
    def loss(
        self, pred: torch.Tensor, y: torch.Tensor, progress: Optional[float] = None
    ) -> torch.Tensor:
        # === 更新 MoE 调度 ===
        if progress is not None and not self.use_dense:
            self.moe_step(progress=progress)

        # === 均衡损失 ===
        aux_loss = torch.tensor(0.0, device=pred.device)
        if self.use_moe_loss and not self.use_dense:
            aux_loss = self.moe_auxiliary_loss()

        # === 主任务损失 ===
        if self.continuous_head == "huber":
            err = F.huber_loss(pred, y, delta=1.0, reduction="none")
            loss = err.mean()
        else:
            raise NotImplementedError()

        return loss + self.aux_loss_weight * aux_loss

    @torch.no_grad()
    def predict(
        self, x: Dict[str, torch.Tensor], context_len: int = 168, pred_len: int = 24
    ):
        preds = self.forward(x, context_len, pred_len)
        if self.continuous_head in ("mse", "huber"):
            return preds, preds
        else:
            return preds[..., :1], preds

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
        router_lr: Optional[float] = None,  # None -> 使用 base_lr
        wd: float = 0.01,
    ):
        """
        返回 optimizer 参数组（parameter groups），用于精细化微调。
        - 不同 group 可设置不同 lr/weight_decay。
        - 自动将 Norm/bias 放入 no_decay 组。
        - 针对 MoE：可只解冻 experts / gate(router)。
        """
        # 1) 先全部冻结
        for p in self.parameters():
            p.requires_grad = False

        # 2) 辅助分类工具
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
                if p is None or (hasattr(p, "dtype") and not p.dtype.is_floating_point):
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

        # 3) 不同策略选择
        param_groups = []

        if strategy == "head":
            add_params(self.head.named_parameters(), head_lr, wd, bucket=param_groups)

        elif strategy == "head_last_block":
            add_params(self.head.named_parameters(), head_lr, wd, bucket=param_groups)
            if self.arch_mode in ("decoder", "encdec") and hasattr(self, "decoder"):
                if hasattr(self.decoder, "layers") and len(self.decoder.layers) > 0:
                    last_block = self.decoder.layers[-1]
                    add_params(
                        last_block.named_parameters(), base_lr, wd, bucket=param_groups
                    )
            elif self.arch_mode == "encoder" and hasattr(self, "encoder"):
                if hasattr(self.encoder, "layers") and len(self.encoder.layers) > 0:
                    last_block = self.encoder.layers[-1]
                    add_params(
                        last_block.named_parameters(), base_lr, wd, bucket=param_groups
                    )

        elif strategy in ("experts_only", "router_experts"):
            train_router = strategy == "router_experts"
            r_lr = router_lr if (router_lr is not None) else base_lr

            def pick_moe_submodules(module: nn.Module):
                for name, mod in module.named_modules():
                    if hasattr(mod, "experts") and isinstance(
                        getattr(mod, "experts"), nn.ModuleList
                    ):
                        add_params(
                            mod.experts.named_parameters(),
                            base_lr,
                            wd,
                            bucket=param_groups,
                        )
                        if train_router and hasattr(mod, "gate"):
                            add_params(
                                mod.gate.named_parameters(),
                                r_lr,
                                wd,
                                bucket=param_groups,
                            )

            if hasattr(self, "encoder"):
                pick_moe_submodules(self.encoder)
            if hasattr(self, "decoder"):
                pick_moe_submodules(self.decoder)
            add_params(self.head.named_parameters(), head_lr, wd, bucket=param_groups)

        elif strategy == "norms_biases":
            for n, p in self.named_parameters():
                if is_norm_or_bias(n, p):
                    p.requires_grad = True
            add_params(self.head.named_parameters(), head_lr, wd, bucket=param_groups)
            others = [
                (n, p)
                for n, p in self.named_parameters()
                if p.requires_grad
                and p not in sum([g["params"] for g in param_groups], [])
            ]
            add_params(others, base_lr, wd, train=False, bucket=param_groups)

        elif strategy == "full":
            add_params(self.head.named_parameters(), head_lr, wd, bucket=param_groups)
            body = []
            head_param_set = {id(p) for g in param_groups for p in g["params"]}
            for n, p in self.named_parameters():
                if id(p) not in head_param_set:
                    body.append((n, p))
            add_params(body, base_lr, wd, bucket=param_groups)

        else:
            raise ValueError(f"Unknown finetune strategy: {strategy}")

        if not param_groups:
            add_params(self.head.named_parameters(), head_lr, wd, bucket=param_groups)

        return param_groups

    def load_from_checkpoint(self, checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu")["model"]
        state = {k.replace("module.", ""): v for k, v in state.items()}
        self.load_state_dict(state)


def moe_static_params(model: torch.nn.Module):
    """
    一次性静态统计 MoE 参数量（无需前向）：
      activated_params_total = shared_params + Σ_layer (top_k[layer] * per_expert_params[layer])

    约定/假设（与你的实现匹配）：
      - 一个 MoE 层满足：模块具有 .experts (nn.ModuleList)
      - "路由专家"指 MoE.experts；shared_experts 视为常驻参数（始终参与计算）
      - 每层专家结构一致；若不一致则以该层“专家平均参数量”作为 per_expert_params[layer]
    """
    import torch.nn as nn

    moe_layers = []  # [(mod_name, mod)]
    for name, mod in model.named_modules():
        if hasattr(mod, "experts") and isinstance(
            getattr(mod, "experts"), nn.ModuleList
        ):
            moe_layers.append((name or "root", mod))

    per_expert_params_per_layer = []
    experts_per_layer = []
    top_k_per_layer = []
    routed_expert_params_total = 0

    for layer_name, mod in moe_layers:
        experts: nn.ModuleList = mod.experts
        n_exp = len(experts)
        experts_per_layer.append(n_exp)

        if n_exp == 0:
            per_expert_params_per_layer.append(0)
        else:
            sizes = []
            for e in experts:
                sizes.append(sum(p.numel() for p in e.parameters() if p.requires_grad))
            if len(set(sizes)) == 1:
                per_exp = sizes[0]
            else:
                per_exp = int(sum(sizes) // n_exp)
            per_expert_params_per_layer.append(per_exp)
            routed_expert_params_total += per_exp * n_exp

        tk = None
        if hasattr(mod, "gate") and hasattr(mod.gate, "topk"):
            try:
                tk = int(mod.gate.topk)
            except Exception:
                tk = None
        top_k_per_layer.append(tk if tk is not None else 0)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    shared_params = total_params - routed_expert_params_total

    activated_expert_params_sum = 0
    for per_exp, tk in zip(per_expert_params_per_layer, top_k_per_layer):
        activated_expert_params_sum += per_exp * int(tk)

    activated_params_total = shared_params + activated_expert_params_sum

    out = {
        "total_params": int(total_params),
        "shared_params": int(shared_params),
        "routed_expert_params_total": int(routed_expert_params_total),
        "num_moe_layers": len(moe_layers),
        "experts_per_layer": experts_per_layer,
        "per_expert_params_per_layer": per_expert_params_per_layer,
        "top_k_per_layer": top_k_per_layer,
        "activated_expert_params_sum": int(activated_expert_params_sum),
        "activated_params_total": int(activated_params_total),
        "activation_rate": float(activated_params_total / max(1, total_params)),
    }
    return out


# ======================= Quick self-test =======================
if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ✅ 默认：单 learnable query（use_multi_query_experts=False）
    model = BuildMoE(arch_mode="encoder", query_mode="multi_expert", use_dense=True).to(
        device
    )

    # 如需启用多专家 query：
    # model = BuildMoE(arch_mode="decoder", use_multi_query_experts=True, query_top_k=2).to(device)

    stats = moe_static_params(model)
    for k, v in stats.items():
        print(f"{k}: {v}")

    B = 4
    ctx, pred = 96, 24
    load = torch.randn(B, ctx + pred, 1, device=device)
    sample = {"load": load}
    target = load[:, -pred:]

    out = model(sample, context_len=ctx, pred_len=pred)
    print("Forward out:", out.shape)

    progress = 0.1
    loss = model.loss(out, target, progress=progress)
    print("Loss (with aux):", float(loss))

    loss.backward()
    print("Backward pass OK")
