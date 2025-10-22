from typing import Dict, Literal, Optional

from buildings_bench.models.balanced_moe_mixin import BalancedMoEMixin
import torch
import torch.nn.functional as F
from buildings_bench.models.base_model import BaseModel
from buildings_bench.models.sub_models_update_02 import (
    Decoder,
    Encoder,
    ModelArgs,
)
from torch import nn


class BuildMoE(BaseModel, BalancedMoEMixin):

    def __init__(
        self,
        max_context_len: int = 336,
        max_pred_len: int = 168,
        context_len: int = 168,
        pred_len: int = 24,
        num_encoder_layers: int = 8,
        num_decoder_layers: int = 22,
        d_model: int = 768,
        nhead: int = 12,
        dim_feedforward: int = 2048,
        num_experts: int = 8,
        top_k: int = 2,
        continuous_loads: bool = True,
        continuous_head: Literal["huber"] = "huber",
        use_dense: bool = False,
        arch_mode: Literal["encdec", "encoder", "decoder"] = "decoder",
        aux_loss_weight: float = 0.01,
        n_shared_experts: int = 0,
        use_moe_balancer: bool = True,
        use_moe_loss: bool = True,
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

        self._init_balancer(use_moe_balancer)

    # --------------------- forward ----------------------
    def forward(
        self,
        x: Dict[str, torch.Tensor],
        context_len: Optional[int] = None,
        pred_len: Optional[int] = None,
    ):
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
            # 常见 LayerNorm/GroupNorm 权重也是 1D，不建议做 weight decay
            if (
                "norm" in n.lower() or "ln" in n.lower() or "layer_norm" in n.lower()
            ) and p.ndim == 1:
                return True
            return False

        def add_params(named_params, lr, weight_decay, train=True, bucket=None):
            decay, no_decay = [], []
            for n, p in named_params:
                if p is None or not p.dtype.is_floating_point:
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
            # 头部
            add_params(self.head.named_parameters(), head_lr, wd, bucket=param_groups)
            # 最后一块（根据架构）
            if self.arch_mode in ("decoder", "encdec") and hasattr(self, "decoder"):
                # TransformerDecoder has .layers: List[DecoderLayer]
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
            # 只训 MoE 的 experts；如果是 router_experts，再训 gate/router
            train_router = strategy == "router_experts"
            r_lr = router_lr if (router_lr is not None) else base_lr

            def pick_moe_submodules(module: nn.Module):
                for name, mod in module.named_modules():
                    # 经验性判断：有 .experts 的就是 MoE 层
                    if hasattr(mod, "experts") and isinstance(
                        getattr(mod, "experts"), nn.ModuleList
                    ):
                        # experts
                        add_params(
                            mod.experts.named_parameters(),
                            base_lr,
                            wd,
                            bucket=param_groups,
                        )
                        # gate/router
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
            # 一般也把 head 打开一点，帮助收敛
            add_params(self.head.named_parameters(), head_lr, wd, bucket=param_groups)

        elif strategy == "norms_biases":
            # 解冻所有 Norm & bias（轻量适配）
            for n, p in self.named_parameters():
                if is_norm_or_bias(n, p):
                    p.requires_grad = True
            # 依然建议把 head 解冻
            add_params(self.head.named_parameters(), head_lr, wd, bucket=param_groups)
            # 其余被标记为 requires_grad=True 的参数放到 base_lr
            others = [
                (n, p)
                for n, p in self.named_parameters()
                if p.requires_grad
                and p not in sum([g["params"] for g in param_groups], [])
            ]
            add_params(others, base_lr, wd, train=False, bucket=param_groups)

        elif strategy == "full":
            # 区分 head 与 body 的学习率；norm/bias 无 weight decay
            add_params(self.head.named_parameters(), head_lr, wd, bucket=param_groups)
            # 剩余主体
            body = []
            head_param_set = {id(p) for g in param_groups for p in g["params"]}
            for n, p in self.named_parameters():
                if id(p) not in head_param_set:
                    body.append((n, p))
            add_params(body, base_lr, wd, bucket=param_groups)

        else:
            raise ValueError(f"Unknown finetune strategy: {strategy}")

        # 4) 兜底：万一没挑到参数（极端情况）
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
    model = BuildMoE().to(device)

    # ==== 统计 MoE 参数量 ====
    stats = moe_static_params(model)
    for k, v in stats.items():
        print(f"{k}: {v}")

    # ==== 伪造输入数据 ====
    B = 4
    ctx, pred = 96, 24
    load = torch.randn(B, ctx + pred, 1, device=device)  # (B, 120, 1)
    sample = {"load": load}
    target = load[:, -pred:]  # 预测的 ground truth

    # ==== 前向推理 ====
    out = model(sample, context_len=ctx, pred_len=pred)  # (B, pred, 1)
    print("Forward out:", out.shape)

    # ==== 损失函数 ====
    progress = 0.1  # 模拟训练进度 (10%)
    loss = model.loss(out, target, progress=progress)
    print("Loss (with aux):", float(loss))

    # ==== 反向传播 ====
    loss.backward()
    print("Backward pass OK")
