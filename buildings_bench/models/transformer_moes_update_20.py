from typing import Dict, Optional, Tuple, Literal, List
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
                enc_layer, num_encoder_layers + 2, enable_nested_tensor=False
            )
        if self.arch_mode == "decoder" or self.arch_mode == "encdec":
            dec_layer = Decoder(self.cfg.n_dense_layers, self.cfg)
            self.decoder = nn.TransformerDecoder(dec_layer, num_decoder_layers + 2)

        out_dim = 1 if continuous_head in ("mse", "huber") else 2
        self.head = nn.Linear(self.cfg.dim, out_dim)

        # 收集门控 logits 用于辅助均衡损失
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
    def loss(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
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
            use_dense=True,
            arch_mode="encdec",
            num_decoder_layers=1,
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
