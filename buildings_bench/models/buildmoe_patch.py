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


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from typing_extensions import Literal

# 假定 ModelArgs / Encoder / Decoder / BaseModel / BalancedMoEMixin 已在其它文件中定义


class DynamicPredHead(nn.Module):
    """
    动态 token 数量的点级预测头：
    - 最大支持 max_tokens 个 patch token；
    - 每个 token 维度为 D；
    - 输出最大长度为 max_pred_len 的 future 序列。

    参数形状:
    - weight: [max_pred_len, max_tokens, D]
    - bias:   [max_pred_len]

    前向时根据当前 N_tokens 切 weight 的前 N 段：
    y[b, p] = sum_{n < N, d < D} tokens[b, n, d] * W[p, n, d] + b[p]
    """

    def __init__(self, max_tokens: int, d_model: int, max_pred_len: int):
        super().__init__()
        self.max_tokens = int(max_tokens)
        self.max_pred_len = int(max_pred_len)

        self.weight = nn.Parameter(
            torch.empty(self.max_pred_len, self.max_tokens, d_model)
        )  # [P, N_max, D]
        self.bias = nn.Parameter(torch.zeros(self.max_pred_len))  # [P]

        # 初始化：把 (N_max * D) 当成 in_features 做 xavier 比较合理
        nn.init.xavier_uniform_(self.weight.view(self.max_pred_len, -1))
        nn.init.zeros_(self.bias)

    def forward(self, tokens: torch.Tensor, pred_len: int) -> torch.Tensor:
        """
        tokens:   [B, N_tokens, D]
        pred_len: 实际需要输出的 future 长度（<= max_pred_len）

        返回: [B, pred_len, 1]
        """
        B, N, D = tokens.shape
        if N > self.max_tokens:
            raise ValueError(
                f"DynamicPredHead: 接收到的 token 数 N={N} 超过 max_tokens={self.max_tokens}，"
                f"请增大 max_context_len/max_pred_len 或减小 patch_stride。"
            )
        if pred_len > self.max_pred_len:
            raise ValueError(
                f"DynamicPredHead: pred_len={pred_len} 超过 max_pred_len={self.max_pred_len}。"
            )

        # 取前 N 个 token 对应的权重，和前 pred_len 个 time step 对应的权重
        # W_slice: [pred_len, N, D]
        W_slice = self.weight[:pred_len, :N, :]
        b_slice = self.bias[:pred_len]  # [pred_len]

        # einsum: "bnd,pnd->bp" → 每个样本、每个 time step p，聚合所有 token 贡献
        # 结果: [B, pred_len]
        full = torch.einsum("bnd,pnd->bp", tokens, W_slice) + b_slice  # broadcasting

        # [B, pred_len] → [B, pred_len, 1]
        return full.unsqueeze(-1)


class BuildMoEBatch(BaseModel, BalancedMoEMixin):
    """
    Patch-based MoE 负荷预测模型（Chronos-style 点级掩码 + 动态 N token Flatten-style Head）

    关键特点：
    - 支持任意 context_len / pred_len（只要不超过 max_*）；
    - 使用点级有效性掩码（mask）区分“真实值的 0”和“padding 的 0”；
    - 所有 patch token（当前 N_total_patches）一起视作一条大特征向量，
      通过 DynamicPredHead 做动态线性解码，完全摒弃 0 填充 / 截断。
    """

    def __init__(
        self,
        max_context_len: int = 336,
        max_pred_len: int = 168,
        # 下面两个只是 BaseModel 接口中的“默认值”，forward/predict 里可以覆盖
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
        arch_mode: Literal["encdec", "encoder", "decoder"] = "encdec",
        aux_loss_weight: float = 0.01,
        n_shared_experts: int = 0,
        use_moe_balancer: bool = True,
        use_moe_loss: bool = True,
        patch_len: int = 24,  # === patch size ===
        patch_stride: Optional[int] = 12,  # === patch stride（可重叠）===
        **kwargs,
    ):
        super().__init__(context_len, pred_len, continuous_loads=continuous_loads)
        self.max_context_len = int(max_context_len)
        self.max_pred_len = int(max_pred_len)
        self.continuous_head = continuous_head
        self.arch_mode = arch_mode
        self.use_dense = use_dense
        self.aux_loss_weight = aux_loss_weight
        self.use_moe_loss = use_moe_loss

        # ===== Patch 设置（size + stride）=====
        self.patch_len = int(patch_len)
        self.patch_stride = (
            self.patch_len if patch_stride is None else int(patch_stride)
        )

        assert self.patch_len > 0 and self.patch_stride > 0, "patch_len/stride 必须 > 0"
        assert (
            self.patch_len <= self.max_context_len
        ), "patch_len 不能大于 max_context_len"
        assert self.patch_len <= self.max_pred_len, "patch_len 不能大于 max_pred_len"

        # 估算 patch token 数上限（注意是“窗口数上限”，用于 MoE 和 DynamicPredHead 统一参数大小）
        # encoder-only / decoder-only：在 ctx+pred 上做滑窗（最大就是 max_context_len + max_pred_len）
        self.n_joint_patches_max = max(
            1,
            (self.max_context_len + self.max_pred_len - self.patch_len)
            // self.patch_stride
            + 1,
        )
        # encdec：encoder 用 ctx，decoder 用 pred 各自滑窗（这里的 “上限” 是严格按 max_* 算出来的）
        self.n_ctx_patches_max = max(
            1, (self.max_context_len - self.patch_len) // self.patch_stride + 1
        )
        self.n_pred_patches_max = max(
            1, (self.max_pred_len - self.patch_len) // self.patch_stride + 1
        )

        # Transformer 的 max_seq_len 取「所有可能路径中最大的 token 数」
        max_tokens = max(
            self.n_ctx_patches_max + self.n_pred_patches_max, self.n_joint_patches_max
        )

        # ------- patch 级编码/解码 -------
        # 一个 patch（长度 patch_len） + 对应的点级 mask（同长） → 拼接后做 Linear
        # 输入 patch_in: [B, N_patches, 2 * patch_len]
        self.patch_embed = nn.Linear(self.patch_len * 2, d_model)

        # MoE Transformer 配置（序列长度按“patch token 数”计）
        self.cfg = ModelArgs(
            max_seq_len=max_tokens,
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

        if self.arch_mode in ("encoder", "encdec"):
            enc_layer = Encoder(self.cfg.n_dense_layers, self.cfg)
            self.encoder = nn.TransformerEncoder(
                enc_layer, num_encoder_layers, enable_nested_tensor=False
            )
        if self.arch_mode in ("decoder", "encdec"):
            dec_layer = Decoder(self.cfg.n_dense_layers, self.cfg)
            self.decoder = nn.TransformerDecoder(dec_layer, num_decoder_layers)

        # 预测端：DynamicPredHead，最大 token 数 = 对应架构下的“真正上限”
        if self.arch_mode == "encdec":
            # encdec：decoder 侧最多看到 n_pred_patches_max 个 query token
            dec_token_max = self.n_pred_patches_max
        else:
            # encoder / decoder-only：ctx + max_pred_len 一起滑窗
            dec_token_max = self.n_joint_patches_max

        self.dec_token_max = dec_token_max
        self.pred_head = DynamicPredHead(
            max_tokens=self.dec_token_max,
            d_model=self.cfg.dim,
            max_pred_len=self.max_pred_len,
        )

        # 初始化 MoE balancer
        self._init_balancer(use_moe_balancer)

    # ====== Patch + 点级 mask 工具函数 ======

    def _build_valid_mask(
        self, length: int, total_len: int, device, dtype
    ) -> torch.Tensor:
        """
        构造点级有效性 mask（1=真实值，0=padding/占位），形状 [1, total_len, 1]。
        length: 真实有效长度
        total_len: 构造的总长度
        """
        valid = torch.zeros(1, total_len, 1, device=device, dtype=dtype)
        valid[:, :length, 0] = 1.0
        return valid

    def _patchify_with_mask(
        self, x: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        x:          [B, L, 1]      单通道负荷
        valid_mask: [B, L, 1]      点级有效性（1=有效, 0=padding/占位）

        - 若 L < patch_len，则在右侧补零到 patch_len，mask 同步补 0；
        - 使用 unfold 得到 patch 序列；
        - 将数值 patch 与 mask patch 在最后一维拼接：
            patch_in = [values, mask]  →  [B, N_patches, 2 * patch_len]
        """
        B, L, C = x.shape
        assert C == 1, "目前只实现单通道负荷"
        assert valid_mask.shape == x.shape, "valid_mask 形状必须与 x 一致"

        if L < self.patch_len:
            pad_len = self.patch_len - L
            pad_x = torch.zeros(B, pad_len, 1, device=x.device, dtype=x.dtype)
            pad_m = torch.zeros(
                B, pad_len, 1, device=valid_mask.device, dtype=valid_mask.dtype
            )
            x = torch.cat([x, pad_x], dim=1)
            valid_mask = torch.cat([valid_mask, pad_m], dim=1)
            L = self.patch_len

        # [B, L, 1] → unfold → [B, 1, N_patches, patch_len] → squeeze → [B, N_patches, patch_len]
        x_patches = (
            x.transpose(1, 2)
            .unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)
            .squeeze(1)
            .contiguous()
        )
        m_patches = (
            valid_mask.transpose(1, 2)
            .unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)
            .squeeze(1)
            .contiguous()
        )

        # 拼接数值与 mask： [B, N, patch_len] → [B, N, 2*patch_len]
        patch_in = torch.cat([x_patches, m_patches], dim=-1)
        return patch_in  # 给 patch_embed 使用

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

        context_len = int(context_len)
        pred_len = int(pred_len)

        if context_len > self.max_context_len:
            raise ValueError(
                f"BuildMoE: 输入 context_len={context_len} 超过 max_context_len={self.max_context_len}。"
                "当前模型仅在不超过该长度的上下文上训练，超出部分未见过，可能导致精度明显下降；"
                "建议裁剪序列或重训模型。"
            )
        if pred_len > self.max_pred_len:
            raise ValueError(
                f"BuildMoE: 输入 pred_len={pred_len} 超过 max_pred_len={self.max_pred_len}。"
                "当前模型仅在不超过该长度的预测地平线上训练，超出部分未见过，可能导致精度明显下降；"
                "建议缩短预测长度或重训模型。"
            )

        load = x["load"]  # (B, T, 1)，与 stl_* 同尺度（Box-Cox）
        assert load.dim() == 3 and load.size(-1) == 1, "load 需要形状 [B, T, 1]"
        assert (
            load.size(1) >= context_len + pred_len
        ), "load 长度需要 ≥ context_len + pred_len"

        B, T, _ = load.shape

        # 上下文段
        ctx = load[:, :context_len, :]  # [B, ctx, 1]

        if self.arch_mode == "encdec":
            # ===== Encoder-Decoder 路径 =====
            # 1) 编码器：只在真实上下文上做滑窗 patch（全部有效）
            ctx_valid = torch.ones_like(ctx)
            ctx_patches_in = self._patchify_with_mask(
                ctx, ctx_valid
            )  # [B, N_ctx_patches, 2*patch_len]
            mem = self.encoder(
                self.patch_embed(ctx_patches_in)
            )  # [B, N_ctx_patches, D]

            # 2) 解码器：对预测 horizon 构造“预测 patch token”
            #    这里直接用全 0 query（可后续加 pos embedding 等），长度为 n_pred_patches_max
            query = torch.zeros(
                B,
                self.n_pred_patches_max,
                self.cfg.dim,
                device=load.device,
                dtype=load.dtype,
            )  # [B, N_pred_patches, D]

            out = self.decoder(query, mem)  # [B, N_pred_patches, D]
            N_tokens = out.size(1)
            assert (
                N_tokens <= self.dec_token_max
            ), f"encdec: N_tokens={N_tokens} 超过 dec_token_max={self.dec_token_max}"

            # 3) DynamicPredHead → pred_len
            y_hat = self.pred_head(out, pred_len)  # [B, pred_len, 1]
            return y_hat

        elif self.arch_mode == "encoder":
            # ===== 纯 Encoder：ctx + 未来占位零 → patch → 因果 encoder =====
            zeros_pred = torch.zeros(
                B, self.max_pred_len, 1, device=load.device, dtype=load.dtype
            )
            inp = torch.cat([ctx, zeros_pred], dim=1)  # [B, ctx + max_pred_len, 1]

            valid_mask = torch.zeros_like(inp)
            valid_mask[:, :context_len, 0] = 1.0  # 只有历史段为有效

            inp_patches_in = self._patchify_with_mask(
                inp, valid_mask
            )  # [B, N_total_patches, 2*patch_len]
            h = self.encoder(
                self.patch_embed(inp_patches_in), is_causal=True
            )  # [B, N_total_patches, D]
            N_tokens = h.size(1)
            assert (
                N_tokens <= self.dec_token_max
            ), f"encoder: N_tokens={N_tokens} 超过 dec_token_max={self.dec_token_max}"

            y_hat = self.pred_head(h, pred_len)  # [B, pred_len, 1]
            return y_hat

        else:  # self.arch_mode == "decoder"
            # ===== 纯 Decoder：ctx + 未来占位零 → patch → 因果 decoder =====
            zeros_pred = torch.zeros(
                B, self.max_pred_len, 1, device=load.device, dtype=load.dtype
            )
            tgt_vals = torch.cat([ctx, zeros_pred], dim=1)  # [B, ctx+max_pred_len, 1]

            valid_mask = torch.zeros_like(tgt_vals)
            valid_mask[:, :context_len, 0] = 1.0

            tgt_patches_in = self._patchify_with_mask(
                tgt_vals, valid_mask
            )  # [B, N_total_patches, 2*patch_len]
            tgt = self.patch_embed(tgt_patches_in)  # [B, N_total_patches, D]

            h = self.decoder(
                tgt, memory=None, tgt_is_causal=True
            )  # [B, N_total_patches, D]
            N_tokens = h.size(1)
            assert (
                N_tokens <= self.dec_token_max
            ), f"decoder: N_tokens={N_tokens} 超过 dec_token_max={self.dec_token_max}"

            y_hat = self.pred_head(h, pred_len)  # [B, pred_len, 1]
            return y_hat

    # --------------------- loss ---------------------
    def loss(
        self, pred: torch.Tensor, y: torch.Tensor, progress: Optional[float] = None
    ) -> torch.Tensor:
        if progress is not None and not self.use_dense:
            self.moe_step(progress=progress)

        aux_loss = torch.tensor(0.0, device=pred.device)
        if self.use_moe_loss and not self.use_dense:
            aux_loss = self.moe_auxiliary_loss()

        if self.continuous_head == "huber":
            # pred / y 预期形状 [B, L, 1]
            if pred.ndim == 3 and pred.size(-1) == 1:
                pred_main = pred
            else:
                pred_main = pred[..., :1]
            if y.ndim == 3 and y.size(-1) == 1:
                y_main = y
            else:
                y_main = y[..., :1]

            err = F.huber_loss(pred_main, y_main, delta=1.0, reduction="none")
            loss = err.mean()
        else:
            raise NotImplementedError()

        return loss + self.aux_loss_weight * aux_loss

    @torch.no_grad()
    def predict(
        self, x: Dict[str, torch.Tensor], context_len: int = 168, pred_len: int = 24
    ):
        preds = self.forward(x, context_len, pred_len)  # [B, pred_len, 1]
        return preds, preds  # 第二个占位，兼容 BuildingsBench 管线

    # --------------------- finetune 参数组 ---------------------
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

        param_groups = []

        if strategy == "head":
            # “头”只包含 pred_head（DynamicPredHead）
            add_params(
                self.pred_head.named_parameters(), head_lr, wd, bucket=param_groups
            )

        elif strategy == "head_last_block":
            add_params(
                self.pred_head.named_parameters(), head_lr, wd, bucket=param_groups
            )
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

            # 输出头（pred_head）也一般解冻
            add_params(
                self.pred_head.named_parameters(), head_lr, wd, bucket=param_groups
            )

        elif strategy == "norms_biases":
            # 先把所有 Norm/bias 打开 requires_grad
            for n, p in self.named_parameters():
                if is_norm_or_bias(n, p):
                    p.requires_grad = True

            # 单独把 pred_head 当作 head 调高 lr
            add_params(
                self.pred_head.named_parameters(), head_lr, wd, bucket=param_groups
            )

            # 其它已被标记为 requires_grad 的参数按 base_lr 打包
            used_params = {id(p) for g in param_groups for p in g["params"]}
            others = [
                (n, p)
                for n, p in self.named_parameters()
                if p.requires_grad and id(p) not in used_params
            ]
            add_params(others, base_lr, wd, train=False, bucket=param_groups)

        elif strategy == "full":
            # 头：pred_head，使用 head_lr
            add_params(
                self.pred_head.named_parameters(), head_lr, wd, bucket=param_groups
            )
            # 其余所有参数归为 body，使用 base_lr
            head_param_set = {id(p) for g in param_groups for p in g["params"]}
            body = []
            for n, p in self.named_parameters():
                if id(p) not in head_param_set:
                    body.append((n, p))
            add_params(body, base_lr, wd, bucket=param_groups)

        else:
            raise ValueError(f"Unknown finetune strategy: {strategy}")

        # 兜底：如果上面没加到任何 group（极端情况），至少解冻 pred_head
        if not param_groups:
            add_params(
                self.pred_head.named_parameters(), head_lr, wd, bucket=param_groups
            )

        return param_groups

    def load_from_checkpoint(self, checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu")["model"]
        state = {k.replace("module.", ""): v for k, v in state.items()}
        self.load_state_dict(state)


def moe_static_params(model: torch.nn.Module):
    """
    一次性静态统计 MoE 参数量（无需前向）：
      activated_params_total = shared_params + Σ_layer (top_k[layer] * per_expert_params[layer])
    """
    import torch.nn as nn

    moe_layers = []
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
    import torch

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ==== 初始化模型 ====
    model = BuildMoE(
        max_context_len=168,
        max_pred_len=168,
        context_len=168,
        pred_len=24,
        patch_len=24,  # 确保能整除
        model_arch="encoder",
    ).to(device)

    # ==== 统计 MoE 参数量 ====
    stats = moe_static_params(model)
    for k, v in stats.items():
        print(f"{k}: {v}")

    # ==== 伪造输入数据 ====
    B = 4
    ctx, pred = 12, 24
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
