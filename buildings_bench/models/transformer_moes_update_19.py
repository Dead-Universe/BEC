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
    """
    不在模型内做分解；使用数据批次中提供的 STL 结果：
      - x['stl_S'] : (B, T, 1)  只用 [:context_len]
      - x['stl_T'] : (B, T, 1)  只用 [:context_len]
      - x['stl_R'] : (B, T, 1)  只用 [:context_len]
    预测目标仍是负荷（与上游保持同一尺度，通常是 Box-Cox 尺度）。
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
    ):
        super().__init__(context_len, pred_len, continuous_loads=continuous_loads)
        self.max_context_len = max_context_len
        self.max_pred_len = max_pred_len
        self.continuous_head = continuous_head

        # ------- 三路编码/解码（结构保持不变） -------
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
            enc_layer, max((num_encoder_layers // 2) - 1, 1), enable_nested_tensor=False
        )
        self.trend_encoder = nn.TransformerEncoder(
            enc_layer, 1, enable_nested_tensor=False
        )
        self.residual_encoder = nn.TransformerEncoder(
            enc_layer, (num_encoder_layers // 2) + 1, enable_nested_tensor=False
        )

        dec_layer = Decoder(self.cfg.n_dense_layers, self.cfg)
        self.seasonal_decoder = nn.TransformerDecoder(
            dec_layer, max((num_decoder_layers // 2) - 1, 1)
        )
        self.trend_decoder = nn.TransformerDecoder(dec_layer, 1)
        self.residual_decoder = nn.TransformerDecoder(
            dec_layer, (num_decoder_layers // 2) + 1
        )

        out_dim = 1 if continuous_head in ("mse", "huber") else 2
        self.seasonal_head = nn.Linear(self.cfg.dim, out_dim)
        self.trend_head = nn.Linear(self.cfg.dim, out_dim)
        self.residual_head = nn.Linear(self.cfg.dim, out_dim)

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

        # —— 直接使用批内提供的 STL 结果（只取 context 段）——
        # 预测段通常为 0 填充（在 collate_fn 里），这里不使用它们，避免泄漏
        try:
            s_ctx = x["stl_S"][:, :context_len]  # (B, ctx, 1)
            t_ctx = x["stl_T"][:, :context_len]
            r_ctx = x["stl_R"][:, :context_len]
        except KeyError as e:
            raise KeyError(
                f"缺少 {e}：本模型不在内部做 STL 分解，需要在 collate_fn 中提供 "
                "'stl_S', 'stl_T', 'stl_R'。"
            )

        # 送入三路编码器
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

    # --------------------- loss（已移除分解正则） ---------------------
    def loss(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
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


# ======================= Quick self-test =======================
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = (
        LoadForecastingTransformerMoE(
            max_context_len=672,
            max_pred_len=168,
            continuous_head="huber",
        )
        .to(device)
        .train()
    )

    ctx, pred = 96, 40
    B = 2
    # 伪造 batch：注意提供 stl_S/T/R
    load = torch.rand(B, ctx + pred, 1, device=device)
    stl_S = torch.randn(B, ctx + pred, 1, device=device)
    stl_T = torch.randn(B, ctx + pred, 1, device=device)
    stl_R = torch.randn(B, ctx + pred, 1, device=device)
    dummy = {"load": load, "stl_S": stl_S, "stl_T": stl_T, "stl_R": stl_R}
    target = dummy["load"][:, -pred:]

    out = model(dummy, context_len=ctx, pred_len=pred)
    loss = model.loss(out, target)
    loss.backward()
    print("Sanity-check OK – loss:", float(loss))

    model.eval()
    preds, _ = model.predict(dummy, context_len=ctx, pred_len=pred)
    print("Inference preds shape:", preds.shape)  # [B, pred, 1]
