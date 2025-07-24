# ─────────────────────────────────────────────────────────────
# load_forecasting_transformer_moe_decomp.py  (可直接替换原文件)
# ─────────────────────────────────────────────────────────────
from typing import Dict, Optional, Tuple, Literal
import math

import torch
import torch.nn.functional as F
from torch import nn

from buildings_bench.models.base_model import BaseModel
from buildings_bench.models.sub_models_update_01 import (
    Encoder,
    Decoder,
    ModelArgs,
)


# ---------------------- 端到端季节‑趋势分解 ----------------------
class DecompositionLayer(nn.Module):
    """Moving-Average based 可学习分解：输出 seasonal & trend"""

    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.moving_avg = nn.AvgPool1d(
            kernel_size, stride=1, padding=kernel_size // 2, count_include_pad=False
        )

    def forward(self, x):  # x:[B,T,1]
        trend = self.moving_avg(x.transpose(1, 2)).transpose(1, 2)
        seasonal = x - trend
        return seasonal, trend


# ------------------------- 主 模 型 -----------------------------
class LoadForecastingTransformerMoE(BaseModel):
    """
    可变长输入 / 输出的 端到端季节‑趋势分解 + MoE Transformer
    """

    # ----------- ① 只保留“最大长度”配置 -----------
    def __init__(
        self,
        max_context_len: int = 336,  # ≈ 2 weeks history
        max_pred_len: int = 168,  # 1 week forecast
        context_len: int = 168,
        pred_len: int = 24,
        vocab_size: int = 2274,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.0,
        activation: str = "gelu",
        continuous_loads: bool = True,
        continuous_head: Literal["mse", "gaussian_nll"] = "mse",
        ignore_spatial: bool = False,
        weather_inputs: list | None = None,  # 仍然保留参数，但不会用
    ):
        # BaseModel 仍需两个数字——传“最大长度”即可
        super().__init__(context_len, pred_len, continuous_loads=continuous_loads)

        # 保存上限，运行时可传任意 ≤ 上限 的长度
        self.max_context_len = max_context_len
        self.max_pred_len = max_pred_len
        self.continuous_head = continuous_head
        self.vocab_size = vocab_size  # 仅分类任务用

        # ---------- ② 分解层 ----------
        self.decomp = DecompositionLayer(kernel_size=25)

        # ---------- ③ 嵌入 ----------
        self.seasonal_embedding = nn.Linear(1, d_model)
        self.trend_embedding = nn.Linear(1, d_model)

        # ---------- ④ 配置对象 ----------
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

        # ---------- ⑤ 编码器 / 解码器 ----------
        enc_layer = Encoder(self.cfg.n_dense_layers, self.cfg)
        self.seasonal_encoder = nn.TransformerEncoder(
            enc_layer, num_encoder_layers // 2, enable_nested_tensor=False
        )
        self.trend_encoder = nn.TransformerEncoder(
            enc_layer, num_encoder_layers // 2, enable_nested_tensor=False
        )

        dec_layer = Decoder(self.cfg.n_dense_layers, self.cfg)
        self.seasonal_decoder = nn.TransformerDecoder(
            dec_layer, num_decoder_layers // 2
        )
        self.trend_decoder = nn.TransformerDecoder(dec_layer, num_decoder_layers // 2)

        # ---------- ⑥ 预测头 ----------
        out_dim = 1 if continuous_head == "mse" or continuous_head == "huber" else 2
        self.seasonal_head = nn.Linear(d_model, out_dim)
        self.trend_head = nn.Linear(d_model, out_dim)

        # ---------- ⑦ 查询向量 (最大的) ----------
        self.query_embed = nn.Embedding(max_pred_len, d_model)

    # --------------------- forward (变长) ----------------------
    def forward(
        self,
        x: Dict[str, torch.Tensor],
        context_len: Optional[int] = None,
        pred_len: Optional[int] = None,
    ):
        """
        Args
        ----
        x["load"] : [B, T, 1]，T == context_len + pred_len
        context_len / pred_len : 若 None 则使用“最大值”
        """
        # ① 解析长度并做检查
        if context_len is None:
            context_len = self.max_context_len
        if pred_len is None:
            pred_len = self.max_pred_len

        load = x["load"]  # [B, T, 1]
        assert (
            load.size(1) == context_len + pred_len
        ), f"输入序列长度 {load.size(1)} ≠ context_len({context_len}) + pred_len({pred_len})"

        # ② 分解
        seasonal, trend = self.decomp(load)  # 两路 [B,T,1]
        s_ctx, t_ctx = seasonal[:, :context_len], trend[:, :context_len]

        # ③ 嵌入 & 编码
        s_mem = self.seasonal_encoder(self.seasonal_embedding(s_ctx))
        t_mem = self.trend_encoder(self.trend_embedding(t_ctx))

        # ④ 生成查询 (切片)
        B = load.size(0)
        query = self.query_embed.weight[:pred_len]  # [pred_len, D]
        query = query.unsqueeze(0).expand(B, -1, -1)  # [B, pred_len, D]

        # ⑤ 解码
        s_out = self.seasonal_decoder(query, s_mem)  # [B, pred_len, D]
        t_out = self.trend_decoder(query, t_mem)

        # ⑥ 预测
        y_hat = self.seasonal_head(s_out) + self.trend_head(
            t_out
        )  # [B, pred_len, out_dim]
        return y_hat

    # --------------------- 损失 & 推理 ------------------------
    def loss(self, pred: torch.Tensor, y: torch.Tensor):
        """
        pred : [B, pred_len, out_dim]   (out_dim = 1 或 2)
        y    : [B, pred_len, 1]
        """
        if self.continuous_loads:
            if self.continuous_head == "huber":
                # delta(=β) 控制 MSE ↔ MAE 过渡区间，常用 1.0
                return F.huber_loss(pred, y, delta=1.0)  # PyTorch ≥ 1.13
                # 若你的版本较旧，请改用:
                # return F.smooth_l1_loss(pred, y, beta=1.0)
            elif self.continuous_head == "mse":
                return F.mse_loss(pred, y)
            elif self.continuous_head == "gaussian_nll":
                return F.gaussian_nll_loss(
                    pred[:, :, 0].unsqueeze(2),
                    y,
                    F.softplus(pred[:, :, 1].unsqueeze(2)) ** 2,
                )
        else:
            return F.cross_entropy(
                pred.reshape(-1, self.vocab_size), y.long().reshape(-1)
            )

    @torch.no_grad()
    def predict(
        self,
        x: Dict[str, torch.Tensor],
        context_len: int = 168,
        pred_len: int = 24,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        preds = self.forward(x, context_len, pred_len)
        if self.continuous_head == "mse" or self.continuous_head == "huber":
            return preds, preds
        else:  # gaussian_nll
            return preds[:, :, 0:1], preds

    # -------------- 微调时统一解冻（保持不变） --------------
    def unfreeze_and_get_parameters_for_finetuning(self):
        return self.parameters()

    # -------------- 从检查点加载（保持不变） --------------
    def load_from_checkpoint(self, checkpoint_path):
        stored_ckpt = torch.load(checkpoint_path, map_location="cpu")
        model_state_dict = stored_ckpt["model"]
        new_state_dict = {
            (k.replace("module.", "") if k.startswith("module.") else k): v
            for k, v in model_state_dict.items()
        }
        self.load_state_dict(new_state_dict)


# -------------------------- 简单自测 ----------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 创建模型
    model = (
        LoadForecastingTransformerMoE(
            max_context_len=672, max_pred_len=168, continuous_head="gaussian_nll"
        )
        .to(device)
        .train()
    )

    # 随机动态长度
    ctx, pred = 96, 40  # 任意 ≤ 最大上限
    B = 2
    T = ctx + pred

    dummy = {"load": torch.rand(B, T, 1, device=device)}
    target = dummy["load"][:, -pred:]

    # 前向 & 反传
    out = model(dummy, context_len=ctx, pred_len=pred)
    loss = model.loss(out, target)
    loss.backward()

    print("Sanity-check OK – loss:", float(loss))

    # 推理
    model.eval()
    preds, _ = model.predict(dummy, context_len=ctx, pred_len=pred)
    print("Inference preds shape:", preds.shape)  # [B, pred, 1]
