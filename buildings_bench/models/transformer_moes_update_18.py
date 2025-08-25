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
    def __init__(self, kernel_size=25, res_ks=None):
        super().__init__()
        self.trend_pool = nn.AvgPool1d(
            kernel_size, stride=1, padding=kernel_size // 2, count_include_pad=False
        )
        self.res_ks = res_ks or max(3, kernel_size // 5)
        self.smooth_res = nn.AvgPool1d(
            self.res_ks, stride=1, padding=self.res_ks // 2, count_include_pad=False
        )

    def forward(self, x):  # x: [B,T,1]
        trend = self.trend_pool(x.transpose(1, 2)).transpose(1, 2)
        seasonal_raw = x - trend
        smooth_season = self.smooth_res(seasonal_raw.transpose(1, 2)).transpose(1, 2)
        residual = seasonal_raw - smooth_season
        return smooth_season, trend, residual


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
        continuous_head: Literal["mse", "gaussian_nll", "huber"] = "mse",
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
        # self.decomp = DecompositionLayer(kernel_size=25)

        # ---------- ③ 嵌入 ----------
        self.seasonal_embedding = nn.Linear(1, d_model)
        self.trend_embedding = nn.Linear(1, d_model)
        self.residual_embedding = nn.Linear(1, d_model)  # NEW

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
        self.residual_encoder = nn.TransformerEncoder(
            enc_layer, 1, enable_nested_tensor=False
        )  # NEW

        dec_layer = Decoder(self.cfg.n_dense_layers, self.cfg)
        self.seasonal_decoder = nn.TransformerDecoder(
            dec_layer, num_decoder_layers // 2
        )
        self.trend_decoder = nn.TransformerDecoder(dec_layer, num_decoder_layers // 2)
        self.residual_decoder = nn.TransformerDecoder(dec_layer, 1)  # NEW

        # ---------- ⑥ 预测头 ----------
        out_dim = 1 if continuous_head == "mse" or continuous_head == "huber" else 2
        self.seasonal_head = nn.Linear(d_model, out_dim)
        self.trend_head = nn.Linear(d_model, out_dim)
        self.residual_head = nn.Linear(d_model, out_dim)  # 仍保留，但输入换成 r_out

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
        # seasonal, trend, residual = self.decomp(load)
        s_ctx, t_ctx, r_ctx = (
            load[:, :context_len],
            load[:, :context_len],
            load[:, :context_len],
        )

        # ③ 嵌入 & 编码
        s_mem = self.seasonal_encoder(self.seasonal_embedding(s_ctx))
        t_mem = self.trend_encoder(self.trend_embedding(t_ctx))
        r_mem = self.residual_encoder(self.residual_embedding(r_ctx))

        # ④ 生成查询 (切片)
        B = load.size(0)
        query = self.query_embed.weight[:pred_len]  # [pred_len, D]
        query = query.unsqueeze(0).expand(B, -1, -1)  # [B, pred_len, D]

        # ⑤ 解码
        s_out = self.seasonal_decoder(query, s_mem)  # [B, pred_len, D]
        t_out = self.trend_decoder(query, t_mem)
        r_out = self.residual_decoder(query, r_mem)

        # ⑥ 预测
        y_hat = (
            self.seasonal_head(s_out)
            + self.trend_head(t_out)
            + self.residual_head(r_out)
        )  # [B, pred_len, out_dim]

        return y_hat

    # --------------------- 损失 & 推理 ------------------------
    def loss(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        pred : [B, pred_len, out_dim]   (out_dim = 1 or 2)
        y    : [B, pred_len, 1]         (连续)  或  [B, pred_len, 1] int (离散)
        先对 “单条序列” 做平均，再对 batch 做平均，
        使不同 pred_len 在梯度中拥有相同权重。
        """
        if self.continuous_loads:
            if self.continuous_head == "huber":
                # per‑token Huber, no reduction
                err = F.huber_loss(pred, y, delta=1.0, reduction="none")
            elif self.continuous_head == "mse":
                err = F.mse_loss(pred, y, reduction="none")
            elif self.continuous_head == "gaussian_nll":
                mu, sigma_sq = pred[..., 0:1], F.softplus(pred[..., 1:2]) ** 2
                # NLL = 0.5*log(2πσ²) + (y‑μ)²/(2σ²)
                err = 0.5 * (
                    torch.log(2 * torch.pi * sigma_sq) + (y - mu) ** 2 / sigma_sq
                )
            # ---- 关键两级平均 ----
            per_seq = err.mean(dim=(-2, -1))  # [B]   先对 (pred_len, out_dim)
            return per_seq.mean()  # 标量   再对 batch
        else:
            # 离散负荷：直接交叉熵，展平后默认 mean 即 per‑token 平均
            B, L, _ = pred.shape
            return F.cross_entropy(
                pred.reshape(B * L, -1),
                y.long().reshape(B * L),
                reduction="mean",
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
            max_context_len=672, max_pred_len=168, continuous_head="huber"
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
