# ─────────────────────────────────────────────────────────────
# load_forecasting_transformer_moe_decomp.py
# 增强版：端到端季节-趋势分解 + MoE Transformer
# ─────────────────────────────────────────────────────────────
from typing import Dict

from buildings_bench.models.base_model import BaseModel
import torch
from torch import nn
import torch.nn.functional as F


from buildings_bench.models.sub_models import (
    Decoder,
    Encoder,
    MoEPositionalEncoding,
    ModelArgs,
)


# ──────────────────────────────────────────────
# 2. 端到端季节-趋势分解层
# ──────────────────────────────────────────────
class DecompositionLayer(nn.Module):
    """Moving-Average based 可学习分解：输出 seasonal & trend"""

    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.moving_avg = nn.AvgPool1d(
            kernel_size, stride=1, padding=kernel_size // 2, count_include_pad=False
        )

    def forward(self, x):  # x:[B,T,1]
        # AvgPool 需要 [B, C, T]
        trend = self.moving_avg(x.transpose(1, 2)).transpose(1, 2)
        seasonal = x - trend
        return seasonal, trend


# ──────────────────────────────────────────────
# 3. 主模型：LoadForecastingTransformerMoE + 分解
# ──────────────────────────────────────────────
class LoadForecastingTransformerMoE(BaseModel):
    """
    端到端季节-趋势分解 + MoE Transformer。
    仅使用 load 序列，分解后两路共享编码器/解码器与预测头。
    """

    def __init__(
        self,
        num_experts: int = 8,
        top_k: int = 2,
        context_len: int = 168,
        pred_len: int = 24,
        vocab_size: int = 2274,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        d_model: int = 256,  # 保持不变
        nhead: int = 8,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
        activation: str = "gelu",
        continuous_loads: bool = False,
        continuous_head: str = "mse",
        ignore_spatial: bool = False,
        weather_inputs=None,  # 仍然保留参数，但不会用
    ):
        super().__init__(context_len, pred_len, continuous_loads=True)

        # 1) 分解层
        self.decomp = DecompositionLayer(kernel_size=25)

        # 2) 嵌入：季节、趋势公用一套线性映射（也可分开）
        # 季节线性映射
        self.seasonal_embedding = nn.Linear(1, d_model)
        # 趋势线性映射
        self.trend_embedding = nn.Linear(1, d_model)

        # 3) 位置编码
        self.pos_encoding = MoEPositionalEncoding(d_model, dropout)

        # 4) 配置对象
        self.cfg = ModelArgs(
            max_seq_len=context_len + pred_len,
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

        # 5) 编码器 / 解码器
        seasonal_enc_layer = Encoder(self.cfg.n_dense_layers, self.cfg)
        self.seasonal_encoder = nn.TransformerEncoder(
            seasonal_enc_layer, num_encoder_layers // 2
        )
        trend_enc_layer = Encoder(self.cfg.n_dense_layers, self.cfg)
        self.trend_encoder = nn.TransformerEncoder(
            trend_enc_layer, num_encoder_layers // 2
        )

        seasonal_dec_layer = Decoder(self.cfg.n_dense_layers, self.cfg)
        self.seasonal_decoder = nn.TransformerDecoder(
            seasonal_dec_layer, num_decoder_layers // 2
        )
        trend_dec_layer = Decoder(self.cfg.n_dense_layers, self.cfg)
        self.trend_decoder = nn.TransformerDecoder(
            trend_dec_layer, num_decoder_layers // 2
        )

        # 6) 预测头
        out_dim = 1 if continuous_head == "mse" or continuous_head == "huber" else 2
        self.seasonal_head = nn.Linear(d_model, out_dim)
        self.trend_head = nn.Linear(d_model, out_dim)
        self.residual_head = nn.Linear(d_model, out_dim)

        self.continuous_head = continuous_head

        # 7) 解码查询向量
        self.query_embed = nn.Embedding(pred_len, d_model)

    # ------------------------------------------------------------------
    # forward 仅接收 x['load']
    # ------------------------------------------------------------------
    def forward(self, x: Dict[str, torch.Tensor]):
        """
        Args:
            x["load"] : [B, T, 1]，T = context_len + pred_len
        """
        load = x["load"]  # [B,T,1]
        seasonal, trend = self.decomp(load)  # 两路 [B,T,1]

        # —— 上下文 / 预测段拆分 ——
        s_ctx, t_ctx = seasonal[:, : self.context_len], trend[:, : self.context_len]

        # 嵌入 + 位置编码
        s_emb = self.pos_encoding(self.seasonal_embedding(s_ctx))
        t_emb = self.pos_encoding(self.trend_embedding(t_ctx))

        # # 编码
        s_mem = self.seasonal_encoder(s_emb)
        t_mem = self.trend_encoder(t_emb)

        # 查询向量
        B = load.size(0)
        query = self.query_embed.weight.unsqueeze(0).expand(
            B, -1, -1
        )  # [B, pred_len, D]
        query = self.pos_encoding(query)

        # # 解码
        s_out = self.seasonal_decoder(query, s_mem)  # [B, pred_len, D]
        t_out = self.trend_decoder(query, t_mem)

        # 预测并相加
        y_hat = (
            self.seasonal_head(s_out)
            + self.trend_head(t_out)
            + self.residual_head(query)
        )  # [B, pred_len, out_dim]

        return y_hat

    # ------------------------------------------------------------------
    # loss / predict / finetune helper 与原实现几乎一致
    # ------------------------------------------------------------------
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

    # ───────── 在类里加入一段私有工具函数 ─────────
    def _postprocess(
        self, pred, context, clamp_q=(0.01, 0.99), smooth_kernel=3  # winsorize 区间
    ):  # 平滑窗口；<=1 则不开
        """
        后处理：清理 nan/inf → winsorize → 可选平滑
        pred    : [B, pred_len, 1]  原始输出
        context : [B, context_len, 1]  最近 context_len 小时负荷
        """
        # 1) nan / inf 替换为 0
        pred = torch.where(torch.isfinite(pred), pred, torch.zeros_like(pred))

        # 2) winsorize
        if clamp_q:
            q_low, q_high = clamp_q
            low = torch.quantile(context, q_low, dim=1, keepdim=True)
            high = torch.quantile(context, q_high, dim=1, keepdim=True)
            pred = torch.max(torch.min(pred, high), low)

        # 3) 简单滑动平均
        if smooth_kernel and smooth_kernel > 1:
            k = smooth_kernel
            weight = torch.ones(1, 1, k, device=pred.device) / k
            pred = F.conv1d(pred.transpose(1, 2), weight, padding=k // 2).transpose(
                1, 2
            )

        return pred

    # ───────── 把 predict 换成下面版本 ─────────
    def predict(
        self,
        x: Dict,
        postprocess: bool = True,
        clamp_q: tuple = (0.01, 0.99),
        smooth_kernel: int = 3,
    ):
        """
        与旧 predict 接口兼容，但多了后处理（可关）。
        返回: (clean_point_forecast, raw_full_output)
        """
        ans = self.forward(x)  # 原始网络输出  [B, pred_len, out_dim]

        # 取 point 预测 (与旧逻辑一致)
        pred_point = (
            ans
            if self.continuous_head == "mse" or self.continuous_head == "huber"
            else ans[:, :, 0].unsqueeze(-1)
        )  # 只取 μ

        # 后处理
        if postprocess:
            context = x["load"][:, -self.context_len :]  # 最近 context_len 小时
            pred_point = self._postprocess(
                pred_point, context, clamp_q=clamp_q, smooth_kernel=smooth_kernel
            )

        return pred_point, ans

    def unfreeze_and_get_parameters_for_finetuning(self):
        # for p in self.parameters():
        #     p.requires_grad_(False)
        # self.logits.requires_grad_(True)
        # return self.logits.parameters()
        return self.parameters()

    def load_from_checkpoint(self, checkpoint_path):
        stored_ckpt = torch.load(checkpoint_path)
        model_state_dict = stored_ckpt["model"]
        new_state_dict = {}
        for k, v in model_state_dict.items():
            # remove string 'module.' from the key
            if "module." in k:
                new_state_dict[k.replace("module.", "")] = v
            else:
                new_state_dict[k] = v
        self.load_state_dict(new_state_dict)
        # print(f"Loaded model checkpoint from {checkpoint_path}...")


# ----------------------------------------------------------------------
# Quick sanity-check
# ----------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    # 模型配置
    cfg = dict(
        context_len=168,
        pred_len=24,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_model=256,
        dim_feedforward=1024,
        num_experts=2,
        top_k=2,
        nhead=8,
        dropout=0.0,
        continuous_loads=True,
        continuous_head="gaussian_nll",
    )

    # 创建模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LoadForecastingTransformerMoE(**cfg).to(device).train()

    # 创建假数据 (B=2, T=192: 168 + 24)
    B = 2
    T = cfg["context_len"] + cfg["pred_len"]

    dummy = {
        "latitude": torch.rand(B, T, 1, device=device) * 2 - 1,
        "longitude": torch.rand(B, T, 1, device=device) * 2 - 1,
        "building_type": torch.zeros(B, T, 1, dtype=torch.long, device=device),
        "day_of_year": torch.rand(B, T, 1, device=device) * 2 - 1,
        "day_of_week": torch.rand(B, T, 1, device=device) * 2 - 1,
        "hour_of_day": torch.rand(B, T, 1, device=device) * 2 - 1,
        "load": torch.rand(B, T, 1, device=device),
    }
    target = dummy["load"][:, -cfg["pred_len"] :, :]

    out = model(dummy)
    loss = model.loss(out, target)
    loss.backward()

    print("sanity-check OK – loss:", float(loss))

    dummy_input = {
        "latitude": torch.rand(B, T, 1, device=device) * 2 - 1,  # 假设的纬度
        "longitude": torch.rand(B, T, 1, device=device) * 2 - 1,  # 假设的经度
        "building_type": torch.zeros(
            B, T, 1, dtype=torch.long, device=device
        ),  # 假设的建筑类型
        "day_of_year": torch.rand(B, T, 1, device=device) * 2 - 1,  # 假设的日期
        "day_of_week": torch.rand(B, T, 1, device=device) * 2 - 1,  # 假设的星期几
        "hour_of_day": torch.rand(B, T, 1, device=device) * 2 - 1,  # 假设的小时
        "load": torch.rand(B, T, 1, device=device),  # 假设的负荷数据
    }

    # 调用模型的predict方法
    model.eval()  # 设置为评估模式
    predictions, a = model.predict(dummy_input)

    # 输出结果
    print("Predictions Shape:", predictions.shape)
    print("Predictions:", predictions)
    print("Distribution Parameters:", a)
