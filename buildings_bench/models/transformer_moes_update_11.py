import torch
from typing import Dict
from torch import nn
import torch.nn.functional as F
from buildings_bench.models.base_model import BaseModel
from buildings_bench.models.sub_models import (
    Decoder,
    Encoder,
    MoEPositionalEncoding,
    ModelArgs,
    TokenEmbedding,
)


class LoadForecastingTransformerMoE(BaseModel):
    """
    与原类同名、同签名；只用 load 作为输入特征。
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
        super().__init__(context_len, pred_len, continuous_loads)

        self.continuous_head = continuous_head
        self.vocab_size = vocab_size

        # ---------- 只保留 load 的 embedding ----------
        if continuous_loads:
            self.power_embedding = nn.Linear(1, d_model)
        else:
            self.power_embedding = TokenEmbedding(vocab_size, d_model)

        # ---------- 位置编码、编码器/解码器保持不变 ----------
        self.positional_encoding = MoEPositionalEncoding(d_model, dropout=dropout)

        self.config = ModelArgs(
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

        enc_layer = Encoder(self.config.n_dense_layers, self.config)
        self.encoder = nn.TransformerEncoder(enc_layer, num_encoder_layers)

        dec_layer = Decoder(self.config.n_dense_layers, self.config)
        self.decoder = nn.TransformerDecoder(dec_layer, num_decoder_layers)

        # 预测头同原实现
        if self.continuous_loads:
            out_dim = 1 if continuous_head == "mse" else 2
            self.logits = nn.Linear(d_model, out_dim)
        else:
            self.logits = nn.Linear(d_model, vocab_size)

        # 查询向量
        self.query_embed = nn.Embedding(pred_len, d_model)

    # ------------------------------------------------------------------
    # forward 只取 load
    # ------------------------------------------------------------------
    def forward(self, x: Dict[str, torch.Tensor]):
        """
        x 仍是字典，只会用到 x['load']。
        其他字段即使传进来也会被忽略。
        """
        # [B, T, d_model]
        load_embed = self.power_embedding(x["load"]).squeeze(2)

        # 拆分上下文 / 预测段
        src_series_inputs = load_embed[:, : self.context_len, :]

        src_series_embed = self.positional_encoding(src_series_inputs)
        memory = self.encoder(src_series_embed)

        batch_size = src_series_inputs.size(0)
        query_idx = torch.arange(self.pred_len, device=src_series_inputs.device)
        tgt_queries = self.query_embed(query_idx)  # [pred_len, d_model]
        tgt = tgt_queries.unsqueeze(0).expand(batch_size, -1, -1)
        tgt_enc = self.positional_encoding(tgt)

        outs = self.decoder(tgt_enc, memory)  # [B, pred_len, d_model]
        return self.logits(outs)

    # 其余 loss / predict / checkpoint 方法完全复用原实现

    def loss(self, pred, y):
        """
        task loss only (aux loss 已移除)  pred:[B, pred_len, ...]
        """
        if self.continuous_loads and self.continuous_head == "mse":
            return F.mse_loss(pred, y)
        elif self.continuous_loads and self.continuous_head == "gaussian_nll":
            return F.gaussian_nll_loss(
                pred[:, :, 0].unsqueeze(2),
                y,
                F.softplus(pred[:, :, 1].unsqueeze(2)) ** 2,
            )
        else:
            return F.cross_entropy(
                pred.reshape(-1, self.vocab_size), y.long().reshape(-1)
            )

    def predict(self, x: Dict):
        ans = self.forward(x)
        return (
            ans if self.continuous_head == "mse" else ans[:, :, 0].unsqueeze(-1)
        ), ans

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
