# DeepSeekFeedForward

import torch
from typing import Tuple, Dict, List
from torch import nn
import torch.nn.functional as F
from buildings_bench.models.base_model import BaseModel
from buildings_bench.models.sub_models import (
    Decoder,
    Encoder,
    MoEPositionalEncoding,
    ModelArgs,
    TimeSeriesSinusoidalPeriodicEmbedding,
    TokenEmbedding,
    ZeroEmbedding,
)


class LoadForecastingTransformerMoE(BaseModel):
    """
    Encoder-Decoder Transformer with DeepSeek-MoE FFN.
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
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
        activation: str = "gelu",
        continuous_loads: bool = False,
        continuous_head: str = "mse",
        ignore_spatial: bool = False,
        weather_inputs: List[str] | None = None,
    ):
        super().__init__(context_len, pred_len, continuous_loads)

        self.continuous_head = continuous_head
        self.vocab_size = vocab_size
        self.ignore_spatial = ignore_spatial

        self.weather_features = weather_inputs
        if weather_inputs:
            self.weather_embedding = nn.Linear(len(weather_inputs), 64)
            d_model += 64

        self.config = ModelArgs(
            max_seq_len=context_len + pred_len,
            dim=d_model,
            inter_dim=dim_feedforward,
            moe_inter_dim=d_model // 2,
            n_encoder_layers=num_encoder_layers,
            n_decoder_layers=num_decoder_layers,
            n_heads=nhead,
            n_routed_experts=num_experts,
            n_shared_experts=2,  # 固定为 2
            n_activated_experts=top_k,
            n_expert_groups=1,  # 固定为 1
            n_limited_groups=1,  # 固定为 1
            score_func="softmax",  # 固定为 softmax
            route_scale=1.0,  # 固定为 1.0
        )

        # ---- Encoder / Decoder (使用新 MoE 层) ------------------------------
        enc_layer = Encoder(self.config.n_dense_layers, self.config)
        self.encoder = nn.TransformerEncoder(enc_layer, num_encoder_layers)

        dec_layer = Decoder(self.config.n_dense_layers, self.config)
        self.decoder = nn.TransformerDecoder(dec_layer, num_decoder_layers)

        scale = d_model // 256

        # ----  Prediction head ---------------------------------------------
        if self.continuous_loads:
            out_dim = 1 if self.continuous_head == "mse" else 2
            self.logits = nn.Linear(d_model, out_dim)
            self.power_embedding = nn.Linear(1, 64 * scale)
        else:
            self.logits = nn.Linear(d_model, self.vocab_size)
            self.power_embedding = TokenEmbedding(self.vocab_size, 64 * scale)

        self.tgt_mask = nn.Transformer.generate_square_subsequent_mask(self.pred_len)
        self.positional_encoding = MoEPositionalEncoding(d_model, dropout=dropout)
        self.building_embedding = nn.Embedding(2, 32 * scale)
        self.lat_embedding = nn.Linear(1, 32 * scale)
        self.lon_embedding = nn.Linear(1, 32 * scale)
        if self.ignore_spatial:
            self.lat_embedding = ZeroEmbedding(32 * scale)
            self.lon_embedding = ZeroEmbedding(32 * scale)

        self.day_of_year_encoding = TimeSeriesSinusoidalPeriodicEmbedding(32 * scale)
        self.day_of_week_encoding = TimeSeriesSinusoidalPeriodicEmbedding(32 * scale)
        self.hour_of_day_encoding = TimeSeriesSinusoidalPeriodicEmbedding(32 * scale)

    def to(self, device):
        self.tgt_mask = self.tgt_mask.to(device)
        return super().to(device)

    def forward(self, x: Dict[str, torch.Tensor]):

        # ---- 构造时间序列 embedding ----
        ts_inputs = [
            self.lat_embedding(x["latitude"]),
            self.lon_embedding(x["longitude"]),
            self.building_embedding(x["building_type"]).squeeze(2),
            self.day_of_year_encoding(x["day_of_year"]),
            self.day_of_week_encoding(x["day_of_week"]),
            self.hour_of_day_encoding(x["hour_of_day"]),
            self.power_embedding(x["load"]).squeeze(2),
        ]
        if self.weather_features:
            ts_inputs.insert(
                -1,
                self.weather_embedding(
                    torch.cat([x[ft] for ft in self.weather_features], dim=2)
                ),
            )

        ts_embed = torch.cat(ts_inputs, dim=2)  # [B, T, d_model]

        src_series_inputs = ts_embed[:, : self.context_len, :]
        tgt_series_inputs = ts_embed[:, self.context_len - 1 : -1, :]

        src_series_embed = self.positional_encoding(src_series_inputs)
        tgt_series_embed = self.positional_encoding(tgt_series_inputs)

        memory = self.encoder(src_series_embed)
        outs = self.decoder(tgt_series_embed, memory, tgt_mask=self.tgt_mask)
        return self.logits(outs)

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

    def predict(self, x: Dict) -> Tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        return self.generate_sample(x, greedy=True)

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

    @torch.no_grad()
    def generate_sample(self, x, temperature=1.0, greedy=False, num_samples=1):
        """Sample from the conditional distribution.

        Use output of decoder at each prediction step as input to the next decoder step.
        Implements greedy decoding and random temperature-controlled sampling.

        Top-k sampling and nucleus sampling are deprecated.

        Args:
            x (Dict): dictionary of input tensors
            temperature (float): temperature for sampling
            greedy (bool): whether to use greedy decoding
            num_samples (int): number of samples to generate

        Returns:
            predictions (torch.Tensor): of shape [batch_size, pred_len, 1] or shape [batch_size, num_samples, pred_len] if num_samples > 1.
            distribution_parameters (torch.Tensor): of shape [batch_size, pred_len, 1]. Not returned if sampling.
        """
        if self.weather_features:
            time_series_inputs = [
                self.lat_embedding(x["latitude"]),
                self.lon_embedding(x["longitude"]),
                self.building_embedding(x["building_type"]).squeeze(2),
                self.day_of_year_encoding(x["day_of_year"]),
                self.day_of_week_encoding(x["day_of_week"]),
                self.hour_of_day_encoding(x["hour_of_day"]),
                self.weather_embedding(
                    torch.cat([x[ft] for ft in self.weather_features], dim=2)
                ),
                self.power_embedding(x["load"]).squeeze(2),
            ]
        else:
            time_series_inputs = [
                self.lat_embedding(x["latitude"]),
                self.lon_embedding(x["longitude"]),
                self.building_embedding(x["building_type"]).squeeze(2),
                self.day_of_year_encoding(x["day_of_year"]),
                self.day_of_week_encoding(x["day_of_week"]),
                self.hour_of_day_encoding(x["hour_of_day"]),
                self.power_embedding(x["load"]).squeeze(2),
            ]
        time_series_embed = torch.cat(time_series_inputs, dim=2)
        # [batch_size, context_len, d_model]
        src_series_inputs = time_series_embed[:, : self.context_len, :]
        tgt_series_inputs = time_series_embed[:, self.context_len - 1 : -1, :]
        src_series_embed = self.positional_encoding(src_series_inputs)

        encoder_output = self.encoder(src_series_embed)
        decoder_input = tgt_series_inputs[:, 0, :].unsqueeze(1)
        if num_samples > 1 and not greedy:
            # [batch_size, 1, emb_size] --> [batch_size * num_sampes, 1, emb_size]
            decoder_input = decoder_input.repeat_interleave(num_samples, dim=0)
            encoder_output = encoder_output.repeat_interleave(num_samples, dim=0)
        all_preds, all_logits = [], []
        for k in range(1, self.pred_len + 1):
            decoder_embed = self.positional_encoding(decoder_input)
            tgt_mask = self.tgt_mask[:k, :k].to(encoder_output.device)
            decoder_output = self.decoder(
                decoder_embed, encoder_output, tgt_mask=tgt_mask
            )
            # [batch_size, 1] if continuous (2 if head is gaussian_nll) or [batch_size, vocab_size] if not continuous_loads
            outputs = self.logits(decoder_output[:, -1, :])
            all_logits += [outputs.unsqueeze(1)]

            if self.continuous_loads:
                if self.continuous_head == "mse":
                    all_preds += [outputs]
                elif self.continuous_head == "gaussian_nll":
                    if greedy:
                        all_preds += [outputs[:, 0].unsqueeze(1)]  # mean only
                        outputs = all_preds[-1]  # [batch_size, 1, 1]
                    else:
                        mean = outputs[:, 0]
                        std = torch.nn.functional.softplus(outputs[:, 1])
                        outputs = (
                            torch.distributions.normal.Normal(mean, std)
                            .sample()
                            .unsqueeze(1)
                        )
                        all_preds += [outputs]

            elif not greedy:
                # Sample from a Categorical distribution with logits outputs
                all_preds += [
                    torch.multinomial(
                        torch.nn.functional.softmax(outputs / temperature, dim=1), 1
                    )
                ]
                # change outputs to the predicted load tokens
                outputs = all_preds[-1]  # [batch_size * num_samples, 1]
            else:
                # outputs are [batch_size, vocab_size]
                # Greedy decoding
                all_preds += [outputs.argmax(dim=1).unsqueeze(1)]
                # change outputs to the predicted load tokens
                outputs = all_preds[-1]

            # [batch_size, d_model]
            if k < self.pred_len:
                # [batch_size, d_model]
                next_decoder_input = tgt_series_inputs[:, k]
                if num_samples > 1 and not greedy:
                    # [batch_size, d_model] --> [batch_size * num_samples, d_model]
                    next_decoder_input = next_decoder_input.repeat_interleave(
                        num_samples, dim=0
                    )
                # Use the embedding predicted load instead of the ground truth load
                embedded_pred = self.power_embedding(outputs)
                if not self.continuous_loads:
                    # [batch_size, 1, 1, 64*scale] --> [batch_size, 64*scale]
                    embedded_pred = embedded_pred.squeeze(2).squeeze(1)
                next_decoder_input = torch.cat(
                    [next_decoder_input[:, : -embedded_pred.shape[-1]], embedded_pred],
                    dim=1,
                )
                # Append the next decoder input to the decoder input
                decoder_input = torch.cat(
                    [decoder_input, next_decoder_input.unsqueeze(1)], dim=1
                )
        if num_samples == 1 or greedy:
            if self.continuous_head == "gaussian_nll":
                # [batch_size, pred_len, 2]
                gaussian_params = torch.stack(all_logits, 1)[:, :, 0, :]
                means = gaussian_params[:, :, 0]
                sigma = torch.nn.functional.softplus(gaussian_params[:, :, 1])
                return torch.stack(all_preds, 1), torch.cat(
                    [means.unsqueeze(2), sigma.unsqueeze(2)], 2
                )
            else:
                return torch.stack(all_preds, 1), torch.stack(all_logits, 1)[:, :, 0, :]
        else:
            # [batch_size, num_samples, pred_len]
            return torch.stack(all_preds, 1).reshape(-1, num_samples, self.pred_len)


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
        continuous_head="mse",
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
    predictions, _ = model.predict(dummy_input)

    # 输出结果
    print("Predictions Shape:", predictions.shape)
    # print("Predictions:", predictions)

    seqlen = 2

    mask = torch.full((seqlen, seqlen), float("-inf"), device=device).triu_(1)

    print("Mask Shape:", mask.shape)
    print(mask)
    print(mask.unsqueeze(1))
