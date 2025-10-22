import math
from types import SimpleNamespace
from typing import Dict, Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from buildings_bench.models.base_model import BaseModel


def compared_version(ver1, ver2):
    """
    :param ver1
    :param ver2
    :return: ver1< = >ver2 False/True
    """
    list1 = str(ver1).split(".")
    list2 = str(ver2).split(".")

    for i in range(len(list1)) if len(list1) < len(list2) else range(len(list2)):
        if int(list1[i]) == int(list2[i]):
            pass
        elif int(list1[i]) < int(list2[i]):
            return -1
        else:
            return 1

    if len(list1) == len(list2):
        return True
    elif len(list1) < len(list2):
        return False
    else:
        return True


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if compared_version(torch.__version__, "1.5.0") else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type="fixed", freq="h"):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == "fixed" else nn.Embedding
        if freq == "t":
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()

        minute_x = (
            self.minute_embed(x[:, :, 4]) if hasattr(self, "minute_embed") else 0.0
        )
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type="timeF", freq="h"):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {"h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type="fixed", freq="h", dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = (
            TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
            if embed_type != "timeF"
            else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = (
            self.value_embedding(x)
            + self.temporal_embedding(x_mark)
            + self.position_embedding(x)
        )
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type="fixed", freq="h", dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = (
            TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
            if embed_type != "timeF"
            else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class AutoCorrelation(nn.Module):
    """
    AutoCorrelation Mechanism with the following two phases:
    (1) period-based dependencies discovery
    (2) time delay aggregation
    This block can replace the self-attention family mechanism seamlessly.
    """

    def __init__(
        self,
        mask_flag=True,
        factor=1,
        scale=None,
        attention_dropout=0.1,
        output_attention=False,
    ):
        super(AutoCorrelation, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def time_delay_agg_training(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the training phase.
        """
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            pattern = torch.roll(tmp_values, -int(index[i]), -1)
            delays_agg = delays_agg + pattern * (
                tmp_corr[:, i]
                .unsqueeze(1)
                .unsqueeze(1)
                .unsqueeze(1)
                .repeat(1, head, channel, length)
            )
        return delays_agg

    def time_delay_agg_inference(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the inference phase.
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = (
            torch.arange(length)
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(batch, head, channel, 1)
            .to(values.device)
        )
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        weights, delay = torch.topk(mean_value, top_k, dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(
                1
            ).repeat(1, head, channel, length)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * (
                tmp_corr[:, i]
                .unsqueeze(1)
                .unsqueeze(1)
                .unsqueeze(1)
                .repeat(1, head, channel, length)
            )
        return delays_agg

    def time_delay_agg_full(self, values, corr):
        """
        Standard version of Autocorrelation
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = (
            torch.arange(length)
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(batch, head, channel, 1)
            .to(values.device)
        )
        # find top k
        top_k = int(self.factor * math.log(length))
        weights, delay = torch.topk(corr, top_k, dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[..., i].unsqueeze(-1)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * (tmp_corr[..., i].unsqueeze(-1))
        return delays_agg

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        if L > S:
            zeros = torch.zeros_like(queries[:, : (L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        # period-based dependencies
        q = queries.permute(0, 2, 3, 1).contiguous()  # (B, H, E, L)
        k = keys.permute(0, 2, 3, 1).contiguous()
        use_fp32_fft = q.dtype in (torch.float16, torch.bfloat16)
        if use_fp32_fft:
            q_fft = torch.fft.rfft(q.float(), dim=-1)  # -> complex64
            k_fft = torch.fft.rfft(k.float(), dim=-1)
            res = q_fft * torch.conj(k_fft)
            corr = torch.fft.irfft(res, n=L, dim=-1).to(q.dtype)  # 回到原始实数精度
        else:
            q_fft = torch.fft.rfft(q, dim=-1)
            k_fft = torch.fft.rfft(k, dim=-1)
            res = q_fft * torch.conj(k_fft)
            corr = torch.fft.irfft(res, n=L, dim=-1)

        # time delay agg
        if self.training:
            V = self.time_delay_agg_training(
                values.permute(0, 2, 3, 1).contiguous(), corr
            ).permute(0, 3, 1, 2)
        else:
            V = self.time_delay_agg_inference(
                values.permute(0, 2, 3, 1).contiguous(), corr
            ).permute(0, 3, 1, 2)

        if self.output_attention:
            return (V.contiguous(), corr.permute(0, 3, 1, 2))
        else:
            return (V.contiguous(), None)


class AutoCorrelationLayer(nn.Module):
    def __init__(self, correlation, d_model, n_heads, d_keys=None, d_values=None):
        super(AutoCorrelationLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_correlation(queries, keys, values, attn_mask)
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class my_Layernorm(nn.Module):
    """
    Special designed layernorm for the seasonal part
    """

    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class EncoderLayer(nn.Module):
    """
    Autoformer encoder layer with the progressive decomposition architecture
    """

    def __init__(
        self,
        attention,
        d_model,
        d_ff=None,
        moving_avg=25,
        dropout=0.1,
        activation="relu",
    ):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(
            in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False
        )
        self.conv2 = nn.Conv1d(
            in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False
        )
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        x, _ = self.decomp1(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        res, _ = self.decomp2(x + y)
        return res, attn


class Encoder(nn.Module):
    """
    Autoformer encoder
    """

    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = (
            nn.ModuleList(conv_layers) if conv_layers is not None else None
        )
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    """
    Autoformer decoder layer with the progressive decomposition architecture
    """

    def __init__(
        self,
        self_attention,
        cross_attention,
        d_model,
        c_out,
        d_ff=None,
        moving_avg=25,
        dropout=0.1,
        activation="relu",
    ):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(
            in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False
        )
        self.conv2 = nn.Conv1d(
            in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False
        )
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.decomp3 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(
            in_channels=d_model,
            out_channels=c_out,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="circular",
            bias=False,
        )
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
        x, trend1 = self.decomp1(x)
        x = x + self.dropout(
            self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0]
        )
        x, trend2 = self.decomp2(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y)

        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(
            1, 2
        )
        return x, residual_trend


class Decoder(nn.Module):
    """
    Autoformer encoder
    """

    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            trend = trend + residual_trend

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x, trend


class AutoformerCore(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """

    def __init__(self, configs):
        super(AutoformerCore, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Decomp
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(
            configs.enc_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )
        self.dec_embedding = DataEmbedding_wo_pos(
            configs.dec_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            True,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True),
        )

    def forward(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
        enc_self_mask=None,
        dec_self_mask=None,
        dec_enc_mask=None,
    ):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros(
            [x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device
        )
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len :, :], mean], dim=1)
        seasonal_init = torch.cat(
            [seasonal_init[:, -self.label_len :, :], zeros], dim=1
        )
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(
            dec_out,
            enc_out,
            x_mask=dec_self_mask,
            cross_mask=dec_enc_mask,
            trend=trend_init,
        )
        # final
        dec_out = trend_part + seasonal_part

        if self.output_attention:
            return dec_out[:, -self.pred_len :, :], attns
        else:
            return dec_out[:, -self.pred_len :, :]  # [B, L, D]


class AutoformerUnified(BaseModel):
    """
    Autoformer (Unified Interface, Scheme-A):
      - Train with fixed context_len=336, pred_len=168
      - Inference: always produce 168 steps and then slice the first `pred_len` (<=168)
      - If provided context_len != 336:
          * >336: take the last 336 steps
          * <336: left-pad by repeating the first value to reach 336
      - Loss: Huber (delta configurable)
      - Time features (x_mark_enc/dec): zeros placeholder with `time_feat_dim`
        (set to match your repo's TimeFeatureEmbedding for the chosen `freq`)
    """

    def __init__(
        self,
        # 方案A固定长度
        context_len: int = 336,
        pred_len: int = 168,
        label_len: int = 48,  # 解码器“已知标签段”长度，常用 48（可调）
        # Autoformer超参
        d_model: int = 768,
        n_heads: int = 12,
        e_layers: int = 8,
        d_layers: int = 10,
        d_ff: int = 2048,
        factor: int = 3,
        dropout: float = 0.0,
        activation: str = "gelu",
        moving_avg: int = 25,
        embed: str = "timeF",  # 官方DataEmbedding_wo_pos的设置
        freq: str = "h",  # 你的数据粒度（小时时 'h'）
        time_feat_dim: int = 4,  # 与freq对应的时间特征维度（常见'h'为4）
        # 训练头部
        huber_delta: float = 1.0,
        continuous_loads: bool = True,
        continuous_head: str = "linear",
        output_attention: bool = False,
        **kwargs,
    ):
        super().__init__(context_len, pred_len, continuous_loads=continuous_loads)

        self.label_len = label_len
        self.time_feat_dim = time_feat_dim
        self.huber_delta = huber_delta
        self.output_attention = output_attention
        self.continuous_head = continuous_head

        # 官方Autoformer配置对象
        cfg = SimpleNamespace(
            seq_len=context_len,
            label_len=label_len,
            pred_len=pred_len,
            output_attention=output_attention,
            moving_avg=moving_avg,
            enc_in=1,
            dec_in=1,
            c_out=1,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            e_layers=e_layers,
            d_layers=d_layers,
            factor=factor,
            dropout=dropout,
            activation=activation,
            embed=embed,
            freq=freq,
        )

        self.core = AutoformerCore(cfg)

    # ---------- helpers ----------
    def _fix_context_len(self, ctx: torch.Tensor) -> torch.Tensor:
        """
        Ensure context length == self.context_len (=336) by cropping/padding.
        Args:
            ctx: (B, Lc, 1)
        Returns:
            (B, 336, 1)
        """
        B, Lc, C = ctx.shape
        if Lc == self.context_len:
            return ctx
        elif Lc > self.context_len:
            return ctx[:, -self.context_len :, :]
        else:
            pad = self.context_len - Lc
            left = ctx[:, :1, :].repeat(1, pad, 1)
            return torch.cat([left, ctx], dim=1)

    def _build_marks(self, B: int, enc_len: int, dec_total_len: int, device, dtype):
        """
        Build zero time-feature marks to satisfy DataEmbedding_wo_pos's signature.
        Shapes:
          x_mark_enc: (B, enc_len, time_feat_dim)
          x_mark_dec: (B, dec_total_len, time_feat_dim)
        """
        x_mark_enc = torch.zeros(
            B, enc_len, self.time_feat_dim, device=device, dtype=dtype
        )
        x_mark_dec = torch.zeros(
            B, dec_total_len, self.time_feat_dim, device=device, dtype=dtype
        )
        return x_mark_enc, x_mark_dec

    # ---------- forward ----------
    def forward(
        self,
        x: Dict[str, torch.Tensor],
        context_len: Optional[int] = None,
        pred_len: Optional[int] = None,
    ) -> torch.Tensor:
        """
        x["load"]: (B, context_len + pred_len, 1)
        return: (B, pred_len, 1)  —— pred_len<=168 时从 168 头部切片
        """
        if context_len is None:
            context_len = self.context_len
        if pred_len is None:
            pred_len = self.pred_len
        if pred_len > self.pred_len:
            raise ValueError(
                f"pred_len({pred_len}) > max {self.pred_len} under Scheme A"
            )

        load = x["load"]  # (B, L, 1)
        assert load.dim() == 3 and load.size(-1) == 1, "expect load shape (B, L, 1)"
        assert load.size(1) >= context_len, "load length must be >= context_len"

        # 取上下文并裁剪/填充到 336
        src = load[:, :context_len, :]  # (B, ctx, 1)
        x_enc = self._fix_context_len(src)  # (B, 336, 1)

        B, _, _ = x_enc.shape
        device, dtype = x_enc.device, x_enc.dtype

        # 构造解码器输入（仅用于形状，无需真实值；官方forward内部会重组）
        zeros_pred = torch.zeros(B, self.pred_len, 1, device=device, dtype=dtype)
        x_dec = torch.cat(
            [x_enc[:, -self.label_len :, :], zeros_pred], dim=1
        )  # (B, label_len+pred_len, 1)

        # 时间标记（零占位）
        x_mark_enc, x_mark_dec = self._build_marks(
            B,
            enc_len=self.context_len,
            dec_total_len=self.label_len + self.pred_len,
            device=device,
            dtype=dtype,
        )

        # 官方Autoformer前向（会返回 (B, 168, 1)）
        y_full = self.core(
            x_enc=x_enc,
            x_mark_enc=x_mark_enc,
            x_dec=x_dec,
            x_mark_dec=x_mark_dec,
            enc_self_mask=None,
            dec_self_mask=None,
            dec_enc_mask=None,
        )
        # core 可能返回 (y, attn) 或 y
        if isinstance(y_full, (tuple, list)):
            y_full = y_full[0]

        # 截取到请求的 pred_len（<=168）
        y_hat = y_full[:, :pred_len, :]
        return y_hat

    # ---------- loss ----------
    def loss(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if not self.continuous_loads:
            raise ValueError("AutoformerUnified is implemented for continuous loads.")
        return F.huber_loss(pred, y, delta=self.huber_delta, reduction="mean")

    # ---------- predict ----------
    @torch.no_grad()
    def predict(
        self,
        x: Dict[str, torch.Tensor],
        context_len: int = 336,
        pred_len: int = 168,
    ):
        preds = self.forward(x, context_len=context_len, pred_len=pred_len)
        return preds, preds

    # ---------- finetune / ckpt ----------
    def unfreeze_and_get_parameters_for_finetuning(self):
        for p in self.parameters():
            p.requires_grad = True
        return self.parameters()

    def load_from_checkpoint(self, checkpoint_path: str):
        state = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        state = {k.replace("module.", ""): v for k, v in state.items()}
        self.load_state_dict(state, strict=False)


class _AFConfigs:
    """最小配置容器；和 AutoformerCore __init__ 里的读取保持字段名一致"""

    def __init__(
        self,
        seq_len: int,
        label_len: int,
        pred_len: int,
        enc_in: int = 1,
        dec_in: int = 1,
        c_out: int = 1,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 2048,
        e_layers: int = 2,
        d_layers: int = 1,
        moving_avg: int = 25,
        factor: int = 3,
        dropout: float = 0.1,
        embed: str = "timeF",  # 未用到位置编码时也可保留
        freq: str = "h",
        activation: str = "gelu",
        output_attention: bool = False,
    ):
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.dec_in = dec_in
        self.c_out = c_out
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.e_layers = e_layers
        self.d_layers = d_layers
        self.moving_avg = moving_avg
        self.factor = factor
        self.dropout = dropout
        self.embed = embed
        self.freq = freq
        self.activation = activation
        self.output_attention = output_attention


class AutoformerBB(BaseModel):
    """
    BuildingsBench 适配版 Autoformer：
    - 输入 batch 至少包含 {"load": [B, T, 1]}，可选 {"x_mark_enc": [B, ctx, Dm], "x_mark_dec": [B, label_len+pred, Dm]}
    - forward(context_len, pred_len) 固定 I/O；内部对 AutoformerCore 的长度动态对齐
    - 输出 [B, pred_len, 1]
    """

    def __init__(
        self,
        max_context_len: int = 336,
        max_pred_len: int = 168,
        context_len: int = 168,
        pred_len: int = 24,
        *,
        # Autoformer 结构参数
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 2048,
        e_layers: int = 2,
        d_layers: int = 1,
        moving_avg: int = 25,
        factor: int = 3,
        dropout: float = 0.1,
        activation: str = "gelu",
        embed: str = "timeF",  # 采用 TimeFeatureEmbedding（float）
        freq: str = "h",  # 与 TimeFeatureEmbedding 的 freq_map 对齐（'h' -> 4 维）
        output_attention: bool = False,
        # BuildingsBench 习惯参数
        continuous_loads: bool = True,
        continuous_head: Literal["huber"] = "huber",
        label_len: int = 48,  # decoder 侧“已知历史”长度上限
        **kwargs,
    ):
        super().__init__(context_len, pred_len, continuous_loads=continuous_loads)
        self.max_context_len = int(max_context_len)
        self.max_pred_len = int(max_pred_len)
        self.continuous_head = continuous_head
        self.label_len_cfg = int(label_len)  # 配置上限，实际会截断到 <= context_len
        self.freq = freq

        # —— 用“最大长度”初始化核心；实际 forward 按需覆盖 —— #
        cfg = _AFConfigs(
            seq_len=self.max_context_len,
            label_len=self.label_len_cfg,
            pred_len=self.max_pred_len,
            enc_in=1,
            dec_in=1,
            c_out=1,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            e_layers=e_layers,
            d_layers=d_layers,
            moving_avg=moving_avg,
            factor=factor,
            dropout=dropout,
            embed=embed,
            freq=freq,
            activation=activation,
            output_attention=output_attention,
        )
        self.core = AutoformerCore(cfg)

    # ---------- helpers ----------

    @staticmethod
    def _ensure_float(x: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        # 强制到 float32（或与 like 一致的 float），避免 long/float 冲突
        want_dtype = torch.float32 if not like.is_floating_point() else like.dtype
        return x.to(dtype=want_dtype)

    def _timefeat_dim(self) -> int:
        # 与 TimeFeatureEmbedding.freq_map 一致
        freq_map = {"h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3}
        if self.freq not in freq_map:
            raise ValueError(f"Unsupported freq '{self.freq}' for timeF embedding.")
        return freq_map[self.freq]

    def _make_time_marks(self, B: int, L: int, device, dtype) -> torch.Tensor:
        # 生成全 0 的 time features（float），形状 [B, L, Dm]
        Dm = self._timefeat_dim()
        return torch.zeros(B, L, Dm, device=device, dtype=dtype)

    # ---------- forward ----------

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

        load = x["load"]  # [B, T, 1]
        assert load.dim() == 3 and load.size(-1) == 1, "load 形状必须是 [B, T, 1]"
        assert load.size(1) >= context_len + pred_len, "序列长度不足以覆盖 ctx+pred"

        B, T, _ = load.shape
        device = load.device

        # 编码器输入：仅上下文
        x_enc = load[:, :context_len, :]
        x_enc = self._ensure_float(x_enc, like=load)

        # 解码器输入：label_len 的已知历史 + pred_len 的占位
        eff_label_len = min(self.label_len_cfg, context_len)
        known_hist = load[:, context_len - eff_label_len : context_len, :]
        known_hist = self._ensure_float(known_hist, like=load)

        zeros_pred = torch.zeros(B, pred_len, 1, device=device, dtype=x_enc.dtype)
        x_dec = torch.cat([known_hist, zeros_pred], dim=1)  # [B, eff_label_len+pred, 1]

        # 时间标记：优先使用 batch 中的 x_mark_*；否则生成占位
        # —— AutoformerCore 使用 DataEmbedding_wo_pos + embed='timeF' → 需要 float —— #
        if "x_mark_enc" in x:
            x_mark_enc = self._ensure_float(x["x_mark_enc"], like=x_enc)
            assert (
                x_mark_enc.shape[0] == B and x_mark_enc.shape[1] == context_len
            ), f"x_mark_enc 形状应为 [B, {context_len}, Dm]"
        else:
            x_mark_enc = self._make_time_marks(B, context_len, device, x_enc.dtype)

        if "x_mark_dec" in x:
            x_mark_dec = self._ensure_float(x["x_mark_dec"], like=x_enc)
            assert (
                x_mark_dec.shape[0] == B
                and x_mark_dec.shape[1] == eff_label_len + pred_len
            ), f"x_mark_dec 形状应为 [B, {eff_label_len + pred_len}, Dm]"
        else:
            x_mark_dec = self._make_time_marks(
                B, eff_label_len + pred_len, device, x_enc.dtype
            )

        # —— 动态覆盖核心长度，保证与本次 forward 完全一致 —— #
        self.core.seq_len = int(context_len)
        self.core.label_len = int(eff_label_len)
        self.core.pred_len = int(pred_len)

        # 前向
        dec_out = self.core(
            x_enc=x_enc,
            x_mark_enc=x_mark_enc,
            x_dec=x_dec,
            x_mark_dec=x_mark_dec,
            enc_self_mask=None,
            dec_self_mask=None,
            dec_enc_mask=None,
        )
        if isinstance(dec_out, tuple):
            dec_out = dec_out[0]  # 丢掉注意力
        dec_out = dec_out[:, -pred_len:, :]  # 稳妥截断
        return dec_out  # [B, pred_len, 1]

    # ---------- loss / predict / finetune / ckpt ----------

    def loss(
        self, pred: torch.Tensor, y: torch.Tensor, progress: Optional[float] = None
    ) -> torch.Tensor:
        if self.continuous_head == "huber":
            if pred.ndim == 3 and pred.size(-1) == 1:
                pred = pred.squeeze(-1)
            if y.ndim == 3 and y.size(-1) == 1:
                y = y.squeeze(-1)
            return F.huber_loss(pred, y, delta=1.0)
        raise NotImplementedError(f"Unsupported head: {self.continuous_head}")

    @torch.no_grad()
    def predict(
        self, x: Dict[str, torch.Tensor], context_len: int = 168, pred_len: int = 24
    ):
        preds = self.forward(x, context_len, pred_len)  # [B, pred, 1]
        return preds, preds  # 第二个返回值占位，兼容训练脚本

    def unfreeze_and_get_parameters_for_finetuning(self):
        # 需要“参数组策略”时可像我之前给 BuildMoE 的那套一样扩展；默认全量
        return self.parameters()

    def load_from_checkpoint(self, checkpoint_path: str):
        state = torch.load(checkpoint_path, map_location="cpu")
        sd = state.get("model", state)
        new_sd = {k.replace("module.", ""): v for k, v in sd.items()}
        self.load_state_dict(new_sd, strict=False)


# if __name__ == "__main__":
#     model = AutoformerUnified()
#     x = torch.randn(16, 300 + 100, 1)
#     y = model({"load": x}, context_len=300, pred_len=100)
#     print(y.shape)
#     loss = model.loss(y, torch.randn(16, 100, 1))
#     print(loss)
#     n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(n_params)


if __name__ == "__main__":
    import torch

    cfgs = [
        # 你已有的 S/M/L ...
        dict(
            name="S",
            d_model=256,
            n_heads=4,
            d_ff=1024,
            e_layers=2,
            d_layers=1,
            moving_avg=25,
            factor=3,
            dropout=0.1,
            label_len=48,
        ),
        dict(
            name="M",
            d_model=512,
            n_heads=8,
            d_ff=2048,
            e_layers=2,
            d_layers=1,
            moving_avg=25,
            factor=3,
            dropout=0.1,
            label_len=48,
        ),
        dict(
            name="L",
            d_model=768,
            n_heads=12,
            d_ff=3072,
            e_layers=3,
            d_layers=2,
            moving_avg=25,
            factor=3,
            dropout=0.1,
            label_len=48,
        ),
        # —— 新增：100M 附近的三档 —— #
        dict(
            name="XL-≈90M",
            d_model=1152,
            n_heads=16,
            d_ff=4608,
            e_layers=3,
            d_layers=2,
            moving_avg=25,
            factor=3,
            dropout=0.1,
            label_len=48,
        ),
        dict(
            name="XL≈100M",
            d_model=1024,
            n_heads=16,
            d_ff=4096,
            e_layers=4,
            d_layers=3,
            moving_avg=25,
            factor=3,
            dropout=0.1,
            label_len=48,
        ),
        dict(
            name="XL+≈111M",
            d_model=1280,
            n_heads=20,
            d_ff=5120,
            e_layers=3,
            d_layers=2,
            moving_avg=25,
            factor=3,
            dropout=0.1,
            label_len=48,
        ),
    ]

    def count_params(m: torch.nn.Module):
        total = sum(p.numel() for p in m.parameters())
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        return total, trainable

    print("== AutoformerBB 参数量检查（含 ~100M 档） ==")
    for c in cfgs:
        model = AutoformerBB(
            max_context_len=336,
            max_pred_len=168,
            context_len=168,
            pred_len=24,
            d_model=c["d_model"],
            n_heads=c["n_heads"],
            d_ff=c["d_ff"],
            e_layers=c["e_layers"],
            d_layers=c["d_layers"],
            moving_avg=c["moving_avg"],
            factor=c["factor"],
            dropout=c["dropout"],
            embed="timeF",
            freq="h",
            label_len=c["label_len"],
            continuous_loads=True,
            continuous_head="huber",
        )
        total, trainable = count_params(model)
        size_mb = total * 4 / (1024**2)  # fp32

        print(
            f"[{c['name']}] "
            f"d_model={c['d_model']}, heads={c['n_heads']}, d_ff={c['d_ff']}, "
            f"E={c['e_layers']}, D={c['d_layers']}, "
            f"Params(total/trainable)={total:,}/{trainable:,} (~{size_mb:.2f} MB, fp32)"
        )
