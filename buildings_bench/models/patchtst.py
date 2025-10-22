from typing import Dict, Literal, Optional
from buildings_bench.models.base_model import BaseModel
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import math


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)


def get_activation_fn(activation):
    if callable(activation):
        return activation()
    elif activation.lower() == "relu":
        return nn.ReLU()
    elif activation.lower() == "gelu":
        return nn.GELU()
    raise ValueError(
        f'{activation} is not available. You can use "relu", "gelu", or a callable'
    )


# decomposition


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


# pos_encoding


def PositionalEncoding(q_len, d_model, normalize=True):
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe


SinCosPosEncoding = PositionalEncoding


def Coord2dPosEncoding(
    q_len, d_model, exponential=False, normalize=True, eps=1e-3, verbose=False
):
    x = 0.5 if exponential else 1
    i = 0
    for i in range(100):
        cpe = (
            2
            * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** x)
            * (torch.linspace(0, 1, d_model).reshape(1, -1) ** x)
            - 1
        )
        pv(f"{i:4.0f}  {x:5.3f}  {cpe.mean():+6.3f}", verbose)
        if abs(cpe.mean()) <= eps:
            break
        elif cpe.mean() > eps:
            x += 0.001
        else:
            x -= 0.001
        i += 1
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe


def Coord1dPosEncoding(q_len, exponential=False, normalize=True):
    cpe = (
        2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** (0.5 if exponential else 1))
        - 1
    )
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe


def positional_encoding(pe, learn_pe, q_len, d_model):
    # Positional encoding
    if pe == None:
        W_pos = torch.empty(
            (q_len, d_model)
        )  # pe = None and learn_pe = False can be used to measure impact of pe
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == "zero":
        W_pos = torch.empty((q_len, 1))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == "zeros":
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == "normal" or pe == "gauss":
        W_pos = torch.zeros((q_len, 1))
        torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
    elif pe == "uniform":
        W_pos = torch.zeros((q_len, 1))
        nn.init.uniform_(W_pos, a=0.0, b=0.1)
    elif pe == "lin1d":
        W_pos = Coord1dPosEncoding(q_len, exponential=False, normalize=True)
    elif pe == "exp1d":
        W_pos = Coord1dPosEncoding(q_len, exponential=True, normalize=True)
    elif pe == "lin2d":
        W_pos = Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True)
    elif pe == "exp2d":
        W_pos = Coord2dPosEncoding(q_len, d_model, exponential=True, normalize=True)
    elif pe == "sincos":
        W_pos = PositionalEncoding(q_len, d_model, normalize=True)
    else:
        raise ValueError(
            f"{pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
        'zeros', 'zero', uniform', 'lin1d', 'exp1d', 'lin2d', 'exp2d', 'sincos', None.)"
        )
    return nn.Parameter(W_pos, requires_grad=learn_pe)


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])  # z: [bs x d_model * patch_num]
                z = self.linears[i](z)  # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)  # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x


class TSTiEncoder(nn.Module):  # i means channel-independent
    def __init__(
        self,
        c_in,
        patch_num,
        patch_len,
        max_seq_len=1024,
        n_layers=3,
        d_model=128,
        n_heads=16,
        d_k=None,
        d_v=None,
        d_ff=256,
        norm="BatchNorm",
        attn_dropout=0.0,
        dropout=0.0,
        act="gelu",
        store_attn=False,
        key_padding_mask="auto",
        padding_var=None,
        attn_mask=None,
        res_attention=True,
        pre_norm=False,
        pe="zeros",
        learn_pe=True,
        verbose=False,
        **kwargs,
    ):

        super().__init__()

        self.patch_num = patch_num
        self.patch_len = patch_len

        # Input encoding
        q_len = patch_num
        self.W_P = nn.Linear(
            patch_len, d_model
        )  # Eq 1: projection of feature vectors onto a d-dim vector space
        self.seq_len = q_len

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(
            q_len,
            d_model,
            n_heads,
            d_k=d_k,
            d_v=d_v,
            d_ff=d_ff,
            norm=norm,
            attn_dropout=attn_dropout,
            dropout=dropout,
            pre_norm=pre_norm,
            activation=act,
            res_attention=res_attention,
            n_layers=n_layers,
            store_attn=store_attn,
        )

    def forward(self, x) -> Tensor:  # x: [bs x nvars x patch_len x patch_num]

        n_vars = x.shape[1]
        # Input encoding
        x = x.permute(0, 1, 3, 2)  # x: [bs x nvars x patch_num x patch_len]
        x = self.W_P(x)  # x: [bs x nvars x patch_num x d_model]

        u = torch.reshape(
            x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        )  # u: [bs * nvars x patch_num x d_model]
        u = self.dropout(u + self.W_pos)  # u: [bs * nvars x patch_num x d_model]

        # Encoder
        z = self.encoder(u)  # z: [bs * nvars x patch_num x d_model]
        z = torch.reshape(
            z, (-1, n_vars, z.shape[-2], z.shape[-1])
        )  # z: [bs x nvars x patch_num x d_model]
        z = z.permute(0, 1, 3, 2)  # z: [bs x nvars x d_model x patch_num]

        return z


# Cell
class TSTEncoder(nn.Module):
    def __init__(
        self,
        q_len,
        d_model,
        n_heads,
        d_k=None,
        d_v=None,
        d_ff=None,
        norm="BatchNorm",
        attn_dropout=0.0,
        dropout=0.0,
        activation="gelu",
        res_attention=False,
        n_layers=1,
        pre_norm=False,
        store_attn=False,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                TSTEncoderLayer(
                    q_len,
                    d_model,
                    n_heads=n_heads,
                    d_k=d_k,
                    d_v=d_v,
                    d_ff=d_ff,
                    norm=norm,
                    attn_dropout=attn_dropout,
                    dropout=dropout,
                    activation=activation,
                    res_attention=res_attention,
                    pre_norm=pre_norm,
                    store_attn=store_attn,
                )
                for i in range(n_layers)
            ]
        )
        self.res_attention = res_attention

    def forward(
        self,
        src: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ):
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers:
                output, scores = mod(
                    output,
                    prev=scores,
                    key_padding_mask=key_padding_mask,
                    attn_mask=attn_mask,
                )
            return output
        else:
            for mod in self.layers:
                output = mod(
                    output, key_padding_mask=key_padding_mask, attn_mask=attn_mask
                )
            return output


class TSTEncoderLayer(nn.Module):
    def __init__(
        self,
        q_len,
        d_model,
        n_heads,
        d_k=None,
        d_v=None,
        d_ff=256,
        store_attn=False,
        norm="BatchNorm",
        attn_dropout=0,
        dropout=0.0,
        bias=True,
        activation="gelu",
        res_attention=False,
        pre_norm=False,
    ):
        super().__init__()
        assert (
            not d_model % n_heads
        ), f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(
            d_model,
            n_heads,
            d_k,
            d_v,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
            res_attention=res_attention,
        )

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2)
            )
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=bias),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=bias),
        )

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2)
            )
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def forward(
        self,
        src: Tensor,
        prev: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:

        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(
                src,
                src,
                src,
                prev,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )
        else:
            src2, attn = self.self_attn(
                src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask
            )
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src = src + self.dropout_attn(
            src2
        )  # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(
            src2
        )  # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src


class _MultiheadAttention(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        d_k=None,
        d_v=None,
        res_attention=False,
        attn_dropout=0.0,
        proj_dropout=0.0,
        qkv_bias=True,
        lsa=False,
    ):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(
            d_model,
            n_heads,
            attn_dropout=attn_dropout,
            res_attention=self.res_attention,
            lsa=lsa,
        )

        # Poject output
        self.to_out = nn.Sequential(
            nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout)
        )

    def forward(
        self,
        Q: Tensor,
        K: Optional[Tensor] = None,
        V: Optional[Tensor] = None,
        prev: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ):

        bs = Q.size(0)
        if K is None:
            K = Q
        if V is None:
            V = Q

        # Linear (+ split in multiple heads)
        q_s = (
            self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        )  # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = (
            self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3, 1)
        )  # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = (
            self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)
        )  # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(
                q_s,
                k_s,
                v_s,
                prev=prev,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )
        else:
            output, attn_weights = self.sdp_attn(
                q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask
            )
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = (
            output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v)
        )  # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(
        self, d_model, n_heads, attn_dropout=0.0, res_attention=False, lsa=False
    ):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim**-0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        prev: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ):
        """
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        """

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = (
            torch.matmul(q, k) * self.scale
        )  # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None:
            attn_scores = attn_scores + prev

        # Attention mask (optional)
        if (
            attn_mask is not None
        ):  # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if (
            key_padding_mask is not None
        ):  # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(
                key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf
            )

        # normalize the attention weights
        attn_weights = F.softmax(
            attn_scores, dim=-1
        )  # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(
            attn_weights, v
        )  # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights


# Cell
class PatchTST_backbone(nn.Module):
    def __init__(
        self,
        c_in: int,
        context_window: int,
        target_window: int,
        patch_len: int,
        stride: int,
        max_seq_len: Optional[int] = 1024,
        n_layers: int = 3,
        d_model=128,
        n_heads=16,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        d_ff: int = 256,
        norm: str = "BatchNorm",
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        act: str = "gelu",
        key_padding_mask: bool | str = "auto",
        padding_var: Optional[int] = None,
        attn_mask: Optional[Tensor] = None,
        res_attention: bool = True,
        pre_norm: bool = False,
        store_attn: bool = False,
        pe: str = "zeros",
        learn_pe: bool = True,
        fc_dropout: float = 0.0,
        head_dropout=0,
        padding_patch=None,
        pretrain_head: bool = False,
        head_type="flatten",
        individual=False,
        revin=True,
        affine=True,
        subtract_last=False,
        verbose: bool = False,
        **kwargs,
    ):

        super().__init__()

        # RevIn
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len) / stride + 1)
        if padding_patch == "end":  # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1

        # Backbone
        self.backbone = TSTiEncoder(
            c_in,
            patch_num=patch_num,
            patch_len=patch_len,
            max_seq_len=max_seq_len,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_k=d_k,
            d_v=d_v,
            d_ff=d_ff,
            attn_dropout=attn_dropout,
            dropout=dropout,
            act=act,
            key_padding_mask=key_padding_mask,
            padding_var=padding_var,
            attn_mask=attn_mask,
            res_attention=res_attention,
            pre_norm=pre_norm,
            store_attn=store_attn,
            pe=pe,
            learn_pe=learn_pe,
            verbose=verbose,
            **kwargs,
        )

        # Head
        self.head_nf = d_model * patch_num
        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual

        if self.pretrain_head:
            self.head = self.create_pretrain_head(
                self.head_nf, c_in, fc_dropout
            )  # custom head passed as a partial func with all its kwargs
        elif head_type == "flatten":
            self.head = Flatten_Head(
                self.individual,
                self.n_vars,
                self.head_nf,
                target_window,
                head_dropout=head_dropout,
            )

    def forward(self, z):  # z: [bs x nvars x seq_len]
        # norm
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, "norm")
            z = z.permute(0, 2, 1)

        # do patching
        if self.padding_patch == "end":
            z = self.padding_patch_layer(z)
        z = z.unfold(
            dimension=-1, size=self.patch_len, step=self.stride
        )  # z: [bs x nvars x patch_num x patch_len]
        z = z.permute(0, 1, 3, 2)  # z: [bs x nvars x patch_len x patch_num]

        # model
        z = self.backbone(z)  # z: [bs x nvars x d_model x patch_num]
        z = self.head(z)  # z: [bs x nvars x target_window]

        # denorm
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, "denorm")
            z = z.permute(0, 2, 1)
        return z

    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(nn.Dropout(dropout), nn.Conv1d(head_nf, vars, 1))


class Model(nn.Module):
    def __init__(
        self,
        configs,
        max_seq_len: Optional[int] = 1024,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        norm: str = "BatchNorm",
        attn_dropout: float = 0.0,
        act: str = "gelu",
        key_padding_mask: bool | str = "auto",
        padding_var: Optional[int] = None,
        attn_mask: Optional[Tensor] = None,
        res_attention: bool = True,
        pre_norm: bool = False,
        store_attn: bool = False,
        pe: str = "zeros",
        learn_pe: bool = True,
        pretrain_head: bool = False,
        head_type="flatten",
        verbose: bool = False,
        **kwargs,
    ):

        super().__init__()

        # load parameters
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len

        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout

        individual = configs.individual

        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch

        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last

        self.model = PatchTST_backbone(
            c_in=c_in,
            context_window=context_window,
            target_window=target_window,
            patch_len=patch_len,
            stride=stride,
            max_seq_len=max_seq_len,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_k=d_k,
            d_v=d_v,
            d_ff=d_ff,
            norm=norm,
            attn_dropout=attn_dropout,
            dropout=dropout,
            act=act,
            key_padding_mask=key_padding_mask,
            padding_var=padding_var,
            attn_mask=attn_mask,
            res_attention=res_attention,
            pre_norm=pre_norm,
            store_attn=store_attn,
            pe=pe,
            learn_pe=learn_pe,
            fc_dropout=fc_dropout,
            head_dropout=head_dropout,
            padding_patch=padding_patch,
            pretrain_head=pretrain_head,
            head_type=head_type,
            individual=individual,
            revin=revin,
            affine=affine,
            subtract_last=subtract_last,
            verbose=verbose,
            **kwargs,
        )

    def forward(self, x):  # x: [Batch, Input length, Channel]
        x = x.permute(0, 2, 1)  # x: [Batch, Channel, Input length]
        x = self.model(x)
        x = x.permute(0, 2, 1)  # x: [Batch, Input length, Channel]
        return x


class LoadForecastingPatchTST_A(BaseModel):
    def __init__(
        self,
        max_context_len: int = 336,
        max_pred_len: int = 168,
        # 下面两个仅用于文档/默认调用，forward/predict 可覆盖
        context_len: int = 336,
        pred_len: int = 168,
        e_layers: int = 6,
        n_heads: int = 4,
        d_model: int = 256,
        d_ff: int = 512,
        dropout: float = 0.0,
        fc_dropout: float = 0.0,
        head_dropout: float = 0.0,
        patch_len: int = 12,
        stride: int = 8,
        padding_patch: Optional[str] = None,
        revin: bool = False,
        affine: bool = True,
        subtract_last: bool = False,
        continuous_loads: bool = True,
        continuous_head: Literal["mse", "gaussian_nll", "huber"] = "huber",
        huber_delta: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            context_len=max_context_len,
            pred_len=max_pred_len,
            continuous_loads=continuous_loads,
        )
        self.max_context_len = max_context_len
        self.max_pred_len = max_pred_len
        self.continuous_head = continuous_head
        self.huber_delta = huber_delta

        # PatchTST configs：固定为最大地平线（方案A）
        from easydict import EasyDict as edict

        configs = edict(
            dict(
                enc_in=1,
                seq_len=max_context_len,
                pred_len=max_pred_len,
                e_layers=e_layers,
                n_heads=n_heads,
                d_model=d_model,
                d_ff=d_ff,
                dropout=dropout,
                fc_dropout=fc_dropout,
                head_dropout=head_dropout,
                individual=False,
                patch_len=patch_len,
                stride=stride,
                padding_patch=padding_patch,
                revin=revin,
                affine=affine,
                subtract_last=subtract_last,
            )
        )
        self.backbone = Model(configs)

        self.out_dim = 1 if continuous_head in ("mse", "huber") else 2
        self.head = nn.Linear(1, self.out_dim)

    def _align_context(self, ctx: torch.Tensor) -> torch.Tensor:
        B, L, C = ctx.shape
        if L == self.max_context_len:
            return ctx
        elif L < self.max_context_len:
            pad_len = self.max_context_len - L
            pad = ctx[:, :1, :].expand(B, pad_len, C)
            return torch.cat([pad, ctx], dim=1)
        else:
            return ctx[:, -self.max_context_len :, :]

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
        if pred_len > self.max_pred_len:
            raise ValueError(
                f"pred_len({pred_len}) > max_pred_len({self.max_pred_len})"
            )

        load = x["load"]  # (B, T, 1)
        assert load.dim() == 3 and load.size(-1) == 1
        assert load.size(1) >= context_len

        ctx = self._align_context(load[:, :context_len, :])  # (B, 336, 1)
        full_pred = self.backbone(ctx)  # (B, 168, 1)
        full_pred = self.head(full_pred)  # (B, 168, out_dim)
        return full_pred[:, :pred_len, :]

    def loss(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if not self.continuous_loads:
            B, L, _ = pred.shape
            return F.cross_entropy(pred.reshape(B * L, -1), y.long().reshape(B * L))
        if self.continuous_head == "huber":
            return F.huber_loss(
                pred[..., :1], y, delta=self.huber_delta, reduction="mean"
            )
        elif self.continuous_head == "mse":
            return F.mse_loss(pred[..., :1], y, reduction="mean")
        else:
            mu, sigma_sq = pred[..., :1], F.softplus(pred[..., 1:]) ** 2
            return (
                0.5 * (torch.log(2 * torch.pi * sigma_sq) + (y - mu) ** 2 / sigma_sq)
            ).mean()

    @torch.no_grad()
    def predict(
        self, x: Dict[str, torch.Tensor], context_len: int = 336, pred_len: int = 168
    ):
        preds = self.forward(x, context_len, pred_len)
        if self.continuous_head in ("mse", "huber"):
            return preds[..., :1], preds
        else:
            return preds[..., :1], preds

    def unfreeze_and_get_parameters_for_finetuning(self):
        return self.parameters()

    def load_from_checkpoint(self, checkpoint_path: str, strict: bool = False):
        state = torch.load(checkpoint_path, map_location="cpu")
        state = state.get("model", state)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        self.load_state_dict(state, strict=strict)


class _PTConfigs:
    """最小配置容器，字段名对齐你上面的 Model.__init__(configs) 读取"""

    def __init__(
        self,
        enc_in: int,
        seq_len: int,
        pred_len: int,
        *,
        # 主干宽度/深度
        e_layers: int = 3,
        n_heads: int = 16,
        d_model: int = 128,
        d_ff: int = 256,
        # dropout
        dropout: float = 0.0,
        fc_dropout: float = 0.0,
        head_dropout: float = 0.0,
        # patching
        patch_len: int = 16,
        stride: int = 8,
        padding_patch: Optional[str] = None,  # 可为 "end" 或 None
        # RevIN
        revin: bool = True,
        affine: bool = True,
        subtract_last: bool = False,
        # 其它
        individual: bool = False,
    ):
        self.enc_in = enc_in
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.e_layers = e_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_ff = d_ff

        self.dropout = dropout
        self.fc_dropout = fc_dropout
        self.head_dropout = head_dropout

        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch

        self.revin = revin
        self.affine = affine
        self.subtract_last = subtract_last

        self.individual = individual


class PatchTSTBB(BaseModel):
    """
    BuildingsBench 适配版 PatchTST：
    - 只依赖 batch['load']，形状 [B, T, 1]
    - 由于 patch/头固定，要求运行时 context_len == 配置 seq_len、pred_len == 配置 pred_len
    - 输出 [B, pred_len, 1]
    """

    def __init__(
        self,
        *,
        # 长度（必须固定）
        context_len: int = 168,
        pred_len: int = 24,
        # PatchTST 结构
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 1024,
        e_layers: int = 3,
        dropout: float = 0.1,
        fc_dropout: float = 0.0,
        head_dropout: float = 0.0,
        # patch 超参
        patch_len: int = 16,
        stride: int = 8,
        padding_patch: Optional[str] = None,  # "end" 或 None
        # RevIN
        revin: bool = True,
        affine: bool = True,
        subtract_last: bool = False,
        # 头部
        individual: bool = False,  # 单变量时 False/True 都可
        # BuildingsBench 习惯参数
        continuous_loads: bool = True,
        continuous_head: Literal["huber"] = "huber",
        **kwargs,
    ):
        super().__init__(
            context_len=context_len,
            pred_len=pred_len,
            continuous_loads=continuous_loads,
        )
        self._cfg_ctx = int(context_len)
        self._cfg_pred = int(pred_len)
        self.continuous_head = continuous_head

        # 配置容器（单变量 enc_in=1）
        cfg = _PTConfigs(
            enc_in=1,
            seq_len=self._cfg_ctx,
            pred_len=self._cfg_pred,
            e_layers=e_layers,
            n_heads=n_heads,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            fc_dropout=fc_dropout,
            head_dropout=head_dropout,
            patch_len=patch_len,
            stride=stride,
            padding_patch=padding_patch,
            revin=revin,
            affine=affine,
            subtract_last=subtract_last,
            individual=individual,
        )

        # 构造 PatchTST 模型
        self.core = Model(cfg)

    # ------------- forward -------------
    def forward(
        self,
        x: Dict[str, torch.Tensor],
        context_len: Optional[int] = None,
        pred_len: Optional[int] = None,
    ):
        # 要求运行时长度与构造一致（PatchTST 的 head/patch 固定）
        if context_len is None:
            context_len = self._cfg_ctx
        if pred_len is None:
            pred_len = self._cfg_pred
        assert (
            int(context_len) == self._cfg_ctx
        ), f"PatchTSTBB: context_len 必须等于构造时的 {self._cfg_ctx}"
        assert (
            int(pred_len) == self._cfg_pred
        ), f"PatchTSTBB: pred_len 必须等于构造时的 {self._cfg_pred}"

        load = x["load"]  # [B, T, 1]
        assert load.dim() == 3 and load.size(-1) == 1, "load 需要形状 [B, T, 1]"
        assert load.size(1) >= self._cfg_ctx, "序列长度不足 context_len"

        # 只取上下文部分（PatchTST 用不到未来区间）
        x_in = load[:, : self._cfg_ctx, :].to(dtype=torch.float32)

        # 调用 PatchTST 的 Model：输入 [B, L, C]，输出 [B, pred_len, C]
        out = self.core(x_in)  # [B, pred_len, 1]
        # 稳妥：裁剪到 pred_len
        out = out[:, -self._cfg_pred :, :]
        return out

    # ------------- loss / predict 等 -------------
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
        preds = self.forward(x, context_len, pred_len)
        return preds, preds  # 第二个占位，兼容你评测管线

    def unfreeze_and_get_parameters_for_finetuning(self):
        # 需要参数组策略时可仿照我之前给 BuildMoE 的实现；默认全量
        return self.parameters()

    def load_from_checkpoint(self, checkpoint_path: str):
        state = torch.load(checkpoint_path, map_location="cpu")
        sd = state.get("model", state)
        new_sd = {k.replace("module.", ""): v for k, v in sd.items()}
        self.load_state_dict(new_sd, strict=False)


# if __name__ == "__main__":
#     model = LoadForecastingPatchTST_A(
#         max_context_len=336,
#         max_pred_len=168,
#         e_layers=18,
#         n_heads=12,
#         d_model=768,
#         d_ff=2048,
#         dropout=0.0,
#         fc_dropout=0.0,
#         head_dropout=0.0,
#         patch_len=12,
#         stride=8,
#         padding_patch=None,
#         revin=True,
#         affine=True,
#         subtract_last=False,
#         continuous_loads=True,
#         continuous_head="huber",
#         huber_delta=1.0,
#     )
#     x = torch.randn(16, 400, 1)
#     out = model({"load": x}, context_len=336, pred_len=168)
#     print(out.shape)
#     n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"Number of trainable parameters: {n_params}")
#     loss = model.loss(out, torch.randn(16, 168, 1))
#     print(loss)


if __name__ == "__main__":
    import torch

    # 说明：
    # - PatchTST 的上下文长度/预测长度在构造时就固定；这里用 BuildingsBench 常用的 168/24
    # - patch 设置 (16, 8) → 对 ctx=168 会得到 patch_num=20
    CTX = 168
    PRED = 24

    cfgs = [
        # —— 100M 附近三档（注意 d_model 必须能被 n_heads 整除）——
        dict(
            name="XL-≈90M",
            d_model=1024,
            n_heads=16,
            d_ff=4096,
            n_layers=7,  # 约 12*7*d^2
            patch_len=16,
            stride=8,
            dropout=0.1,
        ),
        dict(
            name="XL≈100M",
            d_model=1024,
            n_heads=16,
            d_ff=4096,
            n_layers=8,  # 约 12*8*d^2
            patch_len=16,
            stride=8,
            dropout=0.1,
        ),
        dict(
            name="XL+≈111M",
            d_model=1152,
            n_heads=16,
            d_ff=4608,
            n_layers=7,  # 约 12*7*d^2
            patch_len=16,
            stride=8,
            dropout=0.1,
        ),
    ]

    def count_params(m: torch.nn.Module):
        total = sum(p.numel() for p in m.parameters())
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        return total, trainable

    print("== PatchTSTBB 参数量检查（~100M 档） ==")
    for c in cfgs:
        model = PatchTSTBB(
            context_len=CTX,
            pred_len=PRED,
            d_model=c["d_model"],
            n_heads=c["n_heads"],
            d_ff=c["d_ff"],
            e_layers=c["n_layers"],  # PatchTST 用 e_layers 表示层数
            dropout=c["dropout"],
            fc_dropout=0.0,
            head_dropout=0.0,
            patch_len=c["patch_len"],
            stride=c["stride"],
            padding_patch=None,  # 或 "end"
            revin=True,
            affine=True,
            subtract_last=False,
            individual=False,
            continuous_loads=True,
            continuous_head="huber",
        )
        total, trainable = count_params(model)
        size_mb = total * 4 / (1024**2)  # fp32 粗估显存
        print(
            f"[{c['name']}] d_model={c['d_model']}, heads={c['n_heads']}, d_ff={c['d_ff']}, "
            f"L={c['n_layers']}, patch=({c['patch_len']},{c['stride']}), "
            f"Params(total/trainable)={total:,}/{trainable:,} (~{size_mb:.2f} MB, fp32)"
        )

    # 你也可以单独设一个“目标容量”做试探，比如精确打到 ~100M：
    target = dict(
        d_model=1088, n_heads=17, d_ff=4 * 1088, n_layers=7, patch_len=16, stride=8
    )
    # 注意：确保 d_model % n_heads == 0；1088%17 != 0，这里改成 1088/17 不整除，建议用 17→17*64=1088? 不行。
    # 换成 n_heads=17 不可行；用 n_heads=17→1088/17 非整数。改为 n_heads=17 -> d_model=1088 不合法。
    # 选择 n_heads=17 改为 17 的倍数比较罕见，建议改 n_heads=17 为 17 的因数设置，比如 17×64=1088 不是整数；因此改为 n_heads=17 -> n_heads=17 不行。
    # 用 n_heads=17 这行只是演示，请用能整除的组合，如 n_heads=17 改为 17 的因数配置。这里给一个合法示范：
    target = dict(d_model=1088, n_heads=17)  # <-- 演示中断，改为合法示例：
    target = dict(d_model=1088, n_heads=17)  # 占位，请按下方合法示例替换

    # 合法示例（能整除）：d_model=1088, n_heads=17 不合法；用 n_heads=17 的倍数不现实。
    # 用 d_model=1088, n_heads=17 -> 改为 d_model=1088, n_heads=17 不行；改为 n_heads=17 -> 误例。
    # 合法“精调”示例（能整除）：d_model=1088, n_heads=17 -> 改为 d_model=1088, n_heads=17 不行；
    # 直接给个可用的近 100M 组合：
    target = dict(
        d_model=960, n_heads=15, d_ff=3840, n_layers=9, patch_len=16, stride=8
    )  # 约 12*9*960^2 ≈ 99.5M

    model_t = PatchTSTBB(
        context_len=CTX,
        pred_len=PRED,
        d_model=target["d_model"],
        n_heads=target["n_heads"],
        d_ff=target["d_ff"],
        e_layers=target["n_layers"],
        dropout=0.1,
        fc_dropout=0.0,
        head_dropout=0.0,
        patch_len=target["patch_len"],
        stride=target["stride"],
        padding_patch=None,
        revin=True,
        affine=True,
        subtract_last=False,
        individual=False,
        continuous_loads=True,
        continuous_head="huber",
    )
    total_t, trainable_t = count_params(model_t)
    size_mb_t = total_t * 4 / (1024**2)
    print("\n[Target-like 精调]")
    print(
        f"d_model={target['d_model']}, heads={target['n_heads']}, d_ff={target['d_ff']}, "
        f"L={target['n_layers']}, patch=({target['patch_len']},{target['stride']}), "
        f"Params(total/trainable)={total_t:,}/{trainable_t:,} (~{size_mb_t:.2f} MB, fp32)"
    )
