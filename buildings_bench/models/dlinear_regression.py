from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from buildings_bench.models.base_model import BaseModel


class moving_avg(nn.Module):
    """Moving average block to highlight the trend of time series"""

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
    """Series decomposition block"""

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DLinearRegression(BaseModel):
    """
    Decomposition-Linear
    """

    def __init__(self, context_len=168, pred_len=24, continuous_loads=True):
        super(DLinearRegression, self).__init__(context_len, pred_len, continuous_loads)

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        # self.individual = True
        # self.channels = 1

        self.Linear_Seasonal = nn.Linear(context_len, self.pred_len)
        self.Linear_Trend = nn.Linear(context_len, self.pred_len)

        # Use this two lines if you want to visualize the weights
        # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(
        self,
        x,
        context_len: Optional[int] = None,
        pred_len: Optional[int] = None,
    ):
        # x: [Batch, Input length, Channel]
        src_series = x["load"][:, : self.context_len, :]
        seasonal_init, trend_init = self.decompsition(src_series)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(
            0, 2, 1
        )

        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)  # to [Batch, Output length, Channel]

    def loss(self, x, y):
        err = F.huber_loss(x, y, delta=1.0, reduction="none")
        loss = err.mean()
        return loss

    def predict(
        self,
        x,
        context_len: Optional[int] = None,
        pred_len: Optional[int] = None,
    ):
        return self.forward(x, context_len, pred_len), None

    def unfreeze_and_get_parameters_for_finetuning(self):
        for p in self.parameters():
            p.requires_grad = True
        return self.parameters()

    def load_from_checkpoint(self, checkpoint_path):
        return None


class MovingAvg(nn.Module):
    """Moving average block to highlight the trend of time series"""

    def __init__(self, kernel_size: int, stride: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)  # (B, L + k - 1, C)
        x = self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)  # (B, L, C)
        return x


class SeriesDecomp(nn.Module):
    """Series decomposition block"""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x: torch.Tensor):
        # x: (B, L, C)
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DLinearUnified(BaseModel):
    """
    DLinear with fixed training size (context_len=336, pred_len=168), and slicing for shorter horizons.

    - Always uses 336-step context internally:
        * if provided context_len > 336: take the last 336 steps
        * if provided context_len < 336: left-pad by repeating the first value
    - Always predicts 168 steps then slice the first `pred_len` (<= 168).
    - Loss: Huber (delta configurable).
    """

    def __init__(
        self,
        context_len: int = 336,
        pred_len: int = 168,
        kernel_size: int = 25,
        huber_delta: float = 1.0,
        continuous_loads: bool = True,
        continuous_head: str = "huber",
        **kwargs,
    ):
        assert (
            context_len == 336 and pred_len == 168
        ), "方案A设定下，建议用固定的 context_len=336, pred_len=168 进行训练"
        super().__init__(context_len, pred_len, continuous_loads=continuous_loads)

        self.huber_delta = huber_delta
        self.continuous_head = continuous_head

        # decomposition
        self.decomposition = SeriesDecomp(kernel_size)

        # Linear maps time dimension: (B, C=1, 336) -> (B, C=1, 168)
        self.Linear_Seasonal = nn.Linear(self.context_len, self.pred_len)
        self.Linear_Trend = nn.Linear(self.context_len, self.pred_len)

    # --------------------- helpers ----------------------
    def _prepare_context(self, ctx: torch.Tensor) -> torch.Tensor:
        """
        Ensure context length == self.context_len (=336) by cropping/padding.
        Args:
            ctx: (B, Lc, 1)
        Returns:
            ctx_fixed: (B, 336, 1)
        """
        B, Lc, C = ctx.shape
        if Lc == self.context_len:
            return ctx
        elif Lc > self.context_len:
            # take the most recent 336 steps
            return ctx[:, -self.context_len :, :]
        else:
            # left-pad by repeating the first value to reach 336
            pad_len = self.context_len - Lc
            left = ctx[:, :1, :].repeat(1, pad_len, 1)
            return torch.cat([left, ctx], dim=1)

    # --------------------- forward ----------------------
    def forward(
        self,
        x: Dict[str, torch.Tensor],
        context_len: Optional[int] = None,
        pred_len: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            x["load"]: (B, context_len + pred_len, 1)  —— 与你的MoE接口一致
        Returns:
            y_hat_sliced: (B, pred_len, 1)  —— pred_len<=168 时截取前 pred_len
        """
        if context_len is None:
            # 允许外部不传，默认用固定训练长度
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

        # 取出请求的上下文与（可选）目标区间，但内部仅用上下文
        src = load[:, :context_len, :]  # (B, ctx, 1)
        src = self._prepare_context(src)  # (B, 336, 1)

        # DLinear 分解
        seasonal_init, trend_init = self.decomposition(src)  # (B, 336, 1)

        # 线性层期望 (B, C, L)
        seasonal_init = seasonal_init.permute(0, 2, 1)  # (B, 1, 336)
        trend_init = trend_init.permute(0, 2, 1)  # (B, 1, 336)

        seasonal_out = self.Linear_Seasonal(seasonal_init)  # (B, 1, 168)
        trend_out = self.Linear_Trend(trend_init)  # (B, 1, 168)

        y_full = (seasonal_out + trend_out).permute(0, 2, 1)  # (B, 168, 1)
        y_hat = y_full[:, :pred_len, :]  # 截取前 pred_len
        return y_hat

    # --------------------- loss: Huber -------------------
    def loss(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Huber loss on continuous loads.
        pred, y: (B, pred_len, 1)
        """
        if not self.continuous_loads:
            raise ValueError("DLinearUnified is implemented for continuous loads.")
        return F.huber_loss(pred, y, delta=self.huber_delta, reduction="mean")

    # --------------------- predict -----------------------
    @torch.no_grad()
    def predict(
        self,
        x: Dict[str, torch.Tensor],
        context_len: int = 336,
        pred_len: int = 168,
    ):
        preds = self.forward(x, context_len=context_len, pred_len=pred_len)
        return preds, preds

    # ----------------- finetune / ckpt -------------------
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


class DLinearExpert(nn.Module):
    """
    DLinear Expert for MoE
    输入输出都保持 (B, L, d)，和 MLP Expert 一致。
    """

    def __init__(self, d_model: int, kernel_size: int = 25):
        super().__init__()
        self.decomposition = SeriesDecomp(kernel_size)

        # 特征维度线性映射 (不涉及时间维缩放)
        self.Linear_Seasonal = nn.Linear(d_model, d_model)
        self.Linear_Trend = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d)
        seasonal_init, trend_init = self.decomposition(x)  # 分解趋势 & 季节性
        seasonal_out = self.Linear_Seasonal(seasonal_init)  # (B, L, d)
        trend_out = self.Linear_Trend(trend_init)  # (B, L, d)
        return seasonal_out + trend_out


if __name__ == "__main__":
    model = DLinearUnified()
    x = torch.randn(16, 300 + 100, 1)
    y = model({"load": x}, context_len=300, pred_len=100)
    print(y.shape)
    loss = model.loss(y, torch.randn(16, 100, 1))
    print(loss)
    model = DLinearExpert(d_model=8)
    x = torch.randn(16, 100, 8)
    y = model(x)
    print(y.shape)
