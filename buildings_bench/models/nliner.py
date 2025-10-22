import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Literal, Optional
from buildings_bench.models.base_model import BaseModel


class NLinearUnified(BaseModel):
    """
    NLinear with the same interface as LoadForecastingTransformerMoE
    and the Scheme-A behavior:
      - Train with fixed context_len=336, pred_len=168
      - At inference, always produce 168 steps and slice the first pred_len (<=168)
      - If provided context_len != 336:
          * >336: take the last 336 steps
          * <336: left-pad by repeating the first value to reach 336
      - Loss: Huber (delta configurable)
    """

    def __init__(
        self,
        context_len: int = 336,
        pred_len: int = 168,
        huber_delta: float = 1.0,
        continuous_loads: bool = True,
        use_bias: bool = True,
        continuous_head: Literal["mse", "gaussian_nll", "huber"] = "huber",
        **kwargs,
    ):
        assert (
            context_len == 336 and pred_len == 168
        ), "方案A：建议固定 context_len=336, pred_len=168 进行训练"
        super().__init__(context_len, pred_len, continuous_loads=continuous_loads)

        self.huber_delta = huber_delta
        self.continuous_head = continuous_head

        # NLinear: 直接对时间维做线性映射 (B, C=1, 336) -> (B, C=1, 168)
        self.proj = nn.Linear(self.context_len, self.pred_len, bias=use_bias)

        # 训练时常见的 NLinear trick：去中心化再复原（可开关）
        self.use_demean = True

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

    # ---------- forward ----------
    def forward(
        self,
        x: Dict[str, torch.Tensor],
        context_len: Optional[int] = None,
        pred_len: Optional[int] = None,
    ) -> torch.Tensor:
        """
        x["load"]: (B, context_len + pred_len, 1)
        return: (B, pred_len, 1) with pred_len<=168 (slice of 168 head)
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

        src = load[:, :context_len, :]  # (B, ctx, 1)
        src = self._fix_context_len(src)  # (B, 336, 1)

        # NLinear：在时间维线性投影，常见做法是去均值（逐样本逐通道）
        if self.use_demean:
            mean = src.mean(dim=1, keepdim=True)  # (B, 1, 1)
            src_demean = src - mean
        else:
            mean = None
            src_demean = src

        # (B, 336, 1) -> (B, 1, 336)
        src_t = src_demean.permute(0, 2, 1)
        out_t = self.proj(src_t)  # (B, 1, 168)
        y_full = out_t.permute(0, 2, 1)  # (B, 168, 1)

        # 复原均值
        if self.use_demean:
            y_full = y_full + mean

        y_hat = y_full[:, :pred_len, :]  # (B, pred_len, 1)
        return y_hat

    # ---------- loss ----------
    def loss(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if not self.continuous_loads:
            raise ValueError("NLinearUnified is implemented for continuous loads.")
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


class NLinearRegression(BaseModel):
    """
    NLinear-Regression (简洁版)
    - 直接对时间维做线性映射，不做分解
    - 输入: (B, context_len, 1)
    - 输出: (B, pred_len, 1)
    """

    def __init__(
        self,
        context_len: int = 168,
        pred_len: int = 24,
        continuous_loads: bool = True,
        use_bias: bool = True,
        huber_delta: float = 1.0,
    ):
        super().__init__(context_len, pred_len, continuous_loads=continuous_loads)

        self.huber_delta = huber_delta
        self.Linear = nn.Linear(self.context_len, self.pred_len, bias=use_bias)

        # trick: 去中心化再复原
        self.use_demean = True

    def forward(
        self,
        x: Dict[str, torch.Tensor],
        context_len: Optional[int] = None,
        pred_len: Optional[int] = None,
    ) -> torch.Tensor:
        if context_len is None:
            context_len = self.context_len
        if pred_len is None:
            pred_len = self.pred_len

        load = x["load"][:, :context_len, :]  # (B, context_len, 1)
        assert load.dim() == 3 and load.size(-1) == 1

        if self.use_demean:
            mean = load.mean(dim=1, keepdim=True)  # (B,1,1)
            load_demean = load - mean
        else:
            mean = None
            load_demean = load

        # (B, L, 1) -> (B, 1, L)
        src = load_demean.permute(0, 2, 1)
        out = self.Linear(src)  # (B,1,pred_len)
        out = out.permute(0, 2, 1)  # (B,pred_len,1)

        if self.use_demean:
            out = out + mean

        return out

    def loss(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.huber_loss(pred, y, delta=self.huber_delta, reduction="mean")

    @torch.no_grad()
    def predict(
        self,
        x: Dict[str, torch.Tensor],
        context_len: Optional[int] = None,
        pred_len: Optional[int] = None,
    ):
        preds = self.forward(x, context_len=context_len, pred_len=pred_len)
        return preds, None  # 没有 distribution_params

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


if __name__ == "__main__":
    model = NLinearUnified()
    x = torch.randn(2, 400, 1)
    out = model({"load": x}, context_len=400, pred_len=100)
    print(out.shape)  # (2, 100, 1)
    loss = model.loss(out, torch.randn(2, 100, 1))
    print(loss)  # scalar
