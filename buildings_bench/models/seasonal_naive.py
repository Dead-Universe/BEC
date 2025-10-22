# buildings_bench/models/seasonal_naive.py
from typing import Dict, Optional, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F

from buildings_bench.models.base_model import BaseModel


class SeasonalNaive(BaseModel):
    """
    季节持久性（seasonal naive）基线：
    - 给定季节长度 lag（如 24、168），用“最近一个季节窗口”的数值复刻未来。
    - 输入:  batch["load"] 形状 [B, T, 1]，T >= context_len + pred_len
    - 输出:  [B, pred_len, 1]
    - 兼容 Box-Cox：不做任何逆变换，评测脚本会处理。
    - 训练友好：提供一个“哑参数”以兼容优化器与梯度裁剪（不会更新）。
    """

    def __init__(
        self,
        *,
        lag: int = 24,  # 季节长度：24（日季节），168（周季节）
        context_len: int = 168,
        pred_len: int = 24,
        continuous_loads: bool = True,
        continuous_head: Literal["huber"] = "huber",
        **kwargs,
    ):
        super().__init__(
            context_len=context_len,
            pred_len=pred_len,
            continuous_loads=continuous_loads,
        )
        self.lag = int(lag)
        self.continuous_head = continuous_head

        # —— 为了兼容你的训练脚本里“必须要有参数”的 AdamW / 梯度裁剪 —— #
        # 这是一个不会被使用、也不会产生梯度的哑参数；仅用于占位。
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=True)

    def _seasonal_copy(
        self, load: torch.Tensor, context_len: int, pred_len: int
    ) -> torch.Tensor:
        """
        从最近一个季节窗口复制未来：取 [ctx - lag : ctx) 的序列，循环/截断到 pred_len。
        - 这种写法不要求 context_len >= lag + pred_len，只需 context_len >= lag。
        """
        B, T, C = load.shape
        assert C == 1, "SeasonalNaive 仅支持单变量 [B, T, 1]"
        assert context_len <= T, "序列长度不足 context_len"
        assert (
            context_len >= self.lag
        ), f"季节持久性需要 context_len >= lag ({self.lag})"

        ref = load[:, context_len - self.lag : context_len, :]  # [B, lag, 1]
        if self.lag >= pred_len:
            return ref[:, :pred_len, :]  # 直接截断
        # 若 pred_len > lag：循环平铺后再截断
        reps = (pred_len + self.lag - 1) // self.lag  # 向上取整
        ref_tiled = ref.repeat(1, reps, 1)  # [B, reps*lag, 1]
        return ref_tiled[:, :pred_len, :]

    # ------------ BaseModel 接口 ------------
    def forward(
        self,
        x: Dict[str, torch.Tensor],
        context_len: Optional[int] = None,
        pred_len: Optional[int] = None,
    ):
        if context_len is None:
            context_len = self.context_len
        if pred_len is None:
            pred_len = self.pred_len
        load = x["load"]
        assert load.dim() == 3 and load.size(-1) == 1, "load 必须是 [B,T,1]"
        load = load.to(dtype=torch.float32)
        preds = self._seasonal_copy(
            load, int(context_len), int(pred_len)
        )  # [B, pred, 1]
        # 2) 轻触哑参数，挂上计算图；不改变数值
        preds = preds + 0.0 * self._dummy
        return preds

    def loss(
        self, pred: torch.Tensor, y: torch.Tensor, progress: Optional[float] = None
    ) -> torch.Tensor:
        if self.continuous_head == "huber":
            if pred.ndim == 3 and pred.size(-1) == 1:
                pred = pred.squeeze(-1)
            if y.ndim == 3 and y.size(-1) == 1:
                y = y.squeeze(-1)
            base = F.huber_loss(pred, y, delta=1.0)
            # 3) 再轻触一次，保证即便上游被优化裁剪也有梯度流到 dummy
            return base + 0.0 * self._dummy
        raise NotImplementedError(f"Unsupported head: {self.continuous_head}")

    @torch.no_grad()
    def predict(
        self, x: Dict[str, torch.Tensor], context_len: int = 168, pred_len: int = 24
    ):
        preds = self.forward(x, context_len, pred_len)
        return preds, preds  # 第二个是占位，兼容你评测管线的 (preds, dist_params)

    def unfreeze_and_get_parameters_for_finetuning(self):
        # 返回一个“有 param_group”的列表，避免 AdamW 因空参数报错
        return [self._dummy]

    def load_from_checkpoint(self, checkpoint_path: str):
        # 对齐其它模型接口；季节持久性没有实际权重，可忽略
        return


# 便捷别名：日季节 / 周季节
class SeasonalNaive24(SeasonalNaive):
    def __init__(self, **kwargs):
        super().__init__(lag=24, **kwargs)


class SeasonalNaive168(SeasonalNaive):
    def __init__(self, **kwargs):
        super().__init__(lag=168, **kwargs)
