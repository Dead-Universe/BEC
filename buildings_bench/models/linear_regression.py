from typing import Optional
from buildings_bench.models.base_model import BaseModel
import torch.nn as nn
import torch.nn.functional as F


class LinearRegression(BaseModel):
    """Linear regression model that does direct forecasting.

    It has one weight W and one bias b. The output is computed as
    y = Wx + b, where W is a matrix of shape [pred_len, context_len].
    """

    def __init__(self, context_len=168, pred_len=24, continuous_loads=True):
        super(LinearRegression, self).__init__(context_len, pred_len, continuous_loads)
        self.Linear = nn.Linear(context_len, pred_len)

    def forward(
        self,
        x,
        context_len: Optional[int] = None,
        pred_len: Optional[int] = None,
    ):
        x = x["load"][:, : self.context_len, :]
        # src_series: [Batch, Input length, 1]
        x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x  # [Batch, Output length, 1]

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
        out = self.forward(x, context_len, pred_len)
        return out, None

    def unfreeze_and_get_parameters_for_finetuning(self):
        for p in self.parameters():
            p.requires_grad = True
        return self.parameters()

    def load_from_checkpoint(self, checkpoint_path):
        return None
