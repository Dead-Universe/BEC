from typing import Dict, Optional, Literal
import os
import torch
import torch.nn.functional as F
from torch import nn

from momentfm.common import TASKS
from momentfm.models.moment import MOMENT


class MomentAsLoadForecastAdapter(nn.Module):
    """
    与 ChronosAsLoadForecastAdapter 保持相同接口，用 MOMENT 从头预训练做负载预测。
      - forward(x, context_len, pred_len) -> (B, pred_len, 1)
      - predict(x, context_len, pred_len)
      - loss(...): 仅支持 mse / huber
      - unfreeze_and_get_parameters_for_finetuning()
      - load_from_checkpoint(path): 兼容 DDP/DP 保存的前缀
    该版本：不再接收 t5_config 字典，而是用显式超参在内部构造。
    """

    def __init__(
        self,
        # —— 与 Chronos 适配器一致的“对齐参数” —— #
        max_context_len: int = 336,
        max_pred_len: int = 168,
        context_len: int = 168,
        pred_len: int = 24,
        # —— Transformer / MOMENT 关键配置（显式展开） —— #
        d_model: int = 768,
        dim_feedforward: int = 2048,  # T5Config.d_ff
        num_heads: int = 12,
        num_encoder_layers: int = 8,  # T5Config.num_layers（encoder_only时只用这个）
        dropout_rate: float = 0.0,
        feed_forward_proj: str = "relu",  # T5Config.feed_forward_proj
        # — 时序切块相关 — #
        seq_len: int = 512,
        patch_len: int = 8,
        patch_stride_len: int = 8,
        # 任务/模型形态 — #
        transformer_type: str = "encoder_only",  # 'encoder_only' | 'decoder_only' | 'encoder_decoder'
        transformer_backbone: str = "google/flan-t5-large",
        # 训练头与数据形态 — #
        continuous_loads: bool = True,  # 与其他模型接口保持一致；本适配器不使用
        continuous_head: Literal["mse", "huber"] = "huber",
        forecast_horizon: Optional[int] = None,  # 不传则用 max_pred_len
        # MOMENT 其他可选项（保持默认即可从头训练）
        add_positional_embedding: bool = True,
        value_embedding_bias: bool = False,
        revin_affine: bool = False,
        patch_dropout: float = 0.0,
        mask_ratio: float = 0.0,
        enable_gradient_checkpointing: bool = False,
        randomly_initialize_backbone: bool = True,  # 从头训练：通常置 True
        # 设备 / dtype —— #
        torch_dtype: Optional[torch.dtype] = torch.float32,
        device: Optional[torch.device | str] = "cuda",
        **kwargs,
    ):
        super().__init__()

        self.max_context_len = max_context_len
        self.max_pred_len = max_pred_len
        self.continuous_loads = continuous_loads
        self.continuous_head = continuous_head

        # === 在内部构造给 MOMENT 的 t5_config ===
        # 注意：MOMENT._get_transformer_backbone 会执行：
        #   model_config = T5Config.from_dict(config.t5_config)
        #   如果 randomly_initialize_backbone: 用 T5Model(model_config)，否则 T5EncoderModel(model_config)
        #   最终取 get_encoder()，因此 encoder_only 场景只需要设置 num_layers（encoder层数）
        t5_config = {
            "d_model": d_model,
            "d_ff": dim_feedforward,
            "num_heads": num_heads,
            "num_layers": num_encoder_layers,
            "dropout_rate": dropout_rate,
            "feed_forward_proj": feed_forward_proj,
            # 下列字段不是必须；若你在自定义 T5Config 里有用，可在此继续补充：
            # "layer_norm_epsilon": 1e-6,
            # "initializer_factor": 1.0,
        }

        model_kwargs = {
            "task_name": TASKS.FORECASTING,
            "d_model": d_model,
            "seq_len": seq_len,
            "patch_len": patch_len,
            "patch_stride_len": patch_stride_len,
            "transformer_type": transformer_type,
            "transformer_backbone": transformer_backbone,
            "t5_config": t5_config,
            "revin_affine": revin_affine,
            "add_positional_embedding": add_positional_embedding,
            "value_embedding_bias": value_embedding_bias,
            "patch_dropout": patch_dropout,
            "mask_ratio": mask_ratio,
            "enable_gradient_checkpointing": enable_gradient_checkpointing,
            "randomly_initialize_backbone": randomly_initialize_backbone,
            "forecast_horizon": forecast_horizon or max_pred_len,
            "freeze_embedder": False,
            "freeze_encoder": False,
            # classify 的输入需要这些字段，但在 forecasting 下不会用到
            "n_channels": 1,
            "num_class": 2,
        }

        # 重要：传一个空 dict 作为 config，并把 model_kwargs 放在 kwargs，
        # 这样 MOMENT._update_inputs 会走 dict+model_kwargs 的分支
        self.moment = MOMENT(config={}, model_kwargs=model_kwargs)

        if torch_dtype is not None:
            self.moment = self.moment.to(torch_dtype)
        if device is not None:
            self.moment = self.moment.to(device)

    # --------------------- forward（训练图） ----------------------
    def forward(
        self,
        x: Dict[str, torch.Tensor],
        context_len: Optional[int] = None,
        pred_len: Optional[int] = None,
    ) -> torch.Tensor:
        """
        输入:  x['load'] 形状 (B, T, 1)
        输出:  (B, pred_len, 1)
        """
        if context_len is None:
            context_len = self.max_context_len
        if pred_len is None:
            pred_len = self.max_pred_len

        load = x["load"]  # (B, T, 1)
        assert load.dim() == 3 and load.size(-1) == 1, "x['load'] 应为 (B, T, 1)"
        B, T, _ = load.shape

        # 截出历史窗口（尾部去掉 pred_len）
        hist = load[
            :, max(0, T - (context_len + pred_len)) : T - pred_len, 0
        ]  # (B, ctx_eff)

        # === 关键：强制输入长度 == MOMENT 的 seq_len ===
        seq_len = self.moment.config.seq_len
        cur_len = hist.size(-1)

        if cur_len < seq_len:
            # 左侧 padding 到 seq_len；pad 区域 mask=0，真实区域 mask=1
            pad_len = seq_len - cur_len
            hist = F.pad(hist, (pad_len, 0))  # [B, seq_len]
            input_mask = torch.zeros(B, seq_len, device=hist.device, dtype=hist.dtype)
            input_mask[:, pad_len:] = 1.0
        else:
            hist = hist[:, -seq_len:]
            input_mask = torch.ones(B, seq_len, device=hist.device, dtype=hist.dtype)

        # MOMENT 期望 (B, C=1, seq_len)
        x_enc = hist.unsqueeze(1).contiguous()

        out = self.moment.forecast(x_enc=x_enc, input_mask=input_mask)
        # out.forecast: (B, 1, forecast_horizon)
        y_hat = out.forecast.transpose(1, 2)[:, :pred_len, :]  # -> (B, pred_len, 1)
        return y_hat

    # --------------------- loss -------------------------
    def loss(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        仅支持 mse / huber，与原适配器一致。
        """
        if self.continuous_head == "huber":
            err = F.huber_loss(pred, y, delta=1.0, reduction="none")
            return err.mean()
        else:  # 'mse'
            return F.mse_loss(pred, y, reduction="mean")

    # --------------------- predict ----------------------
    @torch.no_grad()
    def predict(
        self,
        x: Dict[str, torch.Tensor],
        context_len: int = 168,
        pred_len: int = 24,
    ):
        preds = self.forward(x, context_len=context_len, pred_len=pred_len)
        return preds, preds

    # ----------------- finetune parameters ---------------
    def unfreeze_and_get_parameters_for_finetuning(self):
        # 从头预训练，不做冻结，直接返回全部参数
        return self.moment.parameters()

    # ----------------- load checkpoint -------------------
    def load_from_checkpoint(self, checkpoint_path: str | bytes | os.PathLike):
        """
        兼容以下几种保存方式：
          - {'model': state_dict}
          - 直接是 state_dict
        并剥离常见前缀：'module.'、'moment.'、'module.moment.'
        """
        sd = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]
        if not isinstance(sd, dict):
            raise ValueError(
                "无效的 checkpoint：应为 state_dict 或 {'model': state_dict}"
            )

        def strip_prefixes(d):
            new_d = {}
            for k, v in d.items():
                if k.startswith("module.moment."):
                    new_d[k[len("module.moment.") :]] = v
                elif k.startswith("moment."):
                    new_d[k[len("moment.") :]] = v
                elif k.startswith("module."):
                    new_d[k[len("module.") :]] = v
                else:
                    new_d[k] = v
            return new_d

        sd = strip_prefixes(sd)
        missing, unexpected = self.moment.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(
                "[MOMENT Adapter] load_state_dict differences:",
                f"missing={missing}, unexpected={unexpected}",
            )


# ============== 简易自检（含反向传播） ==============
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 你的示例超参（可直接替换/复现）
    context_len = 168
    pred_len = 24
    max_context_len = 336
    max_pred_len = 168
    num_encoder_layers = 16
    num_heads = 12
    dim_feedforward = 2048
    d_model = 768
    continuous_loads = True
    continuous_head = "huber"

    adapter = MomentAsLoadForecastAdapter(
        max_context_len=max_context_len,
        max_pred_len=max_pred_len,
        d_model=d_model,
        dim_feedforward=dim_feedforward,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        feed_forward_proj="relu",
        seq_len=512,  # 注意：决定 num_patches = seq_len/patch_len（当 stride==patch_len）
        patch_len=8,
        patch_stride_len=8,
        transformer_type="encoder_only",
        transformer_backbone="google/flan-t5-base",
        continuous_loads=continuous_loads,
        continuous_head=continuous_head,
        randomly_initialize_backbone=True,  # 从头初始化
        device=device,
    ).train()

    count_parameters = sum(p.numel() for p in adapter.parameters() if p.requires_grad)

    print(count_parameters)

    # 构造一批数据（长度小于 seq_len，会触发左侧 padding）
    B = 2
    load = torch.randn(B, context_len + pred_len, 1, device=device)
    batch = {"load": load}

    # —— 训练前向 + 反向 —— #
    y_hat = adapter.forward(
        batch, context_len=context_len, pred_len=pred_len
    )  # (B, pred, 1)
    target = batch["load"][:, -pred_len:, :]
    loss = adapter.loss(y_hat, target)
    loss.backward()
    print("[MOMENT Adapter] train step OK – loss:", float(loss))

    # —— 推理 —— #
    adapter.eval()
    with torch.no_grad():
        preds, _ = adapter.predict(batch, context_len=context_len, pred_len=pred_len)
    print("[MOMENT Adapter] inference preds shape:", preds.shape)  # (B, pred, 1)
