import torch
import torch.distributed
import torch.utils.hooks
import torch.nn as nn
from typing import List, Tuple, Optional


class MoEActivationHook:
    """
    收集并可视化 MoE 专家激活情况的 Hook。
    - 按 Gate（深度顺序）统计每个专家被选中的次数（top-k 每次选择都计数）
    - 可在 DDP 下进行 all-reduce，同步统计到“全局”
    - 提供 plot() 画层×专家热力图
    """

    def __init__(self, model: nn.Module, sync_ddp: bool = True):
        self.model = model
        self.sync_ddp = (
            sync_ddp
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.gates: List[nn.Module] = []
        self.gate_names: List[str] = []
        self.n_experts: Optional[int] = None
        self.stats: Optional[torch.Tensor] = None  # [n_gates, E], float

        # 找到所有 Gate，按出现顺序注册
        for name, m in model.named_modules():
            if m.__class__.__name__ == "Gate":  # 避免硬依赖
                self.gates.append(m)
                self.gate_names.append(name)

        if len(self.gates) == 0:
            raise RuntimeError("未在模型中找到 Gate 模块。")

        # 从第一个 Gate 推断专家数
        g0 = self.gates[0]
        if hasattr(g0, "args") and hasattr(g0.args, "n_routed_experts"):
            self.n_experts = int(g0.args.n_routed_experts)
        elif hasattr(g0, "router") and hasattr(g0.router, "out_features"):
            self.n_experts = int(g0.router.out_features)
        else:
            raise RuntimeError("无法确定专家数量（n_experts）。")

        self.stats = torch.zeros(
            len(self.gates), self.n_experts, dtype=torch.float64, device="cpu"
        )

        # 注册 hook
        for li, g in enumerate(self.gates):
            h = g.register_forward_hook(self._make_hook(li), with_kwargs=False)
            self.handles.append(h)

    def _make_hook(self, layer_idx: int):
        @torch.no_grad()
        def _hook(module, inputs, output):
            if not hasattr(module, "last_logits"):
                return

            logits = module.last_logits.detach()
            E = logits.shape[-1]
            logits = logits.reshape(-1, E)

            score_func = getattr(module, "score_func", "softmax")
            probs = (
                torch.softmax(logits, dim=-1)
                if score_func == "softmax"
                else torch.sigmoid(logits)
            )

            k = getattr(
                module,
                "topk",
                getattr(getattr(module, "args", object()), "n_activated_experts", 1),
            )
            top_idx = torch.topk(probs, k, dim=-1).indices  # [N,k]

            # bincount 在与输入相同的设备上产生计数
            counts = torch.bincount(top_idx.reshape(-1), minlength=E).to(torch.float32)

            # ---- 设备与 DDP 同步处理 ----
            if self.sync_ddp:
                backend = torch.distributed.get_backend()
                if backend == "nccl":
                    # NCCL 需要 GPU 张量
                    counts = counts.to(logits.device)
                    torch.distributed.all_reduce(
                        counts, op=torch.distributed.ReduceOp.SUM
                    )
                    counts = counts.to(self.stats.device)  # 搬回 stats 的设备（CPU）
                else:
                    # 其他后端（如 gloo）可以在 CPU 上规约
                    counts = counts.to(self.stats.device)
                    torch.distributed.all_reduce(
                        counts, op=torch.distributed.ReduceOp.SUM
                    )
            else:
                # 非 DDP：直接搬到 stats 的设备
                counts = counts.to(self.stats.device)

            # 累加（dtype 对齐）
            self.stats[layer_idx] += counts.to(self.stats.dtype)

        return _hook

    def reset(self):
        """清零统计。"""
        if self.stats is not None:
            self.stats.zero_()

    def remove(self):
        """移除所有已注册的 hooks。"""
        for h in self.handles:
            h.remove()
        self.handles.clear()

    @torch.no_grad()
    def usage(self, normalize: bool = True) -> torch.Tensor:
        """
        返回 [n_gates, E] 的统计矩阵。
        normalize=True 时按行归一化为占比（每层和为 1）。
        """
        mat = self.stats.clone()
        if normalize:
            row_sum = mat.sum(-1, keepdim=True).clamp_min(1.0)
            mat = mat / row_sum
        return mat

    def plot(self, normalize: bool = True, figsize: Tuple[int, int] = (10, 4)):
        """
        画层×专家热力图（y=Gate 次序，x=专家 id）。
        注意：这里只依赖 matplotlib，不额外指定样式/颜色，保持默认。
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap

        mat = self.usage(normalize=normalize).numpy()

        # 自定义 colormap
        cmap = LinearSegmentedColormap.from_list("black_blue", ["#FFFFFF", "#093F7F"])

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(mat, aspect="auto", cmap=cmap)

        ax.set_xlabel("Expert ID")
        ax.set_ylabel("Gate (depth)")
        ax.set_xticks(range(self.n_experts))
        ax.set_yticks(range(len(self.gates)))
        short_names = [n if len(n) <= 40 else n[-40:] for n in self.gate_names]
        ax.set_yticklabels(short_names)

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Activation share" if normalize else "Activation count")

        ax.set_title("MoE Expert Activation")
        plt.tight_layout()
        return fig
