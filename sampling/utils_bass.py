# sampling/utils_bass.py
"""
BASS 版工具函数（兼容层）
- 复用 sampling/utils.py 的实现，统一对外命名，避免重复代码与命名发散
- 新名：normalize_logits / multinomial_sample / positive_diff_normalize
- 保留别名：norm_logits / sample / max_fn（为老代码平滑过渡）
"""

import torch
import torch.nn.functional as F
import math

# 复用原 demo 的基础实现
try:
    from .utils import top_k_top_p_filter as _orig_top_k_top_p_filter
    from .utils import norm_logits       as _orig_norm_logits
    from .utils import sample            as _orig_sample
    from .utils import max_fn            as _orig_max_fn
except Exception as e:
    raise ImportError(
        f"[utils_bass] 导入 sampling/utils.py 失败：{e}。"
        "请确认 sampling/utils.py 存在且可以被包导入。"
    )

def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0) -> torch.Tensor:
    """与原实现一致：对 logits 做 Top-K/Top-P 截断（原地修改）。"""
    return _orig_top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)

def normalize_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
) -> torch.Tensor:
    """
    规范化 logits → probs：温度缩放 + TopK/TopP + 稳定 softmax（支持 2D/3D）。
    """
    # 统一成 (B, S, V) 处理
    squeeze_back = False
    if logits.dim() == 2:
        logits = logits.unsqueeze(1)  # (B,1,V)
        squeeze_back = True
    elif logits.dim() == 1:
        logits = logits.unsqueeze(0).unsqueeze(0)  # (1,1,V)
        squeeze_back = True

    B, S, V = logits.shape
    # 温度保护，避免除以 0 或极小值
    t = max(float(temperature), 1e-6)
    # dtype 为 float32 做 softmax 更稳，再转回原 dtype
    logits32 = logits.to(torch.float32) / t

    # 逐步做 TopK/TopP 截断（对 logits）
    for i in range(S):
        _orig_top_k_top_p_filter(logits32[:, i, :], top_k=top_k, top_p=top_p)

    # 稳定 softmax（减去 max）
    probs = safe_softmax(logits32, dim=-1, dtype=torch.float32)
    probs = probs.to(logits.dtype)

    # 最后再做一次消毒 + 归一化，彻底去除 NaN/Inf/负数
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    probs = torch.clamp(probs, min=0.0)
    denom = probs.sum(dim=-1, keepdim=True) + 1e-12
    probs = probs / denom

    if squeeze_back:
        return probs.squeeze(1)
    return probs


def multinomial_sample(
    probs: torch.Tensor,
    num_samples: int = 1,
    *,
    disallow_ids: tuple[int, ...] = (0,),   # 默认屏蔽 id=0，但在极端情况下会自动回退
) -> torch.Tensor:
    """
    从概率分布采样（multinomial），兼容批量：
      - probs: (B, V) 或 (V,)
      - 返回: (B, num_samples) 或 (1, num_samples)
    策略：
      1) 保存原始分布 p0
      2) 将 disallow_ids 概率置 0
      3) 若某行清零后全为 0，则回退用 p0 那一行（不再屏蔽），保证可采样
      4) 归一化后 multinomial
    """
    if probs.dim() == 1:
        probs = probs.unsqueeze(0)  # (1, V)

    p0 = probs.clone()  # 保存原始分布，必要时回退
    probs = probs.clone()

    # 1) 屏蔽不允许的 token
    for tid in disallow_ids:
        if 0 <= tid < probs.size(-1):
            probs[:, tid] = 0

    # 2) 检查是否出现全 0 行
    row_sums = probs.sum(dim=-1, keepdim=True)  # (B,1)
    need_fallback = (row_sums <= 0)

    if need_fallback.any():
        # 对这些行恢复为原始分布（不再屏蔽）
        probs = torch.where(need_fallback, p0, probs)
        row_sums = probs.sum(dim=-1, keepdim=True)

    # 3) 归一化（数值保护）
    probs = probs / (row_sums + 1e-12)

    # 4) 采样
    out = torch.multinomial(probs, num_samples=num_samples)  # (B, num_samples)
    return out if out.size(0) > 1 else out  # (1, num_samples) 时保持形状一致

def positive_diff_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    归一化 max(x,0)。若和为 0，回退为均匀分布，避免数值问题。
    用于推测解码的“拒绝后差分重采样”。
    """
    x_pos = torch.where(x > 0, x, torch.zeros_like(x))
    s = x_pos.sum(dim=-1, keepdim=True)
    ok = (s > eps)
    out = torch.where(ok, x_pos / (s + eps), torch.zeros_like(x_pos))
    if (~ok).any():
        V = x.size(-1)
        uniform = torch.full_like(x_pos, 1.0 / V)
        out = torch.where(ok, out, uniform)
    return out

def safe_softmax(logits: torch.Tensor, dim: int = -1, dtype: torch.dtype | None = None) -> torch.Tensor:
    """数值稳定的 softmax：减去 max 再 softmax，可选 dtype。"""
    if dtype is not None and logits.dtype != dtype:
        logits = logits.to(dtype)
    z = logits - logits.max(dim=dim, keepdim=True).values
    return F.softmax(z, dim=dim)

def get_device(module_or_tensor) -> torch.device:
    """统一设备获取：Module → next(parameters()).device；Tensor → tensor.device。"""
    if hasattr(module_or_tensor, "parameters"):
        return next(module_or_tensor.parameters()).device
    if isinstance(module_or_tensor, torch.Tensor):
        return module_or_tensor.device
    raise TypeError("get_device expects a nn.Module or a torch.Tensor")

# 兼容别名（旧 → 新），供老代码平滑过渡
norm_logits = normalize_logits
sample = multinomial_sample
max_fn = positive_diff_normalize

__all__ = [
    "top_k_top_p_filter",
    "normalize_logits",
    "multinomial_sample",
    "positive_diff_normalize",
    "safe_softmax",
    "get_device",
    # 别名
    "norm_logits",
    "sample",
    "max_fn",
]

class DraftLengthHeuristic:
    """
    论文算法 1：自适应草稿长度 ldraft（内部自带默认超参，不从外部传参）
      缺省：l0=7, lincre=2, lmod=10, llimit=32（论文经验值）
    用法：
      heur = DraftLengthHeuristic()          # 初始化
      gamma = heur.ldraft                    # 当前 γ
      gamma = heur.step(x_vec)               # 用本轮每样本被接受的草稿 token 数 x_vec 更新 γ
    """
    # 内置默认
    _L0     = 7
    _LINCRE = 2
    _LMOD   = 10
    _LLIMIT = 32

    def __init__(self):
        self.ldraft = int(self._L0)
        self.lincre = int(self._LINCRE)
        self.lmod   = int(self._LMOD)
        self.llimit = int(self._LLIMIT)
        self.s      = 0  # 上一轮是否走过“递减分支”的额外 -1

    def step(self, x_vec: torch.Tensor) -> int:
        """
        x_vec: (B,) —— 本轮每样本被接受的草稿 token 数
        返回：更新后的 ldraft（整数）
        """
        if x_vec.numel() == 0:
            return self.ldraft
        x_max = int(x_vec.max().item())

        if x_max == self.ldraft:
            # 满收：增长
            self.ldraft = min(self.ldraft + self.lincre, self.llimit)
            self.s = 0
        else:
            # 非满收：递减
            dec = math.ceil(self.ldraft / self.lmod) + self.s
            self.ldraft = self.ldraft - dec
            # 不得小于 max(1, x_max, ldraft)
            self.ldraft = max(1, x_max, self.ldraft)
            self.s = 1

        return self.ldraft