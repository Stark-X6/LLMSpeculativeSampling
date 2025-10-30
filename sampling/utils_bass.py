# sampling/utils_bass.py
"""
BASS 工具函数（自包含版本）
- 统一暴露：normalize_logits / multinomial_sample / positive_diff_normalize / top_k_top_p_filter
- 兼容别名：norm_logits / sample / max_fn
- 去除对 sampling/utils.py 的依赖
"""
import math
import torch
import torch.nn.functional as F

def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0) -> torch.Tensor:
    """对 logits 做 Top-K/Top-P 截断（原地修改）。支持 2D (B,V) 张量。"""
    if top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.size(-1)))[0]
        logits[logits < kth[:, [-1]]] = float("-inf")
    if top_p > 0.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum_probs > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = 0
        to_rm = remove.scatter(1, sorted_idx, remove)
        logits[to_rm] = float("-inf")
    return logits

def safe_softmax(logits: torch.Tensor, dim: int = -1, dtype: torch.dtype | None = None) -> torch.Tensor:
    """数值稳定 softmax：减去 max 再 softmax，可选 dtype。"""
    if dtype is not None and logits.dtype != dtype:
        logits = logits.to(dtype)
    z = logits - logits.max(dim=dim, keepdim=True).values
    return F.softmax(z, dim=dim)

def normalize_logits(logits: torch.Tensor, temperature: float = 1.0, top_k: int = 0, top_p: float = 0.0) -> torch.Tensor:
    """规范化 logits → probs（支持 1D/2D/3D，内部按 (B,S,V) 处理）"""
    squeeze_back = False
    if logits.dim() == 2:
        logits = logits.unsqueeze(1)  # (B,1,V)
        squeeze_back = True
    elif logits.dim() == 1:
        logits = logits.unsqueeze(0).unsqueeze(0)  # (1,1,V)
        squeeze_back = True

    t = max(float(temperature), 1e-6)
    logits32 = logits.to(torch.float32) / t

    # 逐步做 TopK/TopP
    for i in range(logits32.size(1)):
        top_k_top_p_filter(logits32[:, i, :], top_k=top_k, top_p=top_p)

    probs = safe_softmax(logits32, dim=-1, dtype=torch.float32).to(logits.dtype)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0).clamp_(min=0.0)
    probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-12)

    return probs.squeeze(1) if squeeze_back else probs

def multinomial_sample(probs: torch.Tensor, num_samples: int = 1, *, disallow_ids: tuple[int, ...] = (0,)) -> torch.Tensor:
    """安全采样：屏蔽非法 id、修复 NaN/负数、全 0 行回退、自动归一化。兼容 (B,V) 或 (V,)"""
    if probs.dim() == 1:
        probs = probs.unsqueeze(0)

    p0 = probs.clone()
    probs = probs.clone()
    for tid in disallow_ids:
        if 0 <= tid < probs.size(-1):
            probs[:, tid] = 0.0

    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0).clamp_(min=0.0)

    row_sums = probs.sum(dim=-1, keepdim=True)
    need_fallback = (row_sums <= 0)
    if need_fallback.any():
        probs = torch.where(need_fallback, p0, probs)
        row_sums = probs.sum(dim=-1, keepdim=True)

    probs = probs / (row_sums + 1e-12)

    zero_rows = (probs.sum(dim=-1, keepdim=True) <= 0)
    if zero_rows.any():
        V = probs.size(-1)
        probs = torch.where(zero_rows, torch.full_like(probs, 1.0 / V), probs)

    out = torch.multinomial(probs, num_samples=num_samples)
    return out if out.size(0) > 1 else out

def positive_diff_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """归一化 max(x,0)；若和为 0，回退为均匀分布。"""
    x_pos = torch.where(x > 0, x, torch.zeros_like(x))
    s = x_pos.sum(dim=-1, keepdim=True)
    ok = (s > eps)
    out = torch.where(ok, x_pos / (s + eps), torch.zeros_like(x_pos))
    if (~ok).any():
        V = x.size(-1)
        out = torch.where(ok, out, torch.full_like(x_pos, 1.0 / V))
    return out

def get_device(module_or_tensor) -> torch.device:
    """Module → next(parameters()).device；Tensor → tensor.device。"""
    if hasattr(module_or_tensor, "parameters"):
        return next(module_or_tensor.parameters()).device
    if isinstance(module_or_tensor, torch.Tensor):
        return module_or_tensor.device
    raise TypeError("get_device expects a nn.Module or a torch.Tensor")

# 兼容别名
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