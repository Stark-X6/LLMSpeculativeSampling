# benchmark_bass/metrics.py
import time
import torch
import numpy as np

# ✅ 导入你任务一中实现的批处理推测解码函数
# 路径根据你的项目结构（sampling/speculative_bass.py）调整
from sampling.speculative_bass import speculative_sampling_bass_pad


@torch.no_grad()
def measure_performance(
    draft_model: torch.nn.Module,
    target_model: torch.nn.Module,
    batch_input_ids: torch.Tensor,
    gamma: int | str,
    max_new_tokens: int,
    temperature: float = 0.7,
    top_k: int = 100,
    top_p: float = 0.9,
) -> dict:
    """
    运行一次推测解码并测性能指标。
    基于 speculative_sampling_bass_pad (BASS-PAD 批处理实现)。

    返回:
        dict {
            throughput: tokens/sec,
            latency: ms/token,
            draft_ratio / verify_ratio / qkv_ratio / attn_ratio / ffn_ratio: NaN (占位),
            ttft / tpot: NaN (占位)
        }
    """

    # === 1. CUDA 同步并开始计时 ===
    torch.cuda.synchronize()
    t0 = time.time()

    # === 2. 调用你的 BASS 批处理推测解码函数 ===
    out_ids, out_lengths = speculative_sampling_bass_pad(
        prefixes=batch_input_ids,
        approx_model=draft_model,
        target_model=target_model,
        max_new_tokens=max_new_tokens,
        gamma_init=gamma if isinstance(gamma, int) else 4,  # 启发式时默认初始值 4
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        verbose=False,
    )

    # === 3. 结束计时 ===
    torch.cuda.synchronize()
    t1 = time.time()
    elapsed = t1 - t0  # 单位: 秒

    # === 4. 计算基础指标 ===
    B, L0 = batch_input_ids.shape
    total_new = int((out_lengths - L0).clamp_min(0).sum().item())

    throughput = total_new / elapsed if elapsed > 0 else float("inf")         # tokens / s
    latency = (elapsed / total_new * 1000.0) if total_new > 0 else float("inf")  # ms/token

    # === 5. 结果汇总 ===
    metrics = {
        "throughput": float(throughput),
        "latency": float(latency),
        # 以下指标暂不可测，留空 (后续若加 profile 再填)
        "ttft": float("nan"),
        "tpot": float("nan"),
        "draft_ratio": float("nan"),
        "verify_ratio": float("nan"),
        "qkv_ratio": float("nan"),
        "attn_ratio": float("nan"),
        "ffn_ratio": float("nan"),
    }

    return metrics
