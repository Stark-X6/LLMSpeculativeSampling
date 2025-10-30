from typing import Dict, Any, List
import numpy as np

from .metrics import measure_performance
from .utils_io import log_message
from .runner_support import prepare_batch


def average_metrics(list_of_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys_mean = [
        "throughput", "latency", "ttft", "tpot",
        "draft_ratio", "verify_ratio", "qkv_ratio", "attn_ratio", "ffn_ratio",
        "mean_len", "var_len", "max_len", "min_len"
    ]
    out = {}
    for k in keys_mean:
        vals = [m.get(k, np.nan) for m in list_of_metrics]
        vals = [v for v in vals if v is not None and not np.isnan(v)]
        out[k] = float(np.mean(vals)) if vals else np.nan
    return out


def run_single_experiment(
    model_pair: str,
    draft_model,
    target_model,
    tokenizer,
    batch_size: int,
    gamma: int | str,
    sorted_flag: int,
    repeat: int,
    ctx_len: int,
    max_new_tokens: int
) -> Dict[str, Any]:

    rows = []
    for i in range(repeat):
        # 让每次取不同 offset 的 batch（均匀分布）
        offset = i
        batch_input_ids, len_stats = prepare_batch(
            tokenizer=tokenizer,
            sorted_flag=sorted_flag,
            batch_size=batch_size,
            ctx_len=ctx_len,
            offset=offset
        )
        row = measure_performance(
            draft_model=draft_model,
            target_model=target_model,
            batch_input_ids=batch_input_ids,
            gamma=gamma,
            max_new_tokens=max_new_tokens
        )
        rows.append(row)

    avg = average_metrics(rows)
    avg.update({
        "model_pair": model_pair,
        "batch_size": batch_size,
        "gamma": gamma,
        "sorted": sorted_flag
    })
    avg.update(len_stats)

    log_message(f"[RUN] {model_pair} bs={batch_size} γ={gamma} sorted={sorted_flag} "
                f"→ thr={avg['throughput']:.2f} tok/s, lat={avg['latency']:.4f} ms/tok")
    return avg
