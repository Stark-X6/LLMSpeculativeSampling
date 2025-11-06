# benchmark_heuristic_gamma.py
# 对比固定 γ vs 启发式 γ 的动态分批吞吐/延迟表现
import os
import time
import csv
import argparse
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

# 复用你已有的动态分批与流模拟
from sampling.dynamic_batcher import (
    BucketBatchScheduler,
    load_dolly_dataset,
    simulate_stream,
)

# 直接调用你的 BASS-PAD 推理实现（我们只在这里切换 use_heuristic_gamma）
from sampling.speculative_bass import speculative_sampling_bass_pad


def make_process_fn(tokenizer, small_model, large_model, device, *,
                    mode_name: str, log_path: str,
                    max_new_tokens: int = 20,
                    use_heuristic_gamma: bool = False):
    """
    返回一个 process_fn(batch, gamma_init=4)：
    - 兼容你的 simulate_stream / InferenceWorker 传入的 (batch, gamma_init) 调用方式
    - 在内部调用 speculative_bass_pad(..., use_heuristic_gamma=flag)
    - 将每批的吞吐、延迟、γ 等写入 CSV
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def process_fn(batch, gamma_init: int = 4):
        DEVICE = device
        B = len(batch)
        lengths = [len(r.input_ids) for r in batch]
        L_max = max(lengths)

        padded = torch.full((B, L_max), tokenizer.pad_token_id, dtype=torch.long)
        for i, r in enumerate(batch):
            l = len(r.input_ids)
            padded[i, :l] = torch.tensor(r.input_ids)
        padded = padded.to(DEVICE)

        # ==== 推理阶段 ====
        t0 = time.perf_counter()
        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
            out, new_lengths = speculative_sampling_bass_pad(
                prefixes=padded,
                approx_model=small_model,
                target_model=large_model,
                max_new_tokens=max_new_tokens,
                gamma_init=int(gamma_init),           # 起始 γ（若 simulate_stream 传了就用）
                use_heuristic_gamma=use_heuristic_gamma,  # 这里切换启发式开/关
            )
        t1 = time.perf_counter()

        elapsed = t1 - t0
        total_new = int(new_lengths.sum().item()) - sum(lengths)
        throughput = total_new / elapsed if elapsed > 0 else 0.0
        avg_latency_ms = (elapsed / max(total_new, 1)) * 1000.0

        print(f"[BATCH DONE] mode={mode_name}, batch={B}, avg_len={sum(lengths)/B:.1f}, "
              f"time={elapsed:.3f}s, throughput={throughput:.2f} tok/s, "
              f"avg_latency={avg_latency_ms:.1f} ms/token, gamma0={int(gamma_init)}, "
              f"heuristic={'ON' if use_heuristic_gamma else 'OFF'}")

        # 记录一条示例文本（可选）
        try:
            txt = tokenizer.decode(out[0, :int(new_lengths[0].item())], skip_special_tokens=True)
            print(f"  ▶ sample[0]: {txt[:120]}...")
        except Exception:
            pass

        # ==== 写入 CSV ====
        header = [
            "timestamp", "mode", "batch_size", "avg_input_len",
            "time_sec", "throughput_tok_s", "avg_latency_ms", "gamma0", "heuristic"
        ]
        row = [
            time.strftime("%Y-%m-%d %H:%M:%S"),
            mode_name,
            B,
            round(sum(lengths)/B, 2),
            round(elapsed, 4),
            round(throughput, 3),
            round(avg_latency_ms, 3),
            int(gamma_init),
            int(use_heuristic_gamma),
        ]
        new_file = not os.path.exists(log_path)
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(header)
            w.writerow(row)

        torch.cuda.empty_cache()

    return process_fn


def summarize_csv(path: str, title: str):
    if not os.path.exists(path):
        print(f"[WARN] CSV 未找到：{path}")
        return
    df = pd.read_csv(path)
    # 取中间 50% 批，规避暖机/收尾失真
    n = len(df)
    if n < 10:
        print(f"[INFO] {title} 批次数太少（{n}），直接全量统计")
        mid = df
    else:
        mid = df.iloc[n//5: 7*n//10]

    print(f"\n===== {title} (mid-50%) =====")
    print(f"batches: {len(mid)} / total: {n}")
    print(f"Avg throughput: {mid['throughput_tok_s'].mean():.2f} tok/s")
    print(f"Avg latency:    {mid['avg_latency_ms'].mean():.1f} ms/token")
    if 'gamma0' in mid.columns:
        print(f"Avg gamma0:     {mid['gamma0'].mean():.2f}")
    if 'heuristic' in mid.columns:
        print(f"Heuristic flag: {mid['heuristic'].iloc[0]}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark fixed-γ vs heuristic-γ under dynamic batching")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_samples", type=int, default=1200, help="输入流长度，建议≥1000")
    parser.add_argument("--wait_window_ms", type=int, default=200)
    parser.add_argument("--max_wait_ms", type=int, default=1000)
    parser.add_argument("--outfile_prefix", type=str, default="./results/batch_metrics")
    parser.add_argument("--gamma0", type=int, default=4, help="固定/启发式的初始γ")
    args = parser.parse_args()

    # 设备解析
    if not torch.cuda.is_available():
        DEVICE = "cpu"
        device_index = None
    else:
        try:
            device_index = int(args.device.split(":")[-1])
        except Exception:
            device_index = 0
        n_gpus = torch.cuda.device_count()
        if device_index >= n_gpus:
            device_index = 0
        DEVICE = f"cuda:{device_index}"

    print(f"[INFO] 使用设备: {DEVICE}")

    # 路径（与 dynamic_batcher 的逻辑一致）
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../data"))
    approx_path = os.path.join(DATA_ROOT, "models", "TinyLlama-1B")
    target_path = os.path.join(DATA_ROOT, "models", "Llama-2-7b-raw")

    # Tokenizer / 模型
    tokenizer = AutoTokenizer.from_pretrained(approx_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    small_model = AutoModelForCausalLM.from_pretrained(
        approx_path, torch_dtype=torch.float16, device_map={"": device_index},
        trust_remote_code=True, local_files_only=True
    )
    large_model = AutoModelForCausalLM.from_pretrained(
        target_path, torch_dtype=torch.float16, device_map={"": device_index},
        trust_remote_code=True, local_files_only=True
    )
    print(f"[INFO] small -> {next(small_model.parameters()).device}, large -> {next(large_model.parameters()).device}")

    # 数据
    dolly_path = os.path.join(DATA_ROOT, "dataset", "dolly", "databricks-dolly-15k.jsonl")
    dataset = load_dolly_dataset(dolly_path, tokenizer, args.max_samples)
    print(f"[INFO] 加载 {len(dataset)} 条 Dolly 样本")

    # 两个模式的 CSV
    csv_fixed = f"{args.outfile_prefix}_fixed.csv"
    csv_heur  = f"{args.outfile_prefix}_heuristic.csv"
    for p in [csv_fixed, csv_heur]:
        if os.path.exists(p):
            os.remove(p)

    # 公共调度器配置（控制稳态）
    scheduler_fixed = BucketBatchScheduler(
        small_model=small_model, large_model=large_model, tokenizer=tokenizer,
        wait_window_ms=args.wait_window_ms, max_wait_ms=args.max_wait_ms,
        bucket_config={
            (0, 32): 16,
            (33, 64): 4,
            (65, 128): 4,
            (129, 256): 4,
            (257, 512): 4,
            (513, 1024): 2
        }
    )
    scheduler_heur = BucketBatchScheduler(
        small_model=small_model, large_model=large_model, tokenizer=tokenizer,
        wait_window_ms=args.wait_window_ms, max_wait_ms=args.max_wait_ms,
        bucket_config={
            (0, 32): 16,
            (33, 64): 4,
            (65, 128): 4,
            (129, 256): 4,
            (257, 512): 4,
            (513, 1024): 2
        }
    )

    # 1) 固定 γ（启发式关闭）
    print("\n=== RUN: Fixed γ (heuristic OFF) ===")
    process_fixed = make_process_fn(
        tokenizer, small_model, large_model, DEVICE,
        mode_name="fixed-gamma", log_path=csv_fixed,
        max_new_tokens=20, use_heuristic_gamma=False
    )
    simulate_stream(
        dataset, scheduler_fixed, process_fixed,
        mode="dynamic", device=DEVICE, bs=8, device_index=device_index, log_path=csv_fixed
    )

    # 2) 启发式 γ（启发式开启）
    print("\n=== RUN: Heuristic γ (heuristic ON) ===")
    process_heur = make_process_fn(
        tokenizer, small_model, large_model, DEVICE,
        mode_name="heuristic-gamma", log_path=csv_heur,
        max_new_tokens=20, use_heuristic_gamma=True
    )
    simulate_stream(
        dataset, scheduler_heur, process_heur,
        mode="dynamic", device=DEVICE, bs=8, device_index=device_index, log_path=csv_heur
    )

    # 3) 统计稳态对比
    summarize_csv(csv_fixed,  "Fixed γ")
    summarize_csv(csv_heur,   "Heuristic γ")

    print("\n[DONE] CSV 输出：")
    print(f" - {csv_fixed}")
    print(f" - {csv_heur}")


if __name__ == "__main__":
    # 避免 Pandas 在无显示环境下的警告
    pd.options.mode.chained_assignment = None
    main()
