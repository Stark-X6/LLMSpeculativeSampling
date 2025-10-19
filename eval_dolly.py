#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# ===== 复用你项目里的实现 =====
from sampling.speculative_bass import speculative_sampling_bass_pad
from sampling.speculative_sampling import speculative_sampling, speculative_sampling_v2
from sampling.autoregressive_sampling import autoregressive_sampling  # 仅在需要时使用

def _run_singlewise(func, input_ids, *fargs, **fkwargs):
    """
    把仅支持 batch=1 的解码函数 func，包装成支持 batch=B 的调用。
    返回：堆叠后的 out（pad 到同长）以及 lengths（每条 L0+max_tokens）
    """
    device = input_ids.device
    B, T = input_ids.size(0), input_ids.size(1)
    outs = []
    for b in range(B):
        out_b = func(input_ids[b:b+1, :], *fargs, **fkwargs)  # func 必须返回 token 序列 (1, T_new)
        outs.append(out_b)
    # 右侧 pad 到同长再 cat
    maxlen = max(o.size(1) for o in outs)
    outs_pad = [torch.nn.functional.pad(o, (0, maxlen - o.size(1))) for o in outs]
    out = torch.cat(outs_pad, dim=0).to(device)
    return out


def format_prompt(inst: str, ctx: str) -> str:
    """把 Dolly-15k 的 (instruction, context) 组装成统一的 prompt。"""
    parts = [f"### Instruction:\n{(inst or '').strip()}"]
    if ctx and len(ctx.strip()) > 0:
        parts.append(f"### Context:\n{ctx.strip()}")
    parts.append("### Response:")
    return "\n\n".join(parts)


def report_metrics(mode: str, t0: float, t1: float, lengths: torch.Tensor, L0: int, B: int):
    """打印吞吐/延迟，并返回一个 dict 便于后续汇总。"""
    elapsed = t1 - t0
    new_tokens = (lengths - L0).clamp_min(0)
    total_new = int(new_tokens.sum().item())
    avg_new = float(new_tokens.float().mean().item())
    tput = total_new / elapsed if elapsed > 0 else float("inf")
    per_tok_ms = (elapsed / total_new * 1000.0) if total_new > 0 else float("inf")
    print(f"[METRICS] mode={mode}, batch={B}, total_new={total_new}, "
          f"time={elapsed:.3f}s, throughput={tput:.2f} tok/s, avg_per_seq={avg_new:.2f} tok, "
          f"avg_latency_per_token={per_tok_ms:.2f} ms")
    return {"mode": mode, "time": elapsed, "throughput": tput, "avg_new": avg_new, "total_new": total_new}


def compare_bass(avg_metrics: dict):
    """打印 BASS 相对其它模式的提升倍数（吞吐越大越好，延迟越小越好）。"""
    bass = avg_metrics.get("BASS-PAD")
    if not bass:
        return
    for ref in ("DeepMind", "Google"):
        other = avg_metrics.get(ref)
        if not other:
            continue
        tput_gain = (bass["throughput"] / other["throughput"]) if other["throughput"] > 0 else float("inf")
        lat_gain = (other["time"] / bass["time"]) if bass["time"] > 0 else 0.0
        print(f"[COMPARE] BASS vs {ref}: throughput x{tput_gain:.2f}, latency x{lat_gain:.2f}")


def chunk_iter(it_list, n):
    """把列表按 n 切批。"""
    for i in range(0, len(it_list), n):
        yield it_list[i:i + n]

def _clear_wrappers(*models):
    """清理自定义 wrapper 的缓存，避免显存堆积/碎片化。"""
    for m in models:
        if hasattr(m, "past_key_values"):
            m.past_key_values = None
        if hasattr(m, "prob_history"):
            m.prob_history = None

def _cuda_gc():
    """CUDA 同步 + 清空缓存（对已无引用的块有效）"""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--approx_model_name", required=True, help="小模型（起草模型）名称或本地路径")
    ap.add_argument("--target_model_name", required=True, help="大模型（目标模型）名称或本地路径")
    ap.add_argument("--split", default="train")  # Dolly-15k 只有 train
    ap.add_argument("--samples", type=int, default=128, help="评测样本条数（从数据集头部截取）")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max_tokens", type=int, default=32, help="每条生成的新 token 数")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--gamma_init", type=int, default=7, help="BASS 初始草稿长度（内部会启发式自调）")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir", default="dolly_eval_out")
    # 默认不包含 AR，应老师要求仅比较推测解码三种
    ap.add_argument("--modes", default="DeepMind,Google,BASS",
                    help="逗号分隔，可选: DeepMind,Google,BASS（也支持 AR，如需加入）")
    ap.add_argument("--dataset_path", default="", help="本地 JSONL 路径（提供则优先使用本地）")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = args.device

    # 1) 加载 tokenizer（保持小/大模型共用同一 tokenizer）
    tokenizer = AutoTokenizer.from_pretrained(args.approx_model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2) 加载模型（让 CUDA_VISIBLE_DEVICES 控卡；device_map={'':0} 即可）
    small = AutoModelForCausalLM.from_pretrained(
        args.approx_model_name, torch_dtype=torch.float16, device_map={"": 0}, trust_remote_code=True
    )
    large = AutoModelForCausalLM.from_pretrained(
        args.target_model_name, torch_dtype=torch.float16, device_map={"": 0}, trust_remote_code=True
    )

    # 3) 载入 Dolly-15k
    if args.dataset_path:
        # 本地 JSONL 加载
        data_files = {"train": args.dataset_path}  # 我们统一用 split=train
        ds = load_dataset("json", data_files=data_files, split="train")
    else:
        ds = load_dataset("databricks/databricks-dolly-15k", split=args.split)

    ds = ds.select(range(min(args.samples, len(ds))))

    # 4) 组 prompt（Instruction / Context → Prompt）
    prompts = [format_prompt(ex.get("instruction", ""), ex.get("context", "")) for ex in ds]

    # === Length-based batching：按 token 长度降序排序，减少 PAD 浪费 ===
    MAXLEN_FOR_SORT = 384  # 必须与后面 encode 的 max_length 保持一致
    # 只为算长度做一次轻量编码（CPU 上，padding=False 不会引入 PAD）
    enc_len = tokenizer(prompts, return_tensors="pt",
                        padding=True, truncation=True, max_length=MAXLEN_FOR_SORT)
    # 实际每条长度（padding=False 时就是各行长度；这么写兼容 pad_token_id 为空的情况）
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -1
    lengths = enc_len["input_ids"].ne(pad_id).sum(dim=1)  # shape: [N]

    # 从长到短排序索引
    order = torch.argsort(lengths, descending=True).tolist()
    # 重排 prompts（如有其它并行数组，如 labels/meta，也一并重排）
    prompts_sorted = [prompts[i] for i in order]

    # 如果将来需要把输出还原成原始顺序，可记录反向映射（本版暂不需要）
    # inv_order = [0]*len(order)
    # for new_i, old_i in enumerate(order):
    #     inv_order[old_i] = new_i

    # 5) 输出目录 & 文件
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "metrics.csv")
    jsonl_path = os.path.join(args.out_dir, "generations.jsonl")
    with open(csv_path, "w", encoding="utf-8") as fcsv:
        fcsv.write("mode,batch,total_new,time,throughput,avg_new\n")
    jfh = open(jsonl_path, "w", encoding="utf-8")

    want_modes = set([m.strip().lower() for m in args.modes.split(",")])
    # 收集各批的指标，后面做平均
    all_metrics = {"DeepMind": [], "Google": [], "BASS-PAD": []}
    use_ar = "ar" in want_modes
    if use_ar:
        all_metrics["AR"] = []

    for batch_prompts in chunk_iter(prompts_sorted, args.batch):
        B = len(batch_prompts)
        enc = tokenizer(batch_prompts, return_tensors="pt",
                        padding=True, truncation=True, max_length=384)
        input_ids = enc["input_ids"].to(device)
        L0 = input_ids.size(1)

        # 0) 如果需要 AR（一般不需要）
        if use_ar:
            torch.manual_seed(args.seed)
            t0 = perf_counter()
            out_ar = autoregressive_sampling(input_ids, large, args.max_tokens,
                                             top_k=args.top_k, top_p=args.top_p,
                                             temperature=args.temperature)
            t1 = perf_counter()
            lengths_ar = torch.full((B,), L0 + args.max_tokens, dtype=torch.long, device=input_ids.device)
            m = report_metrics("AR", t0, t1, lengths_ar, L0, B)
            all_metrics["AR"].append(m)
            for b in range(B):
                text = tokenizer.decode(out_ar[b], skip_special_tokens=True)
                jfh.write(json.dumps({"mode": "AR", "prompt": batch_prompts[b], "output": text},
                                     ensure_ascii=False) + "\n")

        # 1) DeepMind 推测解码
        if "deepmind" in want_modes:
            torch.manual_seed(args.seed)
            t0 = perf_counter()
            out_dm = _run_singlewise(
                speculative_sampling_v2, input_ids, small, large, args.max_tokens,
                top_k=args.top_k, top_p=args.top_p, random_seed=args.seed, temperature=args.temperature
            )
            t1 = perf_counter()
            lengths_dm = torch.full((B,), L0 + args.max_tokens, dtype=torch.long, device=input_ids.device)
            m = report_metrics("DeepMind", t0, t1, lengths_dm, L0, B)
            all_metrics["DeepMind"].append(m)
            for b in range(B):
                text = tokenizer.decode(out_dm[b], skip_special_tokens=True)
                jfh.write(json.dumps({"mode": "DeepMind", "prompt": batch_prompts[b], "output": text},
                                     ensure_ascii=False) + "\n")
            del out_dm
            _clear_wrappers(small, large)
            _cuda_gc()

        # 2) Google 推测解码
        if "google" in want_modes:
            torch.manual_seed(args.seed)
            t0 = perf_counter()
            out_gg = _run_singlewise(
                speculative_sampling, input_ids, small, large, args.max_tokens,
                gamma=args.gamma_init, top_k=args.top_k, top_p=args.top_p,
                random_seed=args.seed, verbose=False, temperature=args.temperature
            )
            t1 = perf_counter()
            lengths_gg = torch.full((B,), L0 + args.max_tokens, dtype=torch.long, device=input_ids.device)
            m = report_metrics("Google", t0, t1, lengths_gg, L0, B)
            all_metrics["Google"].append(m)
            for b in range(B):
                text = tokenizer.decode(out_gg[b], skip_special_tokens=True)
                jfh.write(json.dumps({"mode": "Google", "prompt": batch_prompts[b], "output": text},
                                     ensure_ascii=False) + "\n")
            del out_gg
            _clear_wrappers(small, large)
            _cuda_gc()

        # 3) BASS-PAD（内部已做启发式 γ）
        if "bass" in want_modes or "bass-pad" in want_modes:
            torch.manual_seed(args.seed)
            t0 = perf_counter()
            out_bass, lengths_bass = speculative_sampling_bass_pad(
                input_ids, small, large,
                max_new_tokens=args.max_tokens,
                gamma_init=args.gamma_init,
                temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                verbose=False,
            )
            t1 = perf_counter()
            m = report_metrics("BASS-PAD", t0, t1, lengths_bass, L0, B)
            all_metrics["BASS-PAD"].append(m)
            for b in range(B):
                text = tokenizer.decode(out_bass[b, :int(lengths_bass[b].item())], skip_special_tokens=True)
                jfh.write(json.dumps({"mode": "BASS-PAD", "prompt": batch_prompts[b], "output": text},
                                     ensure_ascii=False) + "\n")
            del out_bass, lengths_bass
            _clear_wrappers(small, large)
            _cuda_gc()

        # ★ 批间清缓存（把这一批的编码等释放掉）
        del input_ids
        if 'out_dm' in locals(): del out_dm
        if 'out_gg' in locals(): del out_gg
        if 'out_bass' in locals(): del out_bass
        if 'lengths_bass' in locals(): del lengths_bass
        _clear_wrappers(small, large)
        _cuda_gc()

    jfh.close()

    # === 汇总平均 & 对比 ===
    def avg(arr):
        if not arr:
            return None
        t = sum(x["time"] for x in arr) / len(arr)
        tp = sum(x["throughput"] for x in arr) / len(arr)
        return {"time": t, "throughput": tp}

    epoch_avg = {k: avg(v) for k, v in all_metrics.items() if v}

    print("\n==== AVERAGE OVER ALL BATCHES ====")
    for k, v in epoch_avg.items():
        if v:
            print(f"{k:9s}  avg_time={v['time']:.3f}s   avg_throughput={v['throughput']:.2f} tok/s")

    if "BASS-PAD" in epoch_avg:
        print("\n==== BASS vs Others ====")
        compare_bass(epoch_avg)

    # 逐批明细落 CSV
    with open(csv_path, "a", encoding="utf-8") as fcsv:
        for mode, arr in all_metrics.items():
            for m in arr:
                fcsv.write(f"{mode},{args.batch},{m['total_new']},{m['time']:.6f},{m['throughput']:.6f},{m['avg_new']:.3f}\n")


if __name__ == "__main__":
    main()
