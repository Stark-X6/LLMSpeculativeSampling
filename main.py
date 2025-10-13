
import torch
import argparse
from time import perf_counter
import contexttimer
from colorama import Fore, Style
from transformers import AutoTokenizer, AutoModelForCausalLM

from sampling import autoregressive_sampling, speculative_sampling, speculative_sampling_v2, speculative_sampling_bass_pad
from globals import Decoder




# my local models
MODELZOO = {
    # llama-1
    # https://huggingface.co/PY007/TinyLlama-1.1B-step-50K-105b
    # "llama1b": "/share_nfs/fangjiarui/root/code/hf_models/TinyLlama-1.1B-step-50K-105b",
    # "llama7b": "/share_nfs/tianzhi/code/llama-7b",
    # "llama30b": "/share_nfs/fangjiarui/root/code/hf_models/llama-30b-hf",
    # "llama2-7b" : "/share_nfs/fangjiarui/root/code/hf_models/llama-2-7b-hf",
    # "llama2-70b" : "/share_nfs/fangjiarui/root/code/hf_models/llama-2-70b-hf",
    # "bloom-560m": "/share_nfs/fangjiarui/root/code/hf_models/bloom-560m",
    # "bloom7b": "/share_nfs/fangjiarui/root/code/hf_models/bloomz-7b1",
    # "baichuan-7b": "/share_nfs/duanqiyuan/models/source_models/hf/baichuan-7B",
    # "baichuan-13b": "/share_nfs/duanqiyuan/models/source_models/hf/Baichuan-13B-Base",

    # 组合 1: TinyLlama 1.1B (approx) + LLaMA-2 7B (target)
    "tinyllama1b": "/mnt/sevenT/qinggangw/xiayankang/Project/data/models/TinyLlama-1B",
    "llama7b":     "/mnt/sevenT/qinggangw/xiayankang/Project/data/models/Llama-2-7B",

    # 组合 2: Bloom-560M (approx) + Bloomz-7B1 (target)
    "bloom-560m":  "/mnt/sevenT/qinggangw/xiayankang/Project/data/models/bloom-560m",
    "bloom7b":     "/mnt/sevenT/qinggangw/xiayankang/Project/data/models/bloomz-7b1",
}

def parse_arguments():
    parser = argparse.ArgumentParser(description='args for main.py')

    parser.add_argument('--input', type=str, default="Any recommendations for my holidays in Abu Dhabi?")
    parser.add_argument('--approx_model_name', type=str, default=MODELZOO["bloom-560m"])
    parser.add_argument('--target_model_name', type=str, default=MODELZOO["bloom7b"])
    parser.add_argument('--verbose', '-v', action='store_true', default=False, help='enable verbose mode') #详细日志开关
    parser.add_argument('--seed', '-s', type=int, default=None, help='set a random seed, which can makes the result reproducible')
    parser.add_argument('--benchmark', '-b', action='store_true', default=False, help='show benchmark results.')
    parser.add_argument('--profiling', '-p', action='store_true', default=False, help='collect torch profiler results.')
    parser.add_argument('--max_tokens', '-M', type=int, default=20, help='max token number generated.')
    parser.add_argument('--gamma', '-g', type=int, default=4, help='guess time.')
    parser.add_argument('--bass', action='store_true', default=False, help='use BASS-PAD batched speculative decoding')
    parser.add_argument('--batch', type=int, default=1, help='batch size')
    parser.add_argument('--gamma-min', type=int, default=2)
    parser.add_argument('--gamma-max', type=int, default=8)
    args = parser.parse_args()
    return args


def color_print(text):
    print(Fore.RED + text + Style.RESET_ALL)
    
def benchmark(fn, print_prefix, use_profiler=True, *args, **kwargs):
    TEST_TIME = 10
    profile_filename = f"./profile_logs/{print_prefix}"
    
    with contexttimer.Timer() as t:
        if use_profiler:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=0, warmup=1, active=2, repeat=1, skip_first=0),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_filename),
                record_shapes=False,
                profile_memory=False,
                # with_stack=True
            ) as prof:
                for _ in range(TEST_TIME): 
                    output = fn(*args, **kwargs)
                    prof.step()
        else:
            for _ in range(TEST_TIME): 
                output = fn(*args, **kwargs)

    print(f"\n [benchmark] {print_prefix}, tokens/sec: {len(output[0]) / t.elapsed / TEST_TIME}, {t.elapsed / TEST_TIME} sec generates {len(output[0])} tokens")

def generate(input_text, approx_model_name, target_model_name, num_tokens=20, gamma = 4,
             random_seed = None, verbose = False, use_benchmark = False, use_profiling = False,
             use_bass=False, batch=1, gamma_min=2, gamma_max=8):
    # NOTE() approx_model_name and target_model_name should use the same tokenizer!
    
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    tokenizer = AutoTokenizer.from_pretrained(approx_model_name, trust_remote_code=True)
  
    Decoder().set_tokenizer(tokenizer)
    
    print(f"begin loading models: \n {approx_model_name} \n {target_model_name}")
    device = 0  # 统一到 cuda:0
    small_model = AutoModelForCausalLM.from_pretrained(approx_model_name, 
                                                       torch_dtype=torch.float16,
                                                       device_map={"": device},
                                                       trust_remote_code=True)
    large_model = AutoModelForCausalLM.from_pretrained(target_model_name, 
                                                       torch_dtype=torch.float16,
                                                       device_map={"": device},
                                                       trust_remote_code=True)
    print("finish loading models")
    
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(torch_device)
    # 统计基线：初始长度与批大小
    B = input_ids.size(0)
    L0 = input_ids.size(1)
    metrics = {}  # 收集四种模式的指标

    top_k = 20
    top_p = 0.9

    torch.manual_seed(123)
    output = autoregressive_sampling(input_ids, large_model, num_tokens, top_k = top_k, top_p=top_p)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    color_print(f"large (target) model autoregressive_sampling: {generated_text}")
    
    if use_benchmark:
        benchmark(autoregressive_sampling, "AS_large", use_profiling,
                  input_ids, large_model, num_tokens, top_k = top_k, top_p=top_p)

    torch.manual_seed(123)
    output = autoregressive_sampling(input_ids, small_model, num_tokens, top_k = top_k, top_p=top_p)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    color_print(f"small (approx) model autoregressive_sampling: {generated_text}")
    
    if use_benchmark:
        benchmark(autoregressive_sampling, "AS_small", use_profiling,
                  input_ids, small_model, num_tokens, top_k = top_k, top_p=top_p)
    
    torch.manual_seed(123)
    t0 = perf_counter()
    output = speculative_sampling_v2(input_ids, small_model, large_model, num_tokens, top_k = top_k, top_p=top_p, random_seed = random_seed)
    t1 = perf_counter()
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    color_print(f"deepmind's speculative_sampling: {generated_text}")
    lengths_dm = torch.full((B,), L0 + num_tokens, dtype=torch.long, device=output.device)
    metrics["DeepMind"] = _report_metrics("DeepMind", t0, t1, lengths_dm, L0, B)

    torch.manual_seed(123)
    t0 = perf_counter()
    output = speculative_sampling(input_ids, small_model, large_model, num_tokens, gamma = gamma, top_k = top_k, top_p=top_p, random_seed = random_seed, verbose = verbose)
    t1 = perf_counter()
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    color_print(f"google's speculative_sampling: {generated_text}")
    lengths_gg = torch.full((B,), L0 + num_tokens, dtype=torch.long, device=output.device)
    metrics["Google"] = _report_metrics("Google", t0, t1, lengths_gg, L0, B)
    
    if use_benchmark:
        benchmark(speculative_sampling, "SP", use_profiling,
                  input_ids, small_model, large_model, max_len = num_tokens, gamma = gamma, top_k = top_k, top_p=top_p, random_seed = random_seed)

    torch.manual_seed(123)
    if use_bass or batch > 1:
        prefixes = input_ids.repeat(batch, 1)
        t0 = perf_counter()
        out, lengths = speculative_sampling_bass_pad(
            prefixes, small_model, large_model,
            max_new_tokens=num_tokens,
            gamma_init=gamma, gamma_min=gamma_min, gamma_max=gamma_max,
            top_k=top_k, top_p=top_p,
            adapt_gamma=True, verbose=verbose,
        )
        t1 = perf_counter()
        metrics["BASS-PAD"] = _report_metrics("BASS-PAD", t0, t1, lengths, L0, B)
        # 打印前几条
        for i in range(min(batch, 3)):
            L = int(lengths[i].item())
            txt = tokenizer.decode(out[i, :L], skip_special_tokens=True)
            color_print(f"[BASS-PAD][{i}] {txt[:300]} ...")
        #return out

    # 5) 汇总对比：BASS 相对其它三种的提升倍数
    _compare_against_bass(metrics)

def _report_metrics(mode: str, t0: float, t1: float, lengths, L0: int, B: int):
    """
    打印吞吐与延迟：
      - throughput = 总新生成token数 / 墙钟时长（tokens/s）
      - latency    = 墙钟时长（s）
      - avg ms/token = 1000 * latency / 总新生成token数
    """
    import torch
    elapsed = t1 - t0
    if isinstance(lengths, torch.Tensor):
        new_tokens = (lengths - L0).clamp_min(0)
        total_new = int(new_tokens.sum().item())
        avg_new = float(new_tokens.float().mean().item())
    else:
        total_new = sum(max(0, l - L0) for l in lengths)
        avg_new = total_new / max(B, 1)
    tput = total_new / elapsed if elapsed > 0 else float("inf")
    per_tok_ms = (elapsed / total_new * 1000.0) if total_new > 0 else float("inf")
    print(f"[METRICS] mode={mode}, batch={B}, total_new={total_new}, "
          f"time={elapsed:.3f}s, throughput={tput:.2f} tok/s, avg_per_seq={avg_new:.2f} tok, "
          f"avg_latency_per_token={per_tok_ms:.2f} ms")
    return {"mode": mode, "time": elapsed, "throughput": tput, "total_new": total_new, "avg_new": avg_new}

def _compare_against_bass(results: dict):
    """
    results: dict[mode] -> {"time": float, "throughput": float}
    打印 BASS 相对其它模式的吞吐/延迟倍数（>1 表示更好）
    """
    bass = results.get("BASS-PAD")
    if not bass:
        return
    for ref in ("DeepMind", "Google"):
        other = results.get(ref)
        if not other:
            continue
        tput_gain = (bass["throughput"] / other["throughput"]) if other["throughput"] > 0 else float("inf")
        # 延迟“越小越好”，提升倍数定义为 other_time / bass_time
        lat_gain  = (other["time"] / bass["time"]) if bass["time"] > 0 else 0.0
        print(f"[COMPARE] BASS vs {ref}: throughput x{tput_gain:.2f}, latency x{lat_gain:.2f}")

if __name__ == "__main__":
    args = parse_arguments()
    
    generate(args.input, args.approx_model_name, args.target_model_name, num_tokens=args.max_tokens, gamma=args.gamma,
             random_seed = args.seed, verbose=args.verbose, use_benchmark = args.benchmark,
             use_bass=args.bass, batch=args.batch, gamma_min=args.gamma_min, gamma_max=args.gamma_max,)
