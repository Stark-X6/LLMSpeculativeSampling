"""
Dynamic Batch Scheduler
"""

import os
import time
import json
import random
import uuid
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ==============================
# Request 对象
# ==============================
class Request:
    def __init__(self, req_id, input_ids, arrive_time=None):
        self.req_id = req_id
        self.input_ids = input_ids
        self.arrive_time = arrive_time or time.time()

import threading, queue

class InferenceWorker:
    """
    单消费者推理线程：
      - 在该线程中初始化/绑定 CUDA 上下文
      - 串行处理批次，避免并发访问模型
    """
    def __init__(self, device: str, device_index: int, process_fn):
        self.device = device
        self.device_index = device_index
        self.process_fn = process_fn
        self.q = queue.Queue(maxsize=128)
        self._stop = threading.Event()
        self._started = False
        self._thread = None

    def put(self, batch):
        self.q.put(batch)

    def start(self):
        if self._started:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self._started = True

    def stop(self):
        self._stop.set()
        # 放一个哨兵，确保线程能从阻塞中醒来
        try:
            self.q.put_nowait(None)
        except queue.Full:
            pass
        if self._thread:
            self._thread.join(timeout=60)

    def _loop(self):
        # 在消费者线程里绑定 CUDA 上下文（关键！）
        if self.device.startswith("cuda"):
            try:
                torch.cuda.set_device(self.device_index)
                torch.cuda._lazy_init()
            except Exception as e:
                print(f"[WARN] CUDA init in consumer thread failed: {e}")

        while not self._stop.is_set():
            batch = self.q.get()
            if batch is None:
                break
            try:
                # 执行你传进来的批处理函数（内部会做 CUDA 推理）
                self.process_fn(batch)
            except torch.cuda.OutOfMemoryError:
                print("[ERROR] OOM in consumer; skipping batch")
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"[ERROR] Exception in consumer: {e}")
            finally:
                self.q.task_done()

        # 清空剩余
        try:
            while True:
                _ = self.q.get_nowait()
                self.q.task_done()
        except queue.Empty:
            pass

# ==============================
# 主线程调度器
# ==============================
class BucketBatchScheduler:
    def __init__(self, small_model, large_model, tokenizer: AutoTokenizer,
                 bucket_config=None,
                 wait_window_ms=5000,
                 max_wait_ms=15000):
        self.small_model = small_model
        self.large_model = large_model
        self.tokenizer = tokenizer
        self.bucket_config = bucket_config or {
            (0, 32): 16,
            (33, 64): 4,
            (65, 128): 4,
            (129, 256): 4,
            (257, 512): 4,
            (513, 1024): 2
        }
        self.wait_window = wait_window_ms / 1000.0
        self.max_wait = max_wait_ms / 1000.0
        self.buffer = []

    def add_request(self, req: Request):
        self.buffer.append(req)

    def _get_bucket_bs(self, length):
        for (lo, hi), bs in self.bucket_config.items():
            if lo <= length <= hi:
                return (lo, hi, bs)
        return (4096, 8192, 1)

    def _form_batches(self):
        now = time.time()
        ready_batches, remain = [], []
        buckets = {}

        # 按长度分桶
        for req in self.buffer:
            L = len(req.input_ids)
            key = self._get_bucket_bs(L)
            buckets.setdefault(key, []).append(req)

        for (lo, hi, bs), reqs in buckets.items():
            reqs.sort(key=lambda r: r.arrive_time)
            cur = []
            for r in reqs:
                wait_time = now - r.arrive_time
                if wait_time > self.max_wait:
                    ready_batches.append([r])
                    continue
                cur.append(r)
                if len(cur) == bs:
                    ready_batches.append(cur)
                    cur = []
            remain.extend(cur)

        # 更新缓冲区
        batched_ids = {r.req_id for batch in ready_batches for r in batch}
        self.buffer = [r for r in remain if r.req_id not in batched_ids]
        return ready_batches


# ==============================
# Dolly 数据加载
# ==============================
def load_dolly_dataset(path, tokenizer, max_samples=None, seed=42):
    """
    从数据集中等概率随机抽样（打乱整个数据集，再取前 N 条）：
    - 若 max_samples 为 None：返回全量随机顺序
    - 若 max_samples 为 k：返回随机 k 条
    """
    if seed is not None:
        random.seed(seed)

    # 先把所有样本读入内存（jsonl 每行一个样本）
    raw = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            prompt = obj.get("instruction", "")
            ctx = obj.get("context", "")
            if ctx:
                prompt = f"{ctx.strip()} {prompt.strip()}"
            raw.append(prompt)

    # 全量打乱（这一步确保“在数据集里就乱着取”）
    random.shuffle(raw)

    # 截取 N 条（若给了 max_samples）
    if max_samples is not None:
        raw = raw[:max_samples]

    # 编码 + 过滤过短样本
    data = []
    for text in raw:
        ids = tokenizer.encode(text, truncation=True, max_length=512)
        data.append(ids)

    return data


# ==============================
# 批推理函数
# ==============================
def process_batch_fn(batch, small_model, large_model, tokenizer, device, max_new_tokens=20):
    from sampling.speculative_bass import speculative_sampling_bass_pad
    DEVICE = device
    B = len(batch)
    lengths = [len(r.input_ids) for r in batch]
    L_max = max(lengths)

    padded = torch.full((B, L_max), tokenizer.pad_token_id, dtype=torch.long)
    for i, r in enumerate(batch):
        l = len(r.input_ids)
        padded[i, :l] = torch.tensor(r.input_ids)
    padded = padded.to(DEVICE)

    t0 = time.perf_counter()
    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
        out, new_lengths = speculative_sampling_bass_pad(
            prefixes=padded,
            approx_model=small_model,
            target_model=large_model,
            max_new_tokens=max_new_tokens,
            gamma_init=4,
            top_k=40,
            top_p=0.9,
            use_heuristic_gamma = True,
        )
    t1 = time.perf_counter()
    elapsed = t1 - t0
    total_new = int(new_lengths.sum().item()) - sum(lengths)
    throughput = total_new / elapsed if elapsed > 0 else 0

    print(f"[BATCH DONE] batch={B}, avg_len={sum(lengths)/B:.1f}, "
          f"time={elapsed:.3f}s, throughput={throughput:.2f} tok/s")

    txt = tokenizer.decode(out[0, :int(new_lengths[0].item())], skip_special_tokens=True)
    print(f"  ▶ sample[0]: {txt[:120]}...")

    torch.cuda.empty_cache()


# ==============================
# 模拟输入流（主线程内）
# ==============================
def simulate_stream(dataset, scheduler, process_fn, mode="dynamic", device="cuda:0", bs=8,
                    device_index=0, log_path=None):
    """
    异步输入(生产者) + 主线程调度(组批) + 消费者线程(推理)
    - 只有消费者线程会触发 CUDA 调用（安全）
    - 主线程持续调度，不被推理阻塞 -> 避免“前一大批后全单条”
    """
    print(f"[SIM] 启动输入流模拟，模式={mode}, 设备={device}")

    if mode == "fixed":
        # 固定批：不需要异步消费者，主线程直接攒够就推理
        batch_buf = []
        for ids in dataset:
            batch_buf.append(Request(str(uuid.uuid4()), ids))
            if len(batch_buf) >= bs:
                process_fn(batch_buf)
                batch_buf = []
            time.sleep(random.uniform(0.05, 0.2))
        if batch_buf:
            process_fn(batch_buf)
        return

    # ========== dynamic 模式：使用异步推理队列 ==========
    # 1) 启动消费者线程（推理）
    worker = InferenceWorker(device=device, device_index=device_index, process_fn=process_fn)
    worker.start()

    # 2) 启动输入线程（只往 buffer 加请求）
    produced = {"count": 0}
    def producer():
        for ids in dataset:
            scheduler.add_request(Request(str(uuid.uuid4()), ids))
            produced["count"] += 1
            time.sleep(random.uniform(0.05, 0.2))  # 模拟请求到达间隔
        print(f"[SIM] 所有请求已投递到缓冲区，共 {produced['count']} 条。")

    input_thread = threading.Thread(target=producer, daemon=True)
    input_thread.start()

    # 3) 主线程作为“调度器”：周期性组批，并投递到推理队列
    print("[Scheduler] Dynamic batching started (async infer, main-thread scheduling).")
    consumed = 0
    last_info_ts = time.time()

    try:
        while True:
            # 调度周期
            time.sleep(scheduler.wait_window)
            batches = scheduler._form_batches()
            for b in batches:
                worker.put(b)
                consumed += len(b)

            # 退出条件：输入发完 + 缓冲区空 + 队列空
            no_more_input = (not input_thread.is_alive())
            buffer_empty = (len(scheduler.buffer) == 0)
            queue_empty = worker.q.empty()
            if no_more_input and buffer_empty and queue_empty:
                break

            # 可选：每隔一段打印一次积压状况
            now = time.time()
            if now - last_info_ts > 5:
                print(f"[Scheduler] buffered={len(scheduler.buffer)}, queued={worker.q.qsize()}, "
                      f"produced={produced['count']}, consumed={consumed}")
                last_info_ts = now

    finally:
        worker.stop()
        if input_thread.is_alive():
            input_thread.join(timeout=30)
        print(f"[SIM] 完成：produced={produced['count']}, consumed={consumed}")


# ==============================
# 主程序入口
# ==============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dolly Stream (pure serial)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--mode", type=str, default="dynamic", choices=["dynamic", "fixed"])
    parser.add_argument("--max_samples", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        DEVICE = "cpu"
        device_index = None
    else:
        n_gpus = torch.cuda.device_count()
        try:
            device_index = int(args.device.split(":")[-1])
        except Exception:
            device_index = 0
        if device_index >= n_gpus:
            DEVICE = "cuda:0"
            device_index = 0
        else:
            DEVICE = f"cuda:{device_index}"

    print(f"[INFO] 使用设备: {DEVICE}")

    # 加载模型
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../../data"))
    approx_path = os.path.join(DATA_ROOT, "models", "TinyLlama-1B")
    target_path = os.path.join(DATA_ROOT, "models", "Llama-2-7b-raw")

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

    dolly_path = os.path.join(DATA_ROOT, "dataset", "dolly", "databricks-dolly-15k.jsonl")
    dataset = load_dolly_dataset(dolly_path, tokenizer, args.max_samples)
    print(f"[INFO] 加载 {len(dataset)} 条 Dolly 样本")

    scheduler = BucketBatchScheduler(small_model, large_model, tokenizer)

    simulate_stream(dataset, scheduler,
                    lambda b: process_batch_fn(b, small_model, large_model, tokenizer, DEVICE, 20),
                    mode=args.mode, device=DEVICE)
