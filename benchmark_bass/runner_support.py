# benchmark_bass/runner_support.py
from typing import Tuple, Dict, Any
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# ===== 模型路径 =====
MODELZOO = {
    "bloom": {
        "approx": "/mnt/sevenT/qinggangw/xiayankang/Project/data/models/bloom-560m",
        "target": "/mnt/sevenT/qinggangw/xiayankang/Project/data/models/bloomz-7b1",
    },
    "llama": {
        "approx": "/mnt/sevenT/qinggangw/xiayankang/Project/data/models/TinyLlama-1B",
        "target": "/mnt/sevenT/qinggangw/xiayankang/Project/data/models/Llama-2-7b-raw",
    }
}


# ===== Prompt 拼接逻辑（与 eval_dolly 保持一致） =====
def format_prompt(inst: str, ctx: str) -> str:
    """把 Dolly-15k 的 instruction/context 拼成标准 prompt"""
    parts = [f"### Instruction:\n{(inst or '').strip()}"]
    if ctx and len(ctx.strip()) > 0:
        parts.append(f"### Context:\n{ctx.strip()}")
    parts.append("### Response:")
    return "\n\n".join(parts)


def setup_models(model_pair: str):
    """加载小模型和大模型（保持两侧 tokenizer 一致）"""
    device = 3  # cuda:0
    approx_path = MODELZOO[model_pair]["approx"]
    target_path = MODELZOO[model_pair]["target"]

    tokenizer = AutoTokenizer.from_pretrained(approx_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    draft_model = AutoModelForCausalLM.from_pretrained(
        approx_path, torch_dtype=torch.float16, device_map={"": device}, trust_remote_code=True
    )
    target_model = AutoModelForCausalLM.from_pretrained(
        target_path, torch_dtype=torch.float16, device_map={"": device}, trust_remote_code=True
    )
    return draft_model, target_model, tokenizer


def prepare_batch(
    tokenizer,
    sorted_flag: int,
    batch_size: int,
    ctx_len: int,
    offset: int = 0,
    dataset_path: str = "/mnt/sevenT/qinggangw/xiayankang/Project/data/dataset/dolly/databricks-dolly-15k.jsonl",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    从 Dolly 数据集中构造一个 batch。

    ✅ 全局排序逻辑：
        - 若 sorted_flag == 1：
            先对整个数据集按输入长度升序排列，
            再从排序后的列表中连续取 batch。
        - 若 sorted_flag == 0：
            保持原始顺序，仅随机取起点（模拟无序输入）。

    ✅ 特点：
        - 每个 batch 内样本是连续取的；
        - 不同 batch 之间不会重叠；
        - 排序/不排序逻辑对齐；
        - 输出输入长度统计。
    """
    # === 1. 加载本地 Dolly 数据集 ===
    ds = load_dataset("json", data_files={"train": dataset_path}, split="train")
    total = len(ds)

    # === 2. 生成所有 prompt 并计算长度 ===
    prompts = [format_prompt(ex.get("instruction", ""), ex.get("context", "")) for ex in ds]

    # ⚠️ 不返回 tensor，只取每条样本长度
    encodings = tokenizer(
        prompts,
        padding=False,
        truncation=True,
        max_length=ctx_len,
    )
    lengths = [len(ids) for ids in encodings["input_ids"]]

    # === 3. 若 sorted_flag == 1，则全局排序 ===
    if sorted_flag:
        order = np.argsort(lengths)
        prompts = [prompts[i] for i in order]
        lengths = [lengths[i] for i in order]
    else:
        # 不排序：保持原始顺序，只随机确定起点（随机偏移）
        np.random.seed(offset)  # 保持每次 repeat 可复现
        np.random.shuffle(prompts)
        # 不排序版本保持长度对应打乱
        np.random.seed(offset)
        np.random.shuffle(lengths)

    # === 4. 按 offset 选取 batch（连续片段） ===
    total_batches = max(1, total // batch_size)
    start = (offset % total_batches) * batch_size
    end = min(start + batch_size, total)
    batch_prompts = prompts[start:end]
    batch_lengths = lengths[start:end]

    # === 5. 再次编码 batch（统一 padding） ===
    enc_batch = tokenizer(
        batch_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=ctx_len,
    )
    input_ids = enc_batch["input_ids"].to("cuda")

    # === 6. 计算统计信息 ===
    len_stats = {
        "mean_len": float(np.mean(batch_lengths)),
        "var_len": float(np.var(batch_lengths)),
        "min_len": float(np.min(batch_lengths)),
        "max_len": float(np.max(batch_lengths)),
    }

    return input_ids, len_stats
