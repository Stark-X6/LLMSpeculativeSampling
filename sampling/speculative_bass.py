# sampling/speculative_bass.py
"""
BASS-PAD 批量推测解码
- 流程：草稿 γ 步 → 目标验证 γ 步 → 每条接受/拒绝 → rollback → 继续
- 仅在序列维做裁剪与 PAD，兼容任意 Causal LM（LLaMA/OPT/GPTNeoX/Qwen…）
- 验证阶段逐 token 喂入，语义清晰；如需更快可改为 packed 段一次性验证
"""

import torch
from .batched_kvcache_model import BatchedKVCacheModel
# 采样/差分归一化全部来自 utils_bass（避免重复）
from .utils_bass import multinomial_sample, positive_diff_normalize

@torch.no_grad()
def speculative_sampling_bass_pad(
    prefixes: torch.Tensor,                  # (B, L0)
    approx_model: torch.nn.Module,
    target_model: torch.nn.Module,
    max_new_tokens: int = 32,
    gamma_init: int = 4,
    gamma_min: int = 2,
    gamma_max: int = 8,
    temperature: float = 0.7,
    top_k: int = 0,
    top_p: float = 0.0,
    adapt_gamma: bool = True,
    verbose: bool = False,
):
    device = next(target_model.parameters()).device
    B, L0 = prefixes.shape
    input_ids = prefixes.to(device)

    done = torch.zeros(B, dtype=torch.bool, device=device)
    lengths = torch.full((B,), L0, dtype=torch.long, device=device)
    T_goal = L0 + max_new_tokens

    small = BatchedKVCacheModel(approx_model, temperature, top_k, top_p)
    large = BatchedKVCacheModel(target_model, temperature, top_k, top_p)

    # prefill：建立两侧 KV 与 prob_history
    _ = small.forward_with_cache(input_ids)
    _ = large.forward_with_cache(input_ids)

    gamma = gamma_init
    acc_full_count = 0
    rej_count = 0

    while True:
        if (lengths >= T_goal).all():
            break

        active_mask = (~done).unsqueeze(1)  # (B,1)

        # ===== 1) 草稿：对未完成样本前进 γ 步 =====
        idx = (lengths - 1).clamp_min(0).view(-1, 1)  # (B,1)
        last = input_ids.gather(1, idx)  # (B,1)
        for _ in range(gamma):
            probs_q = small.forward_with_cache(last)            # (B,1,V)
            next_q  = multinomial_sample(probs_q[:, -1, :])  # (B,1)
            next_q  = torch.where(active_mask, next_q, last)    # 完成样本长度不再变化
            input_ids = torch.cat([input_ids, next_q], dim=1)
            last = next_q

        # ===== 2) 目标：验证这 γ 个新增位置（逐 token 喂入） =====
        new_tail = input_ids[:, -gamma:]                        # (B, γ)
        for t in range(gamma):
            _ = large.forward_with_cache(new_tail[:, t:t+1])

        # ===== 3) 每条序列独立接受/拒绝，得到 n_b =====
        n_vec = torch.empty(B, dtype=torch.long, device=device)
        start = lengths.clone()

        for b in range(B):
            if done[b]:
                n_vec[b] = lengths[b] - 1
                continue
            n = start[b] + gamma - 1
            for i in range(gamma):
                j = input_ids[b, start[b] + i]
                p = large.prob_history[b, start[b] + i - 1, j]
                q = small.prob_history[b, start[b] + i - 1, j]
                r = torch.rand((), device=device)
                if r > (p / q).clamp(max=1.0):
                    n = start[b] + i - 1
                    break
            n_vec[b] = n

        # ===== 4) 差分重采 or 全接收再采 1 步 =====
        t_tokens = torch.empty(B, 1, dtype=torch.long, device=device)
        for b in range(B):
            if done[b]:
                last_idx = int(lengths[b].item()) - 1
                t_tokens[b:b + 1, :] = input_ids[b:b + 1, last_idx:last_idx + 1]
                continue
            n = int(n_vec[b].item())
            if n < start[b] + gamma - 1:
                # 拒绝：差分重采
                p = large.prob_history[b, n, :]
                q = small.prob_history[b, n, :]
                t = multinomial_sample(positive_diff_normalize(p - q)).view(1, 1)
                t_tokens[b:b+1, :] = t
                rej_count += 1
            else:
                # 全接收：用目标分布的“最新一格”再采 1 步（防越界）
                last_idx = large.prob_history.size(1) - 1
                t = multinomial_sample(large.prob_history[b, last_idx, :]).view(1, 1)
                t_tokens[b:b+1, :] = t
                acc_full_count += 1

        # 5) 先把 KV 回滚到 n_b（仅保留“最后接受的位置”，不包含 t）
        kv_end = n_vec  # 注意：此处 end_pos = n_b
        small.rollback(kv_end)
        large.rollback(kv_end)

        # 再把输入裁剪到 n_b+1（含最后接受的位置），并拼接 t（此时 KV 还落后一位）
        new_len = n_vec + 1
        rows = []
        for b in range(B):
            keep = int(new_len[b].item())
            rows.append(torch.cat([input_ids[b:b+1, :keep], t_tokens[b:b+1, :]], dim=1))

        # PAD 到同宽（BASS-PAD）
        max_w = max(r.size(1) for r in rows)
        rows_pad = [torch.nn.functional.pad(r, (0, max_w - r.size(1))) for r in rows]
        input_ids = torch.cat(rows_pad, dim=0)

        lengths = new_len + 1
        done = done | (lengths >= T_goal)

        # ===== 6) γ 自适应（简易启发式）=====
        if adapt_gamma:
            if acc_full_count >= (B // 2):
                gamma = min(gamma + 1, gamma_max)
            if rej_count >= (B // 3):
                gamma = max(gamma - 1, gamma_min)
            acc_full_count = 0
            rej_count = 0

        if verbose:
            print(f"[BASS-PAD] active={(~done).sum().item()}, "
                  f"gamma={gamma}, max_len={int(lengths.max().item())}")

    return input_ids, lengths
