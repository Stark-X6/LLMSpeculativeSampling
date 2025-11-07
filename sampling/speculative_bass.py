"""
BASS-PAD: Batched speculative decoding with PAD-only sequence alignment.

- 草稿 γ 步 → 目标验证 γ 步 → 每条独立接受/拒绝 → rollback → 继续
- 仅在序列维裁剪/补齐（PAD），兼容 LLaMA/OPT/GPTNeoX/Qwen/Bloom 等
- 支持启发式 γ（DraftLengthHeuristic），默认先固定 γ
"""

import torch
from .batched_kvcache_model import BatchedKVCacheModel
from .utils_bass import multinomial_sample, positive_diff_normalize, DraftLengthHeuristic

@torch.no_grad()
def speculative_sampling_bass_pad(
    prefixes: torch.Tensor,                  # (B, L0)
    approx_model: torch.nn.Module,
    target_model: torch.nn.Module,
    max_new_tokens: int = 20,
    gamma_init: int = 4,
    temperature: float = 0.7,
    top_k: int = 0,
    top_p: float = 0.0,
    verbose: bool = False,
    use_heuristic_gamma: bool = False,       # 是否使用启发式 gamma
):
    """
    Batched speculative decoding (PAD alignment) with optional heuristic gamma.

    Args:
      prefixes: (B, L0) batch input ids.
      approx_model, target_model: two causal LMs sharing tokenizer.
      max_new_tokens: total new tokens per sequence.
      gamma_init: initial draft length.
      temperature, top_k, top_p: sampling params.
      verbose: print per-round stats.
      use_heuristic_gamma: if True, use DraftLengthHeuristic to adaptively adjust gamma.

    Returns:
      output_ids: (B, L_out_max) padded output ids.
      lengths: (B,) final lengths for each sequence.
    """
    device = next(target_model.parameters()).device
    B, L0 = prefixes.shape
    input_ids = prefixes.to(device)

    done = torch.zeros(B, dtype=torch.bool, device=device)
    lengths = torch.full((B,), L0, dtype=torch.long, device=device)
    T_goal = L0 + max_new_tokens

    small = BatchedKVCacheModel(approx_model, temperature, top_k, top_p)
    large = BatchedKVCacheModel(target_model, temperature, top_k, top_p)

    # prefill
    _ = small.forward_with_cache(input_ids)
    _ = large.forward_with_cache(input_ids)

    heur = DraftLengthHeuristic()
    heur.ldraft = int(gamma_init)
    round_id = 0
    acc_full_count = 0
    rej_count = 0

    # 新逻辑：初始化 gamma 并选择模式
    gamma_cur = int(gamma_init)

    while True:
        # 1️⃣ 确定本轮 gamma
        if use_heuristic_gamma:
            planned_gamma = int(max(1, heur.ldraft))
        else:
            planned_gamma = int(gamma_cur)

        if (lengths >= T_goal).all():
            break

        remain = (T_goal - lengths).clamp_min(0)
        gamma_eff = int(min(planned_gamma, remain.max().item()))
        if gamma_eff <= 0:
            break

        active_mask = (~done).unsqueeze(1)

        # 2️⃣ 草稿阶段
        idx = (lengths - 1).clamp_min(0).view(-1, 1)
        last = input_ids.gather(1, idx)
        for _ in range(gamma_eff):
            probs_q = small.forward_with_cache(last)
            next_q  = multinomial_sample(probs_q[:, -1, :])
            next_q  = torch.where(active_mask, next_q, last)
            input_ids = torch.cat([input_ids, next_q], dim=1)
            last = next_q

        # 3️⃣ 验证阶段
        new_tail = input_ids[:, -gamma_eff:]
        for t in range(gamma_eff):
            _ = large.forward_with_cache(new_tail[:, t:t + 1])

        base = large.prob_history.size(1) - gamma_eff
        assert base >= 0, "prob_history window too short; check W and gamma"

        n_vec = torch.empty(B, dtype=torch.long, device=device)
        start = (lengths - 1).clamp_min(0).clone()

        for b in range(B):
            if done[b]:
                n_vec[b] = lengths[b] - 1
                continue
            n = start[b] + gamma_eff - 1
            for i in range(gamma_eff):
                j = input_ids[b, start[b] + i]
                p = large.prob_history[b, base + i, j]
                q = small.prob_history[b, base + i, j]
                r = torch.rand((), device=device)
                if r > (p / q).clamp(max=1.0):
                    n = start[b] + i - 1
                    break
            n_vec[b] = n

        # 4️⃣ 接受/拒绝生成
        t_tokens = torch.empty(B, 1, dtype=torch.long, device=device)
        for b in range(B):
            if done[b]:
                last_idx = int(lengths[b].item()) - 1
                t_tokens[b:b + 1, :] = input_ids[b:b + 1, last_idx:last_idx + 1]
                continue
            n = int(n_vec[b].item())
            if n < start[b] + gamma_eff - 1:
                n_win = base + (n - start[b])
                p = large.prob_history[b, n_win, :]
                q = small.prob_history[b, n_win, :]
                t = multinomial_sample(positive_diff_normalize(p - q)).view(1, 1)
                t_tokens[b:b+1, :] = t
                rej_count += 1
            else:
                last_idx = large.prob_history.size(1) - 1
                t = multinomial_sample(large.prob_history[b, last_idx, :]).view(1, 1)
                t_tokens[b:b+1, :] = t
                acc_full_count += 1

        # 5️⃣ 回滚 KV
        kv_end = n_vec
        small.rollback(kv_end)
        large.rollback(kv_end)

        # 拼接 t
        new_len = n_vec + 1
        rows = []
        for b in range(B):
            keep = int(new_len[b].item())
            rows.append(torch.cat([input_ids[b:b+1, :keep], t_tokens[b:b+1, :]], dim=1))

        max_w = max(r.size(1) for r in rows)
        rows_pad = [torch.nn.functional.pad(r, (0, max_w - r.size(1))) for r in rows]
        input_ids = torch.cat(rows_pad, dim=0)
        lengths = new_len + 1
        done = done | (lengths >= T_goal)

        # 6️⃣ 启发式更新（仅启用时）
        x_vec = (n_vec - start + 1).clamp(min=0, max=gamma_eff)
        if use_heuristic_gamma:
            heur.step(x_vec)

        if verbose:
            x_max = int(x_vec.max().item())
            print(f"[BASS-PAD] round={round_id + 1}, gamma={planned_gamma}, x_max={x_max}, "
                  f"active={(~done).sum().item()}, max_len={int(lengths.max().item())}")
        round_id += 1

    return input_ids, lengths
