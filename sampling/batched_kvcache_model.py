"""
BatchedKVCacheModel

- 批量 KV Cache 容器（Bloom 与标准布局兼容）
- forward_with_cache: 逐 token 推进 KV & 维护概率滑窗
- rollback: 按每条序列 end_pos_vec 裁剪（仅序列维）
"""

import torch
from typing import List, Tuple

# 工具函数全部来自 utils_bass（避免重复定义）
from .utils_bass import normalize_logits, multinomial_sample

class BatchedKVCacheModel:
    def __init__(self, model: torch.nn.Module,
                 temperature: float = 1.0, top_k: int = 0, top_p: float = 0.0):
        self.model = model
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.past_key_values = None      # tuple(list) of (k,v) per layer
        self.prob_history = None         # (B, T, V)

    @torch.no_grad()
    def forward_with_cache(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: (B, Lnew)
        返回：本次新增段“最后一帧”的概率 (B, 1, V)，并推进 past_key_values & prob_history
        关键点：
        - 首轮/后续，无论 Lnew 有多少，都用逐 token 的方式推进 KV
        - 对每个 token 都计算一次 logits → normalize_logits(...) → 追加到 prob_history
        - 每步都把 prob_history 的时间维裁到与 KV 一致，防无界增长
        """
        B, T = input_ids.shape
        last_probs = None

        for t in range(T):
            out = self.model(
                input_ids[:, t:t + 1],
                use_cache=True,
                past_key_values=self.past_key_values,
                output_attentions=False,
                output_hidden_states=False,
            )
            self.past_key_values = out.past_key_values

            # 只取本步（最后一帧）logits → 归一化成概率（含 temp/top_k/top_p）
            probs_t = normalize_logits(
                out.logits[:, -1:, :],  # (B, 1, V)
                self.temperature, self.top_k, self.top_p
            )
            last_probs = probs_t  # 保留“最后一帧”作为本函数返回值

            # 维护 prob_history：逐步追加
            if self.prob_history is None:
                self.prob_history = probs_t
            else:
                self.prob_history = torch.cat([self.prob_history, probs_t], dim=1)

            # # 概率滑窗到 KV 时间维，保证与 past_key_values 对齐、避免无界增长
            # k0 = self.past_key_values[0][0]
            # T_kv = k0.shape[-1]  # 取 K 的时间维
            # if self.prob_history.size(1) > T_kv:
            #     self.prob_history = self.prob_history[:, -T_kv:, :]

            # ★ 固定小滑窗：只保留最近 W=32 帧，避免 (B×L0×V) 爆显存
            W = 24  # 可调：至少 ≥ 1 + 预期的最大 γ
            if self.prob_history.size(1) > W:
                self.prob_history = self.prob_history[:, -W:, :]

        return last_probs  # (B, 1, V)

    @torch.no_grad()
    def generate(self, x: torch.Tensor, steps: int) -> torch.Tensor:
        """
        对 batch 统一前进 steps 步（每步喂最后一列并采样 1 token）
        - 首次若无 KV，会先对 x 做一次 prefill（不采样）
        """
        if self.past_key_values is None:
            _ = self.forward_with_cache(x)

        for _ in range(steps):
            last_ids = x[:, -1:]
            probs = self.forward_with_cache(last_ids)     # (B,1,V)
            nxt = multinomial_sample(probs[:, -1, :])     # (B,1)
            x = torch.cat([x, nxt], dim=1)
        return x

    @torch.no_grad()
    def rollback(self, end_pos_vec: torch.LongTensor):
        """
        end_pos_vec: (B,) —— 每条序列保留 [0..end_pos]（含），并 PAD 到 batch 内最大长度。
        仅裁剪“序列维 S”，不动 H/D 维，兼容 GQA。
        """
        assert self.past_key_values is not None, "No KV to rollback"
        max_keep = int(end_pos_vec.max().item()) + 1

        # 裁剪概率历史
        self.prob_history = self.prob_history[:, :max_keep, :]

        trimmed: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for k, v in self.past_key_values:
            if k.dim() == 4 and v.dim() == 4:
                # 标准阵营: k,v (B, H, S, D)
                B, H, S, D = k.shape
                k_pad = torch.zeros(B, H, max_keep, D, device=k.device, dtype=k.dtype)
                v_pad = torch.zeros(B, H, max_keep, D, device=v.device, dtype=v.dtype)
                for b in range(B):
                    keep = int(end_pos_vec[b].item()) + 1
                    k_pad[b, :, :keep, :] = k[b, :, :keep, :]
                    v_pad[b, :, :keep, :] = v[b, :, :keep, :]
                trimmed.append((k_pad.contiguous(), v_pad.contiguous()))

            elif k.dim() == 3 and v.dim() == 3:
                # Bloom 阵营: k (B*H, D, S), v (B*H, S, D)
                BH, D, S = k.shape
                B = end_pos_vec.size(0)
                H = BH // B
                k = k.view(B, H, D, S)
                v = v.view(B, H, S, D)

                k_pad = torch.zeros(B, H, D, max_keep, device=k.device, dtype=k.dtype)
                v_pad = torch.zeros(B, H, max_keep, D, device=v.device, dtype=v.dtype)
                for b in range(B):
                    keep = int(end_pos_vec[b].item()) + 1
                    k_pad[b, :, :, :keep] = k[b, :, :, :keep]
                    v_pad[b, :, :keep, :] = v[b, :, :keep, :]
                k_pad = k_pad.view(B * H, D, max_keep).contiguous()
                v_pad = v_pad.view(B * H, max_keep, D).contiguous()
                trimmed.append((k_pad, v_pad))

            else:
                raise NotImplementedError(
                    f"Unsupported KV layout: k{tuple(k.shape)}, v{tuple(v.shape)}"
                )

        self.past_key_values = tuple(trimmed)
