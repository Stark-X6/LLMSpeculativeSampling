# sampling/batched_kvcache_model.py
"""
批量 KV 缓存容器（BASS-PAD）
- 仅在序列维 S 上裁剪/补齐，兼容任意 Causal LM（LLaMA/OPT/GPTNeoX/Qwen…）
- 对 Bloom 的 KV 形状用“按维度判断”的方式兼容
"""

import torch
from typing import List, Tuple

# 工具函数全部来自 utils_bass（避免重复定义）
from .utils_bass import normalize_logits, multinomial_sample

from transformers.models.bloom.modeling_bloom import BloomForCausalLM

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
        返回：新增段的概率 (B, Lnew, V)，并推进 past_key_values & prob_history
        """
        if self.past_key_values is None:
            outputs = self.model(input_ids, use_cache=True)
            probs = normalize_logits(
                outputs.logits[:, -input_ids.size(1):, :],
                self.temperature, self.top_k, self.top_p
            )
            self.past_key_values = outputs.past_key_values
            self.prob_history = probs
            return probs
        else:
            outputs = self.model(
                input_ids, past_key_values=self.past_key_values, use_cache=True
            )
            probs = normalize_logits(
                outputs.logits[:, -input_ids.size(1):, :],
                self.temperature, self.top_k, self.top_p
            )
            self.past_key_values = outputs.past_key_values
            self.prob_history = torch.cat([self.prob_history, probs], dim=1)
            return probs

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
