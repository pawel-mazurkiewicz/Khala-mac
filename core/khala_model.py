"""Vanilla PyTorch KhalaModel — the de-Megatron-ified backbone / super-res transformer.

A standalone `nn.Module` reimplementation of the Megatron-Core GPT used by Khala,
targeting CPU/MPS/CUDA via `core.device_utils`. No Megatron, TransformerEngine,
apex, or flash-attn.

Naming maps to `tools/convert_megatron_to_hf.py` output:
    embed.weight                                  (multi-codebook embedding)
    layers.{i}.input_norm.weight                  pre-attention RMSNorm
    layers.{i}.attn.{q_proj,k_proj,v_proj}.{weight,bias}
    layers.{i}.attn.o_proj.{weight,bias}
    layers.{i}.post_attn_norm.weight              pre-MLP RMSNorm
    layers.{i}.mlp.{gate_proj,up_proj,down_proj}.{weight,bias}
    norm.weight                                   final RMSNorm
    lm_head.weight                                untied output projection

CONVENTIONS NOT YET PARITY-VERIFIED (no CUDA forward goldens captured in this gather):
  * RoPE: implemented as Llama/GPT-NeoX-style rotate_half. Megatron-core's default
    matches this, but the exact half-vs-interleaved split is the #1 source of
    Megatron->HF drift. Flagged; localized to `apply_rotary`.
  * SwiGLU gate/up assignment: assumes silu(gate_proj) * up_proj with
    weight_w -> gate, weight_v -> up. Localized to `KhalaMLP.forward`.
Both are single-point flips once we have a forward fixture to check against.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .khala_config import KhalaConfig
from .khala_embedding import MultiLayerEmbedding


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


def build_rope_cache(seq_len: int, head_dim: int, theta: float, device, dtype):
    """Standard RoPE inverse-frequency cache. Returns (cos, sin) of [seq_len, head_dim]."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)               # [seq_len, head_dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)        # [seq_len, head_dim]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(q, k, cos, sin):
    # q,k: [B, H, S, D]; cos,sin: [S, D] -> broadcast over B,H
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


class KhalaAttention(nn.Module):
    def __init__(self, config: KhalaConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.num_attention_heads
        self.n_kv = config.num_query_groups
        self.head_dim = config.head_dim
        bias = config.add_bias_linear
        self.q_proj = nn.Linear(config.hidden_size, config.q_dim, bias=bias)
        self.k_proj = nn.Linear(config.hidden_size, config.kv_dim, bias=bias)
        self.v_proj = nn.Linear(config.hidden_size, config.kv_dim, bias=bias)
        self.o_proj = nn.Linear(config.q_dim, config.hidden_size, bias=bias)

    def forward(self, x, cos, sin, causal: bool = True):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_kv, self.head_dim).transpose(1, 2)

        q, k = apply_rotary(q, k, cos, sin)

        # GQA: expand KV heads to match Q heads
        rep = self.n_heads // self.n_kv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

        # Backbone is causal; super-res is non-causal (packed bidirectional).
        # TODO(phase-3): KV-cache for incremental decode + padding-mask support.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        out = out.transpose(1, 2).reshape(B, S, self.n_heads * self.head_dim)
        return self.o_proj(out)


class KhalaMLP(nn.Module):
    def __init__(self, config: KhalaConfig):
        super().__init__()
        bias = config.add_bias_linear
        self.gate_proj = nn.Linear(config.hidden_size, config.ffn_hidden_size, bias=bias)
        self.up_proj = nn.Linear(config.hidden_size, config.ffn_hidden_size, bias=bias)
        self.down_proj = nn.Linear(config.ffn_hidden_size, config.hidden_size, bias=bias)
        self.double_bias = config.swiglu_double_bias and bias

    def forward(self, x):
        # SwiGLU: silu(gate) * up. The upstream fused bias_swiglu adds the fc1 bias a
        # second time (see KhalaConfig.swiglu_double_bias) -> verified bit-exact.
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        if self.double_bias:
            gate = gate + self.gate_proj.bias
            up = up + self.up_proj.bias
        return self.down_proj(F.silu(gate) * up)


class KhalaDecoderLayer(nn.Module):
    def __init__(self, config: KhalaConfig):
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.attn = KhalaAttention(config)
        self.post_attn_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.mlp = KhalaMLP(config)

    def forward(self, x, cos, sin, causal: bool = True):
        x = x + self.attn(self.input_norm(x), cos, sin, causal=causal)
        x = x + self.mlp(self.post_attn_norm(x))
        return x


class KhalaModel(nn.Module):
    def __init__(self, config: KhalaConfig):
        super().__init__()
        self.config = config
        self.embed = MultiLayerEmbedding(config)
        self.layers = nn.ModuleList(KhalaDecoderLayer(config) for _ in range(config.num_layers))
        self.norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.padded_vocab_size, bias=False)
        self._rope_cache: tuple | None = None

    def _rope(self, seq_len: int, device, dtype):
        if self._rope_cache is None or self._rope_cache[0].shape[0] < seq_len \
                or self._rope_cache[0].device != device or self._rope_cache[0].dtype != dtype:
            self._rope_cache = build_rope_cache(
                max(seq_len, 1), self.config.head_dim, self.config.rope_theta, device, dtype
            )
        cos, sin = self._rope_cache
        return cos[:seq_len], sin[:seq_len]

    def forward_hidden_states(self, input_ids: torch.Tensor, causal: bool = True) -> torch.Tensor:
        """Embedding -> transformer stack -> final norm. Returns [B, S, H].

        `causal=True` for the backbone; the super-resolution model runs non-causal
        (packed bidirectional) — pass causal=False. This is the hook the super-res
        path projects from (cf. the worker's `generate_superres_manual_projection`).
        Phase-3 will add the token-range slice + sampling on top.
        """
        h = self.embed(input_ids)
        S = h.shape[1]
        cos, sin = self._rope(S, h.device, h.dtype)
        for layer in self.layers:
            h = layer(h, cos, sin, causal=causal)
        return self.norm(h)

    def forward(self, input_ids: torch.Tensor, causal: bool = True) -> torch.Tensor:
        """Full forward to vocab logits [B, S, padded_vocab_size]."""
        return self.lm_head(self.forward_hidden_states(input_ids, causal=causal))
