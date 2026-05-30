"""Shared helpers for the Phase-3 vanilla-runtime tests (plain asserts, no pytest)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.khala_config import KhalaConfig  # noqa: E402


def artifacts_dir() -> Path:
    return Path(os.environ.get("KHALA_VANILLA_WEIGHTS", REPO / "_cuda_artifacts"))


def tiny_config() -> KhalaConfig:
    """Small, fast config for structural tests. q_dim (heads*head_dim) must equal hidden_size."""
    return KhalaConfig(
        hidden_size=64, num_layers=2, num_attention_heads=4, num_query_groups=2,
        head_dim=16, ffn_hidden_size=128, vocab_size=200, padded_vocab_size=256,
        max_position_embeddings=128, num_quantizers=8,
    )


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()
