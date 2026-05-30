"""Phase-3 Unit A: non-causal padding-mask equivalence.
Run: PYTHONPATH=. .venv-mac/bin/python tests/test_attention_mask.py"""
from __future__ import annotations

import torch

from tests._util import cos, tiny_config
from core.khala_model import KhalaModel


def test_padding_mask_matches_prefix():
    """Non-causal forward with padded tail (keys masked) must reproduce, on the valid
    region, a non-causal forward of just the valid prefix."""
    torch.manual_seed(1)
    cfg = tiny_config()
    model = KhalaModel(cfg).eval()

    L, pad = 5, 4
    valid = torch.randint(0, cfg.vocab_size, (1, L))
    padded = torch.cat([valid, torch.randint(0, cfg.vocab_size, (1, pad))], dim=1)  # [1, L+pad]

    mask = torch.zeros(1, 1, 1, L + pad, dtype=torch.bool)
    mask[..., L:] = True  # True = padded / masked

    with torch.no_grad():
        ref = model.forward_hidden_states(valid, causal=False)             # [1, L, H]
        got = model.forward_hidden_states(padded, causal=False, attention_mask=mask)

    c = cos(ref, got[:, :L])
    print(f"  padding_mask cos={c:.6f}")
    assert c > 0.9999, f"padding-mask path diverged from prefix run: cos={c}"
    print("  test_padding_mask_matches_prefix PASS")


if __name__ == "__main__":
    test_padding_mask_matches_prefix()
    print("ALL PASS")
