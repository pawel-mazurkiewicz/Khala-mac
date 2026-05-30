"""Phase-3 Unit A/B: KV cache equivalence, greedy gate, sampler. Run: PYTHONPATH=. .venv-mac/bin/python tests/test_kv_cache.py"""
from __future__ import annotations

import torch

from tests._util import artifacts_dir, cos, tiny_config
from core.khala_model import KhalaModel, KhalaKVCache


def test_cache_equivalence():
    torch.manual_seed(0)
    cfg = tiny_config()
    model = KhalaModel(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 7))

    with torch.no_grad():
        full = model(ids)  # [1, 7, padded_vocab]

    # Incremental: prefill first 3 tokens, then decode the rest one at a time.
    cache = KhalaKVCache(cfg.num_layers)
    inc_logits = []
    with torch.no_grad():
        h = model.forward_hidden_states(ids[:, :3], causal=True, kv_cache=cache)
        inc_logits.append(model.lm_head(h))  # [1,3,V]
        for t in range(3, ids.shape[1]):
            h = model.forward_hidden_states(ids[:, t:t + 1], causal=True, kv_cache=cache)
            inc_logits.append(model.lm_head(h))  # [1,1,V]
    inc = torch.cat(inc_logits, dim=1)

    c = cos(full, inc)
    print(f"  cache_equivalence cos={c:.6f}")
    assert c > 0.9999, f"KV-cache decode diverged from full forward: cos={c}"
    print("  test_cache_equivalence PASS")


def test_cache_equivalence_batch():
    torch.manual_seed(2)
    cfg = tiny_config()
    model = KhalaModel(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (3, 6))  # batch of 3

    with torch.no_grad():
        full = model(ids)

    cache = KhalaKVCache(cfg.num_layers)
    inc = []
    with torch.no_grad():
        h = model.forward_hidden_states(ids[:, :2], causal=True, kv_cache=cache)
        inc.append(model.lm_head(h))
        for t in range(2, ids.shape[1]):
            h = model.forward_hidden_states(ids[:, t:t + 1], causal=True, kv_cache=cache)
            inc.append(model.lm_head(h))
    inc = torch.cat(inc, dim=1)

    c = cos(full, inc)
    print(f"  cache_equivalence_batch cos={c:.6f}")
    assert c > 0.9999, f"batched KV-cache decode diverged: cos={c}"
    print("  test_cache_equivalence_batch PASS")


if __name__ == "__main__":
    test_cache_equivalence()
    test_cache_equivalence_batch()
    print("ALL PASS")
