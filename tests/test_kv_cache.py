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


def test_sampler_greedy_gate():
    """sample_backbone(temperature=0) must reproduce the CUDA greedy golden 64/64."""
    from core.khala_runtime import load_vanilla_model, sample_backbone

    A = artifacts_dir()
    golden = A / "golden_backbone_greedy.pt"
    if not golden.exists():
        print("  SKIP greedy gate (golden_backbone_greedy.pt absent)")
        return
    g = torch.load(golden, map_location="cpu", weights_only=False)
    prompt = [int(x) for x in g["prompt_ids"]]
    gen = g["generated"]
    gen = [int(x) for x in (gen.tolist() if hasattr(gen, "tolist") else gen)]

    model = load_vanilla_model(
        "backbone", A / "khala_backbone.safetensors", A / "backbone_megatron_args.json",
        device=torch.device("cpu"), dtype=torch.float32,
    )
    out = sample_backbone(model, prompt, num_tokens=len(gen), temperature=0.0, top_k=1)
    match = sum(a == b for a, b in zip(out, gen))
    print(f"  sampler_greedy_gate {match}/{len(gen)}")
    assert match == len(gen), f"greedy sampler diverged: {match}/{len(gen)}"
    print("  test_sampler_greedy_gate PASS")


def test_topk_sampler_runs():
    """Stochastic top-k path returns in-vocab tokens of the requested length."""
    from core.khala_runtime import sample_backbone

    torch.manual_seed(0)
    cfg = tiny_config()
    model = KhalaModel(cfg).eval()
    out = sample_backbone(model, [1, 2, 3], num_tokens=5, temperature=0.8, top_k=10)
    assert len(out) == 5 and all(0 <= t < cfg.vocab_size for t in out), out
    print("  test_topk_sampler_runs PASS")


if __name__ == "__main__":
    test_cache_equivalence()
    test_cache_equivalence_batch()
    test_topk_sampler_runs()
    test_sampler_greedy_gate()
    print("ALL PASS")
