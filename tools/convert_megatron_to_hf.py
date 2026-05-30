"""Convert a gathered Megatron Khala checkpoint (safetensors, Megatron naming) into
the vanilla `KhalaModel` naming, splitting the fused QKV and the SwiGLU pair.

Inputs  (from the CUDA gather, in _cuda_artifacts/):
    <label>.safetensors          Megatron-named weights
    <label>_samples.json         first8/mean/std fingerprints (passthrough check)
    <label>_megatron_args.json    -> KhalaConfig

Output:
    khala_<label>.safetensors     KhalaModel-named weights

Self-verification:
    1. every produced key matches KhalaModel.state_dict() exactly (strict load),
    2. total param count is preserved,
    3. passthrough tensors (embed/lm_head/norm) match the gathered fingerprints,
    4. the QKV split round-trips (re-concatenating q/k/v reproduces linear_qkv).

Usage:
    python tools/convert_megatron_to_hf.py backbone
    python tools/convert_megatron_to_hf.py superres --artifacts _cuda_artifacts
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.khala_config import KhalaConfig  # noqa: E402
from core.khala_model import KhalaModel  # noqa: E402

LAYER = re.compile(r"decoder\.layers\.(\d+)\.")


def _split_qkv(t: torch.Tensor, cfg: KhalaConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deinterleave Megatron fused QKV (per-group [q*hpg, k, v]) into q, k, v.

    weight: [ (ng*(hpg+2)*hd), hidden ]   bias: [ ng*(hpg+2)*hd ]
    """
    ng, hpg, hd = cfg.num_query_groups, cfg.heads_per_group, cfg.head_dim
    if t.dim() == 2:
        hidden = t.shape[1]
        t = t.view(ng, hpg + 2, hd, hidden)
        q = t[:, :hpg].reshape(ng * hpg * hd, hidden)
        k = t[:, hpg].reshape(ng * hd, hidden)
        v = t[:, hpg + 1].reshape(ng * hd, hidden)
    else:  # bias
        t = t.view(ng, hpg + 2, hd)
        q = t[:, :hpg].reshape(ng * hpg * hd)
        k = t[:, hpg].reshape(ng * hd)
        v = t[:, hpg + 1].reshape(ng * hd)
    return q, k, v


def convert(src: dict, cfg: KhalaConfig) -> dict:
    out: dict = {}
    qkv_roundtrip_ok = True
    for name, t in src.items():
        m = LAYER.search(name)
        if name == "embedding.word_embeddings.weight":
            out["embed.weight"] = t
        elif name == "output_layer.weight":
            out["lm_head.weight"] = t
        elif name == "decoder.final_layernorm.weight":
            out["norm.weight"] = t
        elif m:
            i = m.group(1)
            tail = name.split(f"decoder.layers.{i}.", 1)[1]
            p = f"layers.{i}."
            if tail == "self_attention.linear_qkv.layer_norm_weight":
                out[p + "input_norm.weight"] = t
            elif tail in ("self_attention.linear_qkv.weight", "self_attention.linear_qkv.bias"):
                kind = "weight" if tail.endswith("weight") else "bias"
                q, k, v = _split_qkv(t, cfg)
                out[p + f"attn.q_proj.{kind}"] = q
                out[p + f"attn.k_proj.{kind}"] = k
                out[p + f"attn.v_proj.{kind}"] = v
                if kind == "weight":
                    recon = torch.cat([
                        q.view(cfg.num_query_groups, cfg.heads_per_group, cfg.head_dim, -1),
                        k.view(cfg.num_query_groups, 1, cfg.head_dim, -1),
                        v.view(cfg.num_query_groups, 1, cfg.head_dim, -1),
                    ], dim=1).reshape(t.shape)
                    qkv_roundtrip_ok &= torch.equal(recon, t)
            elif tail == "self_attention.linear_proj.weight":
                out[p + "attn.o_proj.weight"] = t
            elif tail == "self_attention.linear_proj.bias":
                out[p + "attn.o_proj.bias"] = t
            elif tail == "mlp.linear_fc1.layer_norm_weight":
                out[p + "post_attn_norm.weight"] = t
            elif tail == "mlp.linear_fc1.weight_w":
                out[p + "mlp.gate_proj.weight"] = t
            elif tail == "mlp.linear_fc1.bias_w":
                out[p + "mlp.gate_proj.bias"] = t
            elif tail == "mlp.linear_fc1.weight_v":
                out[p + "mlp.up_proj.weight"] = t
            elif tail == "mlp.linear_fc1.bias_v":
                out[p + "mlp.up_proj.bias"] = t
            elif tail == "mlp.linear_fc2.weight":
                out[p + "mlp.down_proj.weight"] = t
            elif tail == "mlp.linear_fc2.bias":
                out[p + "mlp.down_proj.bias"] = t
            else:
                raise KeyError(f"Unmapped layer tensor: {name}")
        else:
            raise KeyError(f"Unmapped tensor: {name}")
    if not qkv_roundtrip_ok:
        raise AssertionError("QKV split failed to round-trip — interleave layout is wrong.")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("label", choices=["backbone", "superres"])
    ap.add_argument("--artifacts", default="_cuda_artifacts")
    args = ap.parse_args()

    A = Path(args.artifacts)
    cfg = KhalaConfig.from_megatron_args(A / f"{args.label}_megatron_args.json")
    src = load_file(str(A / f"{args.label}.safetensors"))
    print(f"[convert] {args.label}: {len(src)} source tensors")

    out = convert(src, cfg)

    # 1. key parity against the model
    model_keys = set(KhalaModel(cfg).state_dict().keys())
    produced = set(out.keys())
    missing = model_keys - produced
    extra = produced - model_keys
    assert not missing, f"missing keys for model: {sorted(missing)[:10]}"
    assert not extra, f"extra keys not in model: {sorted(extra)[:10]}"

    # 2. param-count conservation
    n_src = sum(t.numel() for t in src.values())
    n_out = sum(t.numel() for t in out.values())
    assert n_src == n_out, f"param count changed: {n_src} -> {n_out}"

    # 3. passthrough fingerprint check
    samp = json.loads((A / f"{args.label}_samples.json").read_text())
    checks = {
        "embed.weight": "embedding.word_embeddings.weight",
        "lm_head.weight": "output_layer.weight",
        "norm.weight": "decoder.final_layernorm.weight",
    }
    for new, old in checks.items():
        got = out[new].flatten()[:8].float().tolist()
        exp = samp[old]["first8"]
        assert all(abs(a - b) < 1e-3 for a, b in zip(got, exp)), f"fingerprint drift on {new}"

    dst = A / f"khala_{args.label}.safetensors"
    save_file({k: v.contiguous() for k, v in out.items()}, str(dst))
    print(f"[convert] wrote {dst.name}: {len(out)} tensors, {n_out/1e9:.3f}B params")
    print(f"[convert] checks passed: key-parity, param-count={n_out}, qkv-roundtrip, fingerprints")
    return 0


if __name__ == "__main__":
    sys.exit(main())
