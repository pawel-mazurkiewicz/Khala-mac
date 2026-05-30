"""Gather the DAC RVQ decoder weights from the Lightning .ckpt.

The decoder is already vanilla PyTorch (no Megatron). We mirror the worker's
load_decoder() filter exactly: keep checkpoint["state_dict"]["generator.*"],
strip the "generator." prefix, and save as a plain .pt that DacRVQ.load_state_dict
consumes directly on the Mac.

Outputs (into $KHALA_GATHER_OUT):
    decoder_weights.pt       generator-only state_dict (DacRVQ-ready)
    decoder_layout.json      name -> shape/dtype
    decoder_samples.json     fingerprints

Usage:
    python tools/gather_decoder.py /workspace/ckpt/dac_rvq_2490000.ckpt
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch

OUT_DIR = Path(os.environ.get("KHALA_GATHER_OUT", "/workspace/khala_gather/out")).resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _log(m: str) -> None:
    print(f"[gather-dec] {m}", flush=True)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    ckpt = Path(sys.argv[1]).resolve()
    if not ckpt.exists():
        _log(f"ERROR: {ckpt} not found")
        return 1
    t0 = time.time()
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd_full = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
    top_prefixes = sorted({k.split(".")[0] for k in sd_full})
    _log(f"top-level prefixes in ckpt state_dict: {top_prefixes}")

    gen = {
        k[len("generator."):]: v
        for k, v in sd_full.items()
        if k.startswith("generator.")
    }
    if not gen:
        _log("WARN: no generator.* keys; saving full state_dict instead")
        gen = dict(sd_full)
    _log(f"generator tensors: {len(gen)}")

    layout, samples = [], {}
    for name, t in gen.items():
        if not hasattr(t, "shape"):
            layout.append({"name": name, "non_tensor": True, "type": str(type(t))})
            continue
        layout.append({"name": name, "shape": list(t.shape), "dtype": str(t.dtype), "numel": int(t.numel())})
        if t.numel():
            flat = t.detach().flatten().to(torch.float32)
            samples[name] = {
                "first8": flat[:8].tolist(),
                "mean": float(flat.mean()),
                "std": float(flat.std()) if flat.numel() > 1 else 0.0,
                "min": float(flat.min()),
                "max": float(flat.max()),
            }
    (OUT_DIR / "decoder_layout.json").write_text(json.dumps(layout, indent=2))
    (OUT_DIR / "decoder_samples.json").write_text(json.dumps(samples, indent=2))
    torch.save(gen, OUT_DIR / "decoder_weights.pt")
    _log(f"wrote decoder_weights.pt ({(OUT_DIR/'decoder_weights.pt').stat().st_size:,} B)")
    _log(f"DONE in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
