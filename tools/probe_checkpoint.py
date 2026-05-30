"""Inspect what's actually inside the downloaded Khala checkpoint directory
so we can pick the right loader. Runs anywhere with stdlib only.

Usage:
    python tools/probe_checkpoint.py                       # uses ./checkpoints
    python tools/probe_checkpoint.py /path/to/checkpoints

Output: a JSON-style printout of the tree, file sizes, and a verdict per
sub-checkpoint of the form:

    [verdict] backbone -> dist-checkpoint (torch.distributed.checkpoint)
    [verdict] superres -> legacy-pt (torch.load)
    [verdict] dac      -> single .pt file (lightning state_dict)

No torch import required, no GPU touched.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _is_dist_checkpoint(d: Path) -> bool:
    # Megatron's distributed checkpoint format ships a `.metadata` file and
    # `*.distcp` shards.
    return (d / ".metadata").exists() or any(d.glob("*.distcp"))


def _is_legacy_pt(d: Path) -> bool:
    # Megatron's legacy format: model_optim_rng.pt under mp_rank_XX_YY/
    return any(d.glob("**/model_optim_rng.pt")) or any(d.glob("mp_rank_*/*.pt"))


def _is_single_pt(d: Path) -> bool:
    return (d / "pytorch_model.bin").exists() or any(d.glob("*.pt"))


def _is_safetensors(d: Path) -> bool:
    return any(d.glob("*.safetensors"))


def _classify(d: Path) -> str:
    if _is_dist_checkpoint(d):
        return "dist-checkpoint (torch.distributed.checkpoint)"
    if _is_legacy_pt(d):
        return "megatron-legacy-pt"
    if _is_safetensors(d):
        return "safetensors (already portable!)"
    if _is_single_pt(d):
        return "single-pt (likely lightning or HF style)"
    return "UNKNOWN"


def _tree(d: Path, depth: int = 0, max_depth: int = 3, max_per_dir: int = 12) -> None:
    indent = "  " * depth
    if depth >= max_depth:
        print(f"{indent}... (depth limit)")
        return
    entries = sorted(d.iterdir())
    for i, p in enumerate(entries):
        if i >= max_per_dir:
            print(f"{indent}... ({len(entries) - max_per_dir} more)")
            break
        if p.is_dir():
            print(f"{indent}{p.name}/")
            _tree(p, depth + 1, max_depth, max_per_dir)
        else:
            size = p.stat().st_size
            print(f"{indent}{p.name}  ({size:,} B)")


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "checkpoints").resolve()
    if not root.exists():
        print(f"ERROR: {root} does not exist")
        return 1

    print(f"=== Probing {root} ===\n")
    print("Top-level tree (max depth 3):")
    _tree(root)
    print()

    print("Verdicts (per top-level subdir):")
    for sub in sorted(root.iterdir()):
        if sub.is_dir():
            verdict = _classify(sub)
            sizes = sum(p.stat().st_size for p in sub.rglob("*") if p.is_file())
            print(f"  {sub.name:25s} -> {verdict:55s} ({sizes/1e9:.2f} GB)")
        else:
            print(f"  {sub.name:25s} -> file ({sub.stat().st_size:,} B)")

    print()
    print("Next step:")
    print("  python tools/gather_cuda_minimal.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
