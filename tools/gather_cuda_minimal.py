"""Minimal-deps gather: dumps Khala checkpoint state_dicts as safetensors WITHOUT
instantiating Megatron, TransformerEngine, apex, or flash-attn.

Designed for rented containers that ship bare CUDA + Python but no NGC stack.
Install requirements before running:

    pip install --upgrade pip
    pip install "torch>=2.4"                 # any CUDA-enabled torch; we don't even need CUDA for this
    pip install numpy safetensors omegaconf transformers
    # OPTIONAL: only if probe says "dist-checkpoint"
    pip install megatron-core

What it captures (subset of the full gather; forward-pass goldens are deferred):

    env.json
    backbone_layout.json + backbone_samples.json + backbone.safetensors
    superres_layout.json + superres_samples.json + superres.safetensors
    decoder_layout.json  + decoder_weights.pt    + decoder_config.yaml
    tokenizer/             + tokenizer_test.json
    GATHER_INFO.txt        + checkpoint_args.json (if discoverable)

What it does NOT capture (requires running the model, which needs TE):
    embedding_test_dim*.pt     <- deferred to a later rental (or skipped entirely)
    golden_backbone_greedy.pt  <- deferred

Usage:
    export KHALA_GATHER_OUT=/host/khala_artifacts   # any writable dir
    python tools/gather_cuda_minimal.py             # uses ./checkpoints
    python tools/gather_cuda_minimal.py /path/to/checkpoints
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DECODER_ROOT = PROJECT_ROOT / "models" / "Decoder"
TOKENIZER_PATH = PROJECT_ROOT / "models" / "Tokenizer"

CKPT_ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "checkpoints").resolve()
OUT_DIR = Path(os.environ.get("KHALA_GATHER_OUT", "/tmp/khala_artifacts")).resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _log(msg: str) -> None:
    print(f"[gather-min] {msg}", flush=True)


def _json_dump(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))
    _log(f"wrote {path.relative_to(OUT_DIR)} ({path.stat().st_size:,} B)")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------- Checkpoint format detection ----------

def _is_dist_checkpoint(d: Path) -> bool:
    return (d / ".metadata").exists() or any(d.glob("*.distcp"))


def _is_legacy_pt(d: Path) -> bool:
    return any(d.glob("**/model_optim_rng.pt"))


def _classify(d: Path) -> str:
    if _is_dist_checkpoint(d):
        return "distcp"
    if _is_legacy_pt(d):
        return "legacy-pt"
    if any(d.glob("*.safetensors")):
        return "safetensors"
    if any(d.glob("*.pt")):
        return "single-pt"
    return "unknown"


# ---------- Loaders, one per format ----------

def _load_distcp(d: Path) -> tuple[dict, dict]:
    """Read a torch.distributed.checkpoint directory.

    Tries megatron-core's load_plain_tensors() first (handles tensor-parallel
    sharded state correctly), falls back to a raw torch.distributed.checkpoint
    read that only works for unsharded checkpoints.
    """
    args_blob = {}
    try:
        from megatron.core.dist_checkpointing import load_plain_tensors

        _log(f"  using megatron-core load_plain_tensors() on {d}")
        sd = load_plain_tensors(str(d))
    except ImportError:
        _log("  megatron-core not installed; trying raw torch.distributed.checkpoint")
        try:
            import torch
            import torch.distributed.checkpoint as dcp
            from torch.distributed.checkpoint import FileSystemReader

            reader = FileSystemReader(str(d))
            # Discover keys + shapes from the metadata
            metadata = reader.read_metadata()
            sd = {}
            for fqn, props in metadata.state_dict_metadata.items():
                if hasattr(props, "size"):
                    sd[fqn] = torch.empty(props.size, dtype=props.properties.dtype)
            dcp.load_state_dict(state_dict=sd, storage_reader=reader)
        except Exception as exc:
            raise RuntimeError(
                f"Could not read dist-checkpoint at {d}. Install megatron-core: pip install megatron-core. Underlying error: {exc!r}"
            ) from exc

    # Look for Megatron args inside the checkpoint dir
    args_file = d / "common.pt"
    if args_file.exists():
        try:
            import torch

            args_blob = torch.load(args_file, map_location="cpu", weights_only=False)
        except Exception as exc:
            _log(f"  WARN: could not read {args_file}: {exc!r}")

    return sd, args_blob


def _load_legacy_pt(d: Path) -> tuple[dict, dict]:
    """Read Megatron's legacy mp_rank_XX_YY/model_optim_rng.pt format."""
    import torch

    pt_files = sorted(d.glob("**/model_optim_rng.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No model_optim_rng.pt under {d}")
    if len(pt_files) > 1:
        _log(f"  WARN: multiple shards under {d}; loading first only. Found: {[str(p) for p in pt_files]}")
    blob = torch.load(pt_files[0], map_location="cpu", weights_only=False)
    sd = blob.get("model", blob.get("state_dict", blob))
    args_blob = blob.get("args", {})
    return sd, args_blob


def _load_single_pt(d: Path) -> tuple[dict, dict]:
    """Read a single .pt file (lightning-style or HF-style)."""
    import torch

    pt = next(d.glob("*.pt"))
    blob = torch.load(pt, map_location="cpu", weights_only=False)
    if isinstance(blob, dict) and "state_dict" in blob:
        return blob["state_dict"], {k: v for k, v in blob.items() if k != "state_dict"}
    return blob if isinstance(blob, dict) else {"model": blob}, {}


def _load_safetensors(d: Path) -> tuple[dict, dict]:
    from safetensors.torch import load_file

    sd = {}
    for st in sorted(d.glob("*.safetensors")):
        sd.update(load_file(str(st)))
    return sd, {}


LOADERS = {
    "distcp": _load_distcp,
    "legacy-pt": _load_legacy_pt,
    "safetensors": _load_safetensors,
    "single-pt": _load_single_pt,
}


# ---------- Layout / samples / safetensors dump ----------

def _layout_and_samples(sd: dict, layout_path: Path, samples_path: Path, label: str) -> None:
    import torch

    layout = []
    samples = {}
    for name, t in sd.items():
        if not hasattr(t, "shape"):
            layout.append({"name": name, "non_tensor": True, "type": str(type(t))})
            continue
        layout.append({
            "name": name,
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "numel": int(t.numel()),
        })
        if t.numel() == 0:
            continue
        flat = t.detach().flatten().to(torch.float32).cpu()
        samples[name] = {
            "first8": flat[:8].tolist(),
            "mean": float(flat.mean().item()),
            "std": float(flat.std().item()) if flat.numel() > 1 else 0.0,
            "min": float(flat.min().item()),
            "max": float(flat.max().item()),
        }
    _json_dump(layout, layout_path)
    _json_dump(samples, samples_path)
    _log(f"{label}: {len(layout)} tensors")


def _safetensors_save(sd: dict, path: Path) -> None:
    if path.exists():
        _log(f"skip {path.name} (already exists: {path.stat().st_size:,} B)")
        return
    from safetensors.torch import save_file

    cpu_sd = {}
    for k, v in sd.items():
        if hasattr(v, "detach"):
            cpu_sd[k] = v.detach().contiguous().cpu()
    save_file(cpu_sd, str(path))
    _log(f"wrote {path.name} ({path.stat().st_size:,} B)")


# ---------- Per-checkpoint pipeline ----------

def _gather_checkpoint(subdir: Path, label: str) -> None:
    if not subdir.exists():
        _log(f"  WARN: {subdir} not found; skipping {label}")
        return
    fmt = _classify(subdir)
    _log(f"\n=== {label}: {subdir} (format: {fmt}) ===")
    if fmt == "unknown":
        _log(f"  WARN: could not classify {subdir}; skipping")
        return
    loader = LOADERS[fmt]
    sd, args_blob = loader(subdir)
    _log(f"  loaded {len(sd)} tensors")

    _layout_and_samples(
        sd, OUT_DIR / f"{label}_layout.json", OUT_DIR / f"{label}_samples.json", label
    )
    _safetensors_save(sd, OUT_DIR / f"{label}.safetensors")

    if args_blob:
        # Simplify to JSON-serialisable values
        simple = {}
        for k, v in (args_blob.items() if isinstance(args_blob, dict) else []):
            try:
                json.dumps(v)
                simple[k] = v
            except (TypeError, ValueError):
                simple[k] = repr(v)
        _json_dump(simple, OUT_DIR / f"{label}_args.json")


# ---------- Tokenizer ----------

def _dump_tokenizer() -> None:
    dst = OUT_DIR / "tokenizer"
    if not dst.exists():
        shutil.copytree(TOKENIZER_PATH, dst)
        _log(f"copied tokenizer to {dst.relative_to(OUT_DIR)}")
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            str(TOKENIZER_PATH), trust_remote_code=True, local_files_only=True
        )
    except Exception as exc:
        _log(f"  WARN: AutoTokenizer failed ({exc!r}); trying PreTrainedTokenizerFast")
        from transformers import PreTrainedTokenizerFast

        tok = PreTrainedTokenizerFast(tokenizer_file=str(TOKENIZER_PATH / "tokenizer.json"))

    fixtures = {
        "empty": "",
        "ascii_short": "hello world",
        "english_song": (
            "A serene piano ballad with soft strings, gentle pace, in the style of Erik Satie."
        ),
    }
    encoded = {
        name: tok.encode(text, add_special_tokens=False) for name, text in fixtures.items()
    }
    _json_dump(
        {"fixtures": fixtures, "encoded": encoded, "tokenizer_class": type(tok).__name__},
        OUT_DIR / "tokenizer_test.json",
    )


# ---------- Decoder (DAC is already vanilla PyTorch) ----------

def _dump_decoder() -> None:
    import torch

    dac_ckpt = CKPT_ROOT / "decoder"
    if not dac_ckpt.exists():
        # The decoder might be a single file
        candidates = list(CKPT_ROOT.glob("**/dac*.pt")) + list(CKPT_ROOT.glob("**/decoder*.pt"))
        if not candidates:
            _log("  WARN: no decoder checkpoint found")
            return
        pt = candidates[0]
    else:
        pts = list(dac_ckpt.glob("*.pt"))
        if not pts:
            _log("  WARN: no .pt under decoder/")
            return
        pt = pts[0]

    _log(f"\n=== decoder: {pt} ===")
    blob = torch.load(pt, map_location="cpu", weights_only=False)
    sd = blob.get("state_dict", blob)
    # Mirror the worker's filter: keep only "generator." entries
    if any(k.startswith("generator.") for k in sd):
        sd = {k[len("generator."):]: v for k, v in sd.items() if k.startswith("generator.")}
    _layout_and_samples(
        sd, OUT_DIR / "decoder_layout.json", OUT_DIR / "decoder_samples.json", "decoder"
    )
    dst = OUT_DIR / "decoder_weights.pt"
    if not dst.exists():
        torch.save(sd, dst)
        _log(f"wrote {dst.name} ({dst.stat().st_size:,} B)")
    shutil.copy(DECODER_ROOT / "dac_rvq_1024_64_golden.yaml", OUT_DIR / "decoder_config.yaml")


# ---------- Env ----------

def _dump_env() -> None:
    import torch

    info = {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "env": {
            k: v for k, v in os.environ.items()
            if k.startswith(("CUDA_", "NCCL_", "MASTER_", "TORCH_", "NVIDIA_"))
        },
        "checkpoint_root": str(CKPT_ROOT),
    }
    _json_dump(info, OUT_DIR / "env.json")


def _summary() -> None:
    lines = ["Khala minimal-deps gather", f"output dir: {OUT_DIR}", ""]
    total = 0
    for path in sorted(OUT_DIR.rglob("*")):
        if path.is_file():
            size = path.stat().st_size
            total += size
            sha = ""
            if path.suffix in (".safetensors", ".pt"):
                sha = _sha256(path)[:16]
            lines.append(f"  {str(path.relative_to(OUT_DIR)):60s} {size:>15,} B  {sha}")
    lines += ["", f"total: {total:,} bytes ({total/1e9:.2f} GB)"]
    summary = "\n".join(lines)
    (OUT_DIR / "GATHER_INFO.txt").write_text(summary)
    print(summary)


def main() -> int:
    t0 = time.time()
    _log(f"checkpoint root: {CKPT_ROOT}")
    _log(f"output dir:      {OUT_DIR}")
    if not CKPT_ROOT.exists():
        _log(f"ERROR: {CKPT_ROOT} does not exist")
        return 1

    _dump_env()

    # Discover top-level checkpoint subdirs
    candidates = {p.name: p for p in CKPT_ROOT.iterdir() if p.is_dir()}
    # Common naming
    name_to_label = {
        "backbone": "backbone",
        "superres": "superres",
        "super_res": "superres",
        "super-res": "superres",
        "decoder": None,  # handled separately
        "dac": None,
        "tokenizer": None,
    }
    handled = set()
    for name, path in candidates.items():
        label = name_to_label.get(name, name)
        if label is None:
            continue
        _gather_checkpoint(path, label)
        handled.add(name)

    _dump_decoder()
    _dump_tokenizer()
    _summary()
    _log(f"DONE in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
