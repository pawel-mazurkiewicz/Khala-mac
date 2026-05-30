"""Torch-only gather of a Megatron torch_dist (DCP) checkpoint -> safetensors.

Reads a checkpoint directory that contains `.metadata` + `*.distcp` shards
(Megatron's `--ckpt-format torch_dist`) using torch.distributed.checkpoint
directly. No megatron-core, no TransformerEngine, no apex, no flash-attn.

Works because the model is tensor-parallel size 1: every model weight is a
full (logically unsharded) tensor; DCP only chunks it physically across the
`.distcp` files and reassembles on load. Optimizer state and TE `_extra_state`
byte blobs are skipped.

Outputs (into $KHALA_GATHER_OUT, default /workspace/khala_gather/out):
    <label>.safetensors      model weights only, Megatron naming verbatim
    <label>_layout.json      name -> shape/dtype/numel
    <label>_samples.json     name -> first8 + mean/std/min/max  (parity fingerprints)

Usage:
    python tools/gather_dcp.py <label> <checkpoint_dir>
    # e.g.
    python tools/gather_dcp.py backbone /workspace/ckpt/backbone/iter_0036000
"""
from __future__ import annotations

import importlib.abc
import importlib.machinery
import json
import os
import sys
import time
import types
from pathlib import Path


# --- Fake `megatron.*` module tree so torch's plain pickle.load(.metadata)
#     can resolve Megatron-namespaced classes (enums, sharded-obj metadata)
#     to harmless stubs. The chunk/offset layout dcp.load actually needs is
#     stored in standard torch metadata classes, so stubbing is safe. ---
class _StubModule(types.ModuleType):
    def __getattr__(self, name: str):
        def __new__(c, *a, **k):
            return object.__new__(c)

        def __init__(self, *a, **k):
            pass

        cls = type(name, (object,), {"__new__": __new__, "__init__": __init__,
                                     "__module__": self.__name__})
        setattr(self, name, cls)
        return cls


class _MegatronStubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "megatron" or fullname.startswith("megatron."):
            return importlib.machinery.ModuleSpec(fullname, self)
        return None

    def create_module(self, spec):
        return _StubModule(spec.name)

    def exec_module(self, module):
        pass


def _install_megatron_stub():
    if not any(isinstance(f, _MegatronStubFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _MegatronStubFinder())


_install_megatron_stub()

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import TensorStorageMetadata

OUT_DIR = Path(os.environ.get("KHALA_GATHER_OUT", "/workspace/khala_gather/out")).resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _log(m: str) -> None:
    print(f"[gather-dcp] {m}", flush=True)


def _ensure_dist() -> None:
    """DCP load needs a (possibly single-rank) process group."""
    if dist.is_available() and not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29577")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group(backend="gloo", rank=0, world_size=1)


def _is_wanted(key: str) -> bool:
    low = key.lower()
    if "optim" in low:
        return False
    if "_extra_state" in key:
        return False
    if key.startswith("rng_state") or "/rng" in key:
        return False
    return True


def load_dcp_weights(ckpt_dir: Path) -> dict:
    _ensure_dist()
    reader = FileSystemReader(str(ckpt_dir))
    meta = reader.read_metadata()
    sdm = meta.state_dict_metadata

    state_dict: dict = {}
    skipped = 0
    for key, md in sdm.items():
        if not _is_wanted(key):
            skipped += 1
            continue
        if isinstance(md, TensorStorageMetadata):
            state_dict[key] = torch.empty(tuple(md.size), dtype=md.properties.dtype)
        else:
            skipped += 1  # BytesStorageMetadata (_extra_state, rng) -> skip
    _log(f"requesting {len(state_dict)} tensors ({skipped} non-weight entries skipped)")

    dcp.load(state_dict, storage_reader=reader)
    return state_dict


def write_layout_samples(sd: dict, label: str) -> None:
    layout = []
    samples = {}
    for name, t in sd.items():
        layout.append(
            {"name": name, "shape": list(t.shape), "dtype": str(t.dtype), "numel": int(t.numel())}
        )
        if t.numel() == 0:
            continue
        flat = t.detach().flatten().to(torch.float32)
        samples[name] = {
            "first8": flat[:8].tolist(),
            "mean": float(flat.mean()),
            "std": float(flat.std()) if flat.numel() > 1 else 0.0,
            "min": float(flat.min()),
            "max": float(flat.max()),
        }
    (OUT_DIR / f"{label}_layout.json").write_text(json.dumps(layout, indent=2))
    (OUT_DIR / f"{label}_samples.json").write_text(json.dumps(samples, indent=2))
    n_params = sum(int(t.numel()) for t in sd.values())
    _log(f"{label}: {len(sd)} tensors, {n_params/1e9:.3f}B params")


def save_safetensors(sd: dict, label: str) -> None:
    from safetensors.torch import save_file

    path = OUT_DIR / f"{label}.safetensors"
    cpu_sd = {k: v.detach().contiguous().cpu() for k, v in sd.items()}
    save_file(cpu_sd, str(path))
    _log(f"wrote {path.name} ({path.stat().st_size:,} B)")


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    label = sys.argv[1]
    ckpt_dir = Path(sys.argv[2]).resolve()
    if not ckpt_dir.exists():
        _log(f"ERROR: {ckpt_dir} does not exist")
        return 1
    t0 = time.time()
    _log(f"label={label} dir={ckpt_dir} out={OUT_DIR}")
    sd = load_dcp_weights(ckpt_dir)
    write_layout_samples(sd, label)
    save_safetensors(sd, label)
    _log(f"DONE {label} in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
