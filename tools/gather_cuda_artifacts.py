"""Dump everything we need from a CUDA box to enable the NVIDIA-free port.

Runs on the same Linux+CUDA host that today serves Khala (i.e. inside the NGC
container, with the Megatron stack importable). Reuses backend_worker's
existing init/load functions so we don't reimplement Megatron bootstrapping.

What it produces (all under $KHALA_GATHER_OUT, default /tmp/khala_artifacts):

  env.json                      torch / cuda / cudnn / device versions
  megatron_args.json            full Megatron args, post-checkpoint-merge
  backbone_layout.json          [{name, shape, dtype, numel}] for backbone
  backbone_samples.json         first-8 elements + min/max/mean/std per tensor
  backbone.safetensors          raw backbone state_dict, Megatron naming
  superres_layout.json          same for super-res
  superres_samples.json         same for super-res
  superres.safetensors          raw super-res state_dict
  decoder_layout.json           DAC layout
  decoder_weights.pt            DAC state_dict (already PyTorch-native)
  decoder_config.yaml           copy of dac_rvq_1024_64_golden.yaml
  tokenizer/                    copy of the prompt tokenizer dir
  tokenizer_test.json           {prompt -> token_ids} for fixed prompts
  embedding_test_dim2.pt        input/output of MultiLayerVocabParallelEmbedding (B,S)
  embedding_test_dim3_c1.pt     same with shape (B,S,1)
  embedding_test_dim3_cN.pt     same with shape (B,S,4)
  golden_backbone_greedy.pt     first 64 generated tokens for a fixed prompt @ temp=0
  GATHER_INFO.txt               summary, file sizes, sha256 of the safetensors

Usage (inside the NGC container, after `bash run_backend.sh` works at least once):

    export KHALA_GATHER_OUT=/tmp/khala_artifacts
    cd /workspace/Khala
    python tools/gather_cuda_artifacts.py

Then tar + scp home:

    tar -cvzf /tmp/khala_artifacts.tar.gz -C /tmp khala_artifacts
    # on your Mac:
    scp <gpubox>:/tmp/khala_artifacts.tar.gz ~/Downloads/

Re-running is safe; large files are not overwritten if they already exist.
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
BACKEND_DIR = PROJECT_ROOT / "backend"
MEGATRON_ROOT = PROJECT_ROOT / "models" / "Megatron"
DECODER_ROOT = PROJECT_ROOT / "models" / "Decoder"
TOKENIZER_PATH = PROJECT_ROOT / "models" / "Tokenizer"

# Match the sys.path that run_backend.sh sets up
for p in (PROJECT_ROOT, MEGATRON_ROOT, DECODER_ROOT, BACKEND_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

OUT_DIR = Path(os.environ.get("KHALA_GATHER_OUT", "/tmp/khala_artifacts")).resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _log(msg: str) -> None:
    print(f"[gather] {msg}", flush=True)


def _json_dump(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))
    _log(f"wrote {path.relative_to(OUT_DIR)} ({path.stat().st_size:,} B)")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _safetensors_save(state_dict, path: Path) -> None:
    """Persist a state_dict with safetensors, preserving Megatron names verbatim.

    Names are NOT remapped here. Remapping happens off-line on the Mac.
    """
    if path.exists():
        _log(f"skip {path.name} (already exists: {path.stat().st_size:,} B)")
        return
    try:
        from safetensors.torch import save_file

        # safetensors requires CPU tensors; cast Megatron's tp-replicated tensors to contiguous
        cpu_sd = {}
        for k, v in state_dict.items():
            if hasattr(v, "detach"):
                cpu_sd[k] = v.detach().contiguous().cpu()
            else:
                # non-tensor entries (rare in inference state_dicts) get dropped here;
                # they'll be visible in *_layout.json so we can deal with them later
                continue
        save_file(cpu_sd, str(path))
    except ImportError:
        _log("safetensors not installed, falling back to torch.save (.pt)")
        import torch as _torch

        alt = path.with_suffix(".pt")
        _torch.save(state_dict, alt)
        path = alt
    _log(f"wrote {path.name} ({path.stat().st_size:,} B)")


def _layout_and_samples(state_dict, layout_path: Path, samples_path: Path, label: str) -> None:
    import torch

    layout = []
    samples = {}
    for name, t in state_dict.items():
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


def _dump_env() -> None:
    import torch

    info = {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": (
            torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        ),
        "device_count": torch.cuda.device_count(),
        "device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "env": {
            k: v for k, v in os.environ.items()
            if k.startswith(("CUDA_", "NCCL_", "MASTER_", "TORCH_", "NVIDIA_"))
        },
    }
    _json_dump(info, OUT_DIR / "env.json")


def _dump_megatron_args() -> None:
    from megatron.training import get_args

    args = get_args()
    simple = {}
    for k, v in vars(args).items():
        try:
            json.dumps(v)
            simple[k] = v
        except (TypeError, ValueError):
            simple[k] = repr(v)
    _json_dump(simple, OUT_DIR / "megatron_args.json")


def _dump_tokenizer_artifacts(backend_worker_module) -> None:
    bw = backend_worker_module

    # Copy the tokenizer dir verbatim (vocab.json, tokenizer.json, etc.)
    dst = OUT_DIR / "tokenizer"
    if not dst.exists():
        shutil.copytree(TOKENIZER_PATH, dst)
        _log(f"copied tokenizer dir to {dst.relative_to(OUT_DIR)}")
    else:
        _log("tokenizer dir already present, skipping copy")

    # Tokenize a few fixed prompts so we can detect tokenizer drift later
    bw.load_text_tokenizer()
    tok = bw.RESOURCES["tokenizer"]
    fixtures = {
        "empty": "",
        "ascii_short": "hello world",
        "english_song": (
            "A serene piano ballad with soft strings, gentle pace, in the style of Erik Satie."
        ),
        "with_specials": f"{bw.BOS}calm jazz{bw.EOM}English{bw.EOM}{bw.BOA}",
        "instrumental": (
            f"{bw.BOS}upbeat electronic{bw.EOM}Instrumental{bw.EOM}"
            f"<|reserved_special_token_60|>{bw.EOT}{bw.BOA}"
        ),
    }
    encoded = {
        name: tok.encode(text, add_special_tokens=False) for name, text in fixtures.items()
    }
    _json_dump({"fixtures": fixtures, "encoded": encoded}, OUT_DIR / "tokenizer_test.json")


def _dump_backbone(backend_worker_module) -> None:
    import torch

    bw = backend_worker_module
    bw.load_backbone("")
    backbone = bw.RESOURCES["backbone"]
    inner = getattr(backbone, "module", backbone)
    sd = inner.state_dict()

    _layout_and_samples(
        sd, OUT_DIR / "backbone_layout.json", OUT_DIR / "backbone_samples.json", "backbone"
    )
    _safetensors_save(sd, OUT_DIR / "backbone.safetensors")

    # Embedding behaviour goldens — covers MultiLayerVocabParallelEmbedding's
    # three input-shape branches.
    embedding = inner.embedding.word_embeddings
    import torch

    from megatron.training import get_args

    args = get_args()
    vocab = args.padded_vocab_size

    torch.manual_seed(42)
    test_dim2 = torch.randint(0, vocab, (2, 16), device="cuda")
    test_dim3_c1 = torch.randint(0, vocab, (2, 16, 1), device="cuda")
    test_dim3_cN = torch.randint(0, vocab, (2, 16, 4), device="cuda")
    # Inject a few -1 (pad) ids so the masking path is also covered
    test_dim2[0, ::3] = -1
    test_dim3_cN[1, ::4, 2] = -1

    with torch.no_grad():
        out_dim2 = embedding(test_dim2)
        out_dim3_c1 = embedding(test_dim3_c1)
        out_dim3_cN = embedding(test_dim3_cN)

    torch.save(
        {"input": test_dim2.cpu(), "output": out_dim2.detach().to(torch.float32).cpu()},
        OUT_DIR / "embedding_test_dim2.pt",
    )
    torch.save(
        {"input": test_dim3_c1.cpu(), "output": out_dim3_c1.detach().to(torch.float32).cpu()},
        OUT_DIR / "embedding_test_dim3_c1.pt",
    )
    torch.save(
        {"input": test_dim3_cN.cpu(), "output": out_dim3_cN.detach().to(torch.float32).cpu()},
        OUT_DIR / "embedding_test_dim3_cN.pt",
    )
    _log("embedding goldens written (dim2, dim3_c1, dim3_cN)")

    # Backbone greedy-generation golden
    try:
        from megatron.core.inference.sampling_params import SamplingParams

        tok = bw.RESOURCES["tokenizer"]
        engine = bw.RESOURCES["backbone_engine"]
        golden_prompt = (
            "A serene piano ballad with soft strings, gentle pace, in the style of Erik Satie."
        )
        prompt_ids = tok.encode(golden_prompt, add_special_tokens=False)
        torch.manual_seed(1283)
        result = engine.generate(
            prompts=[prompt_ids],
            sampling_params=SamplingParams(num_tokens_to_generate=64, temperature=0.0, top_k=1),
        )
        # Result shape varies by Megatron version; serialise whatever we got
        try:
            generated = (
                result[0].generated_tokens
                if hasattr(result[0], "generated_tokens")
                else result[0]
            )
        except Exception:
            generated = result

        torch.save(
            {
                "prompt": golden_prompt,
                "prompt_ids": prompt_ids,
                "generated": generated,
                "sampling": "greedy (temperature=0.0, top_k=1)",
                "seed": 1283,
            },
            OUT_DIR / "golden_backbone_greedy.pt",
        )
        _log("golden_backbone_greedy.pt written")
    except Exception as exc:
        _log(f"WARN: greedy golden capture failed: {exc!r}")


def _dump_superres(backend_worker_module) -> None:
    bw = backend_worker_module
    bw.load_superres("")
    superres = bw.RESOURCES["superres"]
    inner = getattr(superres, "module", superres)
    sd = inner.state_dict()
    _layout_and_samples(
        sd, OUT_DIR / "superres_layout.json", OUT_DIR / "superres_samples.json", "superres"
    )
    _safetensors_save(sd, OUT_DIR / "superres.safetensors")


def _dump_decoder(backend_worker_module) -> None:
    import torch

    bw = backend_worker_module
    bw.load_decoder()
    dac = bw.RESOURCES["decoder"]
    sd = dac.state_dict()
    _layout_and_samples(
        sd, OUT_DIR / "decoder_layout.json", OUT_DIR / "decoder_samples.json", "decoder"
    )
    dst = OUT_DIR / "decoder_weights.pt"
    if not dst.exists():
        torch.save(sd, dst)
        _log(f"wrote {dst.name} ({dst.stat().st_size:,} B)")
    # Also stash the yaml config beside the weights
    src_yaml = DECODER_ROOT / "dac_rvq_1024_64_golden.yaml"
    shutil.copy(src_yaml, OUT_DIR / "decoder_config.yaml")


def _summary() -> None:
    lines = ["Khala CUDA artifact gather", f"output dir: {OUT_DIR}", ""]
    total = 0
    for path in sorted(OUT_DIR.rglob("*")):
        if path.is_file():
            size = path.stat().st_size
            total += size
            sha = ""
            if path.suffix in (".safetensors", ".pt"):
                sha = _sha256(path)[:16]
            lines.append(f"  {path.relative_to(OUT_DIR)!s:60s} {size:>15,} B  {sha}")
    lines += ["", f"total: {total:,} bytes ({total/1e9:.2f} GB)"]
    summary = "\n".join(lines)
    (OUT_DIR / "GATHER_INFO.txt").write_text(summary)
    print(summary)


def _bootstrap_megatron() -> None:
    """Mimic backend_worker.main()'s init path, but stop before uvicorn."""
    import backend_worker as bw
    from megatron.training.initialize import initialize_megatron

    # Mirror run_backend.sh's MEGATRON_ARGS, minus the flags we don't need for
    # introspection (--enable-cuda-graph, --flash-decode, --stream).
    injected = [
        "--worker-port", "0",
        "--runtime-mode", "one_shot",
        "--seed", "1283",
        "--tensor-model-parallel-size", "1",
        "--pipeline-model-parallel-size", "1",
        "--tokenizer-type", "NullTokenizer",
        "--norm-epsilon", "1e-6",
        "--num-tokens-to-generate", "23552",
        "--inference-max-seq-length", "25600",
        "--bf16",
    ]
    sys.argv = [sys.argv[0]] + injected

    bw.patch_language_model_embedding()

    first_backbone = next(iter(bw.BACKBONE_MODELS.values()))
    sys.argv.extend(["--load", first_backbone["path"]])
    sys.argv.extend(["--vocab-size", str(first_backbone["vocab_size"])])
    sys.argv.append("--use-checkpoint-args")

    _log("calling initialize_megatron(...)")
    initialize_megatron(
        extra_args_provider=bw.add_worker_args,
        args_defaults={
            "no_load_rng": True,
            "no_load_optim": True,
            "micro_batch_size": 1,
            "exit_on_missing_checkpoint": True,
        },
    )
    _log("Megatron initialised")
    return bw


def main() -> int:
    t_start = time.time()
    _log(f"OUT_DIR = {OUT_DIR}")

    _dump_env()

    bw = _bootstrap_megatron()

    _dump_megatron_args()
    _dump_tokenizer_artifacts(bw)
    _dump_backbone(bw)
    _dump_superres(bw)
    _dump_decoder(bw)

    _summary()
    _log(f"DONE in {time.time()-t_start:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
