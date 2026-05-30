"""Capture forward-reference goldens from the REAL Megatron backbone.

Run inside the Khala NGC image (Megatron + TransformerEngine importable), after
`bash run_backend.sh` boots at least once. This is the missing piece the minimal
gather deferred: it actually RUNS the model so we can parity-check the vanilla
KhalaModel port on the Mac.

Produces (under $KHALA_GATHER_OUT, default /workspace/khala_gather/out):
  embedding_test_dim2.pt        MultiLayer embedding I/O, [B,S]
  embedding_test_dim3_c1.pt     ... [B,S,1]
  embedding_test_dim3_cN.pt     ... [B,S,4]
  golden_backbone_greedy.pt     64 greedy tokens for a fixed prompt (end-to-end gate)
  forward_trace.pt              FIXED input -> embedding out, layer0 attn/mlp/block
                                outputs, final hidden, logits. Lets us bisect which
                                convention (RoPE half-vs-interleave, SwiGLU gate/up)
                                is wrong if the end-to-end golden mismatches.

Usage:
  export KHALA_GATHER_OUT=/workspace/khala_gather/out
  cd /workspace/Khala && python tools/capture_goldens.py
"""
from __future__ import annotations

import os
import sys
import time
from functools import partial
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
MEGATRON_ROOT = PROJECT_ROOT / "models" / "Megatron"
DECODER_ROOT = PROJECT_ROOT / "models" / "Decoder"
for p in (PROJECT_ROOT, MEGATRON_ROOT, DECODER_ROOT, BACKEND_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

OUT_DIR = Path(os.environ.get("KHALA_GATHER_OUT", "/workspace/khala_gather/out")).resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Fixed, reproducible probe input for the forward trace. Small + deterministic so
# the Mac side replays the exact same ids. Values are arbitrary valid text tokens.
PROBE_IDS = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597]


def _log(m: str) -> None:
    print(f"[goldens] {m}", flush=True)


def _bootstrap_megatron():
    """Same init path as gather_cuda_artifacts._bootstrap_megatron."""
    import backend_worker as bw
    from megatron.training.initialize import initialize_megatron

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
    _log("initialize_megatron(...)")
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


def _to_cpu_f32(x):
    import torch
    if isinstance(x, torch.Tensor):
        return x.detach().to(torch.float32).cpu()
    if isinstance(x, (tuple, list)):
        return type(x)(_to_cpu_f32(i) for i in x)
    return x


def _dump_embedding_goldens(inner) -> None:
    import torch
    from megatron.training import get_args

    vocab = get_args().padded_vocab_size
    embedding = inner.embedding.word_embeddings
    torch.manual_seed(42)
    d2 = torch.randint(0, vocab, (2, 16), device="cuda")
    d3c1 = torch.randint(0, vocab, (2, 16, 1), device="cuda")
    d3cN = torch.randint(0, vocab, (2, 16, 4), device="cuda")
    d2[0, ::3] = -1
    d3cN[1, ::4, 2] = -1
    with torch.no_grad():
        o2, o1, oN = embedding(d2), embedding(d3c1), embedding(d3cN)
    torch.save({"input": d2.cpu(), "output": _to_cpu_f32(o2)}, OUT_DIR / "embedding_test_dim2.pt")
    torch.save({"input": d3c1.cpu(), "output": _to_cpu_f32(o1)}, OUT_DIR / "embedding_test_dim3_c1.pt")
    torch.save({"input": d3cN.cpu(), "output": _to_cpu_f32(oN)}, OUT_DIR / "embedding_test_dim3_cN.pt")
    _log("embedding goldens written")


def _dump_forward_trace(inner) -> None:
    import torch

    S = len(PROBE_IDS)
    input_ids = torch.tensor([PROBE_IDS], device="cuda")            # [1, S]
    position_ids = torch.arange(S, device="cuda")[None, :]          # [1, S]
    # Megatron causal mask: True = disallowed. [1,1,S,S].
    attn_mask = torch.triu(torch.ones(S, S, dtype=torch.bool, device="cuda"), diagonal=1)[None, None]

    captured: dict = {}

    def hook(name):
        def fn(_m, _inp, out):
            captured[name] = _to_cpu_f32(out)
        return fn

    handles = []
    try:
        handles.append(inner.embedding.register_forward_hook(hook("embedding_out")))
        l0 = inner.decoder.layers[0]
        handles.append(l0.register_forward_hook(hook("layer0_out")))
        handles.append(l0.self_attention.register_forward_hook(hook("layer0_attn_out")))
        handles.append(l0.mlp.register_forward_hook(hook("layer0_mlp_out")))
        if len(inner.decoder.layers) > 1:
            handles.append(inner.decoder.layers[1].register_forward_hook(hook("layer1_out")))
    except Exception as exc:  # noqa: BLE001
        _log(f"WARN registering hooks: {exc!r}")

    with torch.no_grad():
        hidden = inner.forward_hidden_states(
            input_ids=input_ids, position_ids=position_ids, attention_mask=attn_mask
        )  # [S, B, H] (Megatron seq-first)
    for h in handles:
        h.remove()

    output_weight = (
        inner.shared_embedding_or_output_weight()
        if getattr(inner, "share_embeddings_and_output_weights", False)
        else inner.output_layer.weight
    )
    hidden_bsh = hidden.transpose(0, 1).contiguous()  # [B, S, H]
    logits = torch.matmul(hidden_bsh.to(output_weight.dtype), output_weight.t()).float()

    torch.save(
        {
            "probe_ids": PROBE_IDS,
            "layout_note": "hidden saved [B,S,H]; embedding/layer hook outputs are raw Megatron (seq-first or tuples)",
            "input_ids": input_ids.cpu(),
            "position_ids": position_ids.cpu(),
            "hidden_final_bsh": _to_cpu_f32(hidden_bsh),
            "logits_bsv": _to_cpu_f32(logits),
            **captured,
        },
        OUT_DIR / "forward_trace.pt",
    )
    _log(f"forward_trace.pt written (captured: {sorted(captured)})")


def _dump_greedy_golden(bw, inner) -> None:
    import torch
    from megatron.core.inference.sampling_params import SamplingParams

    try:
        tok = bw.RESOURCES["tokenizer"]
        engine = bw.RESOURCES["backbone_engine"]
        prompt = "A serene piano ballad with soft strings, gentle pace, in the style of Erik Satie."
        prompt_ids = tok.encode(prompt, add_special_tokens=False)
        torch.manual_seed(1283)
        result = engine.generate(
            prompts=[prompt_ids],
            sampling_params=SamplingParams(num_tokens_to_generate=64, temperature=0.0, top_k=1),
        )
        try:
            generated = result[0].generated_tokens if hasattr(result[0], "generated_tokens") else result[0]
        except Exception:
            generated = result
        torch.save(
            {"prompt": prompt, "prompt_ids": prompt_ids, "generated": generated,
             "sampling": "greedy(temp=0,top_k=1)", "seed": 1283},
            OUT_DIR / "golden_backbone_greedy.pt",
        )
        _log("golden_backbone_greedy.pt written")
    except Exception as exc:  # noqa: BLE001
        _log(f"WARN greedy golden failed: {exc!r}")


def main() -> int:
    t0 = time.time()
    _log(f"OUT_DIR = {OUT_DIR}")
    bw = _bootstrap_megatron()
    bw.load_text_tokenizer()
    bw.load_backbone("")
    backbone = bw.RESOURCES["backbone"]
    inner = getattr(backbone, "module", backbone)

    _dump_embedding_goldens(inner)
    _dump_forward_trace(inner)
    _dump_greedy_golden(bw, inner)

    _log(f"DONE in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
