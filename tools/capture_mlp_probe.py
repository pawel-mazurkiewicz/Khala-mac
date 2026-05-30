"""Deep layer-0 probe: capture inputs/outputs of every sub-module so we can pin
down the ~0.4% MLP mismatch (fc1 projection vs activation vs fc2).

Run on the NGC box after capture_goldens.py works. Reuses the same bootstrap.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in (PROJECT_ROOT, PROJECT_ROOT / "models" / "Megatron",
          PROJECT_ROOT / "models" / "Decoder", PROJECT_ROOT / "backend"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

OUT_DIR = Path(os.environ.get("KHALA_GATHER_OUT", "/workspace/khala_gather/out")).resolve()
PROBE_IDS = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597]


def _bootstrap():
    import backend_worker as bw
    from megatron.training.initialize import initialize_megatron
    sys.argv = [sys.argv[0], "--worker-port", "0", "--runtime-mode", "one_shot", "--seed", "1283",
                "--tensor-model-parallel-size", "1", "--pipeline-model-parallel-size", "1",
                "--tokenizer-type", "NullTokenizer", "--norm-epsilon", "1e-6",
                "--num-tokens-to-generate", "23552", "--inference-max-seq-length", "25600", "--bf16"]
    bw.patch_language_model_embedding()
    fb = next(iter(bw.BACKBONE_MODELS.values()))
    sys.argv += ["--load", fb["path"], "--vocab-size", str(fb["vocab_size"]), "--use-checkpoint-args"]
    initialize_megatron(extra_args_provider=bw.add_worker_args,
                        args_defaults={"no_load_rng": True, "no_load_optim": True,
                                       "micro_batch_size": 1, "exit_on_missing_checkpoint": True})
    return bw


def main() -> int:
    import torch
    bw = _bootstrap()
    bw.load_backbone("")
    inner = getattr(bw.RESOURCES["backbone"], "module", bw.RESOURCES["backbone"])
    l0 = inner.decoder.layers[0]

    def cpu(x):
        if isinstance(x, torch.Tensor):
            return x.detach().float().cpu()
        if isinstance(x, (tuple, list)):
            return type(x)(cpu(i) for i in x)
        return x

    cap = {}

    def mk(name):
        def hook(_m, inp, out):
            cap[name + ".in"] = cpu(inp)
            cap[name + ".out"] = cpu(out)
        return hook

    targets = {
        "mlp": l0.mlp,
        "mlp.linear_fc1": l0.mlp.linear_fc1,
        "mlp.linear_fc2": l0.mlp.linear_fc2,
        "attn.linear_proj": l0.self_attention.linear_proj,
        "attn.linear_qkv": l0.self_attention.linear_qkv,
    }
    # try to also hook the activation function module if present
    for attr in ("activation_func", "activation"):
        if hasattr(l0.mlp, attr) and isinstance(getattr(l0.mlp, attr), torch.nn.Module):
            targets[f"mlp.{attr}"] = getattr(l0.mlp, attr)

    handles = [m.register_forward_hook(mk(n)) for n, m in targets.items()]

    ids = torch.tensor([PROBE_IDS], device="cuda")
    pos = torch.arange(len(PROBE_IDS), device="cuda")[None]
    S = len(PROBE_IDS)
    mask = torch.triu(torch.ones(S, S, dtype=torch.bool, device="cuda"), 1)[None, None]
    with torch.no_grad():
        inner.forward_hidden_states(input_ids=ids, position_ids=pos, attention_mask=mask)
    for h in handles:
        h.remove()

    # also stash the mlp submodule structure + fc1 param names for ground truth
    cap["_mlp_repr"] = repr(l0.mlp)
    cap["_fc1_params"] = [n for n, _ in l0.mlp.linear_fc1.named_parameters()]
    torch.save(cap, OUT_DIR / "mlp_probe.pt")
    print("[mlp-probe] wrote mlp_probe.pt; captured:", sorted(k for k in cap if not k.startswith("_")))
    print("[mlp-probe] mlp repr:\n", cap["_mlp_repr"][:800])
    return 0


if __name__ == "__main__":
    sys.exit(main())
