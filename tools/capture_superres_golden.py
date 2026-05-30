"""Capture a forward golden from the REAL super-resolution model, so we can parity
-check KhalaModel(superres) the same way we did the backbone.

Super-res takes multi-codebook input [B, S, C]; we probe forward_hidden_states with
a fixed [1, S, 4] input and dump hidden + logits. Run on the NGC box.
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
# Deterministic multi-codebook probe, [S, C=4], valid ids well under the superres vocab.
PROBE = [[1, 2, 3, 4], [5, 8, 13, 21], [34, 55, 89, 144], [233, 377, 610, 987],
         [11, 12, 13, 14], [101, 202, 303, 404], [7, 70, 700, 7000], [9, 99, 999, 9999],
         [2, 4, 6, 8], [3, 6, 9, 12], [5, 10, 15, 20], [123, 234, 345, 456]]


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
    bw.load_superres("")
    inner = getattr(bw.RESOURCES["superres"], "module", bw.RESOURCES["superres"])
    ids = torch.tensor([PROBE], device="cuda")          # [1, S, 4]
    S = ids.shape[1]
    pos = torch.arange(S, device="cuda")[None]
    # Super-res is NON-causal: padding mask [1,1,1,S] (True=masked). No padding here.
    mask = torch.zeros(1, 1, 1, S, dtype=torch.bool, device="cuda")
    with torch.no_grad():
        hidden = inner.forward_hidden_states(input_ids=ids, position_ids=pos, attention_mask=mask)
    ow = (inner.shared_embedding_or_output_weight()
          if getattr(inner, "share_embeddings_and_output_weights", False)
          else inner.output_layer.weight)
    hidden_bsh = hidden.transpose(0, 1).contiguous()
    logits = torch.matmul(hidden_bsh.to(ow.dtype), ow.t()).float()
    torch.save({"probe": PROBE, "input_ids": ids.cpu(),
                "hidden_final_bsh": hidden_bsh.float().cpu(), "logits_bsv": logits.cpu()},
               OUT_DIR / "superres_forward_trace.pt")
    print("[superres-golden] wrote superres_forward_trace.pt", tuple(logits.shape))
    return 0


if __name__ == "__main__":
    sys.exit(main())
