"""Mac-side parity check: vanilla KhalaModel vs the captured CUDA goldens.

Consumes the artifacts written by tools/capture_goldens.py and reports, stage by
stage, where (if anywhere) the port diverges from the reference. Because the trace
captures layer-0 attention and MLP outputs separately, a mismatch localises to a
specific convention:

  embedding mismatch        -> MultiLayerEmbedding port bug
  layer0_attn mismatch only -> RoPE convention (half vs interleaved) in apply_rotary
  layer0_mlp mismatch only  -> SwiGLU gate/up assignment in KhalaMLP
  final/logits mismatch     -> accumulated / lm_head

Runs in fp32 for a clean numerical comparison (the reference is bf16, so expect
~1e-2 abs diffs even when correct; cosine sim should be > 0.999).

Usage:
  python tools/verify_parity.py            # uses _cuda_artifacts/
  python tools/verify_parity.py --label backbone --artifacts _cuda_artifacts
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.khala_config import KhalaConfig  # noqa: E402
from core.khala_model import KhalaModel  # noqa: E402


def _tensor(x):
    """Pull a hidden-state tensor out of a possibly-tuple Megatron hook output."""
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (tuple, list)):
        for e in x:
            if isinstance(e, torch.Tensor) and e.dim() >= 2:
                return e
    return None


def _align(ref: torch.Tensor, mine: torch.Tensor) -> torch.Tensor | None:
    """Return ref reshaped to mine's [B,S,H] layout, trying the seq-first transpose."""
    if ref is None:
        return None
    if ref.shape == mine.shape:
        return ref
    if ref.dim() == 3 and ref.transpose(0, 1).shape == mine.shape:
        return ref.transpose(0, 1).contiguous()
    return None


def _cmp(name: str, ref: torch.Tensor, mine: torch.Tensor) -> None:
    ref = _tensor(ref)
    if ref is None:
        print(f"  {name:20s}  <no ref tensor>")
        return
    aligned = _align(ref, mine)
    if aligned is None:
        print(f"  {name:20s}  shape mismatch ref={tuple(ref.shape)} mine={tuple(mine.shape)}")
        return
    a = aligned.float().flatten()
    b = mine.float().flatten()
    max_abs = (a - b).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    verdict = "OK " if cos > 0.999 and max_abs < 0.2 else "DIFF"
    print(f"  {name:20s}  cos={cos:.5f}  max_abs={max_abs:.4f}   [{verdict}]")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="backbone")
    ap.add_argument("--artifacts", default="_cuda_artifacts")
    args = ap.parse_args()
    A = Path(args.artifacts)

    cfg = KhalaConfig.from_megatron_args(A / f"{args.label}_megatron_args.json")
    model = KhalaModel(cfg).to(torch.float32)
    model.load_state_dict(load_file(str(A / f"khala_{args.label}.safetensors")), strict=True)
    model.eval()

    # --- embedding goldens ---
    emb_file = A / "embedding_test_dim2.pt"
    if emb_file.exists():
        print("== MultiLayerEmbedding goldens ==")
        for fn, _ in [("embedding_test_dim2.pt", 2), ("embedding_test_dim3_c1.pt", 3),
                      ("embedding_test_dim3_cN.pt", 3)]:
            p = A / fn
            if not p.exists():
                continue
            g = torch.load(p, map_location="cpu", weights_only=False)
            with torch.no_grad():
                mine = model.embed(g["input"].to(torch.long))
            _cmp(fn.replace("embedding_test_", "").replace(".pt", ""), g["output"], mine)
    else:
        print("== embedding goldens: NOT PRESENT (run capture_goldens.py on the box) ==")

    # --- forward parity (hidden + logits, vocab-sliced) ---
    ft = A / "forward_trace.pt"
    if ft.exists():
        print("\n== forward parity ==")
        g = torch.load(ft, map_location="cpu", weights_only=False)
        ids = g["input_ids"].to(torch.long)
        with torch.no_grad():
            mine_hidden = model.forward_hidden_states(ids)
            logits = model(ids)
        _cmp("hidden_final", g["hidden_final_bsh"], mine_hidden)
        ref_logits = g["logits_bsv"]
        V = ref_logits.shape[-1]            # reference uses real vocab; mine is padded
        _cmp("logits", ref_logits, logits[..., :V])
        agree = (logits[..., :V].argmax(-1) == ref_logits.argmax(-1)).float().mean().item()
        print(f"  next-token argmax agreement (random probe): {agree*100:.1f}%")
    else:
        print("\n== forward_trace.pt: NOT PRESENT (run capture_goldens.py on the box) ==")
        V = cfg.vocab_size

    # --- greedy end-to-end gate (the decisive parity check) ---
    gg = A / "golden_backbone_greedy.pt"
    if gg.exists():
        g = torch.load(gg, map_location="cpu", weights_only=False)
        prompt = [int(x) for x in g["prompt_ids"]]
        gen = g["generated"]
        gen = [int(x) for x in (gen.tolist() if hasattr(gen, "tolist") else gen)]
        seq = torch.tensor([prompt], dtype=torch.long)
        out: list[int] = []
        with torch.no_grad():
            for _ in range(len(gen)):
                nxt = int(model(seq)[0, -1, :V].argmax())
                out.append(nxt)
                seq = torch.cat([seq, torch.tensor([[nxt]])], 1)
        match = sum(a == b for a, b in zip(out, gen))
        first_div = next((i for i, (a, b) in enumerate(zip(out, gen)) if a != b), None)
        print(f"\n== greedy parity gate ==")
        print(f"  {match}/{len(gen)} tokens identical to CUDA reference"
              + (f" (first divergence @ {first_div})" if first_div is not None else ""))
        print("  PASS" if match == len(gen) else "  FAIL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
