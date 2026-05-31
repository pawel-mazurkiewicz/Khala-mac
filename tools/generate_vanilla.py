"""Real-prompt vanilla generation: mirrors the worker's run_generation flow end-to-end
on MPS/CPU, using a PROPERLY composed + tokenized prompt (unlike smoke_e2e_vanilla.py,
which fed synthetic token ids and therefore produced garbage audio).

Flow: load_text_tokenizer -> prepare_prompt_ids (BOS/metadata/lyrics/duration/BOA) ->
generate_backbone -> generate_superres -> decode_to_wav.

Run (MPS auto-selected on Apple Silicon):
  KHALA_BACKEND=vanilla .venv-mac/bin/python -u tools/generate_vanilla.py
Options:
  --genre Pop --language Instrumental --duration 0 --lyrics "" --seed 42
  --top-k-bb 50 --top-k-sr 10 --temperature 1.0 --out NAME.wav
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "backend"))

import numpy as np  # noqa: E402
import torch  # noqa: E402


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    mps = getattr(torch, "mps", None)
    if mps is not None and hasattr(mps, "manual_seed"):
        try:
            mps.manual_seed(seed)
        except Exception:
            pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--genre", default="Pop")
    ap.add_argument("--language", default="Instrumental")
    ap.add_argument("--tags", default="")
    ap.add_argument("--description", default="")
    ap.add_argument("--lyrics", default="")
    ap.add_argument("--duration", type=int, default=0)
    ap.add_argument("--top-k-bb", type=int, default=50)
    ap.add_argument("--top-k-sr", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="vanilla_real_prompt.wav")
    args = ap.parse_args()

    import backend_worker as bw

    assert bw.USE_VANILLA, "set KHALA_BACKEND=vanilla"
    print(f"[gen] device={bw.DEVICE} backend=vanilla seed={args.seed}")
    _seed_everything(args.seed)

    # --- build a REAL prompt exactly like the worker does ---
    bw.load_text_tokenizer()
    backbone_prompt_text, superres_prompt_text = bw.prepare_prompt_texts(
        genre=args.genre, language=args.language, tags=args.tags,
        description=args.description, duration=args.duration, lyrics=args.lyrics,
        superres_text_mode="same_as_backbone",
    )
    print(f"[gen] backbone prompt text: {backbone_prompt_text!r}")
    backbone_prompt_ids, superres_prompt_ids = bw.prepare_prompt_ids(
        genre=args.genre, language=args.language, tags=args.tags,
        description=args.description, duration=args.duration, lyrics=args.lyrics,
        superres_text_mode="same_as_backbone",
    )
    print(f"[gen] backbone_prompt_ids ({len(backbone_prompt_ids)}): {backbone_prompt_ids}")

    # --- backbone ---
    bw.load_backbone("")
    t0 = time.perf_counter()
    _, backbone_tokens = bw.generate_backbone(
        prompt_ids=backbone_prompt_ids, top_k=args.top_k_bb,
        temperature=args.temperature, duration=args.duration,
    )
    print(f"[gen] backbone: {len(backbone_tokens)} tokens in {time.perf_counter()-t0:.1f}s; "
          f"first 8: {backbone_tokens[:8]}")

    # --- super-res ---
    bw.load_superres("")
    t0 = time.perf_counter()
    audio_tokens = bw.generate_superres(
        superres_prompt_ids=superres_prompt_ids, backbone_tokens=backbone_tokens,
        top_k=args.top_k_sr,
    )
    audio_tokens = audio_tokens.cpu()
    print(f"[gen] superres audio_tokens shape={tuple(audio_tokens.shape)} "
          f"in {time.perf_counter()-t0:.1f}s; "
          f"finite={bool(torch.isfinite(audio_tokens.float()).all())} "
          f"min={int(audio_tokens.min())} max={int(audio_tokens.max())}")

    # --- decode ---
    bw.load_decoder()
    wav_path = os.path.join(bw.OUTPUT_DIR, args.out)
    t0 = time.perf_counter()
    bw.decode_to_wav(audio_tokens, wav_path)
    print(f"[gen] decoded in {time.perf_counter()-t0:.1f}s -> {wav_path} "
          f"({os.path.getsize(wav_path)} B)")

    # --- waveform sanity ---
    try:
        import soundfile as sf
        wav, sr = sf.read(wav_path)
        w = np.asarray(wav, dtype=np.float64)
        rms = float(np.sqrt(np.mean(w ** 2)))
        print(f"[gen] wav: shape={w.shape} sr={sr} dur={w.shape[0]/sr:.1f}s "
              f"finite={bool(np.isfinite(w).all())} rms={rms:.4f} "
              f"peak={float(np.max(np.abs(w))):.3f}")
    except Exception as exc:  # noqa: BLE001
        print(f"[gen] (waveform sanity skipped: {exc})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
