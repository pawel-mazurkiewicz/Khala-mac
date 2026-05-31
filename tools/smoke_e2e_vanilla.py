"""End-to-end vanilla smoke: backbone -> super-res -> DAC decode -> wav, on CPU/MPS.
Loads the REAL backbone + superres + decoder (heavy). Run:
  KHALA_DEVICE=cpu KHALA_BACKEND=vanilla .venv-mac/bin/python tools/smoke_e2e_vanilla.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "backend"))

import numpy as np  # noqa: E402


def main() -> int:
    import backend_worker as bw

    assert bw.USE_VANILLA, "set KHALA_BACKEND=vanilla"
    bw.load_backbone("")
    bw.load_superres("")
    bw.load_decoder()

    # Tiny synthetic prompt: a few text ids; ask for a very short clip.
    prompt_ids = [bw.VQ0_START_ID - 5, bw.VQ0_START_ID - 4, bw.VQ0_START_ID - 3]
    _, backbone_tokens = bw.generate_backbone(prompt_ids, top_k=1, temperature=0.0, duration=0)
    # Need an even count of q0/q1 tokens for super-res; trim to a small even length.
    n = max(2, (len(backbone_tokens) // 2) * 2)
    backbone_tokens = backbone_tokens[:n]
    print(f"[smoke] backbone produced {len(backbone_tokens)} tokens")

    audio_tokens = bw.generate_superres(prompt_ids, backbone_tokens, top_k=50)
    print(f"[smoke] superres audio_tokens shape={tuple(audio_tokens.shape)}")

    wav_path = os.path.join(bw.OUTPUT_DIR, "smoke_e2e_vanilla.wav")
    bw.decode_to_wav(audio_tokens, wav_path)
    assert os.path.exists(wav_path) and os.path.getsize(wav_path) > 0
    print(f"[smoke] wrote {wav_path} ({os.path.getsize(wav_path)} B) — E2E OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
