"""Phase-0 smoke test: instantiate the DAC RVQ decoder on the local device
(MPS on Apple Silicon, CUDA elsewhere) and decode random RVQ codes into a wav.

This proves:
- the decoder code path is device-agnostic with the patches we make
- weight_norm Conv1d / ConvTranspose1d work on the active backend
- the @torch.jit.script `snake` activation compiles and runs
- omegaconf can load the canonical decoder yaml
- soundfile can write the resulting waveform

It does NOT need any pretrained checkpoint; weights are random. The output wav
will be noise, but a successful run end-to-end is the toolchain-health signal.

Usage:
    python tools/decode_smoke_mps.py            # auto-detect device
    KHALA_DEVICE=mps python tools/decode_smoke_mps.py
    KHALA_DEVICE=cpu python tools/decode_smoke_mps.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DECODER_DIR = PROJECT_ROOT / "models" / "Decoder"

# Make the decoder modules importable without editing them
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(DECODER_DIR))

import numpy as np
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from core.device_utils import (  # noqa: E402
    clear_memory,
    device_str,
    get_device,
    recommended_dtype,
)


def main() -> int:
    device = get_device()
    dtype = recommended_dtype(device)
    print(f"[smoke] device={device_str()} dtype={dtype}")
    print(f"[smoke] torch={torch.__version__} mps_built={torch.backends.mps.is_built()}")

    # Import DacRVQ lazily so we can patch the surrounding env first
    from dac_rvq import DacRVQ  # noqa: E402

    cfg_path = DECODER_DIR / "dac_rvq_1024_64_golden.yaml"
    cfg = OmegaConf.load(cfg_path)
    # OmegaConf needs explicit resolution for the ${encoder.d_latent} interpolation
    cfg_resolved = OmegaConf.to_container(cfg, resolve=True)
    print(f"[smoke] config: {cfg_resolved}")

    t0 = time.time()
    # DacRVQ subclasses L.LightningModule which subclasses nn.Module; this works on
    # any device. We construct in fp32 then cast.
    model = DacRVQ(cfg_resolved)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[smoke] decoder params: {n_params/1e6:.2f}M, construct time: {time.time()-t0:.2f}s")

    # Move to device
    t0 = time.time()
    model = model.to(device)
    if device.type != "cpu":
        # Only cast on accelerators; fp16 on CPU is slow and not useful for verification.
        model = model.to(dtype)
    print(f"[smoke] moved to {device}/{dtype} in {time.time()-t0:.2f}s")

    # Synthesize random codebook indices: (num_quantizers, B, T)
    # The DAC RVQ expects shape (Q, B, T) per ResidualVectorQuantization.decode.
    num_quantizers = cfg_resolved["quantizer"]["num_quantizers"]
    codebook_size = cfg_resolved["quantizer"]["codebook_size"]
    B = 1
    T = 64  # ~0.5s of latent frames -> ~1s of audio after upsampling
    g = torch.Generator(device="cpu").manual_seed(0)
    codes = torch.randint(
        low=0, high=codebook_size, size=(num_quantizers, B, T), generator=g, dtype=torch.long
    ).to(device)
    print(f"[smoke] codes shape: {tuple(codes.shape)} on {codes.device}")

    t0 = time.time()
    with torch.no_grad():
        audio = model.decode(codes)
    if device.type != "cpu":
        # ensure all async work is finished before timing
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
    print(f"[smoke] decode wall time: {time.time()-t0:.2f}s, audio shape: {tuple(audio.shape)} dtype={audio.dtype}")

    # Cast back to fp32 and write a wav as proof-of-life
    audio_cpu = audio.detach().to(torch.float32).cpu().squeeze(0).numpy()  # (channels, T)
    audio_cpu = audio_cpu.T  # soundfile wants (T, channels)
    out_path = PROJECT_ROOT / "tools" / "smoke_output.wav"
    import soundfile as sf

    sf.write(str(out_path), audio_cpu, samplerate=44100)
    print(f"[smoke] wrote {out_path} ({audio_cpu.shape}, peak={np.abs(audio_cpu).max():.4f})")

    clear_memory(device)
    print("[smoke] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
