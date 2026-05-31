"""Real-prompt vanilla generation: mirrors the worker's run_generation flow end-to-end
on MPS/CPU, using a PROPERLY composed + tokenized prompt (unlike smoke_e2e_vanilla.py,
which fed synthetic token ids and therefore produced garbage audio).

Flow: load_text_tokenizer -> prepare_prompt_ids (BOS/metadata/lyrics/duration/BOA) ->
generate_backbone -> generate_superres -> decode_to_wav.

Requires KHALA_BACKEND=vanilla in the environment; device is auto-selected (MPS on
Apple Silicon, else CPU). Run `--help` for the full argument reference + examples:
  KHALA_BACKEND=vanilla .venv-mac/bin/python -u tools/generate_vanilla.py --help

Quick start (shortest instrumental Pop clip, ~20s):
  KHALA_BACKEND=vanilla .venv-mac/bin/python -u tools/generate_vanilla.py

Gotchas worth knowing up front:
  * --language takes a full NAME, not an ISO code: "English" (not "en"), "Instrumental", ...
  * --language Instrumental omits the lyrics section, so --lyrics is ignored.
  * --duration is a length BUCKET index (a reserved token slot), NOT seconds.
  * The metadata slot is filled by --description, else --tags, else --genre (first non-empty).
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

# Mirror backend_worker.GENRE_OPTIONS / LANGUAGE_OPTIONS (kept local so --help stays
# instant and doesn't require importing the worker). Language is a closed, trained set;
# genre is a canonical list but free text is also accepted (it is just metadata text).
GENRE_CHOICES = [
    "Pop", "Rock", "R&B", "Hip-Hop", "Electronic", "Jazz", "Classical", "Folk",
    "Country", "Metal", "Latin", "Reggae", "Blues", "Funk", "Soul", "Indie",
    "Alternative", "Dance", "Acoustic",
]
LANGUAGE_CHOICES = ["Chinese", "English", "Japanese", "Korean", "Cantonese", "Instrumental"]

EXAMPLES = r"""
examples (every invocation needs KHALA_BACKEND=vanilla in the environment):

  # shortest instrumental Pop clip (~20s), default sampling
  KHALA_BACKEND=vanilla .venv-mac/bin/python -u tools/generate_vanilla.py

  # instrumental Jazz, longer bucket, custom output name
  KHALA_BACKEND=vanilla .venv-mac/bin/python -u tools/generate_vanilla.py \
      --genre Jazz --language Instrumental --duration 4 --out jazz.wav

  # sung English track WITH lyrics ('\n' starts a new line)
  KHALA_BACKEND=vanilla .venv-mac/bin/python -u tools/generate_vanilla.py \
      --genre Pop --language English --duration 2 \
      --lyrics "Walking in the city light\nChasing shadows through the night"

  # drive the style via tags/description instead of --genre
  KHALA_BACKEND=vanilla .venv-mac/bin/python -u tools/generate_vanilla.py \
      --language Instrumental --tags "ambient, dreamy" \
      --description "slow evolving synth pad, no drums"

  # deterministic (greedy) run for reproducible debugging
  KHALA_BACKEND=vanilla .venv-mac/bin/python -u tools/generate_vanilla.py \
      --temperature 0 --top-k-bb 1 --seed 123
"""


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
    ap = argparse.ArgumentParser(
        prog="generate_vanilla.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Generate a track end-to-end with the vanilla (de-Megatron) pipeline on "
            "MPS/CPU, mirroring the worker's run_generation flow.\n"
            "Requires KHALA_BACKEND=vanilla; device is auto-selected (MPS on Apple "
            "Silicon, else CPU)."
        ),
        epilog=EXAMPLES,
    )

    prompt = ap.add_argument_group("prompt content")
    prompt.add_argument(
        "--genre", default="Pop", metavar="GENRE",
        help="Style label; fills the metadata slot when --tags/--description are empty. "
             "Canonical: " + ", ".join(GENRE_CHOICES) + " (free text also accepted). "
             "Default: %(default)s")
    prompt.add_argument(
        "--language", default="Instrumental", choices=LANGUAGE_CHOICES,
        help="Full language NAME, not an ISO code (use 'English', not 'en'). "
             "'Instrumental' omits the lyrics section. Default: %(default)s")
    prompt.add_argument(
        "--lyrics", default="", metavar="TEXT",
        help=r"Lyrics; '\n' separates lines. IGNORED when --language is Instrumental. "
             "Default: empty")
    prompt.add_argument(
        "--tags", default="", metavar="TEXT",
        help="Free-text style tags (e.g. 'ambient, dreamy'). Overrides --genre in the "
             "metadata slot. Default: empty")
    prompt.add_argument(
        "--description", default="", metavar="TEXT",
        help="Free-text description. Highest-priority metadata slot (overrides --tags "
             "and --genre). Default: empty")
    prompt.add_argument(
        "--duration", type=int, default=0, metavar="N",
        help="Length BUCKET index (a reserved duration-token slot), NOT seconds. Larger "
             "=> longer; N=0 is ~20s in practice, worker/UI default is 2. Default: %(default)s")

    sampling = ap.add_argument_group("sampling")
    sampling.add_argument(
        "--top-k-bb", type=int, default=50, metavar="K",
        help="Backbone top-k. K=1 (or --temperature 0) = greedy. Default: %(default)s")
    sampling.add_argument(
        "--top-k-sr", type=int, default=10, metavar="K",
        help="Super-res top-k. Default: %(default)s")
    sampling.add_argument(
        "--temperature", type=float, default=1.0, metavar="T",
        help="Softmax temperature; 0 = greedy. Default: %(default)s")
    sampling.add_argument(
        "--seed", type=int, default=42, metavar="N",
        help="RNG seed (torch / numpy / mps). Default: %(default)s")

    ap.add_argument(
        "--out", default="vanilla_real_prompt.wav", metavar="FILE",
        help="Output WAV filename, written under the worker OUTPUT_DIR. Default: %(default)s")
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
