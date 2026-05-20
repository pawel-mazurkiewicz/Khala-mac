# GPU Box Runbook — Gather Artifacts for the Apple Silicon Port

You need one short rental on a Linux+NVIDIA box with at least 24 GB VRAM
(RTX 4090 / A10G / L4 / A100 all fine). Plan ~30–60 minutes of wall time
once the docker image is pulled. You will end with one tarball that contains
everything Phase 1+ needs — no further GPU access required afterward.

## 0. Provision

Anything works: Vast.ai, RunPod, Lambda, Paperspace, a friend with a 4090.
You need:
- NVIDIA driver + container toolkit
- ~80 GB free disk (model weights + safetensors copy + tarball)
- Outbound network for pulling the image and the HF checkpoints

## 1. Pull the image, clone the repo, mount your code

```bash
docker pull ghcr.io/davidliujiafeng/khala-env:ngc25.02-node24

# Mount a host dir so artifacts survive container exit. /scratch is a
# convention; substitute whatever your host uses.
mkdir -p /scratch/khala
docker run --gpus all -it --rm \
    --name khala-gather \
    -v /scratch/khala:/host \
    -p 30869:30869 -p 8889:8889 \
    ghcr.io/davidliujiafeng/khala-env:ngc25.02-node24
```

Inside the container:

```bash
cd /workspace
git clone <YOUR FORK URL> Khala     # or rsync from your laptop
cd Khala
```

If you want to gather using the **patched** version of `backend_worker.py`
(it works the same on CUDA, just gates the device-specific bits), make sure
your fork has the `core/device_utils.py` + `tools/gather_cuda_artifacts.py`
files in place. If you cloned upstream Khala, just `scp` those two files
plus the edits to `backend_worker.py` over from your Mac.

## 2. Download the checkpoints

```bash
mkdir -p checkpoints
hf download liujiafeng/Khala-MusicGeneration-v1.0 --local-dir checkpoints
```

Expect 30–60 GB on disk depending on what shards exist.

## 3. Sanity: confirm `run_backend.sh` boots at least once

This proves Megatron init works on this host before you spend time gathering.

```bash
cd backend
bash run_backend.sh --gpus 0 --runtime-mode one_shot
# wait for "  X / 1 workers ready", then Ctrl-C and:
bash run_backend.sh stop
```

If that fails, fix the env before continuing. The gather script reuses the
exact same init machinery.

## 4. Install the one extra Python dep we need

The NGC image ships everything Megatron needs but probably not `safetensors`:

```bash
pip install --break-system-packages safetensors
```

## 5. Run the gather script

```bash
cd /workspace/Khala
export KHALA_GATHER_OUT=/host/khala_artifacts
python tools/gather_cuda_artifacts.py 2>&1 | tee /host/gather.log
```

Expect ~5–10 minutes. Watch for `[gather] DONE in N.Ns`.

The script is idempotent — if it crashes partway through, re-run and it will
skip the large `.safetensors` files already written.

## 6. Tar and ship home

```bash
cd /host
tar -cvf khala_artifacts.tar khala_artifacts          # no gzip; weights are already compact
ls -lh khala_artifacts.tar
sha256sum khala_artifacts.tar > khala_artifacts.tar.sha256
```

On your Mac:

```bash
scp <gpubox>:/scratch/khala/khala_artifacts.tar ~/Downloads/
scp <gpubox>:/scratch/khala/khala_artifacts.tar.sha256 ~/Downloads/
sha256sum -c ~/Downloads/khala_artifacts.tar.sha256

mkdir -p /Users/pawelma/code/ai/Khala/_cuda_artifacts
tar -xvf ~/Downloads/khala_artifacts.tar -C /Users/pawelma/code/ai/Khala/_cuda_artifacts
```

`_cuda_artifacts/` is already in `.gitignore`-territory (add it if not).

## 7. (Optional) capture one real end-to-end generation

Before tearing down the box, do **one** full generation through the existing
backend so we get a real (prompt → backbone codes → super-res codes → wav)
artifact for end-to-end parity testing later.

```bash
cd /workspace/Khala/backend
bash run_backend.sh --gpus 0 --runtime-mode one_shot
# in another shell inside the container:
curl -X POST http://127.0.0.1:8889/generate \
    -H 'Content-Type: application/json' \
    -d '{
        "prompt": "A serene piano ballad with soft strings, gentle pace, in the style of Erik Satie.",
        "lyrics": "",
        "duration": 30,
        "language": "Instrumental"
    }'
# wait for the job to finish, then:
cp -r /workspace/Khala/backend/generated_audio /host/khala_real_gen_audio
tar -uvf /host/khala_artifacts.tar -C /host khala_real_gen_audio
bash run_backend.sh stop
```

## What you'll bring back (file map)

```
khala_artifacts/
├── GATHER_INFO.txt              <- read this first; lists every file with sha256
├── env.json                     <- torch/cuda versions, GPU model
├── megatron_args.json           <- canonical hyperparameters
│
├── backbone_layout.json         <- tensor names + shapes + dtypes
├── backbone_samples.json        <- numerical fingerprints (first-8 + stats per tensor)
├── backbone.safetensors         <- THE BIG ONE: full backbone weights, Megatron naming
│
├── superres_layout.json
├── superres_samples.json
├── superres.safetensors
│
├── decoder_layout.json
├── decoder_samples.json
├── decoder_weights.pt           <- DAC is already vanilla PyTorch; .pt is fine
├── decoder_config.yaml
│
├── tokenizer/                   <- byte-identical copy of models/Tokenizer/
├── tokenizer_test.json          <- fixed prompts → token IDs (drift detector)
│
├── embedding_test_dim2.pt       <- input/output for the (B,S) embedding path
├── embedding_test_dim3_c1.pt    <- input/output for the (B,S,1) embedding path
├── embedding_test_dim3_cN.pt    <- input/output for the (B,S,4) multi-codebook path
│
└── golden_backbone_greedy.pt    <- 64 greedy-decoded tokens for the canonical prompt
```

## How each artifact is used downstream

| Artifact                          | Used for                                                   | Phase  |
| --------------------------------- | ---------------------------------------------------------- | ------ |
| `megatron_args.json`              | Build the vanilla `KhalaConfig` (hidden_size, layers, etc.)| 1      |
| `backbone_layout.json`            | Write the Megatron→HF tensor-name remap                    | 1      |
| `backbone_samples.json`           | Verify the remap preserves bytes (numerical fingerprint)   | 1      |
| `backbone.safetensors`            | Source of weights for the converter                        | 1      |
| `superres_*.{json,safetensors}`   | Same as backbone, for super-res                            | 1      |
| `decoder_weights.pt` + config     | Drop straight into the Mac runtime (already portable)      | 0      |
| `tokenizer/` + `tokenizer_test`   | Detect tokenizer drift across `transformers` versions      | 2      |
| `embedding_test_dim*.pt`          | Validate the rewritten `MultiLayerEmbedding` against CUDA  | 2      |
| `golden_backbone_greedy.pt`       | **Top parity gate**: same tokens, byte-identical, on Mac   | 4      |

If the Mac-side rewrite reproduces `golden_backbone_greedy.pt` to within
tolerance, we know the backbone port is correct. If `embedding_test_dim*.pt`
match, the embedding port is correct. Each artifact gates one phase.

## Storage budget (estimates)

Realistic sizes for a backbone in the 3–7B param range plus a ~1–3B super-res:

| File                       | Approx size    |
| -------------------------- | -------------- |
| backbone.safetensors       |  6 – 14 GB     |
| superres.safetensors       |  2 –  6 GB     |
| decoder_weights.pt         |  ~520 MB       |
| everything else            |  < 20 MB       |
| **total tarball**          |  **8 – 21 GB** |

scp at 100 Mbps: 10–30 minutes.

## After you have the tarball

You're done with the GPU. Everything else — the Megatron→HF remap, the
vanilla `KhalaModel`, the worker rewrite, the parity gates — happens on your
Mac with no further CUDA access. We pick up at Phase 1 with
`backbone.safetensors` + `megatron_args.json` as the inputs to a
`tools/convert_megatron_to_hf.py` we'll write next.
