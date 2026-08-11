# Byte Latent Transformer Project

This repository is an educational path from transformer fundamentals to a small
Byte Latent Transformer (BLT). Phase 2 infrastructure is complete: the project
now has deterministic Java reference code, an Apple-Silicon-native MLX training
path, handwritten Metal kernel candidates, correctness and causal tests,
checkpointing, generation, shape tracing, and reproducible performance reports.

It is important to keep the current boundary precise: the trainable model is a
**byte-level GPT baseline**, not yet a BLT. The expensive transformer still runs
at every byte. Dynamic patches and the local-encoder/global-transformer/
local-decoder hierarchy begin in Stage 6.

Read the durable project references:

- [`PROJECT_ROADMAP.md`](PROJECT_ROADMAP.md) — all seven stages, current status,
  finish lines, and next work.
- [`docs/PHASE2_INFRASTRUCTURE.md`](docs/PHASE2_INFRASTRUCTURE.md) — what Phase 2
  built, how the layers fit together, commands, metrics, and limitations.
- [`notebooks/01_stage_1_blt_research_readme.ipynb`](notebooks/01_stage_1_blt_research_readme.ipynb)
  — research framing and BLT architecture deep dive.

## Current and target architectures

```text
Current trainable baseline
UTF-8 bytes -> byte + position embeddings -> byte-level transformer
            -> 256 next-byte logits at every position

Target BLT
UTF-8 bytes -> local byte encoder -> dynamic latent patches
            -> global patch transformer -> local byte decoder
            -> 256 next-byte logits at every position
```

## Quick start on Apple Silicon

Prerequisites are Python 3.12+, [`uv`](https://docs.astral.sh/uv/), Java 22, and
an Apple Silicon Mac with Metal support.

```bash
make setup
make doctor
make phase2-smoke
```

`make phase2-smoke` runs the Python and Java test suites, overfits the
deterministic tiny byte corpus, saves a checkpoint, traces model-derived kernel
shapes, compares MLX with the tiled Metal candidate in interleaved A/B order,
and measures checkpoint generation.

Useful focused commands:

```bash
make test                   # Python + Java correctness suites
make train-smoke            # tiny deterministic overfit + checkpoint
make benchmark-shapes       # training/prefill/decode kernel evidence
make benchmark-generation   # TTFT, TPOT, throughput, and peak memory
make thermal-soak-short     # plumbing check, not a legitimate thermal study
./gradlew run               # original Java forward-reference demo
```

Generated evidence is written beneath `artifacts/`, which is intentionally
ignored by Git. Each benchmark report contains its model configuration,
environment, workload, sample summaries, and promotion decisions.

## Measurement policy

The MLX implementation is the trusted baseline. A custom Metal kernel may be
promoted only for a specific model-derived shape when:

1. its complete output matches MLX;
2. median speedup is at least 1.05x; and
3. p95 latency does not regress.

Passing on one dtype or shape does not authorize model-wide replacement.
Short smoke runs validate the harness, not sustained thermals or general
performance claims.

## Animated patching lab

[`visualizer/`](visualizer/) contains a separate Gradio learning UI. Enter a
prompt to inspect UTF-8 bytes, toy entropy scores, and explicit patch
boundaries. It visualizes Stage 1 routing concepts; it does not load the trained
byte-GPT or claim the Stage 6 BLT exists.

```bash
cd visualizer
uv sync --python 3.12
uv run python app.py
```

The next product-facing milestone is a simple checkpoint-backed generation UI.
The next architecture milestone after the baseline is stable is a fixed-patch
local/global/local model, followed by causal entropy patching.
