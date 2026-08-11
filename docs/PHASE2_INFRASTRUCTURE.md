# Phase 2 Infrastructure: Tensor and Neural-Network Foundations

Last verified: **August 11, 2026**

## Outcome

Phase 2 is complete as an infrastructure stage. The repository has a correct,
deterministic, reproducible foundation for training and measuring a small
byte-level transformer on an M1 Max MacBook Pro.

Phase 2 does **not** claim that the project has implemented a BLT. Its exit
artifact is a trustworthy byte-GPT baseline and the machinery needed to decide
whether hardware-aware operations should replace MLX defaults. Actual latent
patch routing begins later.

## Why there are two implementation tracks

```text
Java reference track
  explicit Tensor operations -> layers -> causal byte-GPT forward pass
  purpose: understand the mechanics and test the mathematics directly

Python/MLX experimental track
  deterministic data -> MLX byte-GPT -> autodiff/AdamW -> checkpoint/generation
  purpose: train, measure, and iterate natively on Apple Silicon

Hardware-aware candidates
  pure Python reference -> MLX baseline -> handwritten Metal candidates
  purpose: prove correctness first, then promote only measured wins
```

The Java code remains an educational forward reference. MLX supplies array
storage, lazy graph execution, automatic differentiation, and optimizer state;
this project defines the model, initialization, data order, training loop,
checkpoint schema, generation control flow, tracing, benchmarks, and custom
Metal candidates.

## Infrastructure inventory

| Layer | Implemented now | Primary files |
| --- | --- | --- |
| Scalar references | Scalar multiply, matmul, stable softmax, layer norm, GELU | [`python/blt_mlx/foundations/reference.py`](../python/blt_mlx/foundations/reference.py) |
| MLX primitives | Deterministic parameters, eager/compiled GELU, softmax, layer norm | [`python/blt_mlx/foundations/primitives.py`](../python/blt_mlx/foundations/primitives.py) |
| Metal candidates | Scalar multiply, fused bias+GELU, naive matmul, tiled-16 matmul, custom VJPs | [`python/blt_mlx/foundations/metal.py`](../python/blt_mlx/foundations/metal.py) |
| Model | Byte/position embeddings, pre-norm blocks, causal attention, MLP, 256-byte head | [`python/blt_mlx/model.py`](../python/blt_mlx/model.py), [`python/blt_mlx/modules.py`](../python/blt_mlx/modules.py) |
| Determinism | Versioned model/training configs, seeded parameters and batches | [`python/blt_mlx/config.py`](../python/blt_mlx/config.py), [`python/blt_mlx/data.py`](../python/blt_mlx/data.py) |
| Training | Next-byte cross-entropy, compiled AdamW step, finite loss/gradient gates, BPB | [`python/blt_mlx/training.py`](../python/blt_mlx/training.py) |
| Persistence | SafeTensors model/optimizer state plus versioned JSON metadata and resume | [`python/blt_mlx/checkpoint.py`](../python/blt_mlx/checkpoint.py) |
| Generation | Greedy, temperature, top-k, deterministic sampling, raw byte output | [`python/blt_mlx/model.py`](../python/blt_mlx/model.py) |
| Shape tracing | Logical training, prefill, and decode operation shapes | [`python/blt_mlx/shapes.py`](../python/blt_mlx/shapes.py) |
| Performance | Interleaved A/B order, p50/p95, correctness gate, memory, generation and soak reports | [`python/blt_mlx/performance.py`](../python/blt_mlx/performance.py) |
| Java trust layer | Seeded tensors/model construction, known-value and causal JUnit tests | [`src/main/java`](../src/main/java), [`src/test/java`](../src/test/java) |
| Reproducible entrypoints | Pinned Python lock, Gradle wrapper, Make targets | [`pyproject.toml`](../pyproject.toml), [`uv.lock`](../uv.lock), [`gradlew`](../gradlew), [`Makefile`](../Makefile) |

## Default path versus hardware-aware candidates

The default model uses `mlx` matmul and `mlx_compiled` fusion. This is the
trusted path because MLX chooses Apple-aware kernels and participates naturally
in MLX autodiff and compilation.

The optional strategy controls are:

| Setting | Values | Meaning |
| --- | --- | --- |
| `dtype` | `float32`, `float16`, `bfloat16` | Model parameter and activation precision. Custom Metal is deliberately gated to float32/float16. |
| `matmul_strategy` | `mlx`, `metal_naive`, `metal_tiled16` | MLX baseline or a handwritten full-output Metal matrix multiplication. |
| `fusion_strategy` | `mlx_eager`, `mlx_compiled`, `metal_fused` | Separate MLX operations, MLX-compiled GELU, or handwritten bias+GELU fusion. |

These controls make an optimization reversible and attributable. They do not
assume the handwritten implementation is faster.

## Correctness and promotion contract

For kernel evidence, the harness:

1. obtains real matrix shapes from model traces rather than generic teaching
   sizes;
2. warms both candidates and forces lazy MLX work to complete with evaluation
   and synchronization;
3. creates a balanced, deterministic, interleaved A/B order;
4. compares the full candidate output with MLX;
5. records median, p95, dispersion, order, dtype, shape, phase, and peak memory;
6. promotes only when output is valid, median speedup is at least 1.05x, and p95
   does not regress.

The decision belongs to a shape, dtype, candidate, machine, and run—not to the
kernel name in the abstract.

## Service-level objectives and measurements

| Scope | Measurements now | Interpretation |
| --- | --- | --- |
| Mathematical | Full-output tolerance, finite training loss, finite gradients | Correctness gates; failure invalidates speed results. |
| Kernel | Median/p95 latency, speedup, peak MLX memory, promotion decision | Operation-level evidence for traced model shapes. |
| Training | Cross-entropy, bits per byte, per-step latency, active/peak memory | Learning and training-system smoke evidence. |
| Generation | TTFT, TPOT, end-to-end latency, bytes/second, peak memory | Request-level byte-generation evidence; a byte is not called a tokenizer token. |
| Sustained | Windowed request latency, active/cache memory, request count | Harness for longer controlled runs; power and ambient conditions remain external metadata. |
| Quality | Bits per byte and saved generated bytes | Tiny-corpus evidence only; no broad language-quality claim. |

No quality-versus-speed result is valid until model, checkpoint, prompt set,
decoding settings, and acceptance rule are frozen together.

## Reproduce Phase 2

```bash
make setup
make doctor
make phase2-smoke
```

Focused experiments:

```bash
# Float32 tiled candidate on model-derived shapes
uv run blt-phase2 benchmark-shapes \
  --samples 100 \
  --dtype float32 \
  --candidate metal_tiled16 \
  --output artifacts/phase2_benchmarks/float32_tiled16.json

# Float16 naive candidate on the same phase suite
uv run blt-phase2 benchmark-shapes \
  --samples 100 \
  --dtype float16 \
  --candidate metal_naive \
  --output artifacts/phase2_benchmarks/float16_naive.json

# Checkpoint-level generation metrics
uv run blt-phase2 benchmark-generation \
  --checkpoint artifacts/phase2_smoke/checkpoint \
  --samples 100 \
  --prompt "Byte " \
  --max-new-bytes 32 \
  --output artifacts/phase2_benchmarks/generation.json

# Sustained run: choose a duration appropriate for the actual thermal question
uv run blt-phase2 thermal-soak \
  --checkpoint artifacts/phase2_smoke/checkpoint \
  --prompt "Byte " \
  --seconds 1800 \
  --window-seconds 60 \
  --output artifacts/phase2_benchmarks/thermal_soak_30m.json
```

`make thermal-soak-short` is only a ten-second command-path check. It cannot
establish steady thermal behavior.

## Verified exit evidence

On the local M1 Max environment on August 11, 2026:

- MLX 0.32.0 detected the Apple GPU and executed float32, float16, and bfloat16.
- 12 Python tests passed, including custom Metal forward/backward agreement,
  deterministic/causal model behavior, training, exact checkpoint logits, and
  optimizer-step resume.
- 12 Java JUnit tests passed through the pinned Gradle 8.10.2 wrapper.
- The 80-step smoke run reduced training loss from 6.09 to 0.63 and evaluation
  bits per byte from 8.86 to 0.98, then wrote a reloadable checkpoint.
- The latest 20-sample shape run produced 15 comparisons per dtype across
  training, prefill, and decode. All outputs were valid. The tiled candidate
  passed 0/15 float32 gates and 3/15 float16 gates in that run.
- The final 20-sample checkpoint generation smoke recorded median TTFT 1.01 ms,
  median TPOT 1.06 ms, median throughput 946 bytes/s, and 582,144 bytes peak MLX
  memory.

The timing values are machine- and run-specific smoke evidence. They are not a
universal M1 Max claim, and they are not BLT performance because the measured
model is the byte-GPT baseline.

## What remains after Phase 2

1. Promote the tiny smoke path into a frozen research baseline with dataset
   hashes, train/validation splits, parameter counts, longer runs, and configs.
2. Build the simple checkpoint-backed generation and byte-inspection UI.
3. Implement fixed patch routing, then the local encoder, patch-level global
   transformer, and local decoder.
4. Prove end-to-end fixed-patch training before introducing causal entropy
   boundaries.
5. Freeze the Stage 7 paired-seed protocol and run the legitimate BLT claim.

Tile-size/threadgroup sweeps, SIMD-group reductions, long thermals, and larger
precision studies are now easy extensions of the evidence harness, but they are
experiments—not prerequisites for the Phase 2 foundation exit.
