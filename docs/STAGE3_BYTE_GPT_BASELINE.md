# Stage 3: Frozen Byte-GPT Baseline

Last verified: **August 11, 2026**

## Outcome

Stage 3 is complete. The project now has one frozen, inspectable byte-GPT
baseline that future fixed-patch and entropy-patched BLTs must compare against.

This stage establishes a baseline contract; it does not claim broad language
quality. The included corpus is small and authored for deterministic software
verification. The validation split was observed while selecting the 10-step
teaching budget, so it is not an untouched Stage 7 test set or pre-registered
research result.

## Frozen configuration

The source of truth is
[`configs/stage3_byte_gpt_tiny.json`](../configs/stage3_byte_gpt_tiny.json).

| Setting | Value |
| --- | ---: |
| Vocabulary | 256 UTF-8 byte values |
| Maximum sequence | 128 bytes |
| Model width | 64 |
| Transformer layers | 2 |
| Attention heads | 4 |
| MLP width | 128 |
| Precision | float32 |
| Matmul | MLX baseline |
| Fusion | MLX compiled |
| Parameters | 108,032 |
| Parameter storage | 432,128 bytes |
| Training sequence | 64 bytes |
| Batch size | 8 |
| Training steps | 10 |
| Learning rate | 0.001 |

Parameter accounting is executable rather than a handwritten estimate:

| Component | Parameters |
| --- | ---: |
| Byte embedding | 16,384 |
| Position embedding | 8,192 |
| Two transformer blocks | 66,944 |
| Final normalization | 128 |
| Byte language-model head | 16,384 |
| **Total** | **108,032** |

The command fails if the constructed model no longer matches the expected
parameter count. This prevents an architecture change from silently changing
the baseline.

## Frozen authored data

The baseline uses committed, human-readable splits:

- [`data/stage3/train.txt`](../data/stage3/train.txt): 819 bytes, 6 documents,
  SHA-256 `3ed7429796ca5c0e1791e93db46e0c22211669c462c9458a74c5d43ef0d33c48`.
- [`data/stage3/validation.txt`](../data/stage3/validation.txt): 365 bytes, 3
  documents, SHA-256
  `260485e7bbe757baceb453c33864a32d989b6e61d1ecaa02a273775b027eb65a`.

Every inspect, train, or evaluation command recomputes these hashes. Modified
data is rejected rather than being treated as the same experiment.

## Packed-document and padding contract

[`PackedByteDataset`](../python/blt_mlx/data.py) can put several documents into
one fixed-size batch while preserving three distinct masks:

```text
attention_mask: which input positions contain data rather than padding
document_ids:   which document owns each input byte
loss_mask:      which next-byte transitions are legitimate targets
```

For two documents `ab` and `cd`, a padded example behaves like:

```text
inputs:       a  b  c  d  PAD
documents:    0  0  1  1  -1
loss mask:    1  0  1  0   0
```

The `b -> c` transition is excluded because it crosses a document boundary.
Attention for bytes in document 1 is also blocked from document 0, even though
the documents occupy the same array. Padding is blocked independently.

This contract will be reused by the BLT local encoder, patch router, global
model, and local decoder.

## Commands

Run the whole stage:

```bash
make stage3-smoke
```

Or run the lifecycle explicitly:

```bash
make stage3-inspect
make stage3-train
make stage3-evaluate
make stage3-generate
```

The CLI equivalents are:

```bash
uv run blt-lab inspect-baseline \
  --config configs/stage3_byte_gpt_tiny.json \
  --output artifacts/stage3/inspection.json

uv run blt-lab train-baseline \
  --config configs/stage3_byte_gpt_tiny.json \
  --checkpoint artifacts/stage3/checkpoint \
  --output artifacts/stage3/training_report.json

uv run blt-lab evaluate-checkpoint \
  --config configs/stage3_byte_gpt_tiny.json \
  --checkpoint artifacts/stage3/checkpoint \
  --output artifacts/stage3/evaluation.json

uv run blt-lab generate \
  --checkpoint artifacts/stage3/checkpoint \
  --prompt "Byte " \
  --max-new-bytes 32 \
  --temperature 0 \
  --output artifacts/stage3/generation.json
```

`blt-lab` is the project-wide research CLI. The `blt-phase2` alias remains for
compatibility with the Phase 2 commands.

## Evidence artifacts

Generated reports live under ignored `artifacts/stage3/`:

| Artifact | Purpose |
| --- | --- |
| `inspection.json` | Environment, config hash, data hashes, parameter inventory, uniform and untrained baselines |
| `training_report.json` | Initial/final evaluation, every training step, checkpoint path, and generation samples |
| `evaluation.json` | Checkpoint-bound held-out BPB and frozen prompts |
| `generation.json` | UTF-8-safe text, validity, raw byte values, hex, decoding settings, and latency |
| `checkpoint/` | Model SafeTensors, optimizer SafeTensors, and versioned training metadata |

Checkpoint metadata stores the exact baseline-config hash and dataset evidence.
Evaluation refuses a checkpoint produced by a different config, even when the
model dimensions happen to match.

Training is also fail-closed for this frozen fixture: the command refuses to
publish a checkpoint if either training loss or validation BPB does not improve.

## Verified result

On the local M1 Max smoke run:

- Uniform-byte reference: `8.00` BPB.
- Deterministic untrained model: `8.59` validation BPB.
- Training loss: `5.92 -> 2.89` over 10 steps.
- Validation BPB: `8.59 -> 4.83`.
- Greedy generation produced deterministic, valid UTF-8 byte sequences.
- 17 Python tests, 12 Java tests, Ruff, and the complete Stage 3 lifecycle
  passed.

The generated text is still repetitive because the corpus and training budget
are deliberately tiny. This result proves configuration, masking, learning,
checkpoint, evaluation, and generation integrity—not useful general-purpose
language modeling.

## Contract for the future BLT

The fixed-patch and entropy-patched models should implement the same external
behavior:

```text
inputs + attention mask + document IDs
    -> per-byte 256-way logits
    -> masked next-byte loss
    -> checkpoint
    -> validation BPB and generation evidence
```

That allows Stage 7 to hold data, seeds, prompts, loss, and report schemas fixed
while changing only the architecture and patching policy.

## Deliberately deferred

- A key/value cache. Current generation recomputes its active context and is the
  honest unoptimized baseline.
- A larger public-domain corpus and untouched test set. These belong to Stage 4
  research-training hardening.
- Broad quality claims. The authored validation split was used during baseline
  development.
- Dynamic patches and local/global/local BLT architecture. These begin in Stage
  6 after the UI and research training protocol are ready.
