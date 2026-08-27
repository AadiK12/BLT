# Phase 4: Research Training System

## Outcome

Phase 4 is complete for the first frozen external-corpus experiment. The
repository can now prepare a public-domain text with exact provenance, train the
108,032-parameter byte-GPT with a recorded optimizer schedule and gradient
clipping, evaluate validation data on a fixed cadence, select a checkpoint
without seeing test data, resume from complete optimizer state, and consume a
sealed test split once.

The result remains a **byte-level GPT baseline**, not a Byte Latent Transformer.
It establishes the training and evaluation contract that the later fixed-patch
and entropy-patched systems must share.

## Frozen experiment

[`configs/phase4_alice_byte_gpt.json`](../configs/phase4_alice_byte_gpt.json) is
the authoritative experiment contract.

| Component | Frozen value |
| --- | --- |
| Corpus | *Alice's Adventures in Wonderland* by Lewis Carroll |
| Source | [Project Gutenberg ebook 11](https://www.gutenberg.org/ebooks/11) |
| Raw source SHA-256 | `01b38ea4c710a84bc18d0bd41271a5a1a92b94e97b2812f4dece97d4a694725e` |
| Vocabulary | 256 raw byte values |
| Context | 128 bytes |
| Model | width 64, 2 blocks, 4 heads, MLP width 128 |
| Parameter count | 108,032, checked at runtime |
| Precision/device | float32 on the MLX Metal device |
| Training | 400 steps, batch 8, sequence length 128 |
| Optimizer | AdamW, beta1 0.9, beta2 0.95, epsilon `1e-8`, weight decay 0.01 |
| Learning rate | warmup from 0 to 0.001 over 40 steps, then cosine decay to 0.0001 |
| Gradient clipping | global norm capped at 1.0 |
| Validation/checkpoint cadence | every 50/every 100 steps |

Project Gutenberg labels the ebook public domain in the USA. The config keeps
that source and license note beside the exact downloaded-file hash; users
outside the USA must check their local law.

## Corpus preparation and split policy

`phase4-prepare` performs a fail-closed sequence:

1. Reuse the local raw source only if its SHA-256 matches the frozen hash.
2. Otherwise download the exact configured URL with three bounded attempts.
3. Reject the download unless all raw bytes match the frozen SHA-256.
4. Normalize line endings and require the configured Gutenberg start/end markers.
5. Require exactly 12 chapter headings.
6. Create deterministic chapter-level splits.
7. Reject any prepared split whose SHA-256 differs from the config.
8. Write a manifest with provenance, paths, byte counts, chapter assignments,
   and hashes.

| Split | Chapters | Bytes | SHA-256 |
| --- | ---: | ---: | --- |
| Train | I-VIII | 101,958 | `042232cec051860ca40c6751cee8ab151ebb22e0f32a5024919608fb515404b4` |
| Validation | IX-X | 25,534 | `4d32d32ac4fd61a6d2c07f4d9ff674ca2cbb416deffc4fe5bd0d3bf6935d17da` |
| Test | XI-XII | 23,093 | `8d187d5ff917d46c03be5f1586666cb4610796560e1aec76c39c9cc67cd90c7e` |

Splitting on chapters keeps complete document units apart. The packed dataset
continues to use independent attention, document-ID, and loss masks, so the
model neither attends across chapters nor learns an artificial target across a
chapter boundary.

## What the training loop now does

[`python/blt_mlx/training.py`](../python/blt_mlx/training.py) contains the common
optimizer mechanics, and
[`python/blt_mlx/research.py`](../python/blt_mlx/research.py) owns the Phase 4
research lifecycle.

For each training step, the system:

1. Selects a deterministic batch from the training split using the frozen seed
   and global step.
2. Computes masked next-byte cross-entropy.
3. Computes the global gradient norm and clips gradients above 1.0.
4. Applies AdamW using the warmup/cosine schedule.
5. Rejects non-finite loss or gradients.
6. Records loss, learning rate, pre-clipping gradient norm, whether clipping
   activated, device memory, and synchronized step latency.

Validation runs at step 0 and every 50 trained steps. Lower validation bits per
byte is the only checkpoint-selection rule. Scheduled checkpoints are written
every 100 steps; a newly best validation checkpoint is also saved even when it
falls between scheduled checkpoints.

## Checkpoint and resume contract

Every checkpoint contains:

- SafeTensors model weights;
- SafeTensors optimizer arrays, including AdamW schedule/step state;
- model and training configuration;
- global step;
- exact experiment-config and corpus-manifest hashes;
- split identities and validation history;
- current best step and validation BPB; and
- an explicit `test_evaluated: false` training-time marker.

Resume reconstructs the model and optimizer, restores both states and the
global step, and continues deterministic batch selection and the original
learning-rate schedule. It refuses checkpoints from another phase, model,
experiment config, or corpus manifest.

`--max-steps-this-run` permits a controlled partial job. For example:

```bash
uv run blt-lab phase4-train \
  --config configs/phase4_alice_byte_gpt.json \
  --run-directory artifacts/phase4/resume-example \
  --max-steps-this-run 100

uv run blt-lab phase4-train \
  --config configs/phase4_alice_byte_gpt.json \
  --run-directory artifacts/phase4/resume-example \
  --resume-checkpoint artifacts/phase4/resume-example/checkpoints/step_000100
```

## Sealed final-test protocol

Routine preparation creates and hashes test bytes, but inspection, training,
validation, and checkpoint selection do not load them. Final evaluation needs
the exact acknowledgement:

```text
I_UNDERSTAND_THIS_CONSUMES_THE_FINAL_TEST_SET
```

After evaluation, `final_test_consumed.json` binds the run to the checkpoint,
report path, and report hash. The training command then refuses to overwrite
that run, and the final-test command refuses another evaluation from it. A new
experiment requires a new versioned config and run directory; test results must
not be used to tune the completed experiment.

## Commands

```bash
make phase4-prepare       # download/cache, verify, split, and manifest
make phase4-inspect       # verify model/data/schedule and measure step-0 validation
make phase4-train         # train, validate, checkpoint, and select without test access
make phase4-smoke         # doctor + tests + inspect + separate 50-step partial run
make phase4-final-test    # explicitly consume test once for the selected checkpoint
```

Generated evidence is under `artifacts/phase4/` and intentionally ignored by
Git. The important reports are:

- `preparation_report.json`: source and split provenance;
- `inspection.json`: environment, parameter contract, schedule samples, and
  untrained validation;
- `run/training_report.json`: all training observations and validation history;
- `run/selection.json`: pre-test checkpoint decision;
- `final_test.json`: held-out BPB, exact split identity, structured generations,
  TTFT, TPOT, throughput, and memory; and
- `run/final_test_consumed.json`: durable one-shot consumption evidence.

## Verified M1 Max result

This run used Python 3.12.11, MLX 0.32.0, and the Metal GPU device on arm64
macOS. Validation alone selected step 350.

| Step | Validation BPB |
| ---: | ---: |
| 0 | 8.6380 |
| 50 | 4.6988 |
| 100 | 3.8598 |
| 150 | 3.6956 |
| 200 | 3.6354 |
| 250 | 3.5875 |
| 300 | 3.5662 |
| **350** | **3.5529** |
| 400 | 3.5529 |

The selected step-350 checkpoint achieved **3.4620 test BPB** across 48 frozen
test batches. During training, gradient clipping activated on 49 of 400 steps.
Median synchronized training-step latency was 5.04 ms and maximum recorded peak
Metal memory was 82,014,000 bytes. These performance numbers describe this one
machine and run; they are not cross-device benchmarks.

Greedy outputs were valid UTF-8 but repetitive, dominated by fragments such as
`the`. That is consistent with a tiny 108K-parameter model trained for only 400
steps on one short book. BPB improvement proves next-byte learning under the
frozen protocol; it does not prove useful prose generation.

## Tests added for Phase 4

The Python suite now directly verifies:

- warmup and cosine schedule endpoints;
- a parameter changes after optimization;
- two independent runs have identical early losses;
- gradient clipping activates under a deliberately small threshold;
- exact raw and split hashes;
- training access to train/validation while test remains sealed;
- training succeeds even when the test file is deliberately corrupted;
- the final path rejects an incorrect acknowledgement;
- partial training resumes with validation history and optimizer step intact;
- final evaluation writes its consumption marker; and
- consumed runs reject more training or test evaluation.

## Claim boundary and next stage

A legitimate Phase 4 claim is:

> On the frozen chapter split of Project Gutenberg ebook 11, this deterministic
> 108,032-parameter MLX byte-GPT improved validation BPB from 8.6380 to 3.5529;
> the validation-selected step-350 checkpoint achieved 3.4620 BPB on the
> one-shot held-out test protocol.

This is a claim about the byte-GPT control and the pipeline. It is not evidence
that BLT patching is better, that the model generalizes beyond this book, or
that its text is high quality. Stage 5 can now place this checkpoint behind a
simple generation/byte-inspection UI. Stage 6 must build the fixed-patch
local/global/local architecture before any BLT comparison is possible.
