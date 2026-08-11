# Byte Latent Transformer Project Roadmap

## Project purpose

This project exists to understand the research behind Byte Latent Transformers (BLTs) and to build a small, understandable BLT implementation from scratch.

The project should answer three questions:

1. How does an ordinary byte-level GPT work?
2. What does BLT add through latent patches, local byte models, and a global patch model?
3. At a small educational scale, what effects can we observe in learning behavior, efficiency, patching, and robustness?

The primary research references are:

- [Byte Latent Transformer: Patches Scale Better Than Tokens](https://arxiv.org/html/2412.09871)
- [Meta's reference BLT implementation](https://github.com/facebookresearch/blt)
- [Fast Byte Latent Transformer](https://arxiv.org/abs/2605.08044), as an optional extension after the original architecture works

## Quick navigation

- [Current project status](#current-project-status)
- [Roadmap overview](#roadmap-overview)
- [Stage 1: Research framing](#stage-1-research-framing)
- [Stage 2: Tensor and neural-network foundations](#stage-2-tensor-and-neural-network-foundations)
- [Stage 3: Byte-level GPT baseline](#stage-3-byte-level-gpt-baseline)
- [Stage 4: Training system](#stage-4-training-system)
- [Stage 5: Tiny trained model and UI](#stage-5-tiny-trained-model-and-ui)
- [Stage 6: Actual Byte Latent Transformer](#stage-6-actual-byte-latent-transformer)
- [Stage 7: Experiments and conclusions](#stage-7-experiments-and-conclusions)
- [Recommended implementation sequence](#recommended-implementation-sequence)
- [Progress log template](#progress-log-template)

## Current project status

Last verified: **August 11, 2026**

Current repository state:

- Branch: `main`
- The deterministic Java reference builds and tests through the pinned Gradle 8.10.2 wrapper; 12 JUnit tests cover tensor math, normalization, initialization, model shape, and causal behavior.
- The self-contained Python/MLX path trains a byte-level GPT on the Apple GPU, saves model and optimizer state, reloads exact logits, resumes the optimizer step, and generates bytes with greedy, temperature, or top-k decoding.
- Twelve Python tests cover pure references, custom Metal forward/backward behavior, model determinism and causality, checkpoint/resume, shape-derived comparisons, and generation metrics.
- The verified 80-step tiny-corpus smoke run reduced loss from 6.09 to 0.63 and evaluation bits per byte from 8.86 to 0.98.
- Performance infrastructure traces training, prefill, and decode shapes; runs balanced interleaved MLX-versus-Metal comparisons; gates on complete-output correctness, median, and p95; and records environment and memory.
- Checkpoint-level measurement exposes TTFT, TPOT, end-to-end latency, bytes per second, peak memory, and windowed thermal-soak plumbing.
- The Stage 1 notebook and separate Gradio patching lab remain the research and visualization surfaces.
- The patching lab can visualize fixed, whitespace-like, and toy bigram-entropy boundaries, but no learned latent patching or BLT-specific local/global model architecture is implemented.
- Generated Gradle/build/class artifacts and notebook checkpoints have been removed from source tracking and are ignored.

The most accurate description is:

> The research framing and Phase 2 foundation are complete. The repository now has a deterministic, trainable, checkpointed MLX byte-GPT baseline plus a tested Java reference and hardware-aware measurement harness. It still does not have the local/global/local latent-patch hierarchy required to call the model a Byte Latent Transformer.

### Current architecture

```text
bytes
  -> byte embeddings + positional embeddings
  -> full transformer applied at every byte
  -> 256 next-byte logits at every position
```

### Target BLT architecture

```text
bytes
  -> lightweight local encoder
  -> dynamically sized latent patches
  -> large global transformer applied to patches
  -> lightweight local decoder
  -> 256 next-byte logits at every position
```

The distinction matters: processing raw bytes does not by itself make a model a Byte Latent Transformer. BLT's defining feature is that expensive global computation operates on latent patches rather than on every byte.

## Roadmap overview

| Stage | Status | Main finish line |
| --- | --- | --- |
| 1. Research framing | Complete | Keep claims and references current as implementation evolves |
| 2. Tensor and neural-network foundations | Complete | Preserve correctness gates while extending the kernel experiments |
| 3. Byte-GPT forward model | MLX baseline complete; Java reference complete for forward learning | Freeze a research-sized configuration and parameter-count report |
| 4. Training system | Tiny deterministic path works; research pipeline partial | Add frozen datasets/splits, hashes, longer configs, and validation reporting |
| 5. Tiny trained model and UI | Checkpoint exists; trained-model UI not started | Load the learned checkpoint in a browser and expose byte-level inspection |
| 6. Actual BLT architecture | Not started; toy patch routing only | Bytes are dynamically grouped into learned latent patches and processed by the local/global/local hierarchy |
| 7. Experiments and conclusions | Designed; execution not started | Freeze the protocol and seed list, run the controlled comparisons, and report the evidence |

---

# Stage 1: Research framing

## Objective

Define the system being built, the research questions being asked, and the boundary around the phrase "from scratch."

There are three related but distinct model families:

1. A token-level GPT processes vocabulary tokens.
2. A byte-level GPT processes one of 256 byte values at every position.
3. A BLT uses lightweight byte-level models around a large transformer that operates on variable-length latent patches.

The current implementation is the second system. The eventual research target is the third.

## What exists now

- The source comments identify the project as a BLT assignment.
- The implementation demonstrates many underlying transformer mechanics.
- This roadmap defines the project purpose, stage boundaries, implementation sequence, and completion criteria.
- [`notebooks/01_stage_1_blt_research_readme.ipynb`](notebooks/01_stage_1_blt_research_readme.ipynb) is an executed, 37-cell research companion covering bytes, tokens, patches, the target architecture, causal constraints, repository gaps, a glossary, and research limitations.
- The notebook distinguishes paper claims from locally testable hypotheses and defines a controlled fixed-versus-entropy patching comparison. This is still a proposed experiment, not a result.
- [`README.md`](README.md) now identifies the repository as an educational path from a byte-GPT baseline to a small BLT and documents the separate patching lab.
- [`visualizer/`](visualizer/) makes Stage 1 patch-routing concepts inspectable without claiming that the learned Stage 6 model exists.

## What remains

- Freeze an exact dataset protocol, seed list, and analysis rule before calling the Stage 7 comparison pre-registered.
- Keep the notebook's architecture description synchronized with the implemented fixed-patch and entropy-patch systems when those stages begin.

The training decision is now final: preserve Java as the explicit forward-pass learning artifact and use Python/MLX as the Apple-Silicon-native trainable research path. MLX provides arrays and automatic differentiation; this project owns the architecture, data ordering, training/checkpoint contracts, experiments, and UI.

## Research questions

The implementation and experiments should eventually answer:

- Can a tiny byte-level model learn a small text corpus?
- Can several bytes be compressed into useful patch representations?
- How do fixed and entropy-based patches differ?
- Does patching reduce the number of global transformer steps?
- What prediction-quality cost comes with patch compression?
- Does entropy patching allocate more computation to difficult transitions?
- Are byte-level models measurably more robust to spelling noise, Unicode, code, or rare strings at this scale?
- Which BLT benefits only emerge at scales much larger than this project?

## Deliverables

- This roadmap.
- A concise architecture section in `README.md`.
- A glossary of BLT terms.
- Notes from the original paper.
- A list of project simplifications.
- A list of hypotheses and experiments.

## Completion criteria

Stage 1 is complete when a new reader can understand:

```text
what exists now -> what BLT adds -> what will be measured
```

---

# Stage 2: Tensor and neural-network foundations

## Objective

Establish a mathematical foundation that is correct, reproducible, and capable of supporting training.

## What exists now

Phase 2 has two complementary tracks:

- The Java reference keeps matrix multiplication, broadcasting, softmax, layer normalization, linear layers, attention, and transformer blocks explicit for learning. Initialization is seeded and 12 JUnit tests cover known values, invalid shapes, normalization, determinism, logits, and causal prefix invariance.
- The Python/MLX path is the trainable Apple-Silicon implementation. It includes pure-Python scalar references, MLX primitives, handwritten Metal candidates, explicit model modules, deterministic data, automatic differentiation, AdamW, SafeTensors checkpoints, generation, shape tracing, and performance reports.

The hardware-aware strategies are configurable rather than hard-wired:

| Decision | Available choices |
| --- | --- |
| Precision | float32, float16, bfloat16 through MLX; float32/float16 for custom Metal |
| Matrix multiplication | MLX, naive Metal, tiled-16 Metal |
| Bias and activation | eager MLX, compiled MLX, fused Metal bias+GELU |

See [`docs/PHASE2_INFRASTRUCTURE.md`](docs/PHASE2_INFRASTRUCTURE.md) for the component map, promotion contract, commands, metrics, and verified evidence.

## What remains

The Phase 2 exit criteria are satisfied. Future foundation work is experimental extension rather than missing infrastructure:

- Add more Metal tile sizes and threadgroup configurations behind the same strategy contract.
- Study SIMD-group reductions where attention or normalization traces justify them.
- Expand precision studies without silently routing bfloat16 into a float16 custom kernel.
- Run longer, controlled thermal experiments with power and ambient observations.
- Continue to treat MLX as the baseline and promote only shape-specific, numerically valid wins.

## Implementation decision: what "from scratch" means here

The decision is now implemented:

- Java owns the explicit forward-reference mechanics.
- Python/MLX owns the trainable and measurable research system on Apple Silicon.
- MLX provides storage, lazy execution, autodiff, and optimizer primitives.
- This repository owns deterministic initialization, layers and attention composition, byte-level model architecture, patch architecture when added, data contracts, training loop, checkpoint schema, generation, metrics, and experiments.
- Handwritten Metal is used only where it teaches a mechanism or passes the evidence gate; replacing all MLX primitives is not a project goal.

## Completion criteria

Stage 2 is complete when:

- [x] Mathematical operations have automated tests.
- [x] Initialization is deterministic.
- [x] The project builds and tests through pinned `uv` and Gradle entrypoints.
- [x] Generated files are not tracked as source.
- [x] The training implementation path is explicitly chosen and exercised.
- [x] Hardware-aware candidates have backward/correctness checks and a conservative promotion gate.

---

# Stage 3: Byte-level GPT baseline

## Objective

Build a conventional byte-level language model that can serve as the baseline for later BLT comparisons.

## What exists now

The authoritative trainable baseline is [`ByteGPT`](python/blt_mlx/model.py). It has versioned configuration, deterministic byte and positional embeddings, explicit pre-normalization transformer blocks, causal multi-head attention, configurable hardware-aware linear/MLP paths, and a 256-value language-model head. The Java classes in this section remain the explicit forward-reference implementation.

### Embeddings

[`GPT`](src/main/java/com/blt/transformer/GPT.java) creates:

- A `vocabSize x dModel` byte embedding table.
- A `maxSequenceLength x dModel` positional embedding table.
- A vocabulary of 256 values when constructed from `Main`.

### Causal multi-head attention

[`MultiHeadAttention`](src/main/java/com/blt/transformer/MultiHeadAttention.java) implements:

- Query, key, and value projections.
- Multiple attention heads.
- Scaled dot-product attention.
- A causal mask.
- Head concatenation through shared output storage.
- A final output projection.

### Transformer block

[`Block`](src/main/java/com/blt/transformer/Block.java) implements a pre-normalization block:

```text
x = x + Attention(LayerNorm(x))
x = x + MLP(LayerNorm(x))
```

The feed-forward network expands to `4 * dModel`, applies GELU, and projects back to `dModel`.

### Language-model output

The final layer normalization and linear head produce 256 next-byte logits for every input position.

### Generation control flow

`generateBytes()`:

1. Converts prompt bytes to unsigned values from 0 through 255.
2. Restricts the active context to the configured maximum sequence length.
3. Runs a full forward pass.
4. Reads the logits at the last position.
5. Selects the largest logit using argmax.
6. Appends the selected byte.

### Verified Java behavior

The console program was recompiled from source with Java 22 on August 11, 2026 and produced:

```text
Matmul Result:
[[19.0, 22.0], [43.0, 50.0]]
Linear output shape: Tensor(2x3)
LayerNorm Result:
[[-0.99998, 0.99998], [-0.99998, 0.99998]]
Attention output shape: Tensor(3x4)
GPT logits shape: Tensor(3x256)
Generated byte count: 7
```

The Java path verifies forward mechanics and shapes. The MLX path additionally has automated causal/determinism tests, batched inputs, saved configuration, training, checkpoint loading, and greedy/temperature/top-k generation.

## What remains

- Freeze a research-baseline model configuration and parameter-count report.
- Add train/validation dataset files and document-boundary or padding masks for larger corpora.
- Robust UTF-8 display and error handling.
- Parameter counting.
- A key/value cache for efficient generation, if desired later.
- Preserve the deterministic Java model as a reference rather than building a second Java training engine.

The MLX smoke checkpoint has learned the deliberately tiny training fixture. It proves the control flow, not general language modeling quality.

## Baseline metric

For 256 equally likely bytes, a uniform random prediction corresponds to approximately:

```text
8 bits per byte
```

This is a useful reference point for later training and evaluation.

## Completion criteria

Stage 3 is complete when:

- [x] The forward pass is covered by automated tests.
- [x] Causal masking is explicitly verified.
- [x] Model initialization is deterministic.
- [x] The model configuration can be saved and reconstructed.
- [x] Generation supports greedy, temperature, and top-k modes.
- [x] Initial loss and bits per byte are recorded by the smoke report.

---

# Stage 4: Training system

## Objective

Turn the byte model into a model that learns next-byte prediction from text.

The deterministic tiny-corpus path is implemented and has crossed its mechanical exit criteria. The broader research training pipeline remains partial: it still needs frozen external data, train/validation splits, dataset hashes, clipping and schedule controls, and longer-run reporting. BLT-specific architecture should not replace this baseline until those comparison inputs are frozen.

## Required components

### 1. Training examples

Convert text to UTF-8 bytes and shift each sequence by one position:

```text
input:  [b0, b1, b2, ..., b(n-1)]
target: [b1, b2, b3, ..., bn]
```

The model predicts the next byte at every input position.

Start with one deliberately tiny corpus, such as repeated variations of:

```text
hello world
hello blt
hello bytes
```

### 2. Cross-entropy loss

For every sequence position:

1. Convert the 256 logits to log probabilities.
2. Select the target byte's negative log probability.
3. Average across positions and batches.

Track:

- Cross-entropy loss.
- Bits per byte, computed as `loss / ln(2)` when the loss uses natural logarithms.

### 3. Backpropagation

Gradients must reach:

- Byte embeddings.
- Positional embeddings.
- Query, key, value, and output projections.
- Feed-forward weights and biases.
- Layer-normalization scale and shift.
- Final language-model head.

MLX supplies automatic differentiation while this project supplies the architecture and compiled training step. Handwritten Metal candidates expose custom vector-Jacobian products so their backward behavior can be checked against MLX.

### 4. Optimizer

Start with AdamW and include:

- Learning rate.
- Beta values.
- Epsilon.
- Weight decay.
- Gradient clipping.
- Optional learning-rate warmup after basic training works.

### 5. Data pipeline

Build the data pipeline incrementally:

1. Repeat one sequence until the model overfits it.
2. Train on several short lines.
3. Train on a small public-domain text file.
4. Add separate training and validation splits.
5. Add shuffled batches and reproducible data ordering.

### 6. Checkpoints

A checkpoint should contain:

- Model parameters.
- Model configuration.
- Optimizer state.
- Training step or epoch.
- Training and validation losses.
- Random seed.
- Dataset identifier or hash.

### 7. Training command

The tiny training system exposes one reproducible command:

```bash
make train-smoke
```

A later research command should accept frozen configuration and dataset files rather than embedding the smoke fixture.

## Tests and verification

Current verification status:

- [x] Loss decreases on the deterministic tiny fixture.
- [x] Every executed step rejects non-finite loss or gradients.
- [x] Saving and loading preserve logits exactly.
- [x] Resuming training preserves the optimizer step and state.
- [x] A deliberately tiny dataset can be overfit.
- [ ] Add a direct parameter-delta assertion after one optimizer step.
- [ ] Add gradient clipping and a clipping test.
- [ ] Add a two-run early-loss reproducibility assertion.

## Completion criteria

Stage 4 is complete when the model can:

- [x] Overfit one deliberately tiny corpus.
- [x] Drive training loss down substantially.
- [x] Generate bytes influenced by the learned tiny fixture.
- [x] Save a checkpoint.
- [x] Reload the checkpoint and produce identical logits.
- [x] Resume training without resetting the optimizer step or state.

These checks complete the tiny training-system slice. The frozen research data and evaluation requirements above remain before Stage 4 can be treated as a complete experimental pipeline.

---

# Stage 5: Tiny trained model and UI

## Objective

Make the trained byte model easy to run, observe, and understand through a simple local browser interface.

## Current status

The trained-model UI has not started, but a learned smoke checkpoint can now be produced with `make train-smoke`, so its prerequisite exists. The repository also contains a separate [`visualizer/`](visualizer/) Gradio lab that explains patch-routing policies with a live in-page animation. That lab is useful Stage 1 infrastructure, but it does not load the checkpoint, generate learned text, or satisfy this stage's completion criteria.

The next UI should load the Phase 2 smoke checkpoint and clearly label its tiny-corpus scope. It should not imply general language quality or a learned BLT.

## Tiny-model proof

The `make train-smoke` checkpoint and report now demonstrate:

- A known training corpus.
- A recorded loss curve.
- Successful overfitting.
- Recognizable byte generation.
- Deterministic checkpoint loading.

The UI should reproduce or load this fixture and display its model/training metadata.

## Generation panel

Controls:

- Prompt.
- Checkpoint selection.
- Maximum new bytes.
- Temperature.
- Top-k.
- Random seed.
- Generate button.
- Stop button.

Results:

- Generated text.
- Raw byte values.
- Hexadecimal representation.
- Generation time.
- Bytes generated per second.
- Active checkpoint and model configuration.

## Inspection panel

Show:

- UTF-8 characters and their underlying bytes.
- Probability assigned to each generated byte.
- Top alternative predictions.
- Context length.
- Model dimensions.
- Parameter count.
- Truncation or invalid UTF-8 warnings.

## BLT additions for later

Once Stage 6 exists, extend the same UI with:

- Color-coded patch boundaries.
- Patch lengths.
- Per-byte entropy.
- Average patch size.
- Number of global transformer steps.
- Fixed-versus-entropy patch comparison.
- Local and global representation dimensions.

## Recommended implementation

For the implemented Python/MLX track:

```text
Gradio UI
  -> checkpoint loader
  -> generation service
  -> model
```

Training should remain a command-line operation initially. The UI should load completed checkpoints rather than managing long-running training jobs.

## Completion criteria

Stage 5 is complete when:

- The UI starts with one documented command.
- It clearly reports which checkpoint is loaded.
- The same prompt, model, settings, and seed reproduce the same output.
- Invalid UTF-8 bytes do not crash rendering.
- Generation can be stopped.
- Byte-level behavior is visible rather than hidden behind plain text.
- The default tiny checkpoint works from a clean checkout.

---

# Stage 6: Actual Byte Latent Transformer

## Objective

Replace expensive transformer computation at every byte with expensive transformer computation at latent patch positions.

## Current status

Implementation has not started for the learned BLT hierarchy. The patching lab already provides inspectable fixed-width, whitespace-like, and toy bigram-entropy boundary policies, so it can serve as a visualization and tracing scaffold later. It does not yet provide a learned entropy model, local encoder, patch-level global transformer, local decoder, end-to-end loss, or training.

This stage changes the project from a byte-level GPT into a BLT.

## Stage 6A: Fixed patching

Begin with a simple fixed-width patcher:

```text
[4 bytes] [4 bytes] [4 bytes] ...
```

Do not begin with entropy patching. Fixed boundaries make it possible to debug the hierarchical architecture without simultaneously debugging an entropy model and a dynamic boundary algorithm.

The patcher should return:

- The original byte sequence.
- Patch start positions.
- Patch end positions.
- Byte-to-patch assignments.
- Patch lengths.
- Masks required by the encoder, global model, and decoder.

Tests should cover:

- Empty and short sequences.
- Sequences shorter than one patch.
- Sequences that do not divide evenly by patch size.
- Maximum patch length.
- Batch boundaries and document boundaries.
- Incremental generation behavior.

## Stage 6B: Local encoder

The lightweight local encoder should:

1. Embed every byte.
2. Optionally augment byte embeddings with local information.
3. Run a small local transformer over byte representations.
4. Pool the bytes belonging to each patch.
5. Produce one latent representation per patch.

Build pooling incrementally:

1. Mean pooling as the simplest debugging baseline.
2. Learned weighted pooling.
3. Encoder cross-attention matching the BLT design.

In encoder cross-attention:

```text
patch representations are queries
byte representations inside each patch are keys and values
```

The attention mask must prevent a patch from pooling bytes that belong to another patch or document.

## Stage 6C: Global latent transformer

The large global transformer operates on patch representations rather than individual bytes.

It needs:

- Patch-level positional information.
- Block-causal attention.
- One output representation per patch.
- A configurable number of heads and layers.
- A larger hidden dimension or deeper network than the local models.

This is the core compute-saving mechanism. If the average patch contains four bytes, the expensive global model receives roughly one quarter as many sequence positions as a transformer running directly on every byte.

## Stage 6D: Local decoder

The lightweight local decoder converts global patch information back into byte-level predictions.

Conceptually:

```text
byte representations query global patch representations
```

The decoder combines:

- Previously observed byte information.
- Local encoder representations.
- Relevant global patch context.
- Causal byte-level decoding.

It must still produce 256 logits for each next-byte prediction.

Build this incrementally:

1. Broadcast each global patch representation to its bytes.
2. Combine broadcast representations with local byte states.
3. Replace broadcasting with decoder cross-attention.
4. Add the lightweight decoder transformer layers.

## Stage 6E: End-to-end fixed-patch training

Before adding entropy patching, prove that the hierarchy can:

- Perform a forward pass.
- Backpropagate through local encoding, patch pooling, the global transformer, and local decoding.
- Overfit the same tiny corpus used by the byte-GPT baseline.
- Save and reload a hierarchical checkpoint.

This isolates architectural correctness from dynamic patching correctness.

## Stage 6F: Entropy patching

After fixed patches work, introduce dynamic boundaries.

A small autoregressive byte model estimates the next-byte distribution and entropy. Boundaries can then be created using:

- A global entropy threshold.
- An entropy increase relative to the previous byte.
- Minimum and maximum patch sizes.

The intended behavior is:

- Predictable regions receive longer patches.
- Difficult transitions receive shorter patches.
- Shorter patches invoke the global model more frequently.
- Computation is therefore allocated according to estimated difficulty.

Generation-time patching must be incremental: a boundary decision for a generated prefix cannot depend on future bytes that have not been generated.

## Stage 6G: BLT-specific enrichments

Add these only after the minimal hierarchy works:

- Hashed byte n-gram embeddings.
- Local attention windows.
- Multiple encoder cross-attention layers.
- Multiple decoder cross-attention layers.
- Variable patch-size constraints.
- Document-aware masks.
- Improved incremental patching.
- More efficient generation caches.

## Completion criteria

Stage 6 is complete when:

- Byte positions map to explicit, inspectable patches.
- The global transformer receives patches rather than bytes.
- The local decoder produces per-byte logits.
- The complete hierarchy trains end to end.
- Fixed and entropy patching both work.
- Boundary decisions are valid during incremental generation.
- The UI displays patch boundaries and entropy.
- The implementation can be compared fairly with the byte-GPT baseline.

---

# Stage 7: Experiments and conclusions

## Objective

Turn the implementation into a reproducible research project rather than only a coding exercise.

## Current status

The experiment is designed but has not been run. The Stage 1 notebook proposes a falsifiable matched-patch-budget comparison between fixed-stride and entropy-patched tiny BLTs, including held-out bits per byte, paired runs, a 2% average-patch-length matching tolerance, a paired confidence interval, and falsification rules. The exact dataset protocol, seed list, and analysis rule still need to be frozen before the comparison is called pre-registered, and no result should be claimed until Stages 4 through 6 are complete.

## Models to compare

Maintain at least three configurations:

1. Vanilla byte-GPT.
2. Fixed-patch BLT.
3. Entropy-patched BLT.

Where practical, compare models using similar parameter counts or explicit compute budgets. Record differences when an exact match is not possible.

## Core metrics

Measure:

- Training loss.
- Validation loss.
- Bits per byte.
- Average patch length.
- Patch-length distribution.
- Global transformer steps per byte.
- Training throughput.
- Generation throughput.
- Peak memory use.
- Parameter count.
- Estimated floating-point computation.
- Output quality on controlled prompts.
- Robustness on corrupted or unusual input.

## Experiment group 1: Learning

Questions:

- Can every model overfit the same tiny corpus?
- Which model converges faster?
- Does patch compression make optimization harder?
- How does validation bits per byte differ?
- Does increasing patch size reduce quality?

## Experiment group 2: Patching

Compare:

- Fixed patch sizes of 2, 4, 6, and 8 bytes.
- Whitespace patching.
- Global entropy thresholds.
- Relative entropy thresholds.
- Minimum and maximum patch sizes.

Measure both prediction quality and the number of global model steps.

## Experiment group 3: Robustness

Test controlled variants involving:

- Misspellings.
- Repeated punctuation.
- Unicode and emoji.
- Mixed languages.
- Source code.
- Random character corruption.
- Very long or rare words.
- Whitespace changes.

## Experiment group 4: Efficiency

For every evaluation sample, compare:

```text
number of bytes
number of patches
average patch size
number of global transformer steps
generation time
bits per byte
```

Avoid making large-scale efficiency claims from tiny-model results. The goal is to understand the mechanism and local tradeoffs, not to reproduce Meta's scaling conclusions on consumer hardware.

## Experiment records

Every run should record:

- Git commit.
- Configuration file.
- Random seed.
- Dataset identifier.
- Hardware and device.
- Start and end timestamps.
- Training metrics.
- Evaluation metrics.
- Checkpoint path.
- Notes about failures or anomalies.

## Final project artifacts

The completed project should contain:

- A research-oriented README.
- Architecture diagrams.
- Reproducible model configurations.
- Training scripts.
- Automated tests.
- Saved tiny checkpoints.
- A local browser demo.
- Experiment result tables and charts.
- A conclusions document explaining what worked and what did not.

## Completion criteria

The project is complete when it can answer:

- What makes BLT different from a byte-level GPT?
- Why do patches reduce expensive computation?
- How does entropy influence where computation is allocated?
- Does the expected behavior appear at tiny scale?
- Which parts of the paper were reproduced?
- Which parts were simplified or omitted?
- What failed, and why?
- What would need to change to scale the experiment further?

---

# Recommended implementation sequence

Work through the project in this order:

1. [Complete] Document the current and target architectures.
2. [Complete] Add the Gradle wrapper, ignore policy, seeded initialization, and tests.
3. [Complete] Establish Java as the forward reference and MLX as the trainable Apple path.
4. [Complete for the tiny fixture] Implement next-byte loss, optimization, and checkpointing.
5. [Complete] Overfit one deliberately tiny corpus.
6. [Complete] Add deterministic sampling, checkpoint loading, resume, and performance tracing.
7. Build the basic generation and inspection UI.
8. Freeze the research byte-GPT config, dataset splits/hashes, seed list, and evaluation prompts.
9. Implement and visualize fixed patching.
10. Add the local encoder, global transformer, and local decoder.
11. Train the fixed-patch hierarchy end to end.
12. Add the entropy model and dynamic patch boundaries.
13. Run controlled baseline comparisons.
14. Write the conclusions and limitations.

## Immediate milestone

The next milestone should remain deliberately narrow:

> Put the learned smoke checkpoint behind a simple byte-inspection UI, then freeze the research byte-GPT dataset and configuration.

The tiny overfit proof now succeeds. Do not make a BLT quality claim until the byte-GPT comparison data, seeds, prompts, and evaluation rules are frozen; otherwise baseline drift will be confounded with patching changes.

## Definition of the three major finish lines

### Finish line A: Trustworthy byte-GPT baseline — achieved for the tiny fixture

- Reproducible build.
- Automated mathematical and causal tests.
- Trainable byte model.
- Saved and reloadable checkpoints.
- Tiny corpus overfit.

### Finish line B: Usable educational demo

- Browser UI.
- Learned generation.
- Byte and probability inspection.
- Deterministic settings.
- Clear model and checkpoint metadata.

### Finish line C: Completed BLT research project

- Fixed and entropy patching.
- Local encoder, global latent transformer, and local decoder.
- End-to-end training.
- Visible patch boundaries and entropy.
- Controlled baseline comparisons.
- Written conclusions and limitations.

---

# Progress log template

Use the following template when completing a milestone:

```markdown
## YYYY-MM-DD: Milestone name

### Goal

What was this milestone intended to prove?

### Changes

- Files added or changed
- Architecture or behavior introduced

### Verification

- Commands run
- Tests passed
- Metrics observed

### Result

What now works?

### Remaining gaps

What is still incomplete, uncertain, or intentionally deferred?

### Next milestone

What is the narrowest next proof?
```
