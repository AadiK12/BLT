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

Last verified: **August 10, 2026**

Current repository state:

- Branch: `main`
- The `main` branch tip matched `origin/main` at verification time.
- Latest committed revision: `0ded4c1` (`simple fun UI`, August 10, 2026).
- The working tree also contains an uncommitted visualizer refactor from Manim-rendered MP4s to a live in-page Gradio animation, plus this roadmap refresh.
- The Java source compiles with Java 22 using `javac -d <temporary-directory> @sources.txt`.
- The console demonstration runs successfully and produces `Tensor(3x256)` GPT logits.
- The executed Stage 1 research notebook contains 37 cells, including 10 sequentially executed code cells with no saved error outputs.
- The current working-tree patching lab has a documented live in-page Gradio workflow and nine passing unit tests.
- The model can perform a forward pass and produce byte logits.
- The model cannot train, save learned weights, or generate meaningful learned text.
- The patching lab can visualize fixed, whitespace-like, and toy bigram-entropy boundaries, but no learned latent patching or BLT-specific local/global model architecture is implemented.
- The Gradle wrapper, deterministic Java initialization, Java correctness tests, and repository cleanup remain incomplete; generated Gradle/build/class artifacts are still tracked.

The most accurate description is:

> The research framing is substantially complete, while the Java foundations and byte-GPT baseline remain partial. The repository has a runnable, forward-only byte-level GPT and an educational patch-routing visualizer, but it does not yet have training, learned generation, or a true Byte Latent Transformer.

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
| 1. Research framing | Substantially complete | Add the concise root-README architecture summary and finalize the training implementation decision |
| 2. Tensor and neural-network foundations | Partial; manually verified | Add deterministic initialization, automated Java tests, a reproducible build, and repository cleanup |
| 3. Byte-GPT forward model | Runnable; baseline incomplete and untrained | Make the baseline deterministic, tested, configurable, and reproducible |
| 4. Training system | Not started | The model learns from text and saves checkpoints |
| 5. Tiny trained model and UI | Not started; educational patching UI exists | Learned text can be generated and inspected in a browser |
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

- Add a concise current-versus-target architecture summary and a direct notebook link to the root `README.md`.
- Finalize whether the trainable implementation will use custom Java gradients or PyTorch autograd. The recommendation remains to preserve Java as the forward-pass learning artifact and use PyTorch for the trainable research model.
- Freeze an exact dataset protocol, seed list, and analysis rule before calling the Stage 7 comparison pre-registered.

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

The custom [`Tensor`](src/main/java/com/blt/tensor/Tensor.java) supports:

- Two-dimensional `float` storage.
- Matrix multiplication.
- Element-wise addition and subtraction.
- Row-vector bias broadcasting.
- Scalar multiplication.
- Transposition.
- Row-wise softmax.
- Random and constant initialization.

The repository also implements:

- [`Linear`](src/main/java/com/blt/nn/Linear.java)
- [`LayerNorm`](src/main/java/com/blt/nn/LayerNorm.java)
- GELU activation inside [`Block`](src/main/java/com/blt/transformer/Block.java)

These operations compiled and produced the expected shapes and smoke-test values during the August 10, 2026 verification.

## What remains

### Automated correctness tests

Add tests for:

- Matrix multiplication against known values.
- Invalid matrix shapes.
- Bias broadcasting.
- Element-wise operations.
- Transposition.
- Numerically stable softmax.
- Softmax rows summing to approximately one.
- Layer normalization producing approximately zero mean and unit variance before affine scaling.
- Linear-layer output dimensions.
- Causal attention preventing future input positions from affecting earlier outputs.
- Invalid model configurations and sequence lengths.

### Deterministic initialization

The current `fillRandom()` method creates an unseeded `Random`, so every model instance differs.

Add:

- A project-level random seed.
- Seeded parameter initialization.
- A model configuration that records the seed.
- Tests that reproduce identical initial weights and logits.

### Repository and build hygiene

The repository now has a `.gitignore` for editor and Python outputs. It still needs:

- A Gradle wrapper (`gradlew`, `gradlew.bat`, and wrapper files).
- Java/Gradle entries in `.gitignore`.
- A clean separation between source and generated files.
- Removal of tracked `.class`, `build/`, and `.gradle/` artifacts.
- Removal of the tracked notebook checkpoint and an `.ipynb_checkpoints/` ignore rule.
- Clarification or removal of the duplicate `solutions` package.
- A single authoritative build and test command.

### Training support

The current tensor stores values but does not support:

- Gradients.
- A computation graph.
- Backward functions.
- Trainable parameter registration.
- Optimizer state.

This is the major decision point before Stage 4.

## Implementation decision: what does "from scratch" mean?

### Option A: Architecture from scratch

Use a standard tensor/autograd library such as PyTorch, but implement the model architecture, patcher, attention structure, training loop, evaluation, and UI logic directly.

Advantages:

- Keeps the project focused on BLT research.
- Makes attention and patching experiments much faster.
- Provides reliable gradients, batching, and device execution.
- Makes a small Gradio UI straightforward.

Tradeoff:

- The tensor engine and automatic differentiation are not built from scratch.

### Option B: Numerical engine from scratch in Java

Continue the existing Java implementation and add either:

- A general automatic-differentiation engine, or
- Explicit backward passes for every operation and layer.

Advantages:

- Provides a deep understanding of forward and backward mechanics.
- Preserves the current implementation language and style.

Tradeoff:

- A large portion of the work becomes gradient-engine development rather than BLT research.
- Attention, layer normalization, and cross-attention backward passes will be substantial and error-prone.

### Recommended boundary

Preserve the current Java code as the forward-pass learning artifact, and implement the trainable experimental model using PyTorch. In that track, the BLT architecture remains from scratch even though tensor gradients come from the framework.

This is a recommendation, not a prerequisite. The pure-Java path remains valid if learning automatic differentiation is itself a project goal.

## Completion criteria

Stage 2 is complete when:

- Mathematical operations have automated tests.
- Initialization is deterministic.
- The project builds and tests from a clean checkout.
- Generated files are not tracked as source.
- The training implementation path is explicitly chosen.

---

# Stage 3: Byte-level GPT baseline

## Objective

Build a conventional byte-level language model that can serve as the baseline for later BLT comparisons.

## What exists now

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

### Verified behavior

The console program was recompiled from source with Java 22 on August 10, 2026 and produced:

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

This verifies forward execution and tensor shapes. It does not verify learning or meaningful generation.

## What remains

- Learned weights.
- A loss function and training process.
- Checkpoint loading.
- Temperature sampling.
- Top-k or top-p sampling.
- Batched sequences.
- Padding and attention masks.
- Robust UTF-8 display and error handling.
- A model configuration object.
- Parameter counting.
- Inference timing.
- Automated causal and numerical tests.
- A key/value cache for efficient generation, if desired later.

The current generator uses freshly randomized weights. Its output length is correct, but its byte choices have no learned relationship to the prompt.

## Baseline metric

For 256 equally likely bytes, a uniform random prediction corresponds to approximately:

```text
8 bits per byte
```

This is a useful reference point for later training and evaluation.

## Completion criteria

Stage 3 is complete when:

- The forward pass is covered by automated tests.
- Causal masking is explicitly verified.
- Model initialization is deterministic.
- The model configuration can be saved and reconstructed.
- Generation supports greedy, temperature, and top-k modes.
- Random baseline loss and bits per byte are recorded.

---

# Stage 4: Training system

## Objective

Turn the forward-only byte model into a model that learns next-byte prediction from text.

This is the next critical stage. BLT-specific architecture should not be added until the ordinary byte model can learn a tiny dataset.

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

If the Java path is selected, this requires the gradient system described in Stage 2. If the PyTorch path is selected, PyTorch supplies automatic differentiation while this project supplies the architecture.

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

The complete system should expose one reproducible command, for example:

```bash
python train.py --config configs/tiny.yaml
```

or, for a pure-Java implementation:

```bash
./gradlew run --args="train --config configs/tiny.json"
```

## Tests and verification

Add checks for:

- Loss decreases on one batch.
- Gradients are finite.
- Parameters change after an optimizer step.
- Gradient clipping works.
- Saving and loading preserve logits.
- Resuming training preserves the optimizer step.
- A fixed seed reproduces early loss values.
- A deliberately tiny dataset can be overfit.

## Completion criteria

Stage 4 is complete when the model can:

- Overfit one short sequence.
- Drive training loss down substantially.
- Generate recognizable continuations.
- Save a checkpoint.
- Reload the checkpoint and produce identical logits.
- Resume training without resetting optimizer state.

---

# Stage 5: Tiny trained model and UI

## Objective

Make the trained byte model easy to run, observe, and understand through a simple local browser interface.

## Current status

The trained-model UI has not started because there is no learned checkpoint yet. The repository does contain a separate [`visualizer/`](visualizer/) Gradio lab that explains patch-routing policies with a live in-page animation. That lab is useful Stage 1 infrastructure, but it does not load a model, generate learned text, or satisfy this stage's completion criteria.

At the August 10, 2026 refresh, all nine visualizer unit tests passed. The live-animation refactor was still uncommitted.

The first UI should be built after the first learned checkpoint. A UI around the current random model would primarily display random bytes and could create the false impression that generation is already functional.

## Tiny-model proof

Before building the UI, preserve a tiny checkpoint that demonstrates:

- A known training corpus.
- A recorded loss curve.
- Successful overfitting.
- Recognizable byte generation.
- Deterministic checkpoint loading.

This checkpoint becomes the UI's default model and a permanent smoke-test fixture.

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

For a Python/PyTorch track:

```text
Gradio UI
  -> checkpoint loader
  -> generation service
  -> model
```

For a pure-Java track:

```text
static HTML and JavaScript
  -> Java HttpServer API
  -> checkpoint loader
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

1. Expand `README.md` with the current and target architectures.
2. Add the Gradle wrapper, `.gitignore`, seeded initialization, and tests.
3. Decide between pure-Java gradients and PyTorch automatic differentiation.
4. Implement next-byte loss, optimization, and checkpointing.
5. Overfit one deliberately tiny corpus.
6. Add deterministic sampling and checkpoint loading.
7. Build the basic generation and inspection UI.
8. Implement and visualize fixed patching.
9. Add the local encoder, global transformer, and local decoder.
10. Train the fixed-patch hierarchy end to end.
11. Add the entropy model and dynamic patch boundaries.
12. Run controlled baseline comparisons.
13. Write the conclusions and limitations.

## Immediate milestone

The next milestone should remain deliberately narrow:

> Make the existing byte-GPT learn and reproduce a tiny byte sequence.

Do not add BLT-specific architecture until this succeeds. Otherwise, failures in gradients, attention, training data, patching, cross-attention, and decoding will become difficult to isolate from one another.

## Definition of the three major finish lines

### Finish line A: Trustworthy byte-GPT baseline

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
