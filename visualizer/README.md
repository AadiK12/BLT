# BLT Patching Lab

This is a local educational UI that turns a prompt into UTF-8 bytes, applies one
of the Stage 1 notebook patching policies, and asks Manim Community to render an
explanatory animation.

## What is real today

- The fixed-width, whitespace, smoothed bigram entropy, percentile-threshold,
  and minimum/maximum patch rules are extracted from
  `scripts/build_stage1_research_notebook.py`.
- Every byte is assigned to one explicit patch and every boundary has an
  inspectable reason.
- The Gradio interface renders a Manim MP4 and a complete patch ledger.

## Scope boundary

This is a patch-routing visualizer, not a trained Byte Latent Transformer. The
entropy source is the notebook's tiny fixed-corpus bigram model. The repository
does not yet contain the Stage 6 learned entropy model, local encoder, global
patch transformer, or local decoder. As those components are built, this UI can
consume their real traces without changing its basic interaction model.

## Run locally

From this directory:

```bash
uv sync --python 3.12
uv run python app.py
```

Open the local URL printed by Gradio. The first animation takes longer because
Manim renders the video; identical settings reuse the cached MP4.

## Verify

```bash
uv run pytest
uv run python render_sample.py
```

The smoke render prints the generated MP4 path. The Gradio app creates the
per-prompt Manim configuration automatically.
