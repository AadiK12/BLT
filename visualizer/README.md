# BLT Patching Lab

This is a live local UI that turns a prompt into UTF-8 bytes, applies one of the
Stage 1 notebook patching policies, and animates the resulting byte-to-patch
trace directly in the browser.

## What is real today

- The fixed-width, whitespace, smoothed bigram entropy, percentile-threshold,
  and minimum/maximum patch rules are extracted from
  `scripts/build_stage1_research_notebook.py`.
- Every byte is assigned to one explicit patch and every boundary has an
  inspectable reason.
- Python and Gradio produce the interface and patch trace. CSS animates the
  stages in place when **Animate patching** is clicked.
- No video, image sequence, or animation file is rendered or stored.

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

Open the local URL printed by Gradio.

## Verify

```bash
uv run pytest
```
