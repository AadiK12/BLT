# Byte Latent Transformer Project

This repository is an educational, from-scratch path from a byte-level GPT
baseline toward a small Byte Latent Transformer. See
[`PROJECT_ROADMAP.md`](PROJECT_ROADMAP.md) for the implementation stages and
their current status.

## Animated patching lab

[`visualizer/`](visualizer/) contains a local Python and Gradio UI with a live
in-page animation. Enter a prompt to watch its UTF-8 bytes receive entropy
scores and be grouped into explicit patches. It creates no video files. The
current trace uses the Stage 1 notebook's toy bigram entropy model; it does not
claim that the Stage 6 learned BLT encoder already exists.

```bash
cd visualizer
uv sync --python 3.12
uv run python app.py
```
