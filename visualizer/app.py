"""Local Gradio UI for rendering BLT patching animations with Manim."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import gradio as gr

from blt_visualizer import DEFAULT_PROMPT, analyze_prompt


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / ".cache" / "renders"
SCENE_FILE = ROOT / "manim_scene.py"

STRATEGY_VALUES = {
    "Entropy · toy bigram": "entropy",
    "Fixed width": "fixed",
    "Whitespace": "whitespace",
}


def _render_video(analysis: dict) -> Path:
    payload = json.dumps(analysis, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha256(
        payload.encode("utf-8") + SCENE_FILE.read_bytes()
    ).hexdigest()[:16]
    render_dir = CACHE_DIR / digest
    config_path = render_dir / "analysis.json"
    render_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(payload, encoding="utf-8")

    existing = sorted(render_dir.rglob("*.mp4"), key=lambda path: path.stat().st_mtime)
    if existing:
        return existing[-1]

    environment = os.environ.copy()
    environment["BLT_VIZ_CONFIG"] = str(config_path)
    command = [
        sys.executable,
        "-m",
        "manim",
        "render",
        "-ql",
        "--disable_caching",
        "--media_dir",
        str(render_dir / "media"),
        "--output_file",
        "patching",
        str(SCENE_FILE),
        "BLTPatchingScene",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout)[-3000:]
        raise RuntimeError(f"Manim could not render this prompt.\n\n{details}")

    videos = sorted(render_dir.rglob("*.mp4"), key=lambda path: path.stat().st_mtime)
    if not videos:
        raise RuntimeError("Manim finished without producing an MP4 file.")
    return videos[-1]


def _summary_html(result) -> str:
    threshold = (
        f"{result.threshold_bits:.2f} bits"
        if result.threshold_bits is not None
        else "not used"
    )
    strategy = {
        "entropy": "Toy entropy",
        "fixed": "Fixed width",
        "whitespace": "Whitespace",
    }[result.strategy]
    reduction = result.nominal_global_step_reduction * 100
    return f"""
    <section class="metric-shell" aria-label="Patching summary">
      <div class="metric"><span>UTF-8 bytes</span><strong>{result.byte_count}</strong></div>
      <div class="metric"><span>Latent patches</span><strong>{result.patch_count}</strong></div>
      <div class="metric"><span>Average size</span><strong>{result.average_patch_size:.2f}<small> B</small></strong></div>
      <div class="metric"><span>Position reduction</span><strong>{reduction:.1f}<small>%</small></strong></div>
      <div class="metric wide"><span>Boundary policy</span><strong>{strategy}</strong><em>threshold {threshold}</em></div>
    </section>
    """


def visualize(
    prompt: str,
    strategy_label: str,
    threshold_percent: int,
    min_patch_size: int,
    max_patch_size: int,
    fixed_patch_size: int,
    progress=gr.Progress(),
):
    strategy = STRATEGY_VALUES[strategy_label]
    progress(0.05, desc="Reading UTF-8 bytes")
    result = analyze_prompt(
        prompt=prompt,
        strategy=strategy,
        threshold_percentile=threshold_percent / 100,
        min_patch_size=int(min_patch_size),
        max_patch_size=int(max_patch_size),
        fixed_patch_size=int(fixed_patch_size),
    )
    progress(0.25, desc="Building patch boundaries")
    analysis = result.to_dict()
    video_path = _render_video(analysis)
    progress(0.95, desc="Preparing the patch ledger")

    patch_rows = [
        [
            f"P{patch.patch_id}",
            f"{patch.start}:{patch.end}",
            patch.text,
            patch.hex,
            patch.length,
            round(patch.mean_entropy, 3),
            patch.boundary_reason,
        ]
        for patch in result.patches
    ]
    note = (
        "<div class='scope-note'><b>What you are seeing:</b> the exact toy "
        "bigram entropy logic from the Stage 1 notebook, now extracted into a "
        "reusable module. It demonstrates routing and patch boundaries; it is "
        "not yet the learned local encoder or trained entropy model from Stage 6.</div>"
    )
    return str(video_path), _summary_html(result), patch_rows, note


CSS = """
:root {
  --surface: #071018;
  --panel: #0c1923;
  --line: #203746;
  --ink: #e7f0f5;
  --muted: #89a0af;
  --cyan: #54d2d2;
  --orange: #ff9b54;
  --yellow: #f7d774;
}
.gradio-container {
  background:
    radial-gradient(circle at 15% 0%, rgba(84,210,210,.10), transparent 30rem),
    radial-gradient(circle at 95% 20%, rgba(174,140,255,.08), transparent 26rem),
    var(--surface) !important;
  color: var(--ink) !important;
  font-family: "Avenir Next", Inter, ui-sans-serif, system-ui, sans-serif !important;
}
.app-shell { max-width: 1240px; margin: 0 auto; padding: 24px 18px 48px; }
.hero { padding: 24px 0 18px; border-bottom: 1px solid var(--line); margin-bottom: 22px; }
.eyebrow { color: var(--cyan); letter-spacing: .16em; font-size: 12px; font-weight: 700; }
.hero h1 { color: var(--ink); font-size: clamp(34px, 5vw, 68px); line-height: .98; letter-spacing: -.04em; margin: 14px 0 10px; }
.hero p { color: var(--muted); max-width: 760px; font-size: 17px; line-height: 1.55; margin: 0; }
.panel { background: rgba(12,25,35,.86); border: 1px solid var(--line); border-radius: 18px; padding: 18px !important; box-shadow: 0 20px 60px rgba(0,0,0,.18); }
.panel-title { color: var(--ink); font-weight: 700; letter-spacing: -.01em; margin-bottom: 8px; }
.render-button { min-height: 48px !important; background: linear-gradient(110deg, #54d2d2, #7ce4bd) !important; color: #071018 !important; border: none !important; font-weight: 800 !important; }
.metric-shell { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 12px 0; }
.metric { background: #0c1923; border: 1px solid #203746; border-radius: 13px; padding: 13px 14px; display: flex; flex-direction: column; min-height: 86px; }
.metric span { color: #89a0af; font-size: 11px; letter-spacing: .08em; text-transform: uppercase; }
.metric strong { color: #e7f0f5; font-size: 25px; margin-top: 7px; line-height: 1; }
.metric small { color: #89a0af; font-size: 13px; }
.metric.wide { grid-column: span 4; min-height: 72px; flex-direction: row; align-items: center; gap: 14px; }
.metric.wide strong { margin: 0; font-size: 18px; }
.metric.wide em { margin-left: auto; color: #f7d774; font-style: normal; font-family: Menlo, monospace; font-size: 12px; }
.scope-note { color: #a9bac5; border-left: 3px solid #ff9b54; background: rgba(255,155,84,.07); padding: 12px 14px; border-radius: 4px 12px 12px 4px; line-height: 1.5; font-size: 13px; }
footer { color: #647a88; font-size: 12px; padding: 20px 2px 0; }
@media (max-width: 760px) {
  .metric-shell { grid-template-columns: repeat(2, 1fr); }
  .metric.wide { grid-column: span 2; align-items: flex-start; flex-direction: column; }
  .metric.wide em { margin-left: 0; }
}
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="BLT Patching Lab", fill_width=True) as demo:
        with gr.Column(elem_classes="app-shell"):
            gr.HTML(
                """
                <header class="hero">
                  <div class="eyebrow">BLT / PATCHING LAB</div>
                  <h1>Watch bytes become patches.</h1>
                  <p>Type a prompt, inspect its UTF-8 byte stream, and render a
                  Manim explanation of how an entropy boundary policy reduces
                  byte positions into latent patch positions.</p>
                </header>
                """
            )
            with gr.Row(equal_height=False):
                with gr.Column(scale=5, elem_classes="panel"):
                    gr.HTML("<div class='panel-title'>Prompt & boundary policy</div>")
                    prompt = gr.Textbox(
                        value=DEFAULT_PROMPT,
                        label="Prompt",
                        lines=4,
                        max_lines=7,
                        placeholder="Type text, Unicode, or code…",
                        autofocus=True,
                    )
                    strategy = gr.Radio(
                        choices=list(STRATEGY_VALUES),
                        value="Entropy · toy bigram",
                        label="Patching strategy",
                    )
                    with gr.Accordion("Boundary controls", open=True):
                        threshold = gr.Slider(
                            45,
                            95,
                            value=70,
                            step=5,
                            label="Entropy threshold percentile",
                            info="Higher values create fewer entropy-triggered boundaries.",
                        )
                        with gr.Row():
                            minimum = gr.Slider(1, 8, value=2, step=1, label="Minimum patch bytes")
                            maximum = gr.Slider(2, 16, value=8, step=1, label="Maximum patch bytes")
                        fixed_size = gr.Slider(1, 12, value=4, step=1, label="Fixed width")
                    render = gr.Button("Animate patching", variant="primary", elem_classes="render-button")
                    gr.Examples(
                        examples=[
                            ["byte models allocate compute at surprising transitions."],
                            ["hello café ☕ — bytes are not characters"],
                            ["def entropy(p): return -sum(x * log2(x) for x in p)"],
                        ],
                        inputs=[prompt],
                    )
                with gr.Column(scale=7):
                    video = gr.Video(
                        label="Manim patching animation",
                        autoplay=True,
                        loop=True,
                        height=460,
                    )
                    summary = gr.HTML()
            with gr.Column(elem_classes="panel"):
                gr.HTML("<div class='panel-title'>Patch ledger</div>")
                table = gr.Dataframe(
                    headers=["Patch", "Byte range", "Visible bytes", "Hex", "Length", "Mean entropy", "Boundary reason"],
                    datatype=["str", "str", "str", "str", "number", "number", "str"],
                    interactive=False,
                    wrap=True,
                    label=None,
                )
                scope_note = gr.HTML()
            gr.HTML("<footer>Manim Community renderer · notebook-derived educational model · local-only prototype</footer>")

        render.click(
            fn=visualize,
            inputs=[prompt, strategy, threshold, minimum, maximum, fixed_size],
            outputs=[video, summary, table, scope_note],
            api_name="visualize_patching",
        )
        prompt.submit(
            fn=visualize,
            inputs=[prompt, strategy, threshold, minimum, maximum, fixed_size],
            outputs=[video, summary, table, scope_note],
            api_name=False,
        )
    return demo


if __name__ == "__main__":
    build_app().queue(default_concurrency_limit=1).launch(inbrowser=False, css=CSS)
