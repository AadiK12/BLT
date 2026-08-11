"""Live Gradio UI for exploring BLT byte-to-patch routing."""

from __future__ import annotations

import html
import time

import gradio as gr

from blt_visualizer import DEFAULT_PROMPT, analyze_prompt


STRATEGY_VALUES = {
    "Entropy · toy bigram": "entropy",
    "Fixed width": "fixed",
    "Whitespace": "whitespace",
}

PATCH_COLORS = ("#54D2D2", "#F7D774", "#FF9B54", "#AE8CFF", "#6ED39D")


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


def _placeholder_html() -> str:
    return """
    <section class="animation-shell animation-placeholder">
      <div class="placeholder-orbit"><span></span><span></span><span></span></div>
      <strong>Your byte stream will animate here.</strong>
      <p>Choose a boundary policy, then press <b>Animate patching</b>.</p>
    </section>
    """


def _animation_html(result) -> str:
    """Build a replayable, browser-native animation from Python analysis."""
    run_id = time.time_ns()
    escaped_prompt = html.escape(result.prompt, quote=True)
    threshold = result.threshold_bits
    threshold_label = f"{threshold:.2f} bits" if threshold is not None else "not used"
    threshold_bottom = 18 + (min(threshold or 0, 8) / 8) * 62

    byte_nodes: list[str] = []
    entropy_nodes: list[str] = []
    for byte in result.bytes:
        color = PATCH_COLORS[byte.patch_id % len(PATCH_COLORS)]
        boundary_class = " is-boundary" if byte.starts_patch and byte.index > 0 else ""
        glyph = html.escape(byte.glyph)
        details = html.escape(
            f"Byte {byte.index} · 0x{byte.hex} · {byte.entropy:.3f} bits · patch P{byte.patch_id}",
            quote=True,
        )
        byte_nodes.append(
            f"""
            <div class="byte-node{boundary_class}" style="--i:{byte.index};--patch:{color}" title="{details}">
              <span class="byte-index">{byte.index}</span>
              <strong>{glyph}</strong>
              <small>{byte.hex}</small>
              {('<i>boundary</i>' if boundary_class else '')}
            </div>
            """
        )

        height = 12 + (min(byte.entropy, 8) / 8) * 62
        hot_class = " is-hot" if threshold is not None and byte.entropy >= threshold else ""
        entropy_nodes.append(
            f"""
            <div class="entropy-column{boundary_class}{hot_class}" style="--i:{byte.index};--height:{height:.1f}px;--patch:{color}" title="{details}">
              <span class="entropy-bar"></span>
              <small>{byte.entropy:.1f}</small>
            </div>
            """
        )

    patch_nodes: list[str] = []
    for patch in result.patches:
        color = PATCH_COLORS[patch.patch_id % len(PATCH_COLORS)]
        visible = html.escape(patch.text)
        reason = html.escape(patch.boundary_reason)
        patch_nodes.append(
            f"""
            <div class="patch-cluster" style="--p:{patch.patch_id};--len:{patch.length};--patch:{color}">
              <div class="patch-heading"><b>P{patch.patch_id}</b><span>{patch.length} bytes</span></div>
              <strong class="patch-text">{visible}</strong>
              <small>{reason}</small>
            </div>
            """
        )

    return f"""
    <section class="animation-shell" data-run="{run_id}" aria-label="Animated byte patching result">
      <header class="animation-header">
        <div>
          <span class="live-dot"></span>
          <b>LIVE PATCH TRACE</b>
        </div>
        <span>click Animate patching to replay</span>
      </header>

      <div class="prompt-card">
        <span>INPUT STRING</span>
        <strong>{escaped_prompt}</strong>
        <em>{result.byte_count} UTF-8 bytes</em>
      </div>

      <div class="animation-stage stage-bytes">
        <div class="stage-heading"><span>01</span><div><b>Encode the prompt</b><small>One tile per UTF-8 byte</small></div></div>
        <div class="horizontal-track byte-track">{''.join(byte_nodes)}</div>
      </div>

      <div class="animation-stage stage-entropy">
        <div class="stage-heading"><span>02</span><div><b>Estimate surprise</b><small>Next-byte entropy from the notebook's toy bigram model</small></div></div>
        <div class="entropy-legend"><span><i class="cool"></i> below threshold</span><span><i class="hot"></i> boundary candidate</span><b>threshold {threshold_label}</b></div>
        <div class="horizontal-track entropy-track">
          {('<div class="threshold-rule" style="bottom:' + f'{threshold_bottom:.1f}' + 'px"><span>' + threshold_label + '</span></div>') if threshold is not None else ''}
          {''.join(entropy_nodes)}
        </div>
      </div>

      <div class="animation-stage stage-patches">
        <div class="stage-heading"><span>03</span><div><b>Form latent patches</b><small>Boundaries become the positions seen by a future global transformer</small></div></div>
        <div class="horizontal-track patch-track">{''.join(patch_nodes)}</div>
      </div>

      <div class="model-flow" aria-label="Conceptual BLT processing path">
        <span>LOCAL BYTE MODEL</span><i>→</i><span>BOUNDARY POLICY</span><i>→</i><span>PATCH ENCODER</span><i>→</i><span>GLOBAL TRANSFORMER</span>
      </div>
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
    if int(max_patch_size) < int(min_patch_size):
        raise gr.Error("Maximum patch bytes must be at least the minimum patch bytes.")

    strategy = STRATEGY_VALUES[strategy_label]
    progress(0.2, desc="Reading UTF-8 bytes")
    result = analyze_prompt(
        prompt=prompt,
        strategy=strategy,
        threshold_percentile=threshold_percent / 100,
        min_patch_size=int(min_patch_size),
        max_patch_size=int(max_patch_size),
        fixed_patch_size=int(fixed_patch_size),
    )
    progress(0.85, desc="Building the live patch trace")

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
        "<div class='scope-note'><b>What you are seeing:</b> Python runs the exact "
        "toy bigram entropy logic extracted from the Stage 1 notebook. The page "
        "animates that trace live—no video is rendered or stored. This demonstrates "
        "routing and boundaries, not yet a learned Stage 6 BLT encoder.</div>"
    )
    return _animation_html(result), _summary_html(result), patch_rows, note


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
.app-shell { max-width: 1320px; margin: 0 auto; padding: 24px 18px 48px; }
.hero { padding: 24px 0 18px; border-bottom: 1px solid var(--line); margin-bottom: 22px; }
.eyebrow { color: var(--cyan); letter-spacing: .16em; font-size: 12px; font-weight: 700; }
.hero h1 { color: var(--ink); font-size: clamp(34px, 5vw, 68px); line-height: .98; letter-spacing: -.04em; margin: 14px 0 10px; }
.hero p { color: var(--muted); max-width: 820px; font-size: 17px; line-height: 1.55; margin: 0; }
.panel { background: rgba(12,25,35,.86); border: 1px solid var(--line); border-radius: 18px; padding: 18px !important; box-shadow: 0 20px 60px rgba(0,0,0,.18); }
.panel-title { color: var(--ink); font-weight: 700; letter-spacing: -.01em; margin-bottom: 8px; }
.render-button { min-height: 48px !important; background: linear-gradient(110deg, #54d2d2, #7ce4bd) !important; color: #071018 !important; border: none !important; font-weight: 800 !important; }

.animation-shell { background: #08131c; border: 1px solid #203746; border-radius: 18px; padding: 18px; min-height: 500px; overflow: hidden; box-shadow: inset 0 1px rgba(255,255,255,.025); }
.animation-header { display:flex; align-items:center; justify-content:space-between; padding-bottom:13px; border-bottom:1px solid #203746; color:#718794; font-size:11px; letter-spacing:.08em; text-transform:uppercase; }
.animation-header > div { display:flex; align-items:center; gap:8px; color:#a9bac5; }
.live-dot { width:7px; height:7px; border-radius:50%; background:#54d2d2; box-shadow:0 0 0 0 rgba(84,210,210,.5); animation:live-pulse 1.8s ease-out infinite; }
.prompt-card { display:grid; grid-template-columns:auto 1fr auto; gap:14px; align-items:center; margin:16px 0 20px; padding:12px 14px; background:#0c1923; border:1px solid #203746; border-radius:12px; }
.prompt-card span,.prompt-card em { color:#718794; font-size:10px; letter-spacing:.08em; font-style:normal; }
.prompt-card strong { color:#e7f0f5; font:500 13px/1.4 Menlo,monospace; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.animation-stage { margin-top:16px; opacity:0; transform:translateY(8px); animation:stage-in .38s ease forwards; }
.stage-entropy { animation-delay:.55s; }
.stage-patches { animation-delay:1.2s; }
.stage-heading { display:flex; align-items:center; gap:10px; margin-bottom:9px; }
.stage-heading > span { display:grid; place-items:center; width:27px; height:27px; border:1px solid #294554; border-radius:7px; color:#54d2d2; font:700 10px Menlo,monospace; }
.stage-heading div { display:flex; flex-direction:column; gap:1px; }
.stage-heading b { color:#e7f0f5; font-size:12px; }
.stage-heading small { color:#718794; font-size:10px; }
.horizontal-track { position:relative; display:flex; align-items:stretch; gap:5px; width:100%; overflow-x:auto; overflow-y:hidden; padding:5px 3px 10px; scrollbar-color:#294554 transparent; }
.byte-node { position:relative; flex:0 0 43px; height:62px; display:grid; place-items:center; grid-template-rows:13px 1fr 15px; border:1px solid color-mix(in srgb,var(--patch) 70%,#203746); border-radius:8px; background:color-mix(in srgb,var(--patch) 7%,#0c1923); opacity:0; transform:translateY(12px) scale(.94); animation:byte-in .3s cubic-bezier(.2,.8,.2,1) forwards; animation-delay:calc(var(--i) * 18ms); }
.byte-node strong { color:#e7f0f5; font:700 15px Menlo,monospace; }
.byte-node small,.byte-index { color:#6f8795; font:8px Menlo,monospace; }
.byte-node i { position:absolute; top:-5px; left:50%; transform:translateX(-50%); width:5px; height:5px; overflow:hidden; border-radius:50%; background:#f7d774; color:transparent; box-shadow:0 0 8px rgba(247,215,116,.7); }
.byte-node.is-boundary { margin-left:7px; }
.byte-node.is-boundary::before,.entropy-column.is-boundary::before { content:""; position:absolute; left:-7px; top:0; bottom:0; width:2px; background:#f7d774; border-radius:2px; opacity:0; animation:boundary-in .25s ease forwards; animation-delay:calc(.72s + var(--i) * 12ms); }

.entropy-legend { display:flex; gap:13px; align-items:center; padding:0 2px 5px; color:#718794; font-size:9px; }
.entropy-legend span { display:flex; align-items:center; gap:5px; }
.entropy-legend i { width:7px; height:7px; border-radius:2px; background:#54d2d2; }
.entropy-legend i.hot { background:#ff9b54; }
.entropy-legend b { margin-left:auto; color:#f7d774; font:9px Menlo,monospace; }
.entropy-track { align-items:flex-end; min-height:96px; }
.entropy-column { position:relative; flex:0 0 43px; height:84px; display:flex; justify-content:flex-end; flex-direction:column; align-items:center; }
.entropy-column.is-boundary { margin-left:7px; }
.entropy-column small { color:#718794; font:8px Menlo,monospace; height:13px; }
.entropy-bar { width:72%; height:var(--height); transform:scaleY(0); transform-origin:bottom; background:#54d2d2; border-radius:4px 4px 1px 1px; opacity:.82; animation:bar-grow .42s cubic-bezier(.2,.8,.2,1) forwards; animation-delay:calc(.62s + var(--i) * 13ms); }
.entropy-column.is-hot .entropy-bar { background:#ff9b54; }
.threshold-rule { position:absolute; left:3px; right:3px; border-top:1px dashed rgba(247,215,116,.72); z-index:2; pointer-events:none; opacity:0; animation:rule-in .35s ease .7s forwards; }
.threshold-rule span { position:absolute; right:2px; bottom:3px; color:#f7d774; background:#08131c; padding:1px 4px; font:8px Menlo,monospace; }

.patch-track { gap:7px; padding-bottom:12px; }
.patch-cluster { flex:0 0 max(82px,calc(var(--len) * 48px)); min-height:71px; display:flex; flex-direction:column; justify-content:center; gap:5px; padding:8px 10px; border:1px solid var(--patch); border-radius:10px; background:color-mix(in srgb,var(--patch) 9%,#0c1923); opacity:0; transform:translateY(13px); animation:patch-in .38s cubic-bezier(.2,.8,.2,1) forwards; animation-delay:calc(1.28s + var(--p) * 55ms); }
.patch-heading { display:flex; justify-content:space-between; align-items:center; gap:10px; }
.patch-heading b { color:var(--patch); font:700 10px Menlo,monospace; }
.patch-heading span { color:#718794; font-size:9px; }
.patch-text { color:#e7f0f5; font:600 12px Menlo,monospace; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.patch-cluster small { color:#718794; font-size:8px; }
.model-flow { display:grid; grid-template-columns:1fr auto 1fr auto 1fr auto 1.2fr; gap:7px; align-items:center; margin-top:15px; opacity:0; animation:stage-in .4s ease 1.75s forwards; }
.model-flow span { min-height:31px; display:grid; place-items:center; padding:5px 7px; border:1px solid #294554; border-radius:8px; color:#a9bac5; font-size:8px; text-align:center; letter-spacing:.04em; }
.model-flow i { color:#54d2d2; font-style:normal; animation:arrow-pulse 1.3s ease-in-out infinite; }

.animation-placeholder { display:grid; place-items:center; align-content:center; text-align:center; color:#718794; }
.animation-placeholder strong { color:#d9e5eb; margin-top:13px; }
.animation-placeholder p { margin:4px 0 0; font-size:12px; }
.placeholder-orbit { display:flex; gap:7px; }
.placeholder-orbit span { width:15px; height:15px; border:1px solid #54d2d2; border-radius:4px; animation:placeholder-wave 1.5s ease-in-out infinite; }
.placeholder-orbit span:nth-child(2) { animation-delay:.15s; border-color:#f7d774; }
.placeholder-orbit span:nth-child(3) { animation-delay:.3s; border-color:#ff9b54; }

.metric-shell { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin:12px 0; }
.metric { background:#0c1923; border:1px solid #203746; border-radius:13px; padding:13px 14px; display:flex; flex-direction:column; min-height:86px; }
.metric span { color:#89a0af; font-size:11px; letter-spacing:.08em; text-transform:uppercase; }
.metric strong { color:#e7f0f5; font-size:25px; margin-top:7px; line-height:1; }
.metric small { color:#89a0af; font-size:13px; }
.metric.wide { grid-column:span 4; min-height:72px; flex-direction:row; align-items:center; gap:14px; }
.metric.wide strong { margin:0; font-size:18px; }
.metric.wide em { margin-left:auto; color:#f7d774; font-style:normal; font-family:Menlo,monospace; font-size:12px; }
.scope-note { color:#a9bac5; border-left:3px solid #ff9b54; background:rgba(255,155,84,.07); padding:12px 14px; border-radius:4px 12px 12px 4px; line-height:1.5; font-size:13px; }
footer { color:#647a88; font-size:12px; padding:20px 2px 0; }

@keyframes stage-in { to { opacity:1; transform:translateY(0); } }
@keyframes byte-in { to { opacity:1; transform:translateY(0) scale(1); } }
@keyframes bar-grow { to { transform:scaleY(1); } }
@keyframes boundary-in { 50% { opacity:1; box-shadow:0 0 10px rgba(247,215,116,.7); } 100% { opacity:.72; } }
@keyframes rule-in { to { opacity:1; } }
@keyframes patch-in { to { opacity:1; transform:translateY(0); } }
@keyframes live-pulse { 70% { box-shadow:0 0 0 7px rgba(84,210,210,0); } 100% { box-shadow:0 0 0 0 rgba(84,210,210,0); } }
@keyframes arrow-pulse { 50% { color:#f7d774; transform:translateX(2px); } }
@keyframes placeholder-wave { 50% { transform:translateY(-7px); } }

@media (max-width:760px) {
  .metric-shell { grid-template-columns:repeat(2,1fr); }
  .metric.wide { grid-column:span 2; align-items:flex-start; flex-direction:column; }
  .metric.wide em { margin-left:0; }
  .prompt-card { grid-template-columns:1fr; gap:5px; }
  .model-flow { grid-template-columns:1fr; }
  .model-flow i { transform:rotate(90deg); justify-self:center; }
}
@media (prefers-reduced-motion:reduce) {
  .animation-shell * { animation-duration:.001ms !important; animation-delay:0ms !important; }
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
                  <p>Type a prompt and play a live, in-page explanation of how
                  the Python entropy policy turns UTF-8 bytes into latent patch
                  positions. Nothing is rendered to video.</p>
                </header>
                """
            )
            with gr.Row(equal_height=False):
                with gr.Column(scale=4, elem_classes="panel"):
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
                    animate = gr.Button("Animate patching", variant="primary", elem_classes="render-button")
                    gr.Examples(
                        examples=[
                            ["byte models allocate compute at surprising transitions."],
                            ["hello café ☕ — bytes are not characters"],
                            ["def entropy(p): return -sum(x * log2(x) for x in p)"],
                        ],
                        inputs=[prompt],
                    )
                with gr.Column(scale=8):
                    animation = gr.HTML(
                        value=_placeholder_html(),
                        elem_classes="animation-output",
                        min_height=500,
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
            gr.HTML("<footer>Python + Gradio · notebook-derived educational model · live in-page animation</footer>")

        animate.click(
            fn=visualize,
            inputs=[prompt, strategy, threshold, minimum, maximum, fixed_size],
            outputs=[animation, summary, table, scope_note],
            api_name="visualize_patching",
        )
        prompt.submit(
            fn=visualize,
            inputs=[prompt, strategy, threshold, minimum, maximum, fixed_size],
            outputs=[animation, summary, table, scope_note],
            api_name=False,
        )
    return demo


if __name__ == "__main__":
    build_app().queue(default_concurrency_limit=1).launch(inbrowser=False, css=CSS)
