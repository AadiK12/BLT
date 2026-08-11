"""Render the default prompt without starting the browser UI."""

from app import _render_video
from blt_visualizer import DEFAULT_PROMPT, analyze_prompt


if __name__ == "__main__":
    analysis = analyze_prompt(DEFAULT_PROMPT)
    print(_render_video(analysis.to_dict()))
