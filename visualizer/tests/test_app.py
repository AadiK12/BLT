from app import _animation_html
from blt_visualizer import analyze_prompt


def test_animation_is_in_page_html_not_video():
    result = analyze_prompt("BLT patches bytes.")
    animation = _animation_html(result)

    assert "patch-animation" not in animation  # old renderer-specific name
    assert "animation-shell" in animation
    assert animation.count('class="byte-node') == result.byte_count
    assert animation.count('class="patch-cluster"') == result.patch_count
    assert "<video" not in animation
    assert ".mp4" not in animation


def test_animation_escapes_prompt_content():
    result = analyze_prompt('<script>alert("no")</script>')
    animation = _animation_html(result)

    assert "<script>" not in animation
    assert "&lt;script&gt;" in animation
