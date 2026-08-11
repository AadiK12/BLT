"""Manim scene rendered by the Gradio app for one patching analysis."""

from __future__ import annotations

import json
import os
from pathlib import Path

from manim import (
    AnimationGroup,
    Arrow,
    Create,
    DashedLine,
    DOWN,
    FadeIn,
    FadeOut,
    GrowFromCenter,
    LaggedStart,
    LEFT,
    Line,
    ORIGIN,
    Rectangle,
    ReplacementTransform,
    RIGHT,
    RoundedRectangle,
    Scene,
    Text,
    UP,
    VGroup,
    config,
)


BACKGROUND = "#071018"
INK = "#E7F0F5"
MUTED = "#8294A3"
CYAN = "#54D2D2"
ORANGE = "#FF9B54"
YELLOW = "#F7D774"
PATCH_COLORS = ["#54D2D2", "#F7D774", "#FF9B54", "#AE8CFF", "#6ED39D"]
MAX_ANIMATED_BYTES = 36

config.background_color = BACKGROUND


def _safe_prompt(text: str, limit: int = 72) -> str:
    clean = text.replace("\n", " ↵ ").replace("\t", " ⇥ ")
    return clean if len(clean) <= limit else clean[: limit - 1] + "…"


class BLTPatchingScene(Scene):
    def construct(self) -> None:
        input_path = os.environ.get("BLT_VIZ_CONFIG")
        if not input_path:
            raise RuntimeError("BLT_VIZ_CONFIG must point to an analysis JSON file")
        analysis = json.loads(Path(input_path).read_text(encoding="utf-8"))
        byte_rows = analysis["bytes"][:MAX_ANIMATED_BYTES]
        truncated = len(analysis["bytes"]) > len(byte_rows)

        title = Text(
            "BYTE LATENT TRANSFORMER",
            font="Avenir Next",
            weight="BOLD",
            font_size=34,
            color=INK,
        ).to_edge(UP, buff=0.28).to_edge(LEFT, buff=0.45)
        kicker = Text(
            "PATCHING LAB  /  NOTEBOOK MODEL",
            font="Avenir Next",
            font_size=15,
            color=CYAN,
        ).next_to(title, DOWN, aligned_edge=LEFT, buff=0.08)
        self.play(FadeIn(title, shift=0.18 * DOWN), FadeIn(kicker), run_time=0.6)

        prompt_label = Text("PROMPT", font="Avenir Next", font_size=13, color=MUTED)
        prompt_text = Text(
            _safe_prompt(analysis["prompt"]),
            font="Menlo",
            font_size=22,
            color=INK,
        )
        prompt_box = RoundedRectangle(
            corner_radius=0.12,
            width=12.4,
            height=0.66,
            stroke_color="#29404F",
            stroke_width=1.4,
            fill_color="#0C1923",
            fill_opacity=1,
        )
        prompt_group = VGroup(prompt_box, prompt_text)
        prompt_text.move_to(prompt_box).align_to(prompt_box, LEFT).shift(0.25 * RIGHT)
        prompt_group.move_to(2.25 * UP)
        prompt_label.next_to(prompt_box, UP, aligned_edge=LEFT, buff=0.08)
        self.play(Create(prompt_box), FadeIn(prompt_label), FadeIn(prompt_text), run_time=0.65)

        byte_label = Text("UTF-8 BYTES", font="Avenir Next", font_size=13, color=MUTED)
        byte_label.move_to(1.55 * UP + 5.75 * LEFT)
        self.play(FadeIn(byte_label), run_time=0.25)

        count = max(len(byte_rows), 1)
        gap = 0.07
        available_width = 12.3
        cell_width = min(0.72, (available_width - gap * (count - 1)) / count)
        cell_height = 0.62
        byte_cells = VGroup()
        for row in byte_rows:
            patch_color = PATCH_COLORS[row["patch_id"] % len(PATCH_COLORS)]
            box = RoundedRectangle(
                corner_radius=0.06,
                width=cell_width,
                height=cell_height,
                stroke_color=patch_color,
                stroke_width=1.5,
                fill_color="#102330",
                fill_opacity=1,
            )
            glyph = Text(
                row["glyph"],
                font="Menlo",
                font_size=max(10, min(20, int(cell_width * 29))),
                color=INK,
            ).move_to(box)
            cell = VGroup(box, glyph)
            byte_cells.add(cell)
        byte_cells.arrange(RIGHT, buff=gap).move_to(1.05 * UP)
        self.play(
            LaggedStart(*(GrowFromCenter(cell) for cell in byte_cells), lag_ratio=0.025),
            run_time=0.9,
        )

        entropy_label = Text("NEXT-BYTE ENTROPY", font="Avenir Next", font_size=13, color=MUTED)
        entropy_label.move_to(0.48 * UP + 5.35 * LEFT)
        self.play(FadeIn(entropy_label), run_time=0.2)

        threshold = analysis.get("threshold_bits")
        entropy_bars = VGroup()
        baseline_y = -0.82
        for cell, row in zip(byte_cells, byte_rows):
            height = 0.18 + (min(float(row["entropy"]), 8.0) / 8.0) * 1.02
            is_hot = threshold is not None and row["entropy"] >= threshold
            bar = Rectangle(
                width=max(cell_width * 0.72, 0.035),
                height=height,
                stroke_width=0,
                fill_color=ORANGE if is_hot else CYAN,
                fill_opacity=0.82,
            )
            bar.move_to([cell.get_center()[0], baseline_y + height / 2, 0])
            entropy_bars.add(bar)
        self.play(
            LaggedStart(*(GrowFromCenter(bar) for bar in entropy_bars), lag_ratio=0.02),
            run_time=0.75,
        )

        if threshold is not None:
            line_y = baseline_y + 0.18 + (min(float(threshold), 8.0) / 8.0) * 1.02
            threshold_line = DashedLine(
                [byte_cells.get_left()[0], line_y, 0],
                [byte_cells.get_right()[0], line_y, 0],
                dash_length=0.08,
                color=YELLOW,
                stroke_width=1.5,
            )
            threshold_text = Text(
                f"threshold  {threshold:.2f} bits",
                font="Menlo",
                font_size=11,
                color=YELLOW,
            ).move_to([byte_cells.get_left()[0] + 0.9, line_y + 0.12, 0])
            self.play(Create(threshold_line), FadeIn(threshold_text), run_time=0.45)

        starts = [index for index, row in enumerate(byte_rows) if row["starts_patch"] and index > 0]
        boundary_lines = VGroup()
        for index in starts:
            x = (byte_cells[index - 1].get_right()[0] + byte_cells[index].get_left()[0]) / 2
            boundary_lines.add(
                Line([x, 1.4, 0], [x, -0.95, 0], color=YELLOW, stroke_width=2.4)
            )
        if boundary_lines:
            self.play(
                LaggedStart(*(Create(line) for line in boundary_lines), lag_ratio=0.08),
                run_time=0.65,
            )

        segments: list[tuple[int, int, int]] = []
        start = 0
        while start < len(byte_rows):
            patch_id = byte_rows[start]["patch_id"]
            end = start + 1
            while end < len(byte_rows) and byte_rows[end]["patch_id"] == patch_id:
                end += 1
            segments.append((patch_id, start, end))
            start = end

        patch_cards = VGroup()
        patch_arrows = VGroup()
        for patch_id, start, end in segments:
            left = byte_cells[start].get_left()[0]
            right = byte_cells[end - 1].get_right()[0]
            width = max(0.55, right - left)
            color = PATCH_COLORS[patch_id % len(PATCH_COLORS)]
            card = RoundedRectangle(
                corner_radius=0.09,
                width=width,
                height=0.64,
                stroke_color=color,
                stroke_width=1.6,
                fill_color=color,
                fill_opacity=0.14,
            )
            card.move_to([(left + right) / 2, -1.55, 0])
            patch_label = (
                f"P{patch_id}" if width < 0.82 else f"P{patch_id} · {end - start}B"
            )
            label = Text(
                patch_label,
                font="Menlo",
                font_size=max(8, min(14, int(width * 11))),
                color=INK,
            ).move_to(card)
            patch_cards.add(VGroup(card, label))
            patch_arrows.add(
                Arrow(
                    start=[(left + right) / 2, baseline_y - 0.05, 0],
                    end=[(left + right) / 2, -1.2, 0],
                    buff=0.03,
                    color=color,
                    stroke_width=1.4,
                    max_tip_length_to_length_ratio=0.22,
                )
            )
        self.play(
            AnimationGroup(
                LaggedStart(*(Create(arrow) for arrow in patch_arrows), lag_ratio=0.06),
                LaggedStart(*(GrowFromCenter(card) for card in patch_cards), lag_ratio=0.06),
                lag_ratio=0.25,
            ),
            run_time=0.95,
        )

        flow_y = -2.6
        stage_specs = [
            ("LOCAL BYTE MODEL", CYAN),
            ("BOUNDARY POLICY", YELLOW),
            ("PATCH ENCODER", ORANGE),
            ("GLOBAL TRANSFORMER", "#AE8CFF"),
        ]
        stages = VGroup()
        stage_arrows = VGroup()
        for label, color in stage_specs:
            rect = RoundedRectangle(
                corner_radius=0.08,
                width=2.6,
                height=0.48,
                stroke_color=color,
                stroke_width=1.2,
                fill_color=color,
                fill_opacity=0.1,
            )
            text = Text(label, font="Avenir Next", font_size=12, color=INK).move_to(rect)
            stages.add(VGroup(rect, text))
        stages.arrange(RIGHT, buff=0.48).move_to([0, flow_y, 0])
        for left_stage, right_stage in zip(stages, stages[1:]):
            stage_arrows.add(
                Arrow(
                    left_stage.get_right(),
                    right_stage.get_left(),
                    buff=0.08,
                    color=MUTED,
                    stroke_width=1.2,
                    max_tip_length_to_length_ratio=0.28,
                )
            )
        self.play(
            LaggedStart(*(FadeIn(stage) for stage in stages), lag_ratio=0.08),
            LaggedStart(*(Create(arrow) for arrow in stage_arrows), lag_ratio=0.12),
            run_time=0.9,
        )

        strategy_name = {
            "entropy": "TOY ENTROPY",
            "fixed": "FIXED WIDTH",
            "whitespace": "WHITESPACE",
        }[analysis["strategy"]]
        metrics = Text(
            f"{analysis['byte_count']} bytes  →  {analysis['patch_count']} patches"
            f"   ·   avg {analysis['average_patch_size']:.2f} bytes/patch"
            f"   ·   {strategy_name}",
            font="Menlo",
            font_size=14,
            color=INK,
        ).move_to(3.25 * DOWN)
        caveat = Text(
            "educational routing demo · not a trained BLT encoder"
            + (" · first 36 bytes shown" if truncated else ""),
            font="Avenir Next",
            font_size=11,
            color=MUTED,
        ).next_to(metrics, DOWN, buff=0.08)
        self.play(FadeIn(metrics, shift=0.1 * UP), FadeIn(caveat), run_time=0.45)
        self.wait(1.25)
