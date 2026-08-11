"""Small, inspectable patchers extracted from the Stage 1 research notebook.

The entropy model here is intentionally educational. It is a smoothed byte
bigram model trained on a tiny fixed corpus, not the learned entropy model from
the BLT paper and not a local encoder.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import math
from typing import Literal


DEFAULT_PROMPT = "byte models allocate compute at surprising transitions."
MAX_PROMPT_BYTES = 160

TOY_CORPUS = (
    "byte models predict the next byte. "
    "byte latent transformers group predictable bytes into patches. "
    "the global model processes patches. "
    "the local model processes bytes. "
    "entropy rises when the continuation is surprising. "
) * 20

Strategy = Literal["entropy", "fixed", "whitespace"]


@dataclass(frozen=True)
class ByteObservation:
    index: int
    value: int
    hex: str
    glyph: str
    entropy: float
    patch_id: int
    starts_patch: bool
    boundary_reason: str | None


@dataclass(frozen=True)
class PatchObservation:
    patch_id: int
    start: int
    end: int
    length: int
    text: str
    hex: str
    mean_entropy: float
    boundary_reason: str


@dataclass(frozen=True)
class PatchingResult:
    prompt: str
    strategy: Strategy
    byte_count: int
    patch_count: int
    average_patch_size: float
    nominal_global_step_reduction: float
    threshold_bits: float | None
    bytes: tuple[ByteObservation, ...]
    patches: tuple[PatchObservation, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def train_bigram_counts(text: str) -> tuple[dict[int, Counter], Counter]:
    data = text.encode("utf-8")
    transition_counts: dict[int, Counter] = defaultdict(Counter)
    global_counts = Counter(data)
    for previous, current in zip(data, data[1:]):
        transition_counts[previous][current] += 1
    return dict(transition_counts), global_counts


TRANSITION_COUNTS, GLOBAL_COUNTS = train_bigram_counts(TOY_CORPUS)


def smoothed_distribution(counts: Counter, alpha: float = 0.01) -> list[float]:
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    denominator = sum(counts.values()) + alpha * 256
    return [(counts[value] + alpha) / denominator for value in range(256)]


def entropy_bits(probabilities: list[float]) -> float:
    return -sum(
        probability * math.log2(probability)
        for probability in probabilities
        if probability > 0
    )


def next_byte_entropies(data: bytes) -> list[float]:
    if not data:
        return []
    global_distribution = smoothed_distribution(GLOBAL_COUNTS)
    entropies = [entropy_bits(global_distribution)]
    for index in range(1, len(data)):
        previous = data[index - 1]
        counts = TRANSITION_COUNTS.get(previous, GLOBAL_COUNTS)
        entropies.append(entropy_bits(smoothed_distribution(counts)))
    return entropies


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be between 0 and 1")
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def _entropy_patch_starts(
    data: bytes,
    entropies: list[float],
    threshold: float,
    min_patch_size: int,
    max_patch_size: int,
) -> tuple[list[int], dict[int, str]]:
    if len(data) != len(entropies):
        raise ValueError("data and entropies must have equal lengths")
    if min_patch_size <= 0:
        raise ValueError("min_patch_size must be positive")
    if max_patch_size < min_patch_size:
        raise ValueError("max_patch_size must be at least min_patch_size")
    if not data:
        return [], {}

    starts = [0]
    reasons = {0: "sequence start"}
    current_start = 0
    for position in range(1, len(data)):
        current_size = position - current_start
        crosses_threshold = (
            entropies[position] >= threshold and current_size >= min_patch_size
        )
        reaches_maximum = current_size >= max_patch_size
        if crosses_threshold or reaches_maximum:
            starts.append(position)
            if reaches_maximum and crosses_threshold:
                reasons[position] = "high entropy + maximum length"
            elif reaches_maximum:
                reasons[position] = "maximum length"
            else:
                reasons[position] = "high entropy"
            current_start = position
    return starts, reasons


def _fixed_patch_starts(data: bytes, patch_size: int) -> tuple[list[int], dict[int, str]]:
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    starts = list(range(0, len(data), patch_size))
    reasons = {
        start: "sequence start" if start == 0 else f"fixed width {patch_size}"
        for start in starts
    }
    return starts, reasons


def _whitespace_patch_starts(data: bytes) -> tuple[list[int], dict[int, str]]:
    if not data:
        return [], {}
    starts = [0]
    reasons = {0: "sequence start"}
    for position, previous in enumerate(data[:-1], start=1):
        if chr(previous).isspace():
            starts.append(position)
            reasons[position] = "after whitespace"
    return starts, reasons


def byte_glyph(value: int) -> str:
    if value == 32:
        return "␠"
    if value == 10:
        return "↵"
    if value == 9:
        return "⇥"
    if 33 <= value <= 126:
        return chr(value)
    return "·"


def visible_bytes(data: bytes) -> str:
    return (
        data.decode("utf-8", errors="replace")
        .replace(" ", "␠")
        .replace("\n", "↵")
        .replace("\t", "⇥")
    )


def analyze_prompt(
    prompt: str,
    strategy: Strategy = "entropy",
    threshold_percentile: float = 0.70,
    min_patch_size: int = 2,
    max_patch_size: int = 8,
    fixed_patch_size: int = 4,
) -> PatchingResult:
    data = prompt.encode("utf-8")
    if not data:
        raise ValueError("Enter at least one character to visualize.")
    if len(data) > MAX_PROMPT_BYTES:
        raise ValueError(
            f"Prompt is {len(data)} UTF-8 bytes; keep it at or below "
            f"{MAX_PROMPT_BYTES} bytes for this visualizer."
        )

    entropies = next_byte_entropies(data)
    threshold: float | None = None
    if strategy == "entropy":
        calibration_values = entropies[1:] or entropies
        threshold = percentile(calibration_values, threshold_percentile)
        starts, reasons = _entropy_patch_starts(
            data,
            entropies,
            threshold,
            min_patch_size,
            max_patch_size,
        )
    elif strategy == "fixed":
        starts, reasons = _fixed_patch_starts(data, fixed_patch_size)
    elif strategy == "whitespace":
        starts, reasons = _whitespace_patch_starts(data)
    else:
        raise ValueError(f"Unknown patching strategy: {strategy}")

    patches: list[PatchObservation] = []
    byte_rows: list[ByteObservation] = []
    for patch_id, start in enumerate(starts):
        end = starts[patch_id + 1] if patch_id + 1 < len(starts) else len(data)
        patch_data = data[start:end]
        patch_entropies = entropies[start:end]
        reason = reasons[start]
        patches.append(
            PatchObservation(
                patch_id=patch_id,
                start=start,
                end=end,
                length=end - start,
                text=visible_bytes(patch_data),
                hex=" ".join(f"{value:02X}" for value in patch_data),
                mean_entropy=sum(patch_entropies) / len(patch_entropies),
                boundary_reason=reason,
            )
        )
        for index in range(start, end):
            value = data[index]
            byte_rows.append(
                ByteObservation(
                    index=index,
                    value=value,
                    hex=f"{value:02X}",
                    glyph=byte_glyph(value),
                    entropy=entropies[index],
                    patch_id=patch_id,
                    starts_patch=index == start,
                    boundary_reason=reason if index == start else None,
                )
            )

    patch_count = len(patches)
    byte_count = len(data)
    return PatchingResult(
        prompt=prompt,
        strategy=strategy,
        byte_count=byte_count,
        patch_count=patch_count,
        average_patch_size=byte_count / patch_count,
        nominal_global_step_reduction=1 - (patch_count / byte_count),
        threshold_bits=threshold,
        bytes=tuple(byte_rows),
        patches=tuple(patches),
    )

