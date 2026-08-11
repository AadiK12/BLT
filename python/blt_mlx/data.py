"""Deterministic byte-level training and evaluation data contracts."""

from __future__ import annotations

import random
from dataclasses import dataclass

import mlx.core as mx


@dataclass(frozen=True)
class ByteDataset:
    data: bytes
    sequence_length: int
    batch_size: int
    seed: int

    def __post_init__(self) -> None:
        if len(self.data) <= self.sequence_length:
            raise ValueError("dataset must contain more bytes than one sequence")
        if self.sequence_length <= 0 or self.batch_size <= 0:
            raise ValueError("sequence_length and batch_size must be positive")

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        sequence_length: int,
        batch_size: int,
        seed: int,
        repeat_to_at_least: int = 4096,
    ) -> ByteDataset:
        encoded = text.encode("utf-8")
        if not encoded:
            raise ValueError("training text must not be empty")
        repeats = max(1, (repeat_to_at_least + len(encoded) - 1) // len(encoded))
        return cls(
            data=encoded * repeats,
            sequence_length=sequence_length,
            batch_size=batch_size,
            seed=seed,
        )

    def batch(self, step: int) -> tuple[mx.array, mx.array]:
        if step < 0:
            raise ValueError("step must be non-negative")
        rng = random.Random(self.seed + step * 1_000_003)
        maximum_start = len(self.data) - self.sequence_length - 1
        inputs: list[list[int]] = []
        targets: list[list[int]] = []
        for _ in range(self.batch_size):
            start = rng.randrange(maximum_start + 1)
            window = self.data[start : start + self.sequence_length + 1]
            inputs.append(list(window[:-1]))
            targets.append(list(window[1:]))
        return mx.array(inputs, dtype=mx.int32), mx.array(targets, dtype=mx.int32)


DEFAULT_TINY_CORPUS = (
    "Byte latent transformers learn directly from UTF-8 bytes. "
    "Phase two proves that this tiny byte model can learn, save, reload, and generate.\n"
)
