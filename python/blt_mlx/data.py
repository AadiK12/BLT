"""Deterministic byte-level training and evaluation data contracts."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol

import mlx.core as mx


@dataclass(frozen=True)
class LanguageModelBatch:
    """A batch plus the masks required for packed, padded language modeling."""

    inputs: mx.array
    targets: mx.array
    attention_mask: mx.array
    loss_mask: mx.array
    document_ids: mx.array


class LanguageModelDataset(Protocol):
    def language_model_batch(self, step: int) -> LanguageModelBatch: ...


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

    def language_model_batch(self, step: int) -> LanguageModelBatch:
        inputs, targets = self.batch(step)
        shape = inputs.shape
        return LanguageModelBatch(
            inputs=inputs,
            targets=targets,
            attention_mask=mx.ones(shape, dtype=mx.bool_),
            loss_mask=mx.ones(shape, dtype=mx.float32),
            document_ids=mx.zeros(shape, dtype=mx.int32),
        )


@dataclass(frozen=True)
class PackedByteDataset:
    """Pack documents without training on or attending across their boundaries."""

    documents: tuple[bytes, ...]
    sequence_length: int
    batch_size: int
    seed: int
    pad_byte: int = 0

    def __post_init__(self) -> None:
        if self.sequence_length <= 0 or self.batch_size <= 0:
            raise ValueError("sequence_length and batch_size must be positive")
        if not self.documents or any(not document for document in self.documents):
            raise ValueError("documents must contain non-empty byte strings")
        if not 0 <= self.pad_byte <= 255:
            raise ValueError("pad_byte must be between zero and 255")
        if not self._examples():
            raise ValueError("documents must contain at least one within-document target")

    @classmethod
    def from_text_documents(
        cls,
        documents: list[str] | tuple[str, ...],
        *,
        sequence_length: int,
        batch_size: int,
        seed: int,
    ) -> PackedByteDataset:
        return cls(
            documents=tuple(document.encode("utf-8") for document in documents),
            sequence_length=sequence_length,
            batch_size=batch_size,
            seed=seed,
        )

    def _examples(self) -> tuple[tuple[list[int], list[int]], ...]:
        window_size = self.sequence_length + 1
        examples: list[tuple[list[int], list[int]]] = []
        tokens: list[int] = []
        document_ids: list[int] = []

        def finish_window() -> None:
            if not tokens:
                return
            padded_tokens = tokens + [self.pad_byte] * (window_size - len(tokens))
            padded_documents = document_ids + [-1] * (window_size - len(document_ids))
            has_target = any(
                left >= 0 and left == right
                for left, right in zip(
                    padded_documents[:-1], padded_documents[1:], strict=True
                )
            )
            if has_target:
                examples.append((padded_tokens, padded_documents))
            tokens.clear()
            document_ids.clear()

        for document_index, document in enumerate(self.documents):
            offset = 0
            while offset < len(document):
                available = window_size - len(tokens)
                take = min(available, len(document) - offset)
                tokens.extend(document[offset : offset + take])
                document_ids.extend([document_index] * take)
                offset += take
                if len(tokens) == window_size:
                    finish_window()
        finish_window()
        return tuple(examples)

    def language_model_batch(self, step: int) -> LanguageModelBatch:
        if step < 0:
            raise ValueError("step must be non-negative")
        examples = self._examples()
        rng = random.Random(self.seed + step * 1_000_003)
        selected = [examples[rng.randrange(len(examples))] for _ in range(self.batch_size)]
        inputs: list[list[int]] = []
        targets: list[list[int]] = []
        attention_masks: list[list[bool]] = []
        loss_masks: list[list[float]] = []
        input_document_ids: list[list[int]] = []
        for tokens, document_ids in selected:
            input_ids = document_ids[:-1]
            target_ids = document_ids[1:]
            inputs.append(tokens[:-1])
            targets.append(tokens[1:])
            attention_masks.append([document_id >= 0 for document_id in input_ids])
            loss_masks.append(
                [
                    1.0 if left >= 0 and left == right else 0.0
                    for left, right in zip(input_ids, target_ids, strict=True)
                ]
            )
            input_document_ids.append(input_ids)
        return LanguageModelBatch(
            inputs=mx.array(inputs, dtype=mx.int32),
            targets=mx.array(targets, dtype=mx.int32),
            attention_mask=mx.array(attention_masks, dtype=mx.bool_),
            loss_mask=mx.array(loss_masks, dtype=mx.float32),
            document_ids=mx.array(input_document_ids, dtype=mx.int32),
        )

    def batch(self, step: int) -> tuple[mx.array, mx.array]:
        batch = self.language_model_batch(step)
        return batch.inputs, batch.targets


DEFAULT_TINY_CORPUS = (
    "Byte latent transformers learn directly from UTF-8 bytes. "
    "Phase two proves that this tiny byte model can learn, save, reload, and generate.\n"
)
