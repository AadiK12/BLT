"""Operation-shape tracing for model-derived benchmark suites."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import mlx.core as mx


@dataclass(frozen=True)
class OperationShape:
    name: str
    kind: str
    phase: str
    left_shape: tuple[int, ...]
    right_shape: tuple[int, ...] | None
    output_shape: tuple[int, ...]
    dtype: str

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["left_shape"] = list(self.left_shape)
        value["right_shape"] = (
            list(self.right_shape) if self.right_shape is not None else None
        )
        value["output_shape"] = list(self.output_shape)
        return value


class ShapeRecorder:
    """Collect logical operation shapes without changing model numerics."""

    def __init__(self, *, phase: str) -> None:
        if phase not in {"training", "prefill", "decode"}:
            raise ValueError("phase must be training, prefill, or decode")
        self.phase = phase
        self.operations: list[OperationShape] = []

    def record(
        self,
        *,
        name: str,
        kind: str,
        left: mx.array,
        output: mx.array,
        right: mx.array | None = None,
    ) -> None:
        self.operations.append(
            OperationShape(
                name=name,
                kind=kind,
                phase=self.phase,
                left_shape=tuple(int(value) for value in left.shape),
                right_shape=(
                    tuple(int(value) for value in right.shape)
                    if right is not None
                    else None
                ),
                output_shape=tuple(int(value) for value in output.shape),
                dtype=str(output.dtype),
            )
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "phase": self.phase,
            "operations": [operation.as_dict() for operation in self.operations],
        }

    def unique_matmul_cases(self) -> list[OperationShape]:
        unique: dict[tuple[Any, ...], OperationShape] = {}
        for operation in self.operations:
            if operation.kind != "matmul" or operation.right_shape is None:
                continue
            key = (
                operation.left_shape,
                operation.right_shape,
                operation.output_shape,
                operation.dtype,
                operation.phase,
            )
            unique.setdefault(key, operation)
        return list(unique.values())
