"""Dependency-free mathematical references for bounded correctness tests."""

from __future__ import annotations

import math
from collections.abc import Sequence


def scalar_multiply(values: Sequence[float], scalar: float) -> list[float]:
    if not values:
        raise ValueError("values must not be empty")
    return [float(value) * float(scalar) for value in values]


def matmul(
    left: Sequence[Sequence[float]],
    right: Sequence[Sequence[float]],
) -> list[list[float]]:
    if not left or not right or not left[0] or not right[0]:
        raise ValueError("matrices must not be empty")
    if any(len(row) != len(left[0]) for row in left):
        raise ValueError("left must be rectangular")
    if any(len(row) != len(right[0]) for row in right):
        raise ValueError("right must be rectangular")
    if len(left[0]) != len(right):
        raise ValueError("matmul inner dimensions must match")
    return [
        [
            sum(
                float(left[row][inner]) * float(right[inner][column])
                for inner in range(len(right))
            )
            for column in range(len(right[0]))
        ]
        for row in range(len(left))
    ]


def stable_softmax(values: Sequence[float]) -> list[float]:
    if not values:
        raise ValueError("values must not be empty")
    maximum = max(values)
    exponentials = [math.exp(float(value) - maximum) for value in values]
    denominator = sum(exponentials)
    return [value / denominator for value in exponentials]


def layer_norm(values: Sequence[float], epsilon: float = 1e-5) -> list[float]:
    if not values:
        raise ValueError("values must not be empty")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    mean = sum(float(value) for value in values) / len(values)
    variance = sum((float(value) - mean) ** 2 for value in values) / len(values)
    inverse_std = 1.0 / math.sqrt(variance + epsilon)
    return [(float(value) - mean) * inverse_std for value in values]


def gelu_tanh(value: float) -> float:
    coefficient = math.sqrt(2.0 / math.pi)
    return 0.5 * value * (
        1.0 + math.tanh(coefficient * (value + 0.044715 * value**3))
    )
