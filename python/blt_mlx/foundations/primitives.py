"""Transparent MLX implementations of the Phase 2 operation contracts."""

from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx


def _require_float(value: mx.array, *, name: str) -> None:
    if value.dtype not in (mx.float16, mx.float32, mx.bfloat16):
        raise TypeError(f"{name} must use float16, float32, or bfloat16")


def deterministic_parameter(
    shape: Sequence[int],
    *,
    seed: int,
    scale: float = 1.0,
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    normalized = tuple(int(dimension) for dimension in shape)
    if not normalized or any(dimension <= 0 for dimension in normalized):
        raise ValueError("shape must contain positive dimensions")
    key = mx.random.key(int(seed))
    return (mx.random.normal(shape=normalized, key=key) * scale).astype(dtype)


def gelu_tanh(values: mx.array) -> mx.array:
    _require_float(values, name="values")
    coefficient = 0.7978845608028654
    return 0.5 * values * (
        1.0 + mx.tanh(coefficient * (values + 0.044715 * values * values * values))
    )


def bias_gelu(values: mx.array, bias: mx.array) -> mx.array:
    _require_float(values, name="values")
    _require_float(bias, name="bias")
    if bias.ndim != 1 or bias.shape[0] != values.shape[-1]:
        raise ValueError("bias must match the final values dimension")
    if values.dtype != bias.dtype:
        raise TypeError("values and bias must use the same dtype")
    return gelu_tanh(values + bias)


compiled_bias_gelu = mx.compile(bias_gelu)


def stable_softmax(values: mx.array, *, axis: int = -1) -> mx.array:
    _require_float(values, name="values")
    shifted = values - mx.max(values, axis=axis, keepdims=True)
    exponentials = mx.exp(shifted)
    return exponentials / mx.sum(exponentials, axis=axis, keepdims=True)


def layer_norm(
    values: mx.array,
    weight: mx.array | None = None,
    bias: mx.array | None = None,
    *,
    epsilon: float = 1e-5,
) -> mx.array:
    _require_float(values, name="values")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    mean = mx.mean(values, axis=-1, keepdims=True)
    centered = values - mean
    variance = mx.mean(centered * centered, axis=-1, keepdims=True)
    normalized = centered * mx.rsqrt(variance + epsilon)
    if weight is not None:
        normalized = normalized * weight
    if bias is not None:
        normalized = normalized + bias
    return normalized
