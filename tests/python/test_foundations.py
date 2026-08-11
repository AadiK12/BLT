from __future__ import annotations

import math

import mlx.core as mx
import pytest
from blt_mlx.foundations.metal import (
    metal_bias_gelu,
    metal_matmul_naive,
    metal_matmul_tiled16,
    metal_scalar_multiply,
)
from blt_mlx.foundations.primitives import bias_gelu
from blt_mlx.foundations.reference import (
    layer_norm,
    matmul,
    scalar_multiply,
    stable_softmax,
)

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(),
    reason="custom Phase 2 kernels require Apple Metal",
)


def _close(left: mx.array, right: mx.array, *, atol: float = 1e-5) -> bool:
    mx.eval(left, right)
    return bool(mx.allclose(left, right, rtol=1e-5, atol=atol).item())


def test_reference_contracts() -> None:
    assert scalar_multiply([1.0, -2.0], 3.0) == [3.0, -6.0]
    assert matmul([[1.0, 2.0]], [[3.0], [4.0]]) == [[11.0]]
    probabilities = stable_softmax([1000.0, 1001.0, 1002.0])
    assert math.isclose(sum(probabilities), 1.0, abs_tol=1e-12)
    normalized = layer_norm([1.0, 2.0, 3.0])
    assert math.isclose(sum(normalized), 0.0, abs_tol=1e-12)


def test_metal_forward_and_backward_contracts() -> None:
    values = mx.reshape(mx.linspace(-2.0, 2.0, 24), (4, 6))
    bias = mx.linspace(-0.25, 0.25, 6)
    scalar = mx.array([1.75], dtype=mx.float32)
    assert _close(metal_scalar_multiply(values, scalar), values * scalar)
    assert _close(metal_bias_gelu(values, bias), bias_gelu(values, bias), atol=2e-5)

    left = mx.reshape(mx.linspace(-1.0, 1.0, 15), (3, 5))
    right = mx.reshape(mx.linspace(-0.5, 0.5, 20), (5, 4))
    for operation in (metal_matmul_naive, metal_matmul_tiled16):
        assert _close(operation(left, right), left @ right, atol=1e-4)
        custom = mx.grad(
            lambda current_left, current_right, current_operation=operation: mx.sum(
                current_operation(current_left, current_right)
            ),
            argnums=(0, 1),
        )(left, right)
        expected = mx.grad(
            lambda current_left, current_right: mx.sum(current_left @ current_right),
            argnums=(0, 1),
        )(left, right)
        assert _close(custom[0], expected[0])
        assert _close(custom[1], expected[1])


def test_custom_kernels_reject_bfloat16() -> None:
    with pytest.raises(TypeError, match="float16 or float32"):
        metal_scalar_multiply(mx.ones((8,), dtype=mx.bfloat16), 2.0)
