"""Handwritten Metal candidates integrated with MLX arrays and autodiff."""

from __future__ import annotations

from typing import Final

import mlx.core as mx

_ELEMENTWISE_THREADS: Final = 256
_TILE: Final = 16
_SUPPORTED_DTYPES: Final = (mx.float16, mx.float32)


def _require_array(value: mx.array, *, name: str) -> None:
    if value.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"{name} must use float16 or float32, got {value.dtype}")
    if value.size <= 0:
        raise ValueError(f"{name} must not be empty")


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


_SCALAR_KERNEL = mx.fast.metal_kernel(
    name="blt_phase2_scalar_multiply",
    input_names=["values", "scalar"],
    output_names=["output"],
    source="""
        uint element = thread_position_in_grid.x;
        uint location = elem_to_loc(element, values_shape, values_strides, values_ndim);
        output[element] = values[location] * scalar[0];
    """,
    ensure_row_contiguous=False,
)


@mx.custom_function
def _metal_scalar(values: mx.array, scalar: mx.array) -> mx.array:
    return _SCALAR_KERNEL(
        inputs=[values, scalar],
        grid=(values.size, 1, 1),
        threadgroup=(min(_ELEMENTWISE_THREADS, values.size), 1, 1),
        output_shapes=[values.shape],
        output_dtypes=[values.dtype],
    )[0]


@_metal_scalar.vjp
def _metal_scalar_vjp(primals, cotangent, _):
    values, scalar = primals
    return cotangent * scalar, mx.reshape(mx.sum(cotangent * values), scalar.shape)


def metal_scalar_multiply(values: mx.array, scalar: float | mx.array) -> mx.array:
    _require_array(values, name="values")
    scalar_array = (
        scalar
        if isinstance(scalar, mx.array)
        else mx.array([float(scalar)], dtype=values.dtype)
    )
    _require_array(scalar_array, name="scalar")
    if scalar_array.size != 1:
        raise ValueError("scalar must contain exactly one value")
    if scalar_array.dtype != values.dtype:
        raise TypeError("values and scalar must use the same dtype")
    return _metal_scalar(values, scalar_array)


_BIAS_GELU_KERNEL = mx.fast.metal_kernel(
    name="blt_phase2_bias_gelu",
    input_names=["values", "bias"],
    output_names=["output"],
    source="""
        uint element = thread_position_in_grid.x;
        uint value_location = elem_to_loc(
            element, values_shape, values_strides, values_ndim);
        uint column = element % values_shape[values_ndim - 1];
        uint bias_location = column * bias_strides[0];
        float value = float(values[value_location]) + float(bias[bias_location]);
        float inner = 0.7978845608028654f *
            (value + 0.044715f * value * value * value);
        output[element] = T(0.5f * value * (1.0f + metal::tanh(inner)));
    """,
    ensure_row_contiguous=False,
)


@mx.custom_function
def _metal_bias_gelu(values: mx.array, bias: mx.array) -> mx.array:
    return _BIAS_GELU_KERNEL(
        inputs=[values, bias],
        template=[("T", values.dtype)],
        grid=(values.size, 1, 1),
        threadgroup=(min(_ELEMENTWISE_THREADS, values.size), 1, 1),
        output_shapes=[values.shape],
        output_dtypes=[values.dtype],
    )[0]


@_metal_bias_gelu.vjp
def _metal_bias_gelu_vjp(primals, cotangent, _):
    values, bias = primals
    biased = values + bias
    coefficient = 0.7978845608028654
    inner = coefficient * (biased + 0.044715 * biased**3)
    tangent = mx.tanh(inner)
    derivative = 0.5 * (1.0 + tangent) + 0.5 * biased * (1.0 - tangent**2) * (
        coefficient * (1.0 + 3.0 * 0.044715 * biased**2)
    )
    values_gradient = cotangent * derivative
    axes = tuple(range(values.ndim - 1))
    bias_gradient = mx.sum(values_gradient, axis=axes) if axes else values_gradient
    return values_gradient, bias_gradient


def metal_bias_gelu(values: mx.array, bias: mx.array) -> mx.array:
    _require_array(values, name="values")
    _require_array(bias, name="bias")
    if bias.ndim != 1 or bias.shape[0] != values.shape[-1]:
        raise ValueError("bias must match the final values dimension")
    if values.dtype != bias.dtype:
        raise TypeError("values and bias must use the same dtype")
    return _metal_bias_gelu(values, bias)


_NAIVE_MATMUL_KERNEL = mx.fast.metal_kernel(
    name="blt_phase2_matmul_naive",
    input_names=["left", "right"],
    output_names=["output"],
    source="""
        uint column = thread_position_in_grid.x;
        uint row = thread_position_in_grid.y;
        uint rows = left_shape[0];
        uint inner = left_shape[1];
        uint columns = right_shape[1];
        if (row >= rows || column >= columns) return;
        float accumulator = 0.0f;
        for (uint k = 0; k < inner; ++k) {
            uint left_location = row * left_strides[0] + k * left_strides[1];
            uint right_location = k * right_strides[0] + column * right_strides[1];
            accumulator += float(left[left_location]) * float(right[right_location]);
        }
        output[row * columns + column] = T(accumulator);
    """,
    ensure_row_contiguous=False,
)


_TILED_MATMUL_KERNEL = mx.fast.metal_kernel(
    name="blt_phase2_matmul_tiled16",
    input_names=["left", "right"],
    output_names=["output"],
    source="""
        constexpr uint tile_size = 16;
        threadgroup T left_tile[tile_size][tile_size];
        threadgroup T right_tile[tile_size][tile_size];
        uint local_column = thread_position_in_threadgroup.x;
        uint local_row = thread_position_in_threadgroup.y;
        uint column = thread_position_in_grid.x;
        uint row = thread_position_in_grid.y;
        uint rows = left_shape[0];
        uint inner = left_shape[1];
        uint columns = right_shape[1];
        uint tile_count = (inner + tile_size - 1) / tile_size;
        float accumulator = 0.0f;
        for (uint tile = 0; tile < tile_count; ++tile) {
            uint left_column = tile * tile_size + local_column;
            uint right_row = tile * tile_size + local_row;
            if (row < rows && left_column < inner) {
                uint location = row * left_strides[0] + left_column * left_strides[1];
                left_tile[local_row][local_column] = left[location];
            } else {
                left_tile[local_row][local_column] = T(0);
            }
            if (right_row < inner && column < columns) {
                uint location = right_row * right_strides[0] +
                    column * right_strides[1];
                right_tile[local_row][local_column] = right[location];
            } else {
                right_tile[local_row][local_column] = T(0);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint k = 0; k < tile_size; ++k) {
                accumulator += float(left_tile[local_row][k]) *
                    float(right_tile[k][local_column]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (row < rows && column < columns) {
            output[row * columns + column] = T(accumulator);
        }
    """,
    ensure_row_contiguous=False,
)


def _validate_matmul(left: mx.array, right: mx.array) -> tuple[int, int]:
    _require_array(left, name="left")
    _require_array(right, name="right")
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("left and right must be rank two")
    if left.shape[1] != right.shape[0]:
        raise ValueError("matmul inner dimensions must match")
    if left.dtype != right.dtype:
        raise TypeError("left and right must use the same dtype")
    return left.shape[0], right.shape[1]


def _launch_matmul(kernel, left: mx.array, right: mx.array) -> mx.array:
    rows, columns = _validate_matmul(left, right)
    return kernel(
        inputs=[left, right],
        template=[("T", left.dtype)],
        grid=(_round_up(columns, _TILE), _round_up(rows, _TILE), 1),
        threadgroup=(_TILE, _TILE, 1),
        output_shapes=[(rows, columns)],
        output_dtypes=[left.dtype],
    )[0]


@mx.custom_function
def metal_matmul_naive(left: mx.array, right: mx.array) -> mx.array:
    return _launch_matmul(_NAIVE_MATMUL_KERNEL, left, right)


@metal_matmul_naive.vjp
def _metal_matmul_naive_vjp(primals, cotangent, _):
    left, right = primals
    return cotangent @ mx.transpose(right), mx.transpose(left) @ cotangent


@mx.custom_function
def metal_matmul_tiled16(left: mx.array, right: mx.array) -> mx.array:
    return _launch_matmul(_TILED_MATMUL_KERNEL, left, right)


@metal_matmul_tiled16.vjp
def _metal_matmul_tiled16_vjp(primals, cotangent, _):
    left, right = primals
    return cotangent @ mx.transpose(right), mx.transpose(left) @ cotangent
