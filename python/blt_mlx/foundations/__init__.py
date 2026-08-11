"""Mathematical, MLX, and custom-Metal operation ladder."""

from blt_mlx.foundations.metal import (
    metal_bias_gelu,
    metal_matmul_naive,
    metal_matmul_tiled16,
    metal_scalar_multiply,
)
from blt_mlx.foundations.primitives import (
    bias_gelu,
    compiled_bias_gelu,
    deterministic_parameter,
    gelu_tanh,
    layer_norm,
    stable_softmax,
)

__all__ = [
    "bias_gelu",
    "compiled_bias_gelu",
    "deterministic_parameter",
    "gelu_tanh",
    "layer_norm",
    "metal_bias_gelu",
    "metal_matmul_naive",
    "metal_matmul_tiled16",
    "metal_scalar_multiply",
    "stable_softmax",
]
