"""Explicit MLX transformer modules with hardware-selection boundaries."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from blt_mlx.config import FusionStrategy, MatmulStrategy
from blt_mlx.foundations.metal import (
    metal_bias_gelu,
    metal_matmul_naive,
    metal_matmul_tiled16,
)
from blt_mlx.foundations.primitives import (
    bias_gelu,
    compiled_bias_gelu,
    deterministic_parameter,
    layer_norm,
    stable_softmax,
)
from blt_mlx.shapes import ShapeRecorder


class SeedStream:
    """Deterministically allocate distinct parameter seeds."""

    def __init__(self, seed: int) -> None:
        self._next = int(seed)

    def take(self) -> int:
        value = self._next
        self._next += 1
        return value


class ByteEmbedding(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        dimensions: int,
        *,
        seed: int,
        dtype: mx.Dtype,
    ) -> None:
        super().__init__()
        self.weight = deterministic_parameter(
            (vocabulary_size, dimensions),
            seed=seed,
            scale=dimensions**-0.5,
            dtype=dtype,
        )

    def __call__(self, token_ids: mx.array) -> mx.array:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        return self.weight[token_ids]


class ExplicitLayerNorm(nn.Module):
    def __init__(self, dimensions: int, *, dtype: mx.Dtype) -> None:
        super().__init__()
        self.weight = mx.ones((dimensions,), dtype=dtype)
        self.bias = mx.zeros((dimensions,), dtype=dtype)

    def __call__(self, values: mx.array) -> mx.array:
        return layer_norm(values, self.weight, self.bias)


class HardwareAwareLinear(nn.Module):
    """Linear projection whose matrix implementation is an explicit policy."""

    def __init__(
        self,
        input_features: int,
        output_features: int,
        *,
        seed_stream: SeedStream,
        dtype: mx.Dtype,
        strategy: MatmulStrategy = "mlx",
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        if input_features <= 0 or output_features <= 0:
            raise ValueError("linear dimensions must be positive")
        self.weight = deterministic_parameter(
            (output_features, input_features),
            seed=seed_stream.take(),
            scale=input_features**-0.5,
            dtype=dtype,
        )
        self.bias = (
            mx.zeros((output_features,), dtype=dtype) if use_bias else None
        )
        self.strategy = strategy

    def __call__(
        self,
        inputs: mx.array,
        *,
        recorder: ShapeRecorder | None = None,
        name: str = "linear",
    ) -> mx.array:
        if inputs.ndim < 2:
            raise ValueError("linear inputs must have at least two dimensions")
        if inputs.shape[-1] != self.weight.shape[1]:
            raise ValueError("linear input feature count does not match its weight")
        original_prefix = tuple(inputs.shape[:-1])
        flattened = mx.reshape(inputs, (-1, inputs.shape[-1]))
        right = mx.transpose(self.weight)
        if self.strategy == "mlx":
            flattened_output = flattened @ right
        elif self.strategy == "metal_naive":
            flattened_output = metal_matmul_naive(flattened, right)
        elif self.strategy == "metal_tiled16":
            flattened_output = metal_matmul_tiled16(flattened, right)
        else:
            raise ValueError(f"unknown matmul strategy: {self.strategy}")
        if self.bias is not None:
            flattened_output = flattened_output + self.bias
        output = mx.reshape(flattened_output, (*original_prefix, self.weight.shape[0]))
        if recorder is not None:
            recorder.record(
                name=name,
                kind="matmul",
                left=flattened,
                right=right,
                output=flattened_output,
            )
        return output


class HardwareAwareMLP(nn.Module):
    def __init__(
        self,
        dimensions: int,
        hidden_size: int,
        *,
        seed_stream: SeedStream,
        dtype: mx.Dtype,
        matmul_strategy: MatmulStrategy,
        fusion_strategy: FusionStrategy,
    ) -> None:
        super().__init__()
        self.input_projection = HardwareAwareLinear(
            dimensions,
            hidden_size,
            seed_stream=seed_stream,
            dtype=dtype,
            strategy=matmul_strategy,
            use_bias=False,
        )
        self.input_bias = mx.zeros((hidden_size,), dtype=dtype)
        self.output_projection = HardwareAwareLinear(
            hidden_size,
            dimensions,
            seed_stream=seed_stream,
            dtype=dtype,
            strategy=matmul_strategy,
        )
        self.fusion_strategy = fusion_strategy

    def __call__(
        self,
        inputs: mx.array,
        *,
        recorder: ShapeRecorder | None = None,
        name: str,
    ) -> mx.array:
        hidden = self.input_projection(
            inputs,
            recorder=recorder,
            name=f"{name}.expand",
        )
        flattened = mx.reshape(hidden, (-1, hidden.shape[-1]))
        if self.fusion_strategy == "mlx_eager":
            activated = bias_gelu(flattened, self.input_bias)
        elif self.fusion_strategy == "mlx_compiled":
            activated = compiled_bias_gelu(flattened, self.input_bias)
        elif self.fusion_strategy == "metal_fused":
            activated = metal_bias_gelu(flattened, self.input_bias)
        else:
            raise ValueError(f"unknown fusion strategy: {self.fusion_strategy}")
        if recorder is not None:
            recorder.record(
                name=f"{name}.bias_gelu",
                kind="bias_gelu",
                left=flattened,
                right=self.input_bias,
                output=activated,
            )
        activated = mx.reshape(activated, hidden.shape)
        return self.output_projection(
            activated,
            recorder=recorder,
            name=f"{name}.contract",
        )


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dimensions: int,
        num_heads: int,
        *,
        seed_stream: SeedStream,
        dtype: mx.Dtype,
        matmul_strategy: MatmulStrategy,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dimensions = dimensions // num_heads
        self.qkv = HardwareAwareLinear(
            dimensions,
            3 * dimensions,
            seed_stream=seed_stream,
            dtype=dtype,
            strategy=matmul_strategy,
        )
        self.output = HardwareAwareLinear(
            dimensions,
            dimensions,
            seed_stream=seed_stream,
            dtype=dtype,
            strategy=matmul_strategy,
        )

    def __call__(
        self,
        inputs: mx.array,
        *,
        recorder: ShapeRecorder | None = None,
        name: str,
    ) -> mx.array:
        batch, sequence, dimensions = inputs.shape
        projected = self.qkv(inputs, recorder=recorder, name=f"{name}.qkv")
        query = projected[..., :dimensions]
        key = projected[..., dimensions : 2 * dimensions]
        value = projected[..., 2 * dimensions :]

        def split_heads(values: mx.array) -> mx.array:
            shaped = mx.reshape(
                values,
                (batch, sequence, self.num_heads, self.head_dimensions),
            )
            return mx.transpose(shaped, (0, 2, 1, 3))

        query = split_heads(query)
        key = split_heads(key)
        value = split_heads(value)
        transposed_key = mx.transpose(key, (0, 1, 3, 2))
        scores = (query @ transposed_key) * (self.head_dimensions**-0.5)
        mask = mx.triu(mx.full((sequence, sequence), -1.0e4, dtype=scores.dtype), k=1)
        probabilities = stable_softmax(scores + mask, axis=-1)
        attended = probabilities @ value
        if recorder is not None:
            recorder.record(
                name=f"{name}.scores",
                kind="batched_matmul",
                left=query,
                right=transposed_key,
                output=scores,
            )
            recorder.record(
                name=f"{name}.values",
                kind="batched_matmul",
                left=probabilities,
                right=value,
                output=attended,
            )
            recorder.record(
                name=f"{name}.softmax",
                kind="softmax",
                left=scores,
                output=probabilities,
            )
        attended = mx.transpose(attended, (0, 2, 1, 3))
        attended = mx.reshape(attended, (batch, sequence, dimensions))
        return self.output(attended, recorder=recorder, name=f"{name}.output")


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dimensions: int,
        num_heads: int,
        hidden_size: int,
        *,
        seed_stream: SeedStream,
        dtype: mx.Dtype,
        matmul_strategy: MatmulStrategy,
        fusion_strategy: FusionStrategy,
    ) -> None:
        super().__init__()
        self.attention_norm = ExplicitLayerNorm(dimensions, dtype=dtype)
        self.attention = CausalSelfAttention(
            dimensions,
            num_heads,
            seed_stream=seed_stream,
            dtype=dtype,
            matmul_strategy=matmul_strategy,
        )
        self.mlp_norm = ExplicitLayerNorm(dimensions, dtype=dtype)
        self.mlp = HardwareAwareMLP(
            dimensions,
            hidden_size,
            seed_stream=seed_stream,
            dtype=dtype,
            matmul_strategy=matmul_strategy,
            fusion_strategy=fusion_strategy,
        )

    def __call__(
        self,
        inputs: mx.array,
        *,
        recorder: ShapeRecorder | None = None,
        name: str,
    ) -> mx.array:
        attended = self.attention(
            self.attention_norm(inputs),
            recorder=recorder,
            name=f"{name}.attention",
        )
        residual = inputs + attended
        transformed = self.mlp(
            self.mlp_norm(residual),
            recorder=recorder,
            name=f"{name}.mlp",
        )
        return residual + transformed
