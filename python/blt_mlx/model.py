"""Deterministic all-MLX byte-level GPT used to close Phase 2."""

from __future__ import annotations

import time
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from blt_mlx.config import ModelConfig, resolve_dtype
from blt_mlx.foundations.primitives import deterministic_parameter
from blt_mlx.modules import (
    ByteEmbedding,
    ExplicitLayerNorm,
    HardwareAwareLinear,
    SeedStream,
    TransformerBlock,
)
from blt_mlx.shapes import ShapeRecorder


class ByteGPT(nn.Module):
    """A conventional byte-GPT baseline; dynamic BLT patching comes later."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        dtype = resolve_dtype(config.dtype)
        seeds = SeedStream(config.seed)
        self.byte_embedding = ByteEmbedding(
            config.vocab_size,
            config.d_model,
            seed=seeds.take(),
            dtype=dtype,
        )
        self.position_embedding = deterministic_parameter(
            (config.max_sequence_length, config.d_model),
            seed=seeds.take(),
            scale=config.d_model**-0.5,
            dtype=dtype,
        )
        self.blocks = [
            TransformerBlock(
                config.d_model,
                config.num_heads,
                config.mlp_hidden_size,
                seed_stream=seeds,
                dtype=dtype,
                matmul_strategy=config.matmul_strategy,
                fusion_strategy=config.fusion_strategy,
            )
            for _ in range(config.num_layers)
        ]
        self.final_norm = ExplicitLayerNorm(config.d_model, dtype=dtype)
        self.lm_head = HardwareAwareLinear(
            config.d_model,
            config.vocab_size,
            seed_stream=seeds,
            dtype=dtype,
            strategy=config.matmul_strategy,
            use_bias=False,
        )

    def __call__(
        self,
        token_ids: mx.array,
        *,
        recorder: ShapeRecorder | None = None,
    ) -> mx.array:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        if token_ids.dtype not in (mx.int32, mx.uint32, mx.int64, mx.uint64):
            raise TypeError("token_ids must use an integer dtype")
        sequence = token_ids.shape[1]
        if sequence <= 0 or sequence > self.config.max_sequence_length:
            raise ValueError("sequence length is outside the model configuration")
        positions = mx.arange(sequence, dtype=mx.int32)
        hidden = self.byte_embedding(token_ids) + self.position_embedding[positions]
        for index, block in enumerate(self.blocks):
            hidden = block(hidden, recorder=recorder, name=f"block_{index}")
        hidden = self.final_norm(hidden)
        return self.lm_head(hidden, recorder=recorder, name="lm_head")

    def trace(self, token_ids: mx.array, *, phase: str) -> ShapeRecorder:
        recorder = ShapeRecorder(phase=phase)
        output = self(token_ids, recorder=recorder)
        mx.eval(output)
        return recorder


@dataclass(frozen=True)
class GenerationResult:
    prompt: bytes
    output: bytes
    generated: bytes
    first_token_ms: float
    token_latencies_ms: tuple[float, ...]
    end_to_end_ms: float
    peak_memory_bytes: int | None

    @property
    def tpot_ms(self) -> float | None:
        later = self.token_latencies_ms[1:]
        return sum(later) / len(later) if later else None

    @property
    def generation_units_per_second(self) -> float:
        if not self.token_latencies_ms:
            return 0.0
        return len(self.token_latencies_ms) / (
            sum(self.token_latencies_ms) / 1000.0
        )


def _sample_token(
    logits: mx.array,
    *,
    temperature: float,
    top_k: int | None,
    key: mx.array,
) -> int:
    if temperature == 0.0:
        return int(mx.argmax(logits).item())
    if temperature < 0.0:
        raise ValueError("temperature must be non-negative")
    scaled = logits / temperature
    if top_k is not None:
        if top_k <= 0 or top_k > logits.shape[-1]:
            raise ValueError("top_k must be between one and the vocabulary size")
        threshold = mx.sort(scaled)[-top_k]
        scaled = mx.where(scaled < threshold, -1.0e9, scaled)
    return int(mx.random.categorical(scaled, key=key).item())


def generate(
    model: ByteGPT,
    prompt: bytes,
    *,
    max_new_bytes: int,
    temperature: float = 0.0,
    top_k: int | None = None,
    seed: int = 20260813,
) -> GenerationResult:
    if not prompt:
        raise ValueError("prompt must not be empty")
    if max_new_bytes < 0:
        raise ValueError("max_new_bytes must be non-negative")
    tokens = [int(value) for value in prompt]
    token_latencies: list[float] = []
    started = time.perf_counter_ns()
    if mx.metal.is_available():
        mx.reset_peak_memory()
    for step in range(max_new_bytes):
        context = tokens[-model.config.max_sequence_length :]
        token_ids = mx.array([context], dtype=mx.int32)
        token_started = time.perf_counter_ns()
        logits = model(token_ids)[0, -1]
        mx.eval(logits)
        mx.synchronize()
        next_token = _sample_token(
            logits,
            temperature=temperature,
            top_k=top_k,
            key=mx.random.key(seed + step),
        )
        tokens.append(next_token)
        token_latencies.append((time.perf_counter_ns() - token_started) / 1_000_000.0)
    end_to_end_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    peak_memory = int(mx.get_peak_memory()) if mx.metal.is_available() else None
    output = bytes(tokens)
    return GenerationResult(
        prompt=prompt,
        output=output,
        generated=output[len(prompt) :],
        first_token_ms=token_latencies[0] if token_latencies else 0.0,
        token_latencies_ms=tuple(token_latencies),
        end_to_end_ms=end_to_end_ms,
        peak_memory_bytes=peak_memory,
    )
