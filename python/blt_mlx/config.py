"""Versioned configuration contracts for the Phase 2 MLX path."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import mlx.core as mx

MatmulStrategy = Literal["mlx", "metal_naive", "metal_tiled16"]
FusionStrategy = Literal["mlx_eager", "mlx_compiled", "metal_fused"]
DTypeName = Literal["float32", "float16", "bfloat16"]


def resolve_dtype(name: DTypeName) -> mx.Dtype:
    mapping = {
        "float32": mx.float32,
        "float16": mx.float16,
        "bfloat16": mx.bfloat16,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {name}") from exc


@dataclass(frozen=True)
class ModelConfig:
    """All values required to reconstruct a tiny byte-level transformer."""

    schema_version: int = 1
    vocab_size: int = 256
    max_sequence_length: int = 128
    d_model: int = 64
    num_layers: int = 2
    num_heads: int = 4
    mlp_hidden_size: int = 128
    seed: int = 20260811
    dtype: DTypeName = "float32"
    matmul_strategy: MatmulStrategy = "mlx"
    fusion_strategy: FusionStrategy = "mlx_compiled"

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported model configuration schema")
        for name in (
            "vocab_size",
            "max_sequence_length",
            "d_model",
            "num_heads",
            "mlp_hidden_size",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_layers < 0:
            raise ValueError("num_layers must be non-negative")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if self.vocab_size != 256:
            raise ValueError("the Phase 2 byte model requires a 256-value vocabulary")
        resolve_dtype(self.dtype)
        if self.dtype == "bfloat16" and self.matmul_strategy != "mlx":
            raise ValueError(
                "bfloat16 requires MLX matmul; custom Metal supports float16 and float32 only"
            )
        if self.dtype == "bfloat16" and self.fusion_strategy == "metal_fused":
            raise ValueError("custom fused Metal currently supports float16 and float32 only")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ModelConfig:
        return cls(**value)


@dataclass(frozen=True)
class TrainingConfig:
    """Deterministic training and evaluation contract."""

    schema_version: int = 1
    steps: int = 100
    batch_size: int = 8
    sequence_length: int = 32
    learning_rate: float = 3e-3
    weight_decay: float = 0.0
    seed: int = 20260812
    compile_step: bool = True
    log_every: int = 10

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported training configuration schema")
        if self.steps <= 0 or self.batch_size <= 0 or self.sequence_length <= 0:
            raise ValueError("steps, batch_size, and sequence_length must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if self.log_every <= 0:
            raise ValueError("log_every must be positive")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TrainingConfig:
        return cls(**value)
