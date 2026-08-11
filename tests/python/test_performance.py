from __future__ import annotations

import mlx.core as mx
from blt_mlx.config import ModelConfig
from blt_mlx.model import ByteGPT, generate
from blt_mlx.performance import (
    balanced_interleaved_order,
    compare_matmul_shape,
    generation_metrics,
)


def model() -> ByteGPT:
    return ByteGPT(
        ModelConfig(
            max_sequence_length=16,
            d_model=16,
            num_layers=1,
            num_heads=4,
            mlp_hidden_size=32,
            fusion_strategy="mlx_eager",
        )
    )


def test_balanced_order_is_deterministic_and_equal() -> None:
    first = balanced_interleaved_order(10, seed=42)
    second = balanced_interleaved_order(10, seed=42)
    assert first == second
    assert first.count("A") == first.count("B") == 10


def test_model_shape_drives_full_output_kernel_comparison() -> None:
    current = model()
    trace = current.trace(mx.zeros((1, 8), dtype=mx.int32), phase="prefill")
    shape = trace.unique_matmul_cases()[0]
    result = compare_matmul_shape(shape, samples=4, seed=123)
    assert result.output_valid
    assert result.promotion_gate in {"pass", "do_not_promote"}
    assert len(result.order) == 8


def test_generation_metrics_keep_kernel_and_request_units_separate() -> None:
    result = generate(model(), b"BLT", max_new_bytes=3)
    metrics = generation_metrics(result)
    assert metrics["output_bytes"] == 3
    assert metrics["ttft_ms"] > 0.0
    assert metrics["generation_bytes_per_second"] > 0.0
