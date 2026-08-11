from __future__ import annotations

import mlx.core as mx
import pytest
from blt_mlx.config import ModelConfig
from blt_mlx.model import ByteGPT, generate


def tiny_config(**overrides) -> ModelConfig:
    values = {
        "max_sequence_length": 32,
        "d_model": 16,
        "num_layers": 1,
        "num_heads": 4,
        "mlp_hidden_size": 32,
        "seed": 123,
        "fusion_strategy": "mlx_eager",
    }
    values.update(overrides)
    return ModelConfig(**values)


def test_model_is_deterministic_and_has_expected_shape() -> None:
    first = ByteGPT(tiny_config())
    second = ByteGPT(tiny_config())
    tokens = mx.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=mx.int32)
    first_logits = first(tokens)
    second_logits = second(tokens)
    mx.eval(first_logits, second_logits)
    assert first_logits.shape == (2, 4, 256)
    assert bool(mx.allclose(first_logits, second_logits, rtol=0.0, atol=0.0).item())


def test_causal_prefix_logits_do_not_change_with_future_bytes() -> None:
    model = ByteGPT(tiny_config())
    prefix = mx.array([[10, 20, 30]], dtype=mx.int32)
    longer = mx.array([[10, 20, 30, 40, 50]], dtype=mx.int32)
    prefix_logits = model(prefix)
    longer_logits = model(longer)[:, :3]
    mx.eval(prefix_logits, longer_logits)
    assert bool(mx.allclose(prefix_logits, longer_logits, rtol=1e-5, atol=1e-5).item())


def test_shape_trace_contains_training_matmuls_and_reductions() -> None:
    model = ByteGPT(tiny_config())
    trace = model.trace(mx.zeros((2, 8), dtype=mx.int32), phase="training")
    kinds = {operation.kind for operation in trace.operations}
    assert {"matmul", "batched_matmul", "softmax", "bias_gelu"} <= kinds
    assert trace.unique_matmul_cases()


def test_generation_is_deterministic() -> None:
    model = ByteGPT(tiny_config())
    first = generate(model, b"BLT", max_new_bytes=4)
    second = generate(model, b"BLT", max_new_bytes=4)
    assert first.output == second.output
    assert len(first.generated) == 4
    assert len(first.token_latencies_ms) == 4


def test_invalid_model_configurations_fail_explicitly() -> None:
    with pytest.raises(ValueError, match="divisible"):
        tiny_config(d_model=15)
    with pytest.raises(ValueError, match="bfloat16"):
        tiny_config(dtype="bfloat16", matmul_strategy="metal_tiled16")
