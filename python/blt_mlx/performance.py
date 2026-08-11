"""Shape-derived kernel and end-to-end evidence infrastructure."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx

from blt_mlx.foundations.metal import metal_matmul_naive, metal_matmul_tiled16
from blt_mlx.model import ByteGPT, GenerationResult, generate
from blt_mlx.shapes import OperationShape, ShapeRecorder


def environment_metadata() -> dict[str, Any]:
    """Capture the runtime facts needed to interpret local measurements."""

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "mlx": importlib.metadata.version("mlx"),
        "device": str(mx.default_device()),
        "metal_available": bool(mx.metal.is_available()),
    }


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires values")
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + weight * (ordered[upper] - ordered[lower])


def summary(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "minimum": min(values),
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "maximum": max(values),
        "mean": statistics.fmean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def balanced_interleaved_order(samples_per_candidate: int, *, seed: int) -> list[str]:
    if samples_per_candidate < 2:
        raise ValueError("samples_per_candidate must be at least two")
    order: list[str] = []
    while len(order) < samples_per_candidate * 2:
        order.extend(["A", "B", "B", "A"])
    order = order[: samples_per_candidate * 2]
    random.Random(seed).shuffle(order)
    counts = {name: order.count(name) for name in ("A", "B")}
    if counts != {"A": samples_per_candidate, "B": samples_per_candidate}:
        order = ["A", "B"] * samples_per_candidate
        random.Random(seed).shuffle(order)
    return order


@dataclass(frozen=True)
class KernelComparison:
    case: dict[str, Any]
    baseline: dict[str, Any]
    candidate: dict[str, Any]
    median_speedup: float
    p95_speedup: float
    output_valid: bool
    promotion_gate: str
    order: tuple[str, ...]


def _complete(operation) -> tuple[mx.array, float]:
    started = time.perf_counter_ns()
    output = operation()
    mx.eval(output)
    mx.synchronize()
    return output, (time.perf_counter_ns() - started) / 1_000_000.0


def compare_matmul_shape(
    shape: OperationShape,
    *,
    samples: int,
    seed: int,
    candidate_name: str = "metal_tiled16",
) -> KernelComparison:
    if shape.right_shape is None or len(shape.left_shape) != 2 or len(shape.right_shape) != 2:
        raise ValueError("custom Metal comparison requires a rank-two matmul shape")
    if shape.dtype == str(mx.float16):
        dtype = mx.float16
    elif shape.dtype == str(mx.float32):
        dtype = mx.float32
    else:
        raise TypeError(
            f"custom Metal comparison supports float16 or float32, got {shape.dtype}"
        )
    left = mx.random.normal(shape=shape.left_shape, key=mx.random.key(seed)).astype(dtype)
    right = mx.random.normal(shape=shape.right_shape, key=mx.random.key(seed + 1)).astype(dtype)
    mx.eval(left, right)
    reference = left @ right
    mx.eval(reference)
    if candidate_name == "metal_tiled16":
        def candidate_operation() -> mx.array:
            return metal_matmul_tiled16(left, right)
    elif candidate_name == "metal_naive":
        def candidate_operation() -> mx.array:
            return metal_matmul_naive(left, right)
    else:
        raise ValueError("candidate_name must be metal_tiled16 or metal_naive")

    def baseline_operation() -> mx.array:
        return left @ right
    _complete(baseline_operation)
    _complete(candidate_operation)
    order = balanced_interleaved_order(samples, seed=seed)
    durations = {"A": [], "B": []}
    valid = True
    if mx.metal.is_available():
        mx.reset_peak_memory()
    for name in order:
        operation = baseline_operation if name == "A" else candidate_operation
        output, duration = _complete(operation)
        durations[name].append(duration)
        valid = valid and bool(mx.allclose(output, reference, rtol=1e-4, atol=1e-4).item())
    baseline = summary(durations["A"])
    candidate = summary(durations["B"])
    median_speedup = float(baseline["median"]) / float(candidate["median"])
    p95_speedup = float(baseline["p95"]) / float(candidate["p95"])
    gate = (
        "pass"
        if valid and median_speedup >= 1.05 and p95_speedup >= 1.0
        else "do_not_promote"
    )
    return KernelComparison(
        case=shape.as_dict(),
        baseline={**baseline, "name": "mlx"},
        candidate={
            **candidate,
            "name": candidate_name,
            "peak_memory_bytes": (
                int(mx.get_peak_memory()) if mx.metal.is_available() else None
            ),
        },
        median_speedup=median_speedup,
        p95_speedup=p95_speedup,
        output_valid=valid,
        promotion_gate=gate,
        order=tuple(order),
    )


def trace_model_shape_suite(model: ByteGPT) -> dict[str, ShapeRecorder]:
    maximum = model.config.max_sequence_length
    training_length = min(32, maximum)
    prefill_length = min(64, maximum)
    return {
        "training": model.trace(
            mx.zeros((4, training_length), dtype=mx.int32),
            phase="training",
        ),
        "prefill": model.trace(
            mx.zeros((1, prefill_length), dtype=mx.int32),
            phase="prefill",
        ),
        "decode": model.trace(
            mx.zeros((1, min(prefill_length + 1, maximum)), dtype=mx.int32),
            phase="decode",
        ),
    }


def generation_metrics(result: GenerationResult) -> dict[str, Any]:
    return {
        "ttft_ms": result.first_token_ms,
        "tpot_ms": result.tpot_ms,
        "end_to_end_ms": result.end_to_end_ms,
        "generation_bytes_per_second": result.generation_units_per_second,
        "output_bytes": len(result.generated),
        "peak_memory_bytes": result.peak_memory_bytes,
        "per_byte_latencies_ms": list(result.token_latencies_ms),
    }


def thermal_soak(
    model: ByteGPT,
    *,
    prompt: bytes,
    duration_seconds: float,
    window_seconds: float = 60.0,
) -> dict[str, Any]:
    if duration_seconds <= 0.0 or window_seconds <= 0.0:
        raise ValueError("duration and window must be positive")
    started = time.monotonic()
    windows: list[dict[str, Any]] = []
    current: list[float] = []
    window_started = started
    requests = 0
    while time.monotonic() - started < duration_seconds:
        result = generate(model, prompt, max_new_bytes=8)
        current.append(result.end_to_end_ms)
        requests += 1
        now = time.monotonic()
        if now - window_started >= window_seconds:
            windows.append(
                {
                    "elapsed_seconds": now - started,
                    "request_latency_ms": summary(current),
                    "active_memory_bytes": int(mx.get_active_memory()),
                    "cache_memory_bytes": int(mx.get_cache_memory()),
                }
            )
            current = []
            window_started = now
    if current:
        windows.append(
            {
                "elapsed_seconds": time.monotonic() - started,
                "request_latency_ms": summary(current),
                "active_memory_bytes": int(mx.get_active_memory()),
                "cache_memory_bytes": int(mx.get_cache_memory()),
            }
        )
    return {
        "duration_seconds": time.monotonic() - started,
        "request_count": requests,
        "windows": windows,
        "interpretation": "sustained evidence only; record power and ambient conditions separately",
    }


def write_json(path: Path, value: Any) -> Path:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return resolved


def comparison_as_dict(value: KernelComparison) -> dict[str, Any]:
    result = asdict(value)
    result["order"] = list(value.order)
    return result
