"""Phase 2 command-line entrypoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx

from blt_mlx.checkpoint import load_checkpoint, save_checkpoint
from blt_mlx.config import ModelConfig, TrainingConfig
from blt_mlx.data import DEFAULT_TINY_CORPUS, ByteDataset
from blt_mlx.model import ByteGPT, generate
from blt_mlx.performance import (
    compare_matmul_shape,
    comparison_as_dict,
    environment_metadata,
    generation_metrics,
    summary,
    thermal_soak,
    trace_model_shape_suite,
    write_json,
)
from blt_mlx.training import Trainer, evaluate_bits_per_byte


def _tiny_model_config(*, dtype: str = "float32") -> ModelConfig:
    return ModelConfig(
        max_sequence_length=64,
        d_model=32,
        num_layers=1,
        num_heads=4,
        mlp_hidden_size=64,
        seed=20260811,
        dtype=dtype,
        fusion_strategy="mlx_compiled",
    )


def doctor() -> int:
    payload = {
        **environment_metadata(),
        "dtypes": {},
    }
    for name, dtype in (
        ("float32", mx.float32),
        ("float16", mx.float16),
        ("bfloat16", mx.bfloat16),
    ):
        try:
            value = mx.ones((16,), dtype=dtype) * 1.5
            mx.eval(value)
            payload["dtypes"][name] = {"available": True}
        except Exception as exc:  # pragma: no cover - capability dependent
            payload["dtypes"][name] = {
                "available": False,
                "reason": f"{type(exc).__name__}: {exc}",
            }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["metal_available"] else 2


def train_smoke(*, steps: int, checkpoint: Path) -> int:
    model_config = _tiny_model_config()
    training_config = TrainingConfig(
        steps=steps,
        batch_size=8,
        sequence_length=24,
        learning_rate=3e-3,
        seed=20260812,
        log_every=10,
    )
    model = ByteGPT(model_config)
    dataset = ByteDataset.from_text(
        DEFAULT_TINY_CORPUS,
        sequence_length=training_config.sequence_length,
        batch_size=training_config.batch_size,
        seed=training_config.seed,
    )
    initial_bpb = evaluate_bits_per_byte(model, dataset, batches=4)
    trainer = Trainer(model, training_config)
    report = trainer.train(dataset)
    final_bpb = evaluate_bits_per_byte(model, dataset, batches=4)
    if not report.loss_improved or final_bpb >= initial_bpb:
        raise RuntimeError("tiny training fixture did not improve")
    checkpoint_path = save_checkpoint(checkpoint, trainer)
    generated = generate(model, b"Byte ", max_new_bytes=16)
    payload = {
        "schema_version": 1,
        "environment": environment_metadata(),
        "model": model_config.as_dict(),
        "training": training_config.as_dict(),
        "initial_loss": report.initial_loss,
        "final_loss": report.final_loss,
        "minimum_loss": report.minimum_loss,
        "initial_bits_per_byte": initial_bpb,
        "final_bits_per_byte": final_bpb,
        "loss_improved": report.loss_improved,
        "checkpoint": str(checkpoint_path),
        "generation": {
            **generation_metrics(generated),
            "prompt_hex": generated.prompt.hex(),
            "generated_hex": generated.generated.hex(),
        },
        "training_steps": [
            {
                "step": observation.step,
                "loss": observation.loss,
                "duration_ms": observation.duration_ms,
                "active_memory_bytes": observation.active_memory_bytes,
                "peak_memory_bytes": observation.peak_memory_bytes,
            }
            for observation in report.steps
        ],
    }
    report_path = write_json(checkpoint_path.parent / "training_report.json", payload)
    print(json.dumps({"status": "complete", "report": str(report_path), **payload}, indent=2))
    return 0


def benchmark_shapes(
    *,
    samples: int,
    output: Path,
    dtype: str,
    candidate: str,
) -> int:
    model = ByteGPT(_tiny_model_config(dtype=dtype))
    traces = trace_model_shape_suite(model)
    comparisons = []
    seen = set()
    for recorder in traces.values():
        for shape in recorder.unique_matmul_cases():
            key = (shape.phase, shape.left_shape, shape.right_shape, shape.dtype)
            if key in seen:
                continue
            seen.add(key)
            comparisons.append(
                comparison_as_dict(
                    compare_matmul_shape(
                        shape,
                        samples=samples,
                        seed=20260814 + len(comparisons),
                        candidate_name=candidate,
                    )
                )
            )
    payload = {
        "schema_version": 1,
        "environment": environment_metadata(),
        "model": model.config.as_dict(),
        "sample_count_per_candidate": samples,
        "interleaved": True,
        "candidate": candidate,
        "promotion_gate": {
            "full_output": True,
            "median_speedup_minimum": 1.05,
            "p95_speedup_minimum": 1.0,
        },
        "traces": {name: trace.as_dict() for name, trace in traces.items()},
        "comparisons": comparisons,
    }
    path = write_json(output, payload)
    print(json.dumps({"status": "complete", "output": str(path)}, indent=2))
    return 0


def benchmark_generation(
    *,
    checkpoint: Path,
    samples: int,
    prompt: str,
    max_new_bytes: int,
    output: Path,
) -> int:
    if samples < 2:
        raise ValueError("samples must be at least two")
    if max_new_bytes < 2:
        raise ValueError("max_new_bytes must be at least two to measure TPOT")
    loaded = load_checkpoint(checkpoint)
    prompt_bytes = prompt.encode("utf-8")
    generate(loaded.model, prompt_bytes, max_new_bytes=max_new_bytes)
    observations = [
        generate(loaded.model, prompt_bytes, max_new_bytes=max_new_bytes)
        for _ in range(samples)
    ]
    metrics = [generation_metrics(observation) for observation in observations]
    payload = {
        "schema_version": 1,
        "environment": environment_metadata(),
        "model": loaded.model.config.as_dict(),
        "checkpoint": str(checkpoint.expanduser().resolve()),
        "workload": {
            "prompt_utf8": prompt,
            "prompt_hex": prompt_bytes.hex(),
            "max_new_bytes": max_new_bytes,
            "samples": samples,
            "warmup_samples": 1,
        },
        "summary": {
            "ttft_ms": summary([float(value["ttft_ms"]) for value in metrics]),
            "tpot_ms": summary(
                [float(value["tpot_ms"]) for value in metrics if value["tpot_ms"] is not None]
            ),
            "end_to_end_ms": summary(
                [float(value["end_to_end_ms"]) for value in metrics]
            ),
            "generation_bytes_per_second": summary(
                [float(value["generation_bytes_per_second"]) for value in metrics]
            ),
            "peak_memory_bytes": max(
                int(value["peak_memory_bytes"])
                for value in metrics
                if value["peak_memory_bytes"] is not None
            ),
        },
        "samples": metrics,
        "generated_hex": observations[0].generated.hex(),
    }
    path = write_json(output, payload)
    print(json.dumps({"status": "complete", "output": str(path)}, indent=2))
    return 0


def run_thermal_soak(
    *,
    checkpoint: Path,
    prompt: str,
    duration_seconds: float,
    window_seconds: float,
    output: Path,
) -> int:
    if not prompt:
        raise ValueError("prompt must not be empty")
    loaded = load_checkpoint(checkpoint)
    payload = {
        "schema_version": 1,
        "environment": environment_metadata(),
        "model": loaded.model.config.as_dict(),
        "checkpoint": str(checkpoint.expanduser().resolve()),
        "prompt_utf8": prompt,
        "soak": thermal_soak(
            loaded.model,
            prompt=prompt.encode("utf-8"),
            duration_seconds=duration_seconds,
            window_seconds=window_seconds,
        ),
    }
    path = write_json(output, payload)
    print(json.dumps({"status": "complete", "output": str(path)}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BLT Phase 2 MLX infrastructure")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Check Apple/MLX capabilities")
    train = commands.add_parser("train-smoke", help="Overfit the deterministic tiny corpus")
    train.add_argument("--steps", type=int, default=80)
    train.add_argument("--checkpoint", type=Path, required=True)
    benchmark = commands.add_parser(
        "benchmark-shapes",
        help="Trace model shapes and run interleaved MLX/Metal comparisons",
    )
    benchmark.add_argument("--samples", type=int, default=20)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    benchmark.add_argument(
        "--candidate",
        choices=("metal_naive", "metal_tiled16"),
        default="metal_tiled16",
    )
    generation = commands.add_parser(
        "benchmark-generation",
        help="Measure checkpoint TTFT, TPOT, throughput, and peak memory",
    )
    generation.add_argument("--checkpoint", type=Path, required=True)
    generation.add_argument("--samples", type=int, default=20)
    generation.add_argument("--prompt", default="Byte ")
    generation.add_argument("--max-new-bytes", type=int, default=32)
    generation.add_argument("--output", type=Path, required=True)
    soak = commands.add_parser(
        "thermal-soak",
        help="Run repeated checkpoint generation in time windows",
    )
    soak.add_argument("--checkpoint", type=Path, required=True)
    soak.add_argument("--prompt", default="Byte ")
    soak.add_argument("--seconds", type=float, required=True)
    soak.add_argument("--window-seconds", type=float, default=60.0)
    soak.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    mx.set_default_device(mx.gpu)
    if args.command == "doctor":
        return doctor()
    if args.command == "train-smoke":
        return train_smoke(steps=args.steps, checkpoint=args.checkpoint)
    if args.command == "benchmark-shapes":
        return benchmark_shapes(
            samples=args.samples,
            output=args.output,
            dtype=args.dtype,
            candidate=args.candidate,
        )
    if args.command == "benchmark-generation":
        return benchmark_generation(
            checkpoint=args.checkpoint,
            samples=args.samples,
            prompt=args.prompt,
            max_new_bytes=args.max_new_bytes,
            output=args.output,
        )
    if args.command == "thermal-soak":
        return run_thermal_soak(
            checkpoint=args.checkpoint,
            prompt=args.prompt,
            duration_seconds=args.seconds,
            window_seconds=args.window_seconds,
            output=args.output,
        )
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
