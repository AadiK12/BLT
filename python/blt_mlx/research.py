"""Phase 4 scheduled training, validation, checkpoint selection, and final test."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

from blt_mlx.baseline import byte_display, sha256_file
from blt_mlx.checkpoint import LoadedCheckpoint, load_checkpoint, save_checkpoint
from blt_mlx.model import ByteGPT, generate
from blt_mlx.performance import (
    environment_metadata,
    generation_metrics,
    summary,
    write_json,
)
from blt_mlx.phase4 import (
    Phase4ExperimentConfig,
    load_phase4_dataset,
    load_phase4_manifest,
)
from blt_mlx.training import Trainer, TrainingStep, evaluate_loss

FINAL_TEST_ACKNOWLEDGEMENT = "I_UNDERSTAND_THIS_CONSUMES_THE_FINAL_TEST_SET"


def _experiment_hash(config: Phase4ExperimentConfig) -> str:
    return sha256_file(config.source_path)


def _manifest_hash(config: Phase4ExperimentConfig) -> str:
    return sha256_file(config.manifest_path)


def _validate_checkpoint_contract(
    config: Phase4ExperimentConfig,
    loaded: LoadedCheckpoint,
) -> dict[str, Any]:
    if loaded.model.config.as_dict() != config.model.as_dict():
        raise ValueError("checkpoint model does not match the Phase 4 experiment")
    metadata = loaded.metadata.get("run_metadata", {})
    if metadata.get("phase") != 4:
        raise ValueError("checkpoint is not a Phase 4 research checkpoint")
    if metadata.get("experiment_config_sha256") != _experiment_hash(config):
        raise ValueError("checkpoint was not trained from this exact Phase 4 config")
    if metadata.get("corpus_manifest_sha256") != _manifest_hash(config):
        raise ValueError("checkpoint corpus manifest does not match the prepared corpus")
    return metadata


def _validation_metrics(config: Phase4ExperimentConfig, model: ByteGPT) -> dict[str, Any]:
    dataset = load_phase4_dataset(config, "validation")
    loss = evaluate_loss(
        model,
        dataset,
        batches=config.evaluation.validation_batches,
    )
    return {
        "cross_entropy_nats": loss,
        "bits_per_byte": loss / math.log(2.0),
        "batches": config.evaluation.validation_batches,
    }


def _generation_evidence(
    config: Phase4ExperimentConfig,
    model: ByteGPT,
) -> list[dict[str, Any]]:
    evidence = []
    for index, prompt in enumerate(config.evaluation.prompts):
        result = generate(
            model,
            prompt.encode("utf-8"),
            max_new_bytes=config.evaluation.max_new_bytes,
            seed=config.model.seed + index,
        )
        evidence.append(
            {
                "prompt": byte_display(result.prompt),
                "generated": byte_display(result.generated),
                "output": byte_display(result.output),
                "metrics": generation_metrics(result),
            }
        )
    return evidence


def _step_as_dict(step: TrainingStep) -> dict[str, Any]:
    return asdict(step)


def _training_summary(steps: list[dict[str, Any]]) -> dict[str, Any]:
    if not steps:
        return {
            "steps_executed": 0,
            "gradient_clipped_steps": 0,
        }
    peak_memory = [
        int(step["peak_memory_bytes"])
        for step in steps
        if step["peak_memory_bytes"] is not None
    ]
    return {
        "steps_executed": len(steps),
        "loss": summary([float(step["loss"]) for step in steps]),
        "step_latency_ms": summary([float(step["duration_ms"]) for step in steps]),
        "gradient_norm": summary([float(step["gradient_norm"]) for step in steps]),
        "gradient_clipped_steps": sum(
            1 for step in steps if bool(step["gradient_clipped"])
        ),
        "initial_learning_rate": steps[0]["learning_rate"],
        "final_learning_rate": steps[-1]["learning_rate"],
        "maximum_peak_memory_bytes": max(peak_memory) if peak_memory else None,
    }


def _checkpoint_metadata(
    *,
    config: Phase4ExperimentConfig,
    manifest: dict[str, Any],
    validation_history: list[dict[str, Any]],
    best_step: int,
    best_validation_bpb: float,
) -> dict[str, Any]:
    return {
        "phase": 4,
        "experiment_name": config.name,
        "experiment_config_sha256": _experiment_hash(config),
        "corpus_manifest_sha256": _manifest_hash(config),
        "corpus": manifest["corpus"],
        "splits": {
            name: {
                "sha256": value["sha256"],
                "bytes": value["bytes"],
                "chapters": value["chapters"],
            }
            for name, value in manifest["splits"].items()
        },
        "validation_history": validation_history,
        "best_step": best_step,
        "best_validation_bits_per_byte": best_validation_bpb,
        "test_evaluated": False,
    }


def train_phase4_experiment(
    *,
    config: Phase4ExperimentConfig,
    run_directory: Path,
    resume_checkpoint: Path | None = None,
    max_steps_this_run: int | None = None,
) -> dict[str, Any]:
    """Train or resume without reading the sealed test split."""

    manifest = load_phase4_manifest(config)
    run_directory = run_directory.expanduser().resolve()
    if (run_directory / "final_test_consumed.json").is_file():
        raise PermissionError(
            "this run is sealed after final-test evaluation; use a new run directory "
            "and versioned experiment instead of overwriting its evidence"
        )
    checkpoint_directory = run_directory / "checkpoints"
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    train_dataset = load_phase4_dataset(config, "train")

    if resume_checkpoint is None:
        model = ByteGPT(config.model)
        parameters = config.validate_model(model)
        trainer = Trainer(model, config.training)
        initial_validation = _validation_metrics(config, model)
        validation_history = [{"step": 0, **initial_validation}]
        best_step = 0
        best_validation_bpb = float(initial_validation["bits_per_byte"])
        initial_path = checkpoint_directory / "step_000000"
        save_checkpoint(
            initial_path,
            trainer,
            run_metadata=_checkpoint_metadata(
                config=config,
                manifest=manifest,
                validation_history=validation_history,
                best_step=best_step,
                best_validation_bpb=best_validation_bpb,
            ),
        )
    else:
        loaded = load_checkpoint(resume_checkpoint)
        metadata = _validate_checkpoint_contract(config, loaded)
        model = loaded.model
        parameters = config.validate_model(model)
        trainer = loaded.trainer
        validation_history = [dict(value) for value in metadata["validation_history"]]
        best_step = int(metadata["best_step"])
        best_validation_bpb = float(metadata["best_validation_bits_per_byte"])

    start_step = trainer.global_step
    if start_step >= config.training.steps:
        raise ValueError("checkpoint has already reached the configured training steps")
    if max_steps_this_run is not None and max_steps_this_run <= 0:
        raise ValueError("max_steps_this_run must be positive")
    target_step = config.training.steps
    if max_steps_this_run is not None:
        target_step = min(target_step, start_step + max_steps_this_run)

    step_records: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    while trainer.global_step < target_step:
        current = trainer.global_step
        next_validation = (
            (current // config.evaluation.validation_every) + 1
        ) * config.evaluation.validation_every
        next_checkpoint = (
            (current // config.evaluation.checkpoint_every) + 1
        ) * config.evaluation.checkpoint_every
        boundary = min(target_step, next_validation, next_checkpoint)
        report = trainer.train(train_dataset, steps=boundary - current)
        step_records.extend(_step_as_dict(step) for step in report.steps)

        validation: dict[str, Any] | None = None
        improved = False
        if trainer.global_step % config.evaluation.validation_every == 0:
            validation = _validation_metrics(config, model)
            validation_history.append({"step": trainer.global_step, **validation})
            improved = float(validation["bits_per_byte"]) < best_validation_bpb
            if improved:
                best_validation_bpb = float(validation["bits_per_byte"])
                best_step = trainer.global_step

        scheduled_checkpoint = (
            trainer.global_step % config.evaluation.checkpoint_every == 0
        )
        if scheduled_checkpoint or improved or trainer.global_step == target_step:
            path = checkpoint_directory / f"step_{trainer.global_step:06d}"
            save_checkpoint(
                path,
                trainer,
                run_metadata=_checkpoint_metadata(
                    config=config,
                    manifest=manifest,
                    validation_history=validation_history,
                    best_step=best_step,
                    best_validation_bpb=best_validation_bpb,
                ),
            )
            checkpoints.append(
                {
                    "step": trainer.global_step,
                    "path": str(path),
                    "scheduled": scheduled_checkpoint,
                    "new_best": improved,
                    "validation": validation,
                }
            )

    best_checkpoint = checkpoint_directory / f"step_{best_step:06d}"
    payload = {
        "schema_version": 1,
        "status": "complete" if trainer.global_step == config.training.steps else "partial",
        "environment": environment_metadata(),
        "experiment": config.as_dict(),
        "experiment_config_sha256": _experiment_hash(config),
        "corpus_manifest": manifest,
        "corpus_manifest_sha256": _manifest_hash(config),
        "parameters": parameters,
        "start_step": start_step,
        "final_step": trainer.global_step,
        "target_step": config.training.steps,
        "resumed_from": (
            str(resume_checkpoint.expanduser().resolve())
            if resume_checkpoint is not None
            else None
        ),
        "validation_history": validation_history,
        "best_validation_bits_per_byte": best_validation_bpb,
        "best_step": best_step,
        "best_checkpoint": str(best_checkpoint),
        "checkpoints_written_this_run": checkpoints,
        "training_steps": step_records,
        "training_summary": _training_summary(step_records),
        "test_evaluated": False,
        "interpretation": (
            "validation selects the checkpoint; the sealed test split is not read here"
        ),
    }
    write_json(run_directory / "training_report.json", payload)
    write_json(
        run_directory / "selection.json",
        {
            "schema_version": 1,
            "best_step": best_step,
            "best_validation_bits_per_byte": best_validation_bpb,
            "best_checkpoint": str(best_checkpoint),
            "test_evaluated": False,
        },
    )
    return payload


def evaluate_phase4_final_test(
    *,
    config: Phase4ExperimentConfig,
    checkpoint: Path,
    output: Path,
    acknowledgement: str,
) -> dict[str, Any]:
    if acknowledgement != FINAL_TEST_ACKNOWLEDGEMENT:
        raise PermissionError(
            "final test remains sealed; pass the exact acknowledgement string"
        )
    checkpoint = checkpoint.expanduser().resolve()
    output = output.expanduser().resolve()
    consumption_marker = checkpoint.parent.parent / "final_test_consumed.json"
    if consumption_marker.is_file():
        raise PermissionError(
            f"this Phase 4 run already consumed its final test; use the recorded "
            f"report in {consumption_marker}"
        )
    if output.is_file():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if (
            existing.get("test_evaluated") is True
            and existing.get("checkpoint") == str(checkpoint)
            and existing.get("experiment_config_sha256") == _experiment_hash(config)
        ):
            write_json(
                consumption_marker,
                {
                    "schema_version": 1,
                    "test_evaluated": True,
                    "checkpoint": str(checkpoint),
                    "report": str(output),
                    "report_sha256": sha256_file(output),
                },
            )
            return existing
        raise FileExistsError(f"refusing to overwrite existing final-test output: {output}")
    manifest = load_phase4_manifest(config)
    loaded = load_checkpoint(checkpoint)
    checkpoint_metadata = _validate_checkpoint_contract(config, loaded)
    test_dataset = load_phase4_dataset(config, "test", allow_test=True)
    test_loss = evaluate_loss(
        loaded.model,
        test_dataset,
        batches=config.evaluation.test_batches,
    )
    payload = {
        "schema_version": 1,
        "environment": environment_metadata(),
        "experiment_name": config.name,
        "experiment_config_sha256": _experiment_hash(config),
        "corpus_manifest_sha256": _manifest_hash(config),
        "checkpoint": str(checkpoint),
        "checkpoint_step": loaded.trainer.global_step,
        "checkpoint_selection": {
            "best_step": checkpoint_metadata["best_step"],
            "best_validation_bits_per_byte": checkpoint_metadata[
                "best_validation_bits_per_byte"
            ],
        },
        "test": {
            "cross_entropy_nats": test_loss,
            "bits_per_byte": test_loss / math.log(2.0),
            "batches": config.evaluation.test_batches,
            "split": manifest["splits"]["test"],
        },
        "generations": _generation_evidence(config, loaded.model),
        "test_evaluated": True,
        "interpretation": (
            "single final evaluation of the validation-selected checkpoint; do not tune "
            "against this result without defining a new experiment version"
        ),
    }
    write_json(output, payload)
    write_json(
        consumption_marker,
        {
            "schema_version": 1,
            "test_evaluated": True,
            "checkpoint": str(checkpoint),
            "report": str(output),
            "report_sha256": sha256_file(output),
        },
    )
    return payload
