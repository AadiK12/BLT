"""Portable checkpoint directories without arbitrary-code serialization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from blt_mlx.config import ModelConfig, TrainingConfig
from blt_mlx.model import ByteGPT
from blt_mlx.training import Trainer


@dataclass(frozen=True)
class LoadedCheckpoint:
    model: ByteGPT
    trainer: Trainer


def save_checkpoint(path: Path, trainer: Trainer) -> Path:
    resolved = path.expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    trainer.model.save_weights(str(resolved / "model.safetensors"))
    optimizer_arrays = {
        name: value
        for name, value in tree_flatten(trainer.optimizer.state)
        if isinstance(value, mx.array)
    }
    mx.save_safetensors(str(resolved / "optimizer.safetensors"), optimizer_arrays)
    metadata = {
        "schema_version": 1,
        "global_step": trainer.global_step,
        "model": trainer.model.config.as_dict(),
        "training": trainer.config.as_dict(),
    }
    (resolved / "training_state.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return resolved


def load_checkpoint(path: Path) -> LoadedCheckpoint:
    resolved = path.expanduser().resolve()
    metadata_path = resolved / "training_state.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"checkpoint metadata does not exist: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1:
        raise ValueError("unsupported checkpoint schema")
    model = ByteGPT(ModelConfig.from_dict(metadata["model"]))
    model.load_weights(str(resolved / "model.safetensors"))
    trainer = Trainer(
        model,
        TrainingConfig.from_dict(metadata["training"]),
        start_step=int(metadata["global_step"]),
    )
    optimizer_values = mx.load(str(resolved / "optimizer.safetensors"))
    trainer.optimizer.state = tree_unflatten(sorted(optimizer_values.items()))
    trainer.rebuild_compiled_step()
    mx.eval(model.parameters(), trainer.optimizer.state)
    return LoadedCheckpoint(model=model, trainer=trainer)
