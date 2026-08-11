"""Deterministic training, evaluation, and finite-gradient contracts."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from blt_mlx.config import TrainingConfig
from blt_mlx.data import ByteDataset
from blt_mlx.model import ByteGPT


def language_model_loss(
    model: ByteGPT,
    inputs: mx.array,
    targets: mx.array,
) -> mx.array:
    logits = model(inputs)
    return mx.mean(
        nn.losses.cross_entropy(
            mx.reshape(logits, (-1, logits.shape[-1])),
            mx.reshape(targets, (-1,)),
        )
    )


def _all_finite(tree: object) -> bool:
    leaves = [value for _, value in tree_flatten(tree) if isinstance(value, mx.array)]
    if not leaves:
        return False
    checks = [mx.all(mx.isfinite(value)) for value in leaves]
    mx.eval(*checks)
    return all(bool(check.item()) for check in checks)


@dataclass(frozen=True)
class TrainingStep:
    step: int
    loss: float
    duration_ms: float
    active_memory_bytes: int | None
    peak_memory_bytes: int | None


@dataclass(frozen=True)
class TrainingReport:
    initial_loss: float
    final_loss: float
    minimum_loss: float
    steps: tuple[TrainingStep, ...]

    @property
    def loss_improved(self) -> bool:
        return self.final_loss < self.initial_loss


class Trainer:
    """Own model, optimizer, compiled state, and deterministic global step."""

    def __init__(
        self,
        model: ByteGPT,
        config: TrainingConfig,
        *,
        start_step: int = 0,
    ) -> None:
        self.model = model
        self.config = config
        self.global_step = int(start_step)
        self.optimizer = optim.AdamW(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.optimizer.init(model.trainable_parameters())
        self._loss_and_grad = nn.value_and_grad(model, language_model_loss)
        self._configure_step_function()

    def _configure_step_function(self) -> None:
        self._step_function = self._uncompiled_step
        if self.config.compile_step:
            state = [self.model.state, self.optimizer.state]
            self._step_function = partial(
                mx.compile,
                inputs=state,
                outputs=state,
            )(self._uncompiled_step)

    def rebuild_compiled_step(self) -> None:
        """Recapture model and optimizer state after checkpoint restoration."""

        self._configure_step_function()

    def _uncompiled_step(
        self,
        inputs: mx.array,
        targets: mx.array,
    ) -> tuple[mx.array, dict]:
        loss, gradients = self._loss_and_grad(self.model, inputs, targets)
        self.optimizer.update(self.model, gradients)
        return loss, gradients

    def step(self, inputs: mx.array, targets: mx.array) -> TrainingStep:
        if mx.metal.is_available():
            mx.reset_peak_memory()
        started = time.perf_counter_ns()
        loss, gradients = self._step_function(inputs, targets)
        mx.eval(loss, self.model.parameters(), self.optimizer.state)
        mx.synchronize()
        duration_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        if not bool(mx.isfinite(loss).item()):
            raise FloatingPointError("training loss became non-finite")
        if not _all_finite(gradients):
            raise FloatingPointError("training gradients became non-finite")
        self.global_step += 1
        return TrainingStep(
            step=self.global_step,
            loss=float(loss.item()),
            duration_ms=duration_ms,
            active_memory_bytes=(
                int(mx.get_active_memory()) if mx.metal.is_available() else None
            ),
            peak_memory_bytes=(
                int(mx.get_peak_memory()) if mx.metal.is_available() else None
            ),
        )

    def train(self, dataset: ByteDataset, *, steps: int | None = None) -> TrainingReport:
        step_count = self.config.steps if steps is None else int(steps)
        if step_count <= 0:
            raise ValueError("steps must be positive")
        observations: list[TrainingStep] = []
        for _ in range(step_count):
            inputs, targets = dataset.batch(self.global_step)
            observations.append(self.step(inputs, targets))
        losses = [observation.loss for observation in observations]
        return TrainingReport(
            initial_loss=losses[0],
            final_loss=losses[-1],
            minimum_loss=min(losses),
            steps=tuple(observations),
        )


def evaluate_loss(
    model: ByteGPT,
    dataset: ByteDataset,
    *,
    batches: int = 8,
) -> float:
    if batches <= 0:
        raise ValueError("batches must be positive")
    losses = []
    for index in range(batches):
        inputs, targets = dataset.batch(index)
        loss = language_model_loss(model, inputs, targets)
        mx.eval(loss)
        losses.append(float(loss.item()))
    return sum(losses) / len(losses)


def evaluate_bits_per_byte(
    model: ByteGPT,
    dataset: ByteDataset,
    *,
    batches: int = 8,
) -> float:
    return evaluate_loss(model, dataset, batches=batches) / math.log(2.0)
