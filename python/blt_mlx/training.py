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
from blt_mlx.data import LanguageModelDataset
from blt_mlx.model import ByteGPT


def learning_rate_schedule(config: TrainingConfig):
    """Build the exact optimizer schedule recorded by the training config."""

    if config.learning_rate_schedule == "constant":
        return config.learning_rate
    warmup = optim.linear_schedule(
        0.0,
        config.learning_rate,
        config.warmup_steps,
    )
    decay = optim.cosine_decay(
        config.learning_rate,
        config.steps - config.warmup_steps,
        config.learning_rate * config.minimum_learning_rate_ratio,
    )
    return optim.join_schedules([warmup, decay], [config.warmup_steps])


def language_model_loss(
    model: ByteGPT,
    inputs: mx.array,
    targets: mx.array,
    attention_mask: mx.array,
    loss_mask: mx.array,
    document_ids: mx.array,
) -> mx.array:
    logits = model(
        inputs,
        attention_mask=attention_mask,
        document_ids=document_ids,
    )
    token_losses = nn.losses.cross_entropy(logits, targets, reduction="none")
    denominator = mx.sum(loss_mask)
    return mx.sum(token_losses * loss_mask) / mx.maximum(denominator, 1.0)


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
    learning_rate: float
    gradient_norm: float
    gradient_clipped: bool


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
            learning_rate=learning_rate_schedule(config),
            betas=[config.beta1, config.beta2],
            eps=config.epsilon,
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
        attention_mask: mx.array,
        loss_mask: mx.array,
        document_ids: mx.array,
    ) -> tuple[mx.array, dict, mx.array]:
        loss, gradients = self._loss_and_grad(
            self.model,
            inputs,
            targets,
            attention_mask,
            loss_mask,
            document_ids,
        )
        if self.config.gradient_clip_norm is not None:
            gradients, gradient_norm = optim.clip_grad_norm(
                gradients,
                self.config.gradient_clip_norm,
            )
        else:
            gradient_norm_squared = sum(
                mx.sum(value * value)
                for _, value in tree_flatten(gradients)
                if isinstance(value, mx.array)
            )
            gradient_norm = mx.sqrt(gradient_norm_squared)
        self.optimizer.update(self.model, gradients)
        return loss, gradients, gradient_norm

    def step(
        self,
        inputs: mx.array,
        targets: mx.array,
        *,
        attention_mask: mx.array | None = None,
        loss_mask: mx.array | None = None,
        document_ids: mx.array | None = None,
    ) -> TrainingStep:
        batch_shape = inputs.shape
        attention_mask = (
            mx.ones(batch_shape, dtype=mx.bool_)
            if attention_mask is None
            else attention_mask
        )
        loss_mask = (
            mx.ones(batch_shape, dtype=mx.float32) if loss_mask is None else loss_mask
        )
        document_ids = (
            mx.zeros(batch_shape, dtype=mx.int32)
            if document_ids is None
            else document_ids
        )
        if mx.metal.is_available():
            mx.reset_peak_memory()
        started = time.perf_counter_ns()
        loss, gradients, gradient_norm = self._step_function(
            inputs,
            targets,
            attention_mask,
            loss_mask,
            document_ids,
        )
        mx.eval(
            loss,
            gradient_norm,
            self.model.parameters(),
            self.optimizer.state,
        )
        mx.synchronize()
        duration_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        if not bool(mx.isfinite(loss).item()):
            raise FloatingPointError("training loss became non-finite")
        if not _all_finite(gradients):
            raise FloatingPointError("training gradients became non-finite")
        gradient_norm_value = float(gradient_norm.item())
        learning_rate_value = float(self.optimizer.learning_rate.item())
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
            learning_rate=learning_rate_value,
            gradient_norm=gradient_norm_value,
            gradient_clipped=(
                self.config.gradient_clip_norm is not None
                and gradient_norm_value > self.config.gradient_clip_norm
            ),
        )

    def train(
        self,
        dataset: LanguageModelDataset,
        *,
        steps: int | None = None,
    ) -> TrainingReport:
        step_count = self.config.steps if steps is None else int(steps)
        if step_count <= 0:
            raise ValueError("steps must be positive")
        observations: list[TrainingStep] = []
        for _ in range(step_count):
            batch = dataset.language_model_batch(self.global_step)
            observations.append(
                self.step(
                    batch.inputs,
                    batch.targets,
                    attention_mask=batch.attention_mask,
                    loss_mask=batch.loss_mask,
                    document_ids=batch.document_ids,
                )
            )
        losses = [observation.loss for observation in observations]
        return TrainingReport(
            initial_loss=losses[0],
            final_loss=losses[-1],
            minimum_loss=min(losses),
            steps=tuple(observations),
        )


def evaluate_loss(
    model: ByteGPT,
    dataset: LanguageModelDataset,
    *,
    batches: int = 8,
) -> float:
    if batches <= 0:
        raise ValueError("batches must be positive")
    losses = []
    for index in range(batches):
        batch = dataset.language_model_batch(index)
        loss = language_model_loss(
            model,
            batch.inputs,
            batch.targets,
            batch.attention_mask,
            batch.loss_mask,
            batch.document_ids,
        )
        mx.eval(loss)
        losses.append(float(loss.item()))
    return sum(losses) / len(losses)


def evaluate_bits_per_byte(
    model: ByteGPT,
    dataset: LanguageModelDataset,
    *,
    batches: int = 8,
) -> float:
    return evaluate_loss(model, dataset, batches=batches) / math.log(2.0)
