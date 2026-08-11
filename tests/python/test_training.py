from __future__ import annotations

import mlx.core as mx
from blt_mlx.checkpoint import load_checkpoint, save_checkpoint
from blt_mlx.config import ModelConfig, TrainingConfig
from blt_mlx.data import ByteDataset
from blt_mlx.model import ByteGPT
from blt_mlx.training import Trainer, evaluate_bits_per_byte


def configs() -> tuple[ModelConfig, TrainingConfig]:
    return (
        ModelConfig(
            max_sequence_length=24,
            d_model=16,
            num_layers=1,
            num_heads=4,
            mlp_hidden_size=32,
            seed=77,
            fusion_strategy="mlx_eager",
        ),
        TrainingConfig(
            steps=30,
            batch_size=4,
            sequence_length=12,
            learning_rate=5e-3,
            seed=88,
            compile_step=True,
            log_every=10,
        ),
    )


def dataset(training: TrainingConfig) -> ByteDataset:
    return ByteDataset.from_text(
        "abcabcabcabcabcabcabcabc\n",
        sequence_length=training.sequence_length,
        batch_size=training.batch_size,
        seed=training.seed,
        repeat_to_at_least=1024,
    )


def test_training_reduces_loss_and_checkpoint_round_trips(tmp_path) -> None:
    model_config, training_config = configs()
    model = ByteGPT(model_config)
    data = dataset(training_config)
    initial_bpb = evaluate_bits_per_byte(model, data, batches=2)
    trainer = Trainer(model, training_config)
    report = trainer.train(data)
    final_bpb = evaluate_bits_per_byte(model, data, batches=2)
    assert report.loss_improved
    assert final_bpb < initial_bpb

    inputs, _ = data.batch(999)
    expected = model(inputs)
    checkpoint = save_checkpoint(tmp_path / "checkpoint", trainer)
    restored = load_checkpoint(checkpoint)
    actual = restored.model(inputs)
    mx.eval(expected, actual)
    assert restored.trainer.global_step == training_config.steps
    assert bool(mx.allclose(expected, actual, rtol=0.0, atol=0.0).item())

    resumed = restored.trainer.train(data, steps=2)
    assert resumed.steps[-1].step == training_config.steps + 2
