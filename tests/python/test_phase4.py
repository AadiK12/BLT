from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import pytest
from blt_mlx.baseline import parameter_report
from blt_mlx.config import ModelConfig, TrainingConfig
from blt_mlx.data import ByteDataset
from blt_mlx.model import ByteGPT
from blt_mlx.phase4 import (
    Phase4ExperimentConfig,
    load_phase4_dataset,
    prepare_phase4_corpus,
    sha256_bytes,
)
from blt_mlx.research import (
    FINAL_TEST_ACKNOWLEDGEMENT,
    evaluate_phase4_final_test,
    train_phase4_experiment,
)
from blt_mlx.training import Trainer, learning_rate_schedule
from mlx.utils import tree_flatten

START_MARKER = "*** START OF THE PROJECT GUTENBERG EBOOK TEST CORPUS ***"
END_MARKER = "*** END OF THE PROJECT GUTENBERG EBOOK TEST CORPUS ***"
SEPARATOR = "\n<|document|>\n"
CHAPTERS = (
    "CHAPTER I.\nAlpha alpha alpha alpha.\n",
    "CHAPTER II.\nBeta beta beta beta.\n",
    "CHAPTER III.\nGamma gamma gamma gamma.\n",
    "CHAPTER IV.\nDelta delta delta delta.\n",
)


def _split_bytes(chapters: tuple[int, ...]) -> bytes:
    return (SEPARATOR.join(CHAPTERS[index - 1] for index in chapters).rstrip() + "\n").encode()


def _write_tiny_phase4_config(tmp_path: Path) -> Path:
    raw = (
        f"Header\r\n{START_MARKER}\r\n\r\n"
        + "\r\n".join(chapter.replace("\n", "\r\n").rstrip("\r\n") for chapter in CHAPTERS)
        + f"\r\n{END_MARKER}\r\nFooter\r\n"
    ).encode()
    source = tmp_path / "source.txt"
    source.write_bytes(raw)
    model_config = ModelConfig(
        max_sequence_length=16,
        d_model=16,
        num_layers=1,
        num_heads=4,
        mlp_hidden_size=32,
        seed=701,
        fusion_strategy="mlx_eager",
    )
    payload = {
        "schema_version": 1,
        "name": "phase4-test",
        "expected_parameter_count": parameter_report(ByteGPT(model_config))[
            "total_parameters"
        ],
        "model": model_config.as_dict(),
        "training": {
            "schema_version": 2,
            "steps": 4,
            "batch_size": 2,
            "sequence_length": 8,
            "learning_rate": 0.005,
            "weight_decay": 0.01,
            "seed": 702,
            "compile_step": True,
            "log_every": 1,
            "learning_rate_schedule": "warmup_cosine",
            "warmup_steps": 1,
            "minimum_learning_rate_ratio": 0.1,
            "gradient_clip_norm": 1.0,
            "beta1": 0.9,
            "beta2": 0.95,
            "epsilon": 1e-8,
        },
        "corpus": {
            "name": "phase4-test-corpus",
            "title": "Test Corpus",
            "author": "Test Author",
            "source_page": source.resolve().as_uri(),
            "source_url": source.resolve().as_uri(),
            "license": "test fixture",
            "raw_sha256": sha256_bytes(raw),
            "start_marker": START_MARKER,
            "end_marker": END_MARKER,
            "expected_chapters": 4,
            "document_separator": SEPARATOR,
            "prepared_directory": "prepared",
            "train": {
                "chapters": [1, 2],
                "expected_sha256": sha256_bytes(_split_bytes((1, 2))),
            },
            "validation": {
                "chapters": [3],
                "expected_sha256": sha256_bytes(_split_bytes((3,))),
            },
            "test": {
                "chapters": [4],
                "expected_sha256": sha256_bytes(_split_bytes((4,))),
            },
        },
        "evaluation": {
            "validation_every": 2,
            "validation_batches": 1,
            "checkpoint_every": 2,
            "test_batches": 1,
            "max_new_bytes": 2,
            "prompts": ["A"],
        },
    }
    path = tmp_path / "phase4.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_warmup_clip_parameter_delta_and_early_loss_reproducibility() -> None:
    model_config = ModelConfig(
        max_sequence_length=16,
        d_model=16,
        num_layers=1,
        num_heads=4,
        mlp_hidden_size=32,
        seed=703,
        fusion_strategy="mlx_eager",
    )
    training_config = TrainingConfig(
        schema_version=2,
        steps=6,
        batch_size=2,
        sequence_length=8,
        learning_rate=0.01,
        seed=704,
        compile_step=True,
        log_every=1,
        learning_rate_schedule="warmup_cosine",
        warmup_steps=2,
        minimum_learning_rate_ratio=0.1,
        gradient_clip_norm=1e-8,
    )
    schedule = learning_rate_schedule(training_config)
    assert float(schedule(mx.array(0)).item()) == 0.0
    assert float(schedule(mx.array(2)).item()) == pytest.approx(0.01)
    assert float(schedule(mx.array(6)).item()) == pytest.approx(0.001)

    data = ByteDataset.from_text(
        "repeatable byte training data\n",
        sequence_length=training_config.sequence_length,
        batch_size=training_config.batch_size,
        seed=training_config.seed,
    )
    first_model = ByteGPT(model_config)
    second_model = ByteGPT(model_config)
    before = {
        name: value.tolist()
        for name, value in tree_flatten(first_model.trainable_parameters())
    }
    first = Trainer(first_model, training_config).train(data, steps=2)
    second = Trainer(second_model, training_config).train(data, steps=2)
    assert [step.loss for step in first.steps] == pytest.approx(
        [step.loss for step in second.steps],
        rel=0.0,
        abs=0.0,
    )
    assert all(step.gradient_clipped for step in first.steps)
    assert any(
        value.tolist() != before[name]
        for name, value in tree_flatten(first_model.trainable_parameters())
    )


def test_phase4_corpus_training_resume_and_sealed_test(tmp_path: Path) -> None:
    config_path = _write_tiny_phase4_config(tmp_path)
    config = Phase4ExperimentConfig.load(config_path)
    manifest = prepare_phase4_corpus(config)
    assert manifest["splits"]["train"]["chapters"] == [1, 2]
    assert manifest["splits"]["validation"]["chapters"] == [3]
    assert manifest["splits"]["test"]["chapters"] == [4]
    assert load_phase4_dataset(config, "train").documents
    with pytest.raises(PermissionError, match="sealed"):
        load_phase4_dataset(config, "test")

    test_path = Path(manifest["splits"]["test"]["path"])
    test_bytes = test_path.read_bytes()
    test_path.write_bytes(b"corrupted test bytes")
    run_directory = tmp_path / "run"
    partial = train_phase4_experiment(
        config=config,
        run_directory=run_directory,
        max_steps_this_run=2,
    )
    assert partial["status"] == "partial"
    assert partial["final_step"] == 2
    assert partial["test_evaluated"] is False

    test_path.write_bytes(test_bytes)
    resumed = train_phase4_experiment(
        config=config,
        run_directory=run_directory,
        resume_checkpoint=run_directory / "checkpoints" / "step_000002",
    )
    assert resumed["status"] == "complete"
    assert resumed["start_step"] == 2
    assert resumed["final_step"] == 4
    assert [value["step"] for value in resumed["validation_history"]] == [0, 2, 4]
    assert json.loads((run_directory / "selection.json").read_text())[
        "test_evaluated"
    ] is False

    final_checkpoint = run_directory / "checkpoints" / "step_000004"
    with pytest.raises(PermissionError, match="exact acknowledgement"):
        evaluate_phase4_final_test(
            config=config,
            checkpoint=final_checkpoint,
            output=tmp_path / "final.json",
            acknowledgement="not yet",
        )
    final = evaluate_phase4_final_test(
        config=config,
        checkpoint=final_checkpoint,
        output=tmp_path / "final.json",
        acknowledgement=FINAL_TEST_ACKNOWLEDGEMENT,
    )
    assert final["test_evaluated"] is True
    assert final["checkpoint_step"] == 4
    assert final["test"]["bits_per_byte"] > 0.0
    assert (run_directory / "final_test_consumed.json").is_file()
    with pytest.raises(PermissionError, match="already consumed"):
        evaluate_phase4_final_test(
            config=config,
            checkpoint=final_checkpoint,
            output=tmp_path / "another-final.json",
            acknowledgement=FINAL_TEST_ACKNOWLEDGEMENT,
        )
    with pytest.raises(PermissionError, match="run is sealed"):
        train_phase4_experiment(
            config=config,
            run_directory=run_directory,
            resume_checkpoint=final_checkpoint,
        )
