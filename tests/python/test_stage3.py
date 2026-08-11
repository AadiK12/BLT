from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import pytest
from blt_mlx.baseline import (
    Stage3BaselineConfig,
    byte_display,
    parameter_report,
    sha256_file,
)
from blt_mlx.cli import main
from blt_mlx.config import ModelConfig
from blt_mlx.data import PackedByteDataset
from blt_mlx.model import ByteGPT


def test_packed_dataset_masks_padding_and_document_transitions() -> None:
    dataset = PackedByteDataset(
        documents=(b"ab", b"cd"),
        sequence_length=5,
        batch_size=1,
        seed=1,
    )
    batch = dataset.language_model_batch(0)
    assert batch.inputs.tolist() == [[ord("a"), ord("b"), ord("c"), ord("d"), 0]]
    assert batch.attention_mask.tolist() == [[True, True, True, True, False]]
    assert batch.document_ids.tolist() == [[0, 0, 1, 1, -1]]
    assert batch.loss_mask.tolist() == [[1.0, 0.0, 1.0, 0.0, 0.0]]


def test_document_mask_prevents_cross_document_attention() -> None:
    model = ByteGPT(
        ModelConfig(
            max_sequence_length=8,
            d_model=16,
            num_layers=1,
            num_heads=4,
            mlp_hidden_size=32,
            seed=303,
            fusion_strategy="mlx_eager",
        )
    )
    first = mx.array([[10, 20, 30, 40]], dtype=mx.int32)
    changed_previous_document = mx.array([[90, 80, 30, 40]], dtype=mx.int32)
    attention_mask = mx.ones((1, 4), dtype=mx.bool_)
    document_ids = mx.array([[0, 0, 1, 1]], dtype=mx.int32)
    first_logits = model(
        first,
        attention_mask=attention_mask,
        document_ids=document_ids,
    )[:, 2:]
    changed_logits = model(
        changed_previous_document,
        attention_mask=attention_mask,
        document_ids=document_ids,
    )[:, 2:]
    mx.eval(first_logits, changed_logits)
    assert bool(mx.allclose(first_logits, changed_logits, rtol=0.0, atol=0.0).item())


def test_padding_does_not_change_valid_prefix_logits() -> None:
    model = ByteGPT(
        ModelConfig(
            max_sequence_length=8,
            d_model=16,
            num_layers=1,
            num_heads=4,
            mlp_hidden_size=32,
            seed=304,
            fusion_strategy="mlx_eager",
        )
    )
    short = model(mx.array([[10, 20]], dtype=mx.int32))
    padded = model(
        mx.array([[10, 20, 0, 0]], dtype=mx.int32),
        attention_mask=mx.array([[True, True, False, False]], dtype=mx.bool_),
        document_ids=mx.array([[0, 0, -1, -1]], dtype=mx.int32),
    )[:, :2]
    mx.eval(short, padded)
    assert bool(mx.allclose(short, padded, rtol=0.0, atol=0.0).item())


def test_parameter_report_and_utf8_display_are_explicit() -> None:
    config = Stage3BaselineConfig.load(Path("configs/stage3_byte_gpt_tiny.json"))
    report = parameter_report(ByteGPT(config.model))
    assert report["total_parameters"] == config.expected_parameter_count == 108_032
    assert sum(report["by_component"].values()) == report["total_parameters"]
    assert byte_display("café".encode())["valid_utf8"] is True
    invalid = byte_display(b"\xffBLT")
    assert invalid["valid_utf8"] is False
    assert invalid["byte_values"] == [255, 66, 76, 84]


def _write_tiny_baseline(tmp_path: Path) -> Path:
    train = tmp_path / "train.txt"
    validation = tmp_path / "validation.txt"
    train.write_text("abcabcabc\n<|document|>\ndefdefdef\n", encoding="utf-8")
    validation.write_text("abcabc\n<|document|>\ndefdef\n", encoding="utf-8")
    model_config = ModelConfig(
        max_sequence_length=16,
        d_model=16,
        num_layers=1,
        num_heads=4,
        mlp_hidden_size=32,
        seed=404,
        fusion_strategy="mlx_eager",
    )
    expected_parameters = parameter_report(ByteGPT(model_config))["total_parameters"]
    payload = {
        "schema_version": 1,
        "name": "stage3-cli-test",
        "expected_parameter_count": expected_parameters,
        "model": model_config.as_dict(),
        "training": {
            "schema_version": 1,
            "steps": 3,
            "batch_size": 2,
            "sequence_length": 8,
            "learning_rate": 0.005,
            "weight_decay": 0.0,
            "seed": 405,
            "compile_step": True,
            "log_every": 1,
        },
        "dataset": {
            "name": "stage3-cli-test-data",
            "document_separator": "\n<|document|>\n",
            "train": {"path": "train.txt", "sha256": sha256_file(train)},
            "validation": {
                "path": "validation.txt",
                "sha256": sha256_file(validation),
            },
        },
        "evaluation": {"batches": 1, "max_new_bytes": 4, "prompts": ["a"]},
    }
    config_path = tmp_path / "baseline.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return config_path


def test_stage3_cli_inspect_train_evaluate_and_generate(tmp_path: Path) -> None:
    config = _write_tiny_baseline(tmp_path)
    checkpoint = tmp_path / "checkpoint"
    inspection = tmp_path / "inspection.json"
    training = tmp_path / "training.json"
    evaluation = tmp_path / "evaluation.json"
    generation = tmp_path / "generation.json"

    assert main(["inspect-baseline", "--config", str(config), "--output", str(inspection)]) == 0
    assert (
        main(
            [
                "train-baseline",
                "--config",
                str(config),
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(training),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "evaluate-checkpoint",
                "--config",
                str(config),
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(evaluation),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "generate",
                "--checkpoint",
                str(checkpoint),
                "--prompt",
                "a",
                "--max-new-bytes",
                "4",
                "--output",
                str(generation),
            ]
        )
        == 0
    )
    for path in (inspection, training, evaluation, generation):
        assert path.is_file()
    state = json.loads((checkpoint / "training_state.json").read_text())
    assert state["global_step"] == 3
    assert state["run_metadata"]["stage"] == 3

    modified = json.loads(config.read_text())
    modified["name"] = "stage3-cli-test-modified"
    config.write_text(json.dumps(modified), encoding="utf-8")
    with pytest.raises(ValueError, match="exact baseline config"):
        main(
            [
                "evaluate-checkpoint",
                "--config",
                str(config),
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(evaluation),
            ]
        )
