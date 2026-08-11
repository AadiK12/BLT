"""Frozen Stage 3 baseline, dataset provenance, and inspection contracts."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import mlx.core as mx
from mlx.utils import tree_flatten

from blt_mlx.config import ModelConfig, TrainingConfig
from blt_mlx.data import PackedByteDataset
from blt_mlx.model import ByteGPT

SplitName = Literal["train", "validation"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class DataSplitConfig:
    path: str
    sha256: str


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    document_separator: str
    train: DataSplitConfig
    validation: DataSplitConfig


@dataclass(frozen=True)
class EvaluationConfig:
    batches: int
    max_new_bytes: int
    prompts: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.batches <= 0 or self.max_new_bytes <= 0:
            raise ValueError("evaluation batches and max_new_bytes must be positive")
        if not self.prompts or any(not prompt for prompt in self.prompts):
            raise ValueError("evaluation prompts must be non-empty")


@dataclass(frozen=True)
class Stage3BaselineConfig:
    schema_version: int
    name: str
    expected_parameter_count: int
    model: ModelConfig
    training: TrainingConfig
    dataset: DatasetConfig
    evaluation: EvaluationConfig
    source_path: Path

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported Stage 3 baseline schema")
        if not self.name:
            raise ValueError("baseline name must not be empty")
        if self.expected_parameter_count <= 0:
            raise ValueError("expected_parameter_count must be positive")
        if self.training.sequence_length > self.model.max_sequence_length:
            raise ValueError("training sequence length exceeds the model maximum")

    @classmethod
    def load(cls, path: Path) -> Stage3BaselineConfig:
        resolved = path.expanduser().resolve()
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        dataset = payload["dataset"]
        evaluation = payload["evaluation"]
        return cls(
            schema_version=int(payload["schema_version"]),
            name=str(payload["name"]),
            expected_parameter_count=int(payload["expected_parameter_count"]),
            model=ModelConfig.from_dict(payload["model"]),
            training=TrainingConfig.from_dict(payload["training"]),
            dataset=DatasetConfig(
                name=str(dataset["name"]),
                document_separator=str(dataset["document_separator"]),
                train=DataSplitConfig(**dataset["train"]),
                validation=DataSplitConfig(**dataset["validation"]),
            ),
            evaluation=EvaluationConfig(
                batches=int(evaluation["batches"]),
                max_new_bytes=int(evaluation["max_new_bytes"]),
                prompts=tuple(str(prompt) for prompt in evaluation["prompts"]),
            ),
            source_path=resolved,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "expected_parameter_count": self.expected_parameter_count,
            "model": self.model.as_dict(),
            "training": self.training.as_dict(),
            "dataset": asdict(self.dataset),
            "evaluation": asdict(self.evaluation),
            "source_path": str(self.source_path),
        }

    def _split_config(self, split: SplitName) -> DataSplitConfig:
        return self.dataset.train if split == "train" else self.dataset.validation

    def resolve_split(self, split: SplitName) -> Path:
        configured = self._split_config(split)
        return (self.source_path.parent / configured.path).resolve()

    def dataset_evidence(self, split: SplitName) -> dict[str, Any]:
        configured = self._split_config(split)
        path = self.resolve_split(split)
        if not path.is_file():
            raise FileNotFoundError(f"{split} dataset does not exist: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != configured.sha256:
            raise ValueError(
                f"{split} dataset SHA-256 mismatch: expected {configured.sha256}, "
                f"got {actual_hash}"
            )
        text = path.read_text(encoding="utf-8")
        documents = tuple(
            document for document in text.split(self.dataset.document_separator) if document
        )
        if not documents:
            raise ValueError(f"{split} dataset contains no documents")
        return {
            "split": split,
            "path": str(path),
            "sha256": actual_hash,
            "bytes": len(text.encode("utf-8")),
            "documents": len(documents),
        }

    def load_dataset(self, split: SplitName) -> PackedByteDataset:
        evidence = self.dataset_evidence(split)
        text = Path(evidence["path"]).read_text(encoding="utf-8")
        documents = tuple(
            document for document in text.split(self.dataset.document_separator) if document
        )
        seed = self.training.seed if split == "train" else self.training.seed + 10_000
        return PackedByteDataset.from_text_documents(
            documents,
            sequence_length=self.training.sequence_length,
            batch_size=self.training.batch_size,
            seed=seed,
        )


def parameter_report(model: ByteGPT) -> dict[str, Any]:
    parameters: list[dict[str, Any]] = []
    groups: dict[str, int] = {}
    total = 0
    total_bytes = 0
    dtype_widths = {
        str(mx.float32): 4,
        str(mx.float16): 2,
        str(mx.bfloat16): 2,
    }
    for name, value in tree_flatten(model.parameters()):
        count = int(value.size)
        dtype = str(value.dtype)
        width = dtype_widths.get(dtype)
        if width is None:
            raise TypeError(f"unsupported parameter dtype for accounting: {dtype}")
        component = name.split(".", maxsplit=1)[0]
        groups[component] = groups.get(component, 0) + count
        total += count
        total_bytes += count * width
        parameters.append(
            {
                "name": name,
                "shape": [int(dimension) for dimension in value.shape],
                "dtype": dtype,
                "count": count,
                "storage_bytes": count * width,
            }
        )
    return {
        "total_parameters": total,
        "trainable_parameters": total,
        "parameter_storage_bytes": total_bytes,
        "by_component": dict(sorted(groups.items())),
        "parameters": parameters,
    }


def validate_parameter_count(config: Stage3BaselineConfig, model: ByteGPT) -> dict[str, Any]:
    report = parameter_report(model)
    actual = int(report["total_parameters"])
    if actual != config.expected_parameter_count:
        raise ValueError(
            f"parameter-count mismatch: expected {config.expected_parameter_count}, got {actual}"
        )
    return report


def byte_display(value: bytes) -> dict[str, Any]:
    try:
        text = value.decode("utf-8", errors="strict")
        valid_utf8 = True
    except UnicodeDecodeError:
        text = value.decode("utf-8", errors="replace")
        valid_utf8 = False
    return {
        "text": text,
        "valid_utf8": valid_utf8,
        "hex": value.hex(),
        "byte_values": list(value),
        "byte_count": len(value),
    }


def uniform_byte_baseline() -> dict[str, float]:
    cross_entropy = math.log(256.0)
    return {
        "cross_entropy_nats": cross_entropy,
        "bits_per_byte": cross_entropy / math.log(2.0),
    }
