"""Phase 4 corpus provenance and research-training configuration contracts."""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from blt_mlx.baseline import parameter_report, sha256_file
from blt_mlx.config import ModelConfig, TrainingConfig
from blt_mlx.data import PackedByteDataset
from blt_mlx.model import ByteGPT

CorpusSplit = Literal["train", "validation", "test"]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class CorpusSplitConfig:
    chapters: tuple[int, ...]
    expected_sha256: str

    def __post_init__(self) -> None:
        if not self.chapters or any(chapter <= 0 for chapter in self.chapters):
            raise ValueError("corpus split chapters must be positive")
        if len(self.expected_sha256) != 64:
            raise ValueError("corpus split expected_sha256 must be a SHA-256 digest")


@dataclass(frozen=True)
class ExternalCorpusConfig:
    name: str
    title: str
    author: str
    source_page: str
    source_url: str
    license: str
    raw_sha256: str
    start_marker: str
    end_marker: str
    expected_chapters: int
    document_separator: str
    prepared_directory: str
    train: CorpusSplitConfig
    validation: CorpusSplitConfig
    test: CorpusSplitConfig

    def __post_init__(self) -> None:
        if not self.name or not self.source_url or not self.license:
            raise ValueError("corpus name, source URL, and license are required")
        if len(self.raw_sha256) != 64:
            raise ValueError("raw_sha256 must be a SHA-256 digest")
        if self.expected_chapters <= 0:
            raise ValueError("expected_chapters must be positive")
        assigned = self.train.chapters + self.validation.chapters + self.test.chapters
        if len(set(assigned)) != len(assigned):
            raise ValueError("train, validation, and test chapters must be disjoint")
        if set(assigned) != set(range(1, self.expected_chapters + 1)):
            raise ValueError("corpus splits must assign every expected chapter exactly once")


@dataclass(frozen=True)
class Phase4EvaluationConfig:
    validation_every: int
    validation_batches: int
    checkpoint_every: int
    test_batches: int
    max_new_bytes: int
    prompts: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "validation_every",
            "validation_batches",
            "checkpoint_every",
            "test_batches",
            "max_new_bytes",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not self.prompts or any(not prompt for prompt in self.prompts):
            raise ValueError("Phase 4 prompts must be non-empty")


@dataclass(frozen=True)
class Phase4ExperimentConfig:
    schema_version: int
    name: str
    expected_parameter_count: int
    model: ModelConfig
    training: TrainingConfig
    corpus: ExternalCorpusConfig
    evaluation: Phase4EvaluationConfig
    source_path: Path

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported Phase 4 experiment schema")
        if self.training.schema_version != 2:
            raise ValueError("Phase 4 requires training configuration schema 2")
        if self.training.sequence_length > self.model.max_sequence_length:
            raise ValueError("training sequence length exceeds model maximum")
        if self.training.steps % self.evaluation.validation_every != 0:
            raise ValueError("training steps must be divisible by validation_every")
        if self.training.steps % self.evaluation.checkpoint_every != 0:
            raise ValueError("training steps must be divisible by checkpoint_every")

    @classmethod
    def load(cls, path: Path) -> Phase4ExperimentConfig:
        resolved = path.expanduser().resolve()
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        corpus = payload["corpus"]
        evaluation = payload["evaluation"]
        return cls(
            schema_version=int(payload["schema_version"]),
            name=str(payload["name"]),
            expected_parameter_count=int(payload["expected_parameter_count"]),
            model=ModelConfig.from_dict(payload["model"]),
            training=TrainingConfig.from_dict(payload["training"]),
            corpus=ExternalCorpusConfig(
                name=str(corpus["name"]),
                title=str(corpus["title"]),
                author=str(corpus["author"]),
                source_page=str(corpus["source_page"]),
                source_url=str(corpus["source_url"]),
                license=str(corpus["license"]),
                raw_sha256=str(corpus["raw_sha256"]),
                start_marker=str(corpus["start_marker"]),
                end_marker=str(corpus["end_marker"]),
                expected_chapters=int(corpus["expected_chapters"]),
                document_separator=str(corpus["document_separator"]),
                prepared_directory=str(corpus["prepared_directory"]),
                train=CorpusSplitConfig(
                    chapters=tuple(int(value) for value in corpus["train"]["chapters"]),
                    expected_sha256=str(corpus["train"]["expected_sha256"]),
                ),
                validation=CorpusSplitConfig(
                    chapters=tuple(
                        int(value) for value in corpus["validation"]["chapters"]
                    ),
                    expected_sha256=str(corpus["validation"]["expected_sha256"]),
                ),
                test=CorpusSplitConfig(
                    chapters=tuple(int(value) for value in corpus["test"]["chapters"]),
                    expected_sha256=str(corpus["test"]["expected_sha256"]),
                ),
            ),
            evaluation=Phase4EvaluationConfig(
                validation_every=int(evaluation["validation_every"]),
                validation_batches=int(evaluation["validation_batches"]),
                checkpoint_every=int(evaluation["checkpoint_every"]),
                test_batches=int(evaluation["test_batches"]),
                max_new_bytes=int(evaluation["max_new_bytes"]),
                prompts=tuple(str(value) for value in evaluation["prompts"]),
            ),
            source_path=resolved,
        )

    @property
    def prepared_directory(self) -> Path:
        return (self.source_path.parent / self.corpus.prepared_directory).resolve()

    @property
    def manifest_path(self) -> Path:
        return self.prepared_directory / "manifest.json"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "expected_parameter_count": self.expected_parameter_count,
            "model": self.model.as_dict(),
            "training": self.training.as_dict(),
            "corpus": asdict(self.corpus),
            "evaluation": asdict(self.evaluation),
            "source_path": str(self.source_path),
        }

    def validate_model(self, model: ByteGPT) -> dict[str, Any]:
        report = parameter_report(model)
        actual = int(report["total_parameters"])
        if actual != self.expected_parameter_count:
            raise ValueError(
                f"Phase 4 parameter-count mismatch: expected "
                f"{self.expected_parameter_count}, got {actual}"
            )
        return report


def _download_source(config: Phase4ExperimentConfig) -> bytes:
    request = urllib.request.Request(
        config.corpus.source_url,
        headers={"User-Agent": "blt-mlx-lab/0.2 reproducible research corpus"},
    )
    value: bytes | None = None
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
                value = response.read()
            break
        except (TimeoutError, urllib.error.HTTPError, urllib.error.URLError) as error:
            last_error = error
            if attempt < 3:
                time.sleep(attempt)
    if value is None:
        raise ConnectionError(
            f"failed to download pinned corpus after 3 attempts: {last_error}"
        ) from last_error
    actual = sha256_bytes(value)
    if actual != config.corpus.raw_sha256:
        raise ValueError(
            f"raw corpus SHA-256 mismatch: expected {config.corpus.raw_sha256}, "
            f"got {actual}"
        )
    return value


def _extract_chapters(config: Phase4ExperimentConfig, raw: bytes) -> tuple[str, ...]:
    text = raw.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    start = text.find(config.corpus.start_marker)
    end = text.find(config.corpus.end_marker)
    if start < 0 or end < 0 or end <= start:
        raise ValueError("Project Gutenberg start/end markers were not found")
    body = text[start + len(config.corpus.start_marker) : end].strip()
    matches = list(re.finditer(r"(?m)^CHAPTER ([IVX]+)\.\n", body))
    if len(matches) != config.corpus.expected_chapters:
        raise ValueError(
            f"expected {config.corpus.expected_chapters} chapters, found {len(matches)}"
        )
    chapters = []
    for index, match in enumerate(matches):
        chapter_end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        chapters.append(body[match.start() : chapter_end].strip() + "\n")
    return tuple(chapters)


def _split_bytes(
    config: Phase4ExperimentConfig,
    chapters: tuple[str, ...],
    split: CorpusSplit,
) -> bytes:
    split_config = getattr(config.corpus, split)
    selected = [chapters[index - 1] for index in split_config.chapters]
    text = config.corpus.document_separator.join(selected).rstrip() + "\n"
    return text.encode("utf-8")


def prepare_phase4_corpus(config: Phase4ExperimentConfig) -> dict[str, Any]:
    """Download, verify, chapter-split, and freeze the external corpus."""

    output = config.prepared_directory
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "source.txt"
    if raw_path.is_file() and sha256_file(raw_path) == config.corpus.raw_sha256:
        raw = raw_path.read_bytes()
    else:
        raw = _download_source(config)
        raw_path.write_bytes(raw)
    chapters = _extract_chapters(config, raw)
    split_evidence: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        value = _split_bytes(config, chapters, split)
        digest = sha256_bytes(value)
        split_config = getattr(config.corpus, split)
        if digest != split_config.expected_sha256:
            raise ValueError(
                f"prepared {split} SHA-256 mismatch: expected "
                f"{split_config.expected_sha256}, got {digest}"
            )
        path = output / f"{split}.txt"
        path.write_bytes(value)
        split_evidence[split] = {
            "path": str(path),
            "sha256": digest,
            "bytes": len(value),
            "chapters": list(split_config.chapters),
            "documents": len(split_config.chapters),
        }
    manifest = {
        "schema_version": 1,
        "corpus": {
            "name": config.corpus.name,
            "title": config.corpus.title,
            "author": config.corpus.author,
            "source_page": config.corpus.source_page,
            "source_url": config.corpus.source_url,
            "license": config.corpus.license,
            "raw_sha256": config.corpus.raw_sha256,
            "raw_bytes": len(raw),
            "expected_chapters": config.corpus.expected_chapters,
        },
        "splits": split_evidence,
        "test_policy": (
            "test bytes are prepared and hashed here but are not read by training or "
            "validation; final evaluation requires explicit acknowledgement"
        ),
    }
    config.manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_phase4_manifest(
    config: Phase4ExperimentConfig,
    *,
    verify_test_file: bool = False,
) -> dict[str, Any]:
    if not config.manifest_path.is_file():
        raise FileNotFoundError(
            f"prepared corpus manifest does not exist: {config.manifest_path}; "
            "run phase4-prepare first"
        )
    manifest = json.loads(config.manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported prepared corpus manifest schema")
    if manifest.get("corpus", {}).get("raw_sha256") != config.corpus.raw_sha256:
        raise ValueError("prepared corpus manifest does not match the configured source")
    splits_to_verify = ("train", "validation", "test") if verify_test_file else (
        "train",
        "validation",
    )
    for split in splits_to_verify:
        evidence = manifest["splits"][split]
        path = Path(evidence["path"])
        actual = sha256_file(path)
        expected = getattr(config.corpus, split).expected_sha256
        if actual != expected or actual != evidence["sha256"]:
            raise ValueError(f"prepared {split} corpus hash verification failed")
    return manifest


def load_phase4_dataset(
    config: Phase4ExperimentConfig,
    split: CorpusSplit,
    *,
    allow_test: bool = False,
) -> PackedByteDataset:
    if split == "test" and not allow_test:
        raise PermissionError("test split is sealed outside final evaluation")
    manifest = load_phase4_manifest(config, verify_test_file=split == "test")
    path = Path(manifest["splits"][split]["path"])
    text = path.read_text(encoding="utf-8")
    documents = tuple(
        document for document in text.split(config.corpus.document_separator) if document
    )
    seed_offsets = {"train": 0, "validation": 10_000, "test": 20_000}
    return PackedByteDataset.from_text_documents(
        documents,
        sequence_length=config.training.sequence_length,
        batch_size=config.training.batch_size,
        seed=config.training.seed + seed_offsets[split],
    )
