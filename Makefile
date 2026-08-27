.PHONY: setup doctor test test-python test-java lint java-run train-smoke phase2-smoke benchmark-shapes benchmark-generation thermal-soak-short stage3-inspect stage3-train stage3-evaluate stage3-generate stage3-smoke phase4-prepare phase4-inspect phase4-train phase4-smoke phase4-final-test clean-generated

setup:
	uv sync --frozen --extra apple --extra dev

doctor:
	uv run blt-phase2 doctor

test-python:
	uv run pytest tests/python

test-java:
	./gradlew test

test: test-python test-java

lint:
	uv run ruff check python tests/python

java-run:
	./gradlew run

train-smoke:
	uv run blt-phase2 train-smoke \
		--steps 80 \
		--checkpoint artifacts/phase2_smoke/checkpoint

benchmark-shapes:
	uv run blt-phase2 benchmark-shapes \
		--samples 20 \
		--output artifacts/phase2_benchmarks/latest.json

benchmark-generation: train-smoke
	uv run blt-phase2 benchmark-generation \
		--checkpoint artifacts/phase2_smoke/checkpoint \
		--samples 20 \
		--prompt "Byte " \
		--max-new-bytes 32 \
		--output artifacts/phase2_benchmarks/generation.json

thermal-soak-short: train-smoke
	uv run blt-phase2 thermal-soak \
		--checkpoint artifacts/phase2_smoke/checkpoint \
		--prompt "Byte " \
		--seconds 10 \
		--window-seconds 2 \
		--output artifacts/phase2_benchmarks/thermal_soak_short.json

phase2-smoke: doctor test train-smoke benchmark-shapes benchmark-generation

stage3-inspect:
	uv run blt-lab inspect-baseline \
		--config configs/stage3_byte_gpt_tiny.json \
		--output artifacts/stage3/inspection.json

stage3-train:
	uv run blt-lab train-baseline \
		--config configs/stage3_byte_gpt_tiny.json \
		--checkpoint artifacts/stage3/checkpoint \
		--output artifacts/stage3/training_report.json

stage3-evaluate: stage3-train
	uv run blt-lab evaluate-checkpoint \
		--config configs/stage3_byte_gpt_tiny.json \
		--checkpoint artifacts/stage3/checkpoint \
		--output artifacts/stage3/evaluation.json

stage3-generate: stage3-train
	uv run blt-lab generate \
		--checkpoint artifacts/stage3/checkpoint \
		--prompt "Byte " \
		--max-new-bytes 32 \
		--output artifacts/stage3/generation.json

stage3-smoke: doctor test stage3-inspect stage3-train stage3-evaluate stage3-generate

phase4-prepare:
	uv run blt-lab phase4-prepare \
		--config configs/phase4_alice_byte_gpt.json \
		--output artifacts/phase4/preparation_report.json

phase4-inspect: phase4-prepare
	uv run blt-lab phase4-inspect \
		--config configs/phase4_alice_byte_gpt.json \
		--output artifacts/phase4/inspection.json

phase4-train: phase4-prepare
	uv run blt-lab phase4-train \
		--config configs/phase4_alice_byte_gpt.json \
		--run-directory artifacts/phase4/run

phase4-smoke: doctor test phase4-inspect
	uv run blt-lab phase4-train \
		--config configs/phase4_alice_byte_gpt.json \
		--run-directory artifacts/phase4/smoke-run \
		--max-steps-this-run 50

phase4-final-test:
	uv run blt-lab phase4-final-test \
		--config configs/phase4_alice_byte_gpt.json \
		--selection artifacts/phase4/run/selection.json \
		--acknowledgement I_UNDERSTAND_THIS_CONSUMES_THE_FINAL_TEST_SET \
		--output artifacts/phase4/final_test.json

clean-generated:
	./gradlew clean
