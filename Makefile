.PHONY: setup doctor test test-python test-java lint java-run train-smoke phase2-smoke benchmark-shapes benchmark-generation thermal-soak-short clean-generated

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

clean-generated:
	./gradlew clean
