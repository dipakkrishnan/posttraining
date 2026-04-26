SHELL := /bin/bash

.PHONY: help check test lint typecheck verify \
	countdown-smoke countdown-smoke-llama countdown-smoke-qwen35 \
	notebook-kernel dashboard

MODEL ?= meta-llama/Llama-3.2-1B
RUN_NAME ?= rlvr_countdown_smoke
BATCH_SIZE ?= 8
GROUP_SIZE ?= 8
MAX_STEPS ?= 3
MAX_TOKENS ?= 256
LORA_RANK ?= 8
EVAL_EVERY ?= 0
SAVE_EVERY ?= 0
LOG_PATH ?= outputs/$(RUN_NAME)

help:
	@echo "Targets:"
	@echo "  make check"
	@echo "  make verify"
	@echo "  make countdown-smoke MODEL=... RUN_NAME=... MAX_TOKENS=..."
	@echo "  make countdown-smoke-llama"
	@echo "  make countdown-smoke-qwen35 MAX_TOKENS=512"
	@echo "  make notebook-kernel"
	@echo "  make dashboard"

check:
	uv run posttraining check

lint:
	uv run ruff check .

typecheck:
	uv run pyright

test:
	uv run pytest

verify: lint typecheck test

countdown-smoke:
	POSTTRAINING_DEFAULT_MODEL="$(MODEL)" \
	POSTTRAINING_COUNTDOWN_BATCH_SIZE="$(BATCH_SIZE)" \
	POSTTRAINING_COUNTDOWN_GROUP_SIZE="$(GROUP_SIZE)" \
	uv run python -m posttraining.rlvr.countdown_train \
		max_steps="$(MAX_STEPS)" \
		lora_rank="$(LORA_RANK)" \
		max_tokens="$(MAX_TOKENS)" \
		eval_every="$(EVAL_EVERY)" \
		save_every="$(SAVE_EVERY)" \
		log_path="$(LOG_PATH)"

countdown-smoke-llama:
	$(MAKE) countdown-smoke \
		MODEL=meta-llama/Llama-3.2-1B \
		RUN_NAME=rlvr_countdown_llama32_1b_smoke \
		MAX_TOKENS=256

countdown-smoke-qwen35:
	$(MAKE) countdown-smoke \
		MODEL=Qwen/Qwen3.5-4B \
		RUN_NAME=rlvr_countdown_qwen35_4b_smoke_tok$(MAX_TOKENS)

notebook-kernel:
	uv run python -m ipykernel install --user --name posttraining --display-name "posttraining"

dashboard:
	@echo "Open notebooks/rlvr_run_dashboard.ipynb and set:"
	@echo "RUN_DIR = Path(\"../$(LOG_PATH)\").resolve()"
