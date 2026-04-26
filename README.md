# posttraining

Post-training workspace for open source models using
[Tinker](https://github.com/thinking-machines-lab/tinker-cookbook).

## Setup

```bash
uv sync
cp .env.example .env
```

Add your Tinker API key to `.env`:

```bash
TINKER_API_KEY=...
```

Then verify the local environment:

```bash
uv run posttraining check
```

## Layout

```text
configs/              Experiment configs and templates
data/raw/             Local raw datasets, ignored by git
data/processed/       Prepared training/eval JSONL files, ignored by git
outputs/              Training outputs, ignored by git
checkpoints/          Local checkpoint artifacts, ignored by git
src/posttraining/     Project utilities
tests/                Lightweight setup tests
```

Create the ignored local directories with:

```bash
uv run posttraining init-dirs
```

## Development

```bash
uv run ruff check .
uv run pyright
uv run pytest
```

## Notes

- `tinker-cookbook` is installed as the primary dependency and brings in the
  `tinker` SDK.
- Keep secrets in `.env`; the file is intentionally ignored by git.
- `configs/sft.example.yaml` is a starting point for supervised fine-tuning
  experiments, not a finished recipe.

## RLVR Track

The staged RLVR plan is in [docs/rlvr-roadmap.md](docs/rlvr-roadmap.md).
Stage 1 starts with Countdown arithmetic:

```bash
uv run python -m posttraining.rlvr.countdown_train
```

The first-run defaults use `stzhao/tinyzero-countdown-data`, 3-number examples,
`POSTTRAINING_DEFAULT_MODEL`, and logs under `outputs/rlvr_countdown_3num`.
