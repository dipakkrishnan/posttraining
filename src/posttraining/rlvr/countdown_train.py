from __future__ import annotations

import asyncio
import os
import sys
from typing import Literal

import chz
from tinker_cookbook import cli_utils
from tinker_cookbook.rl import train

from posttraining.config import load_settings
from posttraining.rlvr.countdown import CountdownDatasetBuilder, default_model_and_renderer


def build_config_blueprint() -> chz.Blueprint[train.Config]:
    settings = load_settings()
    model_name, renderer_name = default_model_and_renderer(settings.default_model)
    builder = CountdownDatasetBuilder(
        batch_size=_int_env("POSTTRAINING_COUNTDOWN_BATCH_SIZE", 16),
        group_size=_int_env("POSTTRAINING_COUNTDOWN_GROUP_SIZE", 8),
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        max_train_examples=_optional_int_env("POSTTRAINING_COUNTDOWN_MAX_TRAIN_EXAMPLES", 4096),
        max_eval_examples=_optional_int_env("POSTTRAINING_COUNTDOWN_MAX_EVAL_EXAMPLES", 256),
        num_count=_num_count_env("POSTTRAINING_COUNTDOWN_NUM_COUNT", 3),
    )

    return chz.Blueprint(train.Config).apply(
        {
            "model_name": model_name,
            "renderer_name": renderer_name,
            "log_path": str(settings.output_dir / "rlvr_countdown_3num"),
            "dataset_builder": builder,
            "learning_rate": 4e-5,
            "max_tokens": 256,
            "eval_every": 10,
            "save_every": 20,
            "max_steps": 50,
            "num_groups_to_log": 4,
        }
    )


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def _optional_int_env(name: str, default: int | None) -> int | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    if value.strip().lower() in {"none", "null"}:
        return None
    return int(value)


def _num_count_env(name: str, default: Literal[3, 4] | None) -> Literal[3, 4] | None:
    value = _optional_int_env(name, default)
    if value in (3, 4, None):
        return value
    raise ValueError(f"{name} must be 3, 4, none, or unset")


def main(config: train.Config) -> None:
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    blueprint = build_config_blueprint()
    blueprint.make_from_argv(sys.argv[1:])
    main(blueprint.make())
