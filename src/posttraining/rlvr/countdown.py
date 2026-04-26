from __future__ import annotations

import ast
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import partial
from typing import Any, Literal, cast

import chz
from datasets import Dataset, load_dataset
from tinker_cookbook import model_info, renderers
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer

ANSWER_RE = re.compile(r"<answer>\s*(?P<answer>.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class CountdownVerification:
    """Structured result for a Countdown answer check."""

    answer_tag: bool
    parse: bool
    numbers: bool
    correct: bool
    expression: str | None = None
    value: Fraction | None = None
    error: str | None = None


def extract_answer_expression(sample: str) -> tuple[str | None, bool]:
    """Return the last `<answer>...</answer>` expression, if present."""

    matches = list(ANSWER_RE.finditer(sample))
    if not matches:
        return None, False
    expression = matches[-1].group("answer").strip()
    if expression == "":
        return None, True
    return expression, True


def verify_countdown(sample: str, nums: Sequence[int], target: int) -> CountdownVerification:
    """Verify a Countdown response exactly.

    A valid response must put an arithmetic expression in `<answer>` tags, use
    exactly the provided numbers with the same multiplicities, and evaluate
    exactly to `target` using `+`, `-`, `*`, `/`, and parentheses.
    """

    expression, has_answer_tag = extract_answer_expression(sample)
    if expression is None:
        return CountdownVerification(
            answer_tag=has_answer_tag,
            parse=False,
            numbers=False,
            correct=False,
            error="missing answer expression",
        )

    try:
        value, used_numbers = _evaluate_expression(expression)
    except CountdownExpressionError as exc:
        return CountdownVerification(
            answer_tag=has_answer_tag,
            parse=False,
            numbers=False,
            correct=False,
            expression=expression,
            error=str(exc),
        )

    number_match = Counter(used_numbers) == Counter(nums)
    correct = number_match and value == Fraction(target, 1)
    return CountdownVerification(
        answer_tag=has_answer_tag,
        parse=True,
        numbers=number_match,
        correct=correct,
        expression=expression,
        value=value,
    )


class CountdownExpressionError(ValueError):
    """Expression could not be parsed or safely evaluated."""


def _evaluate_expression(expression: str) -> tuple[Fraction, list[int]]:
    normalized = expression.replace("×", "*").replace("÷", "/")
    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise CountdownExpressionError("invalid expression syntax") from exc
    return _eval_ast(tree.body)


def _eval_ast(node: ast.AST) -> tuple[Fraction, list[int]]:
    if isinstance(node, ast.BinOp):
        left_value, left_nums = _eval_ast(node.left)
        right_value, right_nums = _eval_ast(node.right)
        if isinstance(node.op, ast.Add):
            value = left_value + right_value
        elif isinstance(node.op, ast.Sub):
            value = left_value - right_value
        elif isinstance(node.op, ast.Mult):
            value = left_value * right_value
        elif isinstance(node.op, ast.Div):
            if right_value == 0:
                raise CountdownExpressionError("division by zero")
            value = left_value / right_value
        else:
            raise CountdownExpressionError("unsupported operator")
        return value, left_nums + right_nums

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int):
            raise CountdownExpressionError("only integer constants are allowed")
        return Fraction(node.value, 1), [node.value]

    raise CountdownExpressionError(f"unsupported expression node: {type(node).__name__}")


class CountdownEnv(ProblemEnv):
    """Single-turn Countdown arithmetic environment."""

    def __init__(
        self,
        nums: Sequence[int],
        target: int,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        format_coef: float = 0.1,
    ):
        super().__init__(renderer, convo_prefix=convo_prefix, format_coef=format_coef)
        self.nums = tuple(int(num) for num in nums)
        self.target = int(target)

    def get_question(self) -> str:
        numbers = ", ".join(str(num) for num in self.nums)
        return (
            f"Using the numbers [{numbers}], create an equation that equals {self.target}. "
            "Use each provided number exactly once. You may use +, -, *, /, and parentheses. "
            "Show your work in <think> </think> tags and put only the final expression in "
            "<answer> </answer> tags."
        )

    def check_answer(self, sample_str: str) -> bool:
        return verify_countdown(sample_str, self.nums, self.target).correct

    def check_format(self, sample_str: str) -> bool:
        result = verify_countdown(sample_str, self.nums, self.target)
        return result.answer_tag and result.parse

    def get_reference_answer(self) -> str:
        return f"target={self.target}; nums={list(self.nums)}"

    @staticmethod
    def standard_fewshot_prefix() -> list[renderers.Message]:
        return [
            {
                "role": "user",
                "content": (
                    "Using the numbers [2, 3, 4], create an equation that equals 10. "
                    "Use each provided number exactly once. You may use +, -, *, /, and "
                    "parentheses. Show your work in <think> </think> tags and put only "
                    "the final expression in <answer> </answer> tags."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "<think>3 * 4 = 12, and 12 - 2 = 10.</think>"
                    "<answer>(3 * 4) - 2</answer>"
                ),
            },
        ]


class CountdownDataset(RLDataset):
    def __init__(
        self,
        *,
        ds: Dataset,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        format_coef: float = 0.1,
    ):
        self.ds = ds
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.convo_prefix = convo_prefix
        self.format_coef = format_coef

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.ds))
        assert batch_start < batch_end, "batch index out of range"
        return [
            self._make_env_group_builder(cast(dict[str, Any], row), self.group_size)
            for row in self.ds.select(range(batch_start, batch_end))
        ]

    def __len__(self) -> int:
        return math.ceil(len(self.ds) / self.batch_size)

    def _make_env_group_builder(self, row: dict[str, Any], group_size: int) -> ProblemGroupBuilder:
        nums = [int(num) for num in row["nums"]]
        target = int(row["target"])
        return ProblemGroupBuilder(
            env_thunk=partial(
                CountdownEnv,
                nums=nums,
                target=target,
                renderer=self.renderer,
                convo_prefix=self.convo_prefix,
                format_coef=self.format_coef,
            ),
            num_envs=group_size,
            dataset_name=f"countdown_{len(nums)}num",
        )


@chz.chz
class CountdownDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    dataset_name: str = "stzhao/tinyzero-countdown-data"
    split: str = "train"
    max_train_examples: int | None = 4096
    max_eval_examples: int | None = 512
    eval_fraction: float = 0.05
    seed: int = 0
    num_count: Literal[3, 4] | None = 3
    convo_prefix: list[renderers.Message] | None | Literal["standard"] = "standard"
    format_coef: float = 0.1

    async def __call__(self) -> tuple[CountdownDataset, CountdownDataset]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        convo_prefix = (
            CountdownEnv.standard_fewshot_prefix()
            if self.convo_prefix == "standard"
            else self.convo_prefix
        )

        ds = cast(Dataset, load_dataset(self.dataset_name, split=self.split))
        if self.num_count is not None:
            ds = ds.filter(lambda row: len(row["nums"]) == self.num_count)
        ds = ds.shuffle(seed=self.seed)

        eval_size = max(1, int(len(ds) * self.eval_fraction))
        if self.max_eval_examples is not None:
            eval_size = min(eval_size, self.max_eval_examples)
        train_start = eval_size
        train_end = len(ds)
        if self.max_train_examples is not None:
            train_end = min(train_end, train_start + self.max_train_examples)

        eval_ds = ds.select(range(0, eval_size))
        train_ds = ds.select(range(train_start, train_end))
        return (
            CountdownDataset(
                ds=train_ds,
                batch_size=self.batch_size,
                group_size=self.group_size,
                renderer=renderer,
                convo_prefix=convo_prefix,
                format_coef=self.format_coef,
            ),
            CountdownDataset(
                ds=eval_ds,
                batch_size=self.batch_size,
                group_size=1,
                renderer=renderer,
                convo_prefix=convo_prefix,
                format_coef=self.format_coef,
            ),
        )


def default_model_and_renderer(model_name: str) -> tuple[str, str]:
    return model_name, model_info.get_recommended_renderer_name(model_name)
