from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from posttraining.rlvr.countdown import (
    CountdownExpressionError,
    expression_ast_dump,
    extract_answer_expression,
    verify_countdown,
)

app = typer.Typer(no_args_is_help=True, help="Play with the Countdown RLVR verifier.")
console = Console()


@dataclass(frozen=True)
class Example:
    name: str
    nums: tuple[int, ...]
    target: int
    response: str
    lesson: str


EXAMPLES = [
    Example(
        name="valid",
        nums=(2, 3, 4),
        target=10,
        response="<think>3 * 4 = 12, then subtract 2.</think><answer>(3 * 4) - 2</answer>",
        lesson="A clean solution: valid tags, parseable expression, exact numbers, exact target.",
    ),
    Example(
        name="wrong-target",
        nums=(2, 3, 4),
        target=10,
        response="<think>Add everything.</think><answer>2 + 3 + 4</answer>",
        lesson="The expression is valid and uses the right numbers, but evaluates to 9.",
    ),
    Example(
        name="reused-number",
        nums=(2, 3, 4),
        target=10,
        response="<answer>4 + 4 + 2</answer>",
        lesson="The value is 10, but the model reused 4 and skipped 3.",
    ),
    Example(
        name="unsafe",
        nums=(2, 3, 4),
        target=10,
        response="<answer>__import__('os').system('echo nope')</answer>",
        lesson="Function calls are rejected by the AST whitelist.",
    ),
    Example(
        name="fractional-intermediate",
        nums=(8, 4, 6),
        target=8,
        response="<answer>8 / 4 + 6</answer>",
        lesson="Fraction arithmetic allows exact division without floating point drift.",
    ),
]


@app.command("examples")
def examples() -> None:
    """Show instructive verifier examples."""

    for example in EXAMPLES:
        console.rule(example.name)
        _print_problem(example.nums, example.target)
        console.print(Panel(example.response, title="Model response", expand=False))
        _print_result(example.response, example.nums, example.target)
        console.print(f"[bold]Lesson:[/bold] {example.lesson}")


@app.command("score")
def score(
    nums: str = typer.Option(..., help="Comma-separated numbers, e.g. 2,3,4"),
    target: int = typer.Option(..., help="Target value."),
    response: str = typer.Argument(
        ..., help="Model response or bare expression. Bare expressions are wrapped in <answer>."
    ),
) -> None:
    """Score a model response or expression against a Countdown problem."""

    parsed_nums = _parse_nums(nums)
    sample = _ensure_answer_tags(response)
    _print_problem(parsed_nums, target)
    console.print(Panel(sample, title="Response checked", expand=False))
    _print_result(sample, parsed_nums, target)


@app.command("ast")
def ast_view(expression: str = typer.Argument(..., help="Expression to parse, e.g. '(3*4)-2'")):
    """Show the Python AST shape used by the safe verifier."""

    try:
        console.print(expression_ast_dump(expression))
    except CountdownExpressionError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command("play")
def play() -> None:
    """Try expressions against a few local Countdown problems."""

    console.print("[bold]Countdown verifier playground[/bold]")
    console.print("Enter a bare expression. The CLI will wrap it in <answer> tags.\n")
    for example in EXAMPLES[:3]:
        _print_problem(example.nums, example.target)
        answer = typer.prompt("expression", default=example.response)
        sample = _ensure_answer_tags(answer)
        _print_result(sample, example.nums, example.target)
        console.print()


def _print_problem(nums: tuple[int, ...], target: int) -> None:
    console.print(f"[bold]Problem:[/bold] nums={list(nums)} target={target}")


def _print_result(sample: str, nums: tuple[int, ...], target: int) -> None:
    result = verify_countdown(sample, nums, target)
    table = Table(title="Verifier result")
    table.add_column("Check", style="bold")
    table.add_column("Value")
    table.add_row("answer_tag", _yes_no(result.answer_tag))
    table.add_row("parse", _yes_no(result.parse))
    table.add_row("numbers", _yes_no(result.numbers))
    table.add_row("correct", _yes_no(result.correct))
    table.add_row("expression", result.expression or "")
    table.add_row("value", str(result.value) if result.value is not None else "")
    table.add_row("error", result.error or "")
    console.print(table)

    expression, _ = extract_answer_expression(sample)
    if expression is not None:
        with suppress(CountdownExpressionError):
            console.print(Panel(expression_ast_dump(expression), title="AST", expand=False))


def _parse_nums(nums: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(part.strip()) for part in nums.split(",") if part.strip())
    except ValueError as exc:
        raise typer.BadParameter("nums must be comma-separated integers") from exc
    if not parsed:
        raise typer.BadParameter("nums must include at least one integer")
    return parsed


def _ensure_answer_tags(response: str) -> str:
    if "<answer>" in response.lower() and "</answer>" in response.lower():
        return response
    return f"<answer>{response}</answer>"


def _yes_no(value: bool) -> str:
    return "[green]yes[/green]" if value else "[red]no[/red]"
