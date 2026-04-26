from fractions import Fraction

from posttraining.rlvr.countdown import extract_answer_expression, verify_countdown


def test_extracts_last_answer_tag() -> None:
    sample = "<answer>1 + 1</answer> text <answer>(3 * 4) - 2</answer>"

    expression, has_tag = extract_answer_expression(sample)

    assert has_tag is True
    assert expression == "(3 * 4) - 2"


def test_verifies_correct_expression() -> None:
    result = verify_countdown(
        "<think>work</think><answer>(3 * 4) - 2</answer>",
        nums=[2, 3, 4],
        target=10,
    )

    assert result.answer_tag is True
    assert result.parse is True
    assert result.numbers is True
    assert result.correct is True
    assert result.value == Fraction(10, 1)


def test_rejects_number_reuse() -> None:
    result = verify_countdown("<answer>4 + 4 + 2</answer>", nums=[2, 3, 4], target=10)

    assert result.parse is True
    assert result.numbers is False
    assert result.correct is False


def test_rejects_wrong_target() -> None:
    result = verify_countdown("<answer>2 + 3 + 4</answer>", nums=[2, 3, 4], target=10)

    assert result.parse is True
    assert result.numbers is True
    assert result.correct is False


def test_rejects_unsafe_expression() -> None:
    result = verify_countdown(
        "<answer>__import__('os').system('echo nope')</answer>",
        nums=[2, 3, 4],
        target=10,
    )

    assert result.parse is False
    assert result.correct is False


def test_supports_fractional_intermediate_values() -> None:
    result = verify_countdown("<answer>8 / 4 + 6</answer>", nums=[8, 4, 6], target=8)

    assert result.correct is True
