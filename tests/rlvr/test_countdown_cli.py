from typer.testing import CliRunner

from posttraining.cli import app

runner = CliRunner()


def test_countdown_score_accepts_bare_expression() -> None:
    result = runner.invoke(
        app,
        ["countdown", "score", "--nums", "2,3,4", "--target", "10", "(3 * 4) - 2"],
    )

    assert result.exit_code == 0
    assert "correct" in result.output
    assert "yes" in result.output


def test_countdown_ast_shows_binop() -> None:
    result = runner.invoke(app, ["countdown", "ast", "(3 * 4) - 2"])

    assert result.exit_code == 0
    assert "BinOp" in result.output
