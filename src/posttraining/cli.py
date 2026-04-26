from __future__ import annotations

import importlib.metadata
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from posttraining.config import load_settings
from posttraining.rlvr.countdown_cli import app as countdown_app

app = typer.Typer(no_args_is_help=True)
app.add_typer(countdown_app, name="countdown")
console = Console()


@app.command()
def check() -> None:
    """Check the local Tinker post-training environment."""
    settings = load_settings()

    table = Table(title="Posttraining Environment")
    table.add_column("Item", style="bold")
    table.add_column("Value")

    table.add_row("Tinker SDK", _package_version("tinker"))
    table.add_row("Tinker Cookbook", _package_version("tinker-cookbook"))
    table.add_row("API key configured", "yes" if settings.has_tinker_api_key else "no")
    table.add_row("Tinker base URL", settings.tinker_base_url or "default service")
    table.add_row("Default model", settings.default_model)
    table.add_row("Output directory", str(settings.output_dir))

    console.print(table)


@app.command("init-dirs")
def init_dirs() -> None:
    """Create local data and experiment output directories."""
    paths = [
        Path("configs"),
        Path("data/raw"),
        Path("data/processed"),
        Path("outputs"),
        Path("checkpoints"),
    ]
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
        console.print(f"created {path}")


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"
