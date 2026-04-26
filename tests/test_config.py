from pathlib import Path

from posttraining.config import load_settings


def test_load_settings_uses_defaults_without_env(monkeypatch) -> None:
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    monkeypatch.delenv("TINKER_BASE_URL", raising=False)
    monkeypatch.delenv("POSTTRAINING_DEFAULT_MODEL", raising=False)
    monkeypatch.delenv("POSTTRAINING_OUTPUT_DIR", raising=False)

    settings = load_settings(env_file=Path("does-not-exist.env"))

    assert settings.has_tinker_api_key is False
    assert settings.tinker_base_url is None
    assert settings.default_model == "Qwen/Qwen3-0.6B"
    assert settings.output_dir == Path("outputs")


def test_load_settings_reads_environment(monkeypatch) -> None:
    monkeypatch.setenv("TINKER_API_KEY", "tml-test")
    monkeypatch.setenv("TINKER_BASE_URL", "http://localhost:8000")
    monkeypatch.setenv("POSTTRAINING_DEFAULT_MODEL", "Qwen/Qwen3-8B")
    monkeypatch.setenv("POSTTRAINING_OUTPUT_DIR", "runs")

    settings = load_settings(env_file=Path("does-not-exist.env"))

    assert settings.has_tinker_api_key is True
    assert settings.tinker_base_url == "http://localhost:8000"
    assert settings.default_model == "Qwen/Qwen3-8B"
    assert settings.output_dir == Path("runs")
