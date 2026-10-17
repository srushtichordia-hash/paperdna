"""Smoke tests: imports, version, configuration, and logging."""

import logging

import paperdna
from paperdna.config import load_settings
from paperdna.logging import setup_logging


def test_import():
    assert paperdna.__name__ == "paperdna"


def test_version_is_string():
    assert isinstance(paperdna.__version__, str)
    assert paperdna.__version__


def test_default_settings():
    settings = load_settings()
    assert settings.log_level == "INFO"


def test_log_level_from_env(monkeypatch):
    monkeypatch.setenv("PAPERDNA_LOG_LEVEL", "debug")
    assert load_settings().log_level == "DEBUG"


def test_invalid_log_level_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("PAPERDNA_LOG_LEVEL", "not-a-level")
    assert load_settings().log_level == "INFO"


def test_setup_logging_returns_logger():
    logger = setup_logging("WARNING")
    assert logger.name == "paperdna"
    assert logger.level == logging.WARNING
