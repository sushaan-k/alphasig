"""Tests for sigint._logging -- Centralised logging configuration."""

from __future__ import annotations

import logging

import structlog

from sigint._logging import configure_logging


class TestConfigureLogging:
    """Tests for configure_logging()."""

    def test_default_verbosity_sets_warning(self) -> None:
        configure_logging(verbosity=0)
        root = logging.getLogger()
        assert root.level == logging.WARNING

    def test_verbosity_one_sets_info(self) -> None:
        configure_logging(verbosity=1)
        root = logging.getLogger()
        assert root.level == logging.INFO

    def test_verbosity_two_sets_debug(self) -> None:
        configure_logging(verbosity=2)
        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_high_verbosity_still_debug(self) -> None:
        configure_logging(verbosity=5)
        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_json_mode_does_not_raise(self) -> None:
        # Should complete without error
        configure_logging(verbosity=0, json=True)
        root = logging.getLogger()
        assert root.level == logging.WARNING

    def test_console_mode_does_not_raise(self) -> None:
        configure_logging(verbosity=1, json=False)
        root = logging.getLogger()
        assert root.level == logging.INFO

    def test_noisy_loggers_silenced(self) -> None:
        configure_logging(verbosity=2)
        for noisy_name in ("httpx", "httpcore", "asyncio"):
            noisy_logger = logging.getLogger(noisy_name)
            assert noisy_logger.level == logging.WARNING

    def test_handler_replaced_on_reconfigure(self) -> None:
        configure_logging(verbosity=0)
        root = logging.getLogger()
        handler_count_1 = len(root.handlers)

        configure_logging(verbosity=1)
        handler_count_2 = len(root.handlers)

        # Should have exactly one handler each time (old cleared)
        assert handler_count_1 == 1
        assert handler_count_2 == 1

    def test_structlog_produces_logger(self) -> None:
        configure_logging(verbosity=1)
        log = structlog.get_logger()
        assert log is not None
