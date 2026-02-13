# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Structured logging configuration for ESSlivedata."""

import logging
import sys
from logging.handlers import RotatingFileHandler

import structlog


def configure_logging(
    *,
    level: int = logging.INFO,
    json_file: str | None = None,
    disable_stdout: bool = False,
) -> None:
    """
    Configure structured logging for the application.

    Parameters
    ----------
    level:
        The minimum log level to output.
    json_file:
        Path to write JSON-formatted logs to. If provided, creates a rotating
        file handler (10MB max, 5 backups).
    disable_stdout:
        If True, disable logging to stdout.
    """
    # Shared processors for both structlog and stdlib logging
    shared_processors: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Add stdout handler with pretty console output (unless disabled)
    if not disable_stdout:
        console_renderer = structlog.dev.ConsoleRenderer(colors=True)
        console_formatter = structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared_processors,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                console_renderer,
            ],
        )
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

    # Add file handler with JSON output (if path provided)
    if json_file is not None:
        json_renderer = structlog.processors.JSONRenderer()
        json_formatter = structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared_processors,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.format_exc_info,
                json_renderer,
            ],
        )
        file_handler = RotatingFileHandler(
            json_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
        )
        file_handler.setFormatter(json_formatter)
        root_logger.addHandler(file_handler)

    root_logger.setLevel(level)
