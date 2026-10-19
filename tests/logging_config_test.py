# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import io
import logging
import sys
from collections.abc import Iterator

import pytest
import structlog

from ess.livedata.logging_config import configure_logging


class TtyStringIO(io.StringIO):
    def isatty(self) -> bool:
        return True


@pytest.fixture(autouse=True)
def restore_logging() -> Iterator[None]:
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)
    structlog.reset_defaults()


def log_to(stream: io.StringIO, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setattr(sys, 'stdout', stream)
    configure_logging()
    structlog.get_logger('test').info('hello', key='value')
    return stream.getvalue()


def test_no_escape_codes_when_stdout_is_not_a_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = log_to(io.StringIO(), monkeypatch)
    assert 'hello' in output
    assert '\x1b[' not in output


def test_colors_when_stdout_is_a_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    output = log_to(TtyStringIO(), monkeypatch)
    assert 'hello' in output
    assert '\x1b[' in output
