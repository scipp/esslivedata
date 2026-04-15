# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
# ruff: noqa: E402, I

import importlib.metadata
import re

try:
    __version__ = importlib.metadata.version("esslivedata")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

del importlib

# Pattern for setuptools_scm dev versions:
#   "26.4.2.dev0+g68b165851.d20260410" → base="26.4.2", hash="68b16585"
_DEV_VERSION_RE = re.compile(
    r'^(?P<base>\d+\.\d+\.\d+)\.dev\d+\+g(?P<hash>[0-9a-f]{7,})'
)


def format_version(version: str) -> str:
    """Format a version string for display, shortening dev versions.

    Release versions pass through unchanged. Dev versions are shortened from
    e.g. ``26.4.2.dev0+g68b165851.d20260410`` to ``26.4.2-dev (68b16585)``.
    """
    m = _DEV_VERSION_RE.match(version)
    if m is None:
        return version
    return f"{m['base']}-dev ({m['hash'][:8]})"


from .core import (
    Accumulator,
    IdentityProcessor,
    JobBasedPreprocessorFactoryBase,
    Message,
    MessageSink,
    MessageSource,
    PreprocessorFactory,
    Processor,
    Service,
    ServiceBase,
    StreamId,
    StreamKind,
    compact_messages,
)

__all__ = [
    "Accumulator",
    "IdentityProcessor",
    "JobBasedPreprocessorFactoryBase",
    "Message",
    "MessageSink",
    "MessageSource",
    "PreprocessorFactory",
    "Processor",
    "Service",
    "ServiceBase",
    "StreamId",
    "StreamKind",
    "compact_messages",
    "format_version",
]
