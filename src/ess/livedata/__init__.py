# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
# ruff: noqa: E402, I

import importlib.metadata

try:
    __version__ = importlib.metadata.version(__package__ or __name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

del importlib

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
]
