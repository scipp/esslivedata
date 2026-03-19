# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Shared color palette for status indicators across dashboard widgets."""


class StatusColors:
    """Semantic color constants for status badges and indicators."""

    ERROR = "#dc3545"  # Red
    SUCCESS = "#28a745"  # Green
    WARNING = "#ffc107"  # Yellow
    INFO = "#6c757d"  # Gray
    PENDING = "#17a2b8"  # Blue
    MUTED = "#6c757d"  # Gray (alias for default/stopped)
    PRIMARY = "#007bff"  # Blue (actions, "you" indicator)
