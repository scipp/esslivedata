# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Shared color palette and style constants for dashboard widgets.

All dashboard widgets should import colors from here rather than
hard-coding hex values. This keeps the palette consistent and makes
future theming changes a single-file edit.
"""


class StatusColors:
    """Semantic color constants for status badges and indicators."""

    ERROR = "#dc3545"  # Red
    SUCCESS = "#28a745"  # Green
    WARNING = "#ffc107"  # Yellow
    INFO = "#6c757d"  # Gray
    PENDING = "#17a2b8"  # Blue
    MUTED = "#6c757d"  # Gray (alias for default/stopped)
    PRIMARY = "#007bff"  # Blue (actions, "you" indicator)


class HoverColors:
    """Translucent hover backgrounds derived from StatusColors."""

    ERROR = "rgba(220, 53, 69, 0.1)"
    SUCCESS = "rgba(40, 167, 69, 0.1)"
    PRIMARY = "rgba(0, 123, 255, 0.1)"
    MUTED = "rgba(108, 117, 125, 0.1)"


class Colors:
    """Neutral palette for borders, backgrounds, and text."""

    BORDER = "#dee2e6"
    BG_LIGHT = "#f8f9fa"
    BG_MUTED = "#e9ecef"
    TEXT_DARK = "#212529"
    TEXT = "#495057"
    TEXT_MUTED = "#6c757d"
    TAB_BORDER = "#2c5aa0"
    TAB_ACTIVE_BG = "#e8f4f8"


class ErrorBox:
    """Colors for error alert boxes (Bootstrap-style danger alert)."""

    BG = "#f8d7da"
    BORDER = "#f5c6cb"
    TEXT = "#721c24"


class WarningBox:
    """Colors for warning alert boxes."""

    BG = "#fff3cd"
    BORDER = "#ffc107"
    TEXT = "#856404"
