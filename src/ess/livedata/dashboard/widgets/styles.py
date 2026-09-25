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


class StatusPill:
    """Color bands for the cell status pill and per-layer status badges.

    ``FRESH``/``STALE``/``OLD`` band live data age. ``STOPPED`` marks a
    deliberately stopped job whose frozen snapshot is still valid — neutral
    gray, no alarm. ``ERROR`` flags a failed layer and shares the alarm red
    with ``OLD``. Each band is ``(background, text, dot)``.
    """

    FRESH = ("rgba(40, 167, 69, 0.16)", "#1e7e34", StatusColors.SUCCESS)
    STALE = ("rgba(255, 193, 7, 0.18)", "#946c00", StatusColors.WARNING)
    OLD = ("rgba(220, 53, 69, 0.16)", "#b21f2d", StatusColors.ERROR)
    STOPPED = ("rgba(108, 117, 125, 0.16)", Colors.TEXT, StatusColors.MUTED)
    ERROR = OLD


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


class ModalSizing:
    """Shared sizing constants for modal dialogs with sticky header/footer."""

    WIDTH = 800
    SCROLL_BODY_MAX_HEIGHT = 650
    # Keeps a footer's buttons in view when the whole dialog has to scroll,
    # which it does on screens shorter than the dialog (see ``design.py``).
    STICKY_FOOTER_STYLES = {  # noqa: RUF012
        'position': 'sticky',
        'bottom': '0',
        'z-index': '1',
        'background-color': 'white',
    }


class PhoneLayout:
    """Spacing of the phone layout (``?layout=phone``), in pixels.

    The open plot's height is derived from these, so a change here keeps the
    plot fitting the screen.
    """

    # Band in the header's color above the content (``dashboard.py``).
    TOP_BAND = 8
    # Padding around the tab content (``plot_grid_tabs.py``).
    TAB_CONTENT_PADDING = 4
    # Edge of an icon-only tab: the icon plus its padding (``plot_grid_tabs.py``).
    TAB_ICON = 24
    TAB_PADDING = 12
    TAB_SIZE = TAB_ICON + 2 * TAB_PADDING
