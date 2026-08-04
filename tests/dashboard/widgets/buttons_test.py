# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for tool button creation helpers."""

from ess.livedata.dashboard.widgets.buttons import (
    ButtonStyles,
    create_download_button,
    create_tool_button,
)


def test_tool_button_carries_stable_css_classes() -> None:
    # These classes are a committed convention for automation/tests to target
    # label-less icon buttons. Do not drop them without updating consumers.
    button = create_tool_button(
        icon_name='settings',
        button_color=ButtonStyles.PRIMARY_BLUE,
        hover_color=ButtonStyles.PRIMARY_HOVER,
        on_click_callback=lambda: None,
    )
    assert 'lt-tool' in button.css_classes
    assert 'lt-tool-settings' in button.css_classes


def test_tool_button_appends_caller_css_classes() -> None:
    button = create_tool_button(
        icon_name='x',
        button_color=ButtonStyles.DANGER_RED,
        hover_color=ButtonStyles.DANGER_HOVER,
        on_click_callback=lambda: None,
        css_classes=['lt-wf-monitor_histogram'],
    )
    assert 'lt-tool' in button.css_classes
    assert 'lt-tool-x' in button.css_classes
    assert 'lt-wf-monitor_histogram' in button.css_classes


def test_download_button_carries_stable_css_classes() -> None:
    # A FileDownload renders as a label-less icon like any other tool button,
    # so it follows the same hook convention.
    button = create_download_button(
        filename='grid.yaml',
        callback=lambda: '',
        css_classes=['lt-grid-detectors'],
    )
    assert 'lt-tool' in button.css_classes
    assert 'lt-tool-download' in button.css_classes
    assert 'lt-grid-detectors' in button.css_classes
