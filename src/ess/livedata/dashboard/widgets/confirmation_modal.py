# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Generic yes/no confirmation modal.

A minimal :class:`panel.Modal` wrapper presenting a title, a message, and a
Cancel/Confirm footer. ``on_confirm`` runs only when the user confirms; closing
via Cancel, the X button, or Escape (handled globally by
:class:`~ess.livedata.dashboard.widgets.modal_escape_closer.ModalEscapeCloser`)
does nothing. Mirrors the modal pattern of
:class:`~ess.livedata.dashboard.widgets.cell_properties_modal.CellPropertiesModal`.
"""

from __future__ import annotations

from collections.abc import Callable

import panel as pn

from .styles import Colors, WarningBox


class ConfirmationModal:
    """Modal asking the user to confirm or cancel a disruptive action.

    Parameters
    ----------
    title:
        Heading shown at the top of the modal.
    message:
        HTML body explaining the consequences of confirming.
    on_confirm:
        Invoked once when the user clicks Confirm. Not called on cancel/close.
    confirm_label:
        Label for the confirm button.
    confirm_css_classes:
        Extra CSS classes for the confirm button, for automation hooks.
    on_close:
        Optional callback invoked after the modal closes (confirm or cancel),
        e.g. to tear down the shared modal container.
    """

    def __init__(
        self,
        *,
        title: str,
        message: str,
        on_confirm: Callable[[], None],
        confirm_label: str = 'Confirm',
        confirm_css_classes: list[str] | None = None,
        on_close: Callable[[], None] | None = None,
    ) -> None:
        self._on_confirm = on_confirm
        self._on_close = on_close
        self._done = False

        confirm_btn = pn.widgets.Button(
            label=confirm_label,
            color='danger',
            css_classes=confirm_css_classes or [],
        )
        confirm_btn.on_click(lambda _: self._finish(confirm=True))
        cancel_btn = pn.widgets.Button(
            label='Cancel', color='light', css_classes=['lt-confirm-cancel']
        )
        cancel_btn.on_click(lambda _: self._finish(confirm=False))

        title_html = pn.pane.HTML(
            f'<div style="font-size: 16px; font-weight: 700; '
            f'color: {Colors.TEXT_DARK}; margin-bottom: 4px;">{title}</div>'
        )
        warning_html = pn.pane.HTML(
            f'<div style="background: {WarningBox.BG}; '
            f'border-left: 4px solid {WarningBox.BORDER}; '
            f'color: {WarningBox.TEXT}; padding: 10px 12px; border-radius: 4px; '
            f'font-size: 13px; line-height: 1.45;">{message}</div>',
            sizing_mode='stretch_width',
        )

        self.modal = pn.Modal(
            pn.Column(
                title_html,
                warning_html,
                pn.Row(
                    pn.Spacer(sizing_mode='stretch_width'),
                    cancel_btn,
                    confirm_btn,
                    margin=(10, 0, 0, 0),
                ),
                sizing_mode='stretch_width',
            ),
            name=title,
            margin=20,
            width=460,
        )
        # Closing via the X button or Escape counts as cancel.
        self.modal.param.watch(
            lambda event: None if event.new else self._finish(confirm=False), 'open'
        )

    def show(self) -> None:
        """Open the modal."""
        self.modal.open = True

    def _finish(self, *, confirm: bool) -> None:
        if self._done:
            return
        self._done = True
        self.modal.open = False
        if confirm:
            self._on_confirm()
        if self._on_close is not None:
            self._on_close()
