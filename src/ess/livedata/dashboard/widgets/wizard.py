# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Generic multi-step wizard component."""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum, auto
from typing import Any, Protocol

import panel as pn


class WizardState(Enum):
    """State of the wizard workflow."""

    ACTIVE = auto()
    COMPLETED = auto()
    CANCELLED = auto()


class WizardStep(Protocol):
    """Protocol for wizard step components."""

    def render(self) -> pn.Column:
        """Render the step's UI content."""
        ...

    def is_valid(self) -> bool:
        """Whether step data allows advancement."""
        ...

    def on_enter(self) -> None:
        """Called when step becomes active."""
        ...


class Wizard:
    """
    Generic multi-step wizard component.

    The wizard manages navigation between steps and handles completion/cancellation
    callbacks. Steps receive callbacks to signal advancement and share data via a
    context object.

    Parameters
    ----------
    steps:
        List of wizard steps to display in sequence
    context:
        Shared data object (typically a dataclass) that steps read/write
    on_complete:
        Called with context when wizard completes successfully
    on_cancel:
        Called when wizard is cancelled
    """

    def __init__(
        self,
        steps: list[WizardStep],
        context: Any,
        on_complete: Callable[[Any], None],
        on_cancel: Callable[[], None],
    ) -> None:
        self._steps = steps
        self._context = context
        self._on_complete = on_complete
        self._on_cancel = on_cancel

        # State tracking
        self._current_step_index = 0
        self._state = WizardState.ACTIVE

        # Navigation buttons
        self._back_button = pn.widgets.Button(
            name="Back",
            button_type="light",
            sizing_mode='fixed',
            width=100,
        )
        self._back_button.on_click(self._on_back_clicked)

        self._next_button = pn.widgets.Button(
            name="Next",
            button_type="primary",
            sizing_mode='fixed',
            width=120,
        )
        self._next_button.on_click(self._on_next_clicked)

        self._cancel_button = pn.widgets.Button(
            name="Cancel",
            button_type="light",
            sizing_mode='fixed',
            width=100,
        )
        self._cancel_button.on_click(self._on_cancel_clicked)

        # Content container
        self._content = pn.Column(sizing_mode='stretch_width')

    def advance(self) -> None:
        """Move to next step if current step is valid."""
        if not self._current_step.is_valid():
            return

        if self._current_step_index < len(self._steps) - 1:
            self._current_step_index += 1
            self._update_content()
        else:
            # Already on last step, advancement means completion
            self.complete()

    def back(self) -> None:
        """Go to previous step."""
        if self._current_step_index > 0:
            self._current_step_index -= 1
            self._update_content()

    def complete(self) -> None:
        """Complete wizard successfully."""
        self._state = WizardState.COMPLETED
        self._on_complete(self._context)

    def cancel(self) -> None:
        """Cancel wizard."""
        self._state = WizardState.CANCELLED
        self._on_cancel()

    def reset(self) -> None:
        """Reset wizard to first step."""
        self._current_step_index = 0
        self._state = WizardState.ACTIVE
        self._update_content()

    def render(self) -> pn.Column:
        """Render the wizard content."""
        return self._content

    @property
    def context(self) -> Any:
        """Get the shared context object."""
        return self._context

    @property
    def _current_step(self) -> WizardStep:
        """Get the current step."""
        return self._steps[self._current_step_index]

    @property
    def _is_first_step(self) -> bool:
        """Check if on first step."""
        return self._current_step_index == 0

    @property
    def _is_last_step(self) -> bool:
        """Check if on last step."""
        return self._current_step_index == len(self._steps) - 1

    def refresh_ui(self) -> None:
        """
        Refresh the UI to reflect current state.

        Call this when step validity changes (e.g., after user selection).
        """
        self._next_button.disabled = not self._current_step.is_valid()

    def _update_content(self) -> None:
        """Update modal content for current step."""
        self._current_step.on_enter()
        self._render_step()

    def _render_step(self) -> None:
        """Render current step with navigation buttons."""
        self._content.clear()

        # Add step content
        self._content.append(self._current_step.render())

        # Update next button state
        self._next_button.disabled = not self._current_step.is_valid()

        # Build navigation row
        nav_buttons = [pn.Spacer()]

        if not self._is_first_step:
            nav_buttons.append(self._back_button)

        nav_buttons.append(self._cancel_button)

        if not self._is_last_step:
            nav_buttons.append(self._next_button)

        self._content.append(pn.Row(*nav_buttons, margin=(10, 0)))

    def _on_next_clicked(self, event) -> None:
        """Handle next button click."""
        self.advance()

    def _on_back_clicked(self, event) -> None:
        """Handle back button click."""
        self.back()

    def _on_cancel_clicked(self, event) -> None:
        """Handle cancel button click."""
        self.cancel()
