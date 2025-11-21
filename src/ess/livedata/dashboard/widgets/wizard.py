# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Generic multi-step wizard component."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Generic, TypeVar

import panel as pn

TInput = TypeVar('TInput')
TOutput = TypeVar('TOutput')


class WizardStep(ABC, Generic[TInput, TOutput]):
    """
    Base class for wizard step components.

    Each step transforms input from the previous step into output for the next step.
    The first step receives None as input.

    Type Parameters
    ----------------
    TInput:
        Type of input data from previous step (None for first step)
    TOutput:
        Type of output data to pass to next step
    """

    def __init__(self) -> None:
        self._on_ready_changed: Callable[[bool], None] | None = None
        self._step_number: int | None = None

    def on_ready_changed(self, callback: Callable[[bool], None]) -> None:
        """Register callback to be notified when ready state changes."""
        self._on_ready_changed = callback

    def _notify_ready_changed(self, is_ready: bool) -> None:
        """Notify wizard of ready state change."""
        if self._on_ready_changed:
            self._on_ready_changed(is_ready)

    @property
    @abstractmethod
    def name(self) -> str:
        """Display name for this step (e.g., 'Select Job and Output')."""

    @property
    def description(self) -> str | None:
        """Optional description text shown below the step header."""
        return None

    def render(self, step_number: int) -> pn.Column:
        """
        Render the step's UI with automatic header generation.

        Parameters
        ----------
        step_number:
            The 1-based step number to display in the header

        Returns
        -------
        :
            Column containing header and step content
        """
        self._step_number = step_number

        # Build header
        header_parts = [f"<h3>Step {step_number}: {self.name}</h3>"]
        if self.description:
            header_parts.append(f"<p>{self.description}</p>")

        return pn.Column(
            pn.pane.HTML("".join(header_parts)),
            self.render_content(),
            sizing_mode='stretch_width',
        )

    @abstractmethod
    def render_content(self) -> pn.Column | pn.viewable.Viewable:
        """Render the step's content (without header)."""

    @abstractmethod
    def is_valid(self) -> bool:
        """Whether step data allows advancement."""

    @abstractmethod
    def commit(self) -> TOutput | None:
        """
        Commit this step's data for the pipeline.

        Called when the user advances from this step. This method should package
        the step's current state into output data for the next step. For the final
        step, this may also trigger side effects (e.g., creating a plot).

        Returns
        -------
        :
            Output data to pass to next step, or None if commit failed
        """

    @abstractmethod
    def on_enter(self, input_data: TInput) -> None:
        """
        Called when step becomes active.

        Parameters
        ----------
        input_data:
            Output from the previous step (None for first step)
        """


class Wizard:
    """
    Generic multi-step wizard component.

    The wizard manages navigation between steps, threading data from each step's
    execution to the next step's input. Each step transforms input data to output
    data, creating a pipeline of transformations.

    Parameters
    ----------
    steps:
        List of wizard steps to display in sequence
    on_complete:
        Called with final step's output when wizard completes successfully
    on_cancel:
        Called when wizard is cancelled
    action_button_label:
        Optional label for the action button on the last step (e.g., "Create Plot").
        If None, no action button is shown on the last step.
    """

    def __init__(
        self,
        steps: list[WizardStep[Any, Any]],
        on_complete: Callable[[Any], None],
        on_cancel: Callable[[], None],
        action_button_label: str | None = None,
    ) -> None:
        self._steps = steps
        self._on_complete = on_complete
        self._on_cancel = on_cancel
        self._action_button_label = action_button_label

        # State tracking
        self._current_step_index = 0
        self._finished = False
        self._step_results: list[Any] = []  # Results from executed steps

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

        # Content container - stretch both to fill modal and allow proper scrolling
        self._content = pn.Column(sizing_mode='stretch_both')

    def advance(self) -> None:
        """Move to next step if current step is valid."""
        if not self._current_step.is_valid():
            return

        # Commit current step and get result
        result = self._current_step.commit()
        if result is None:
            return  # Commit failed, don't advance

        # Store result for this step
        if self._current_step_index < len(self._step_results):
            self._step_results[self._current_step_index] = result
        else:
            self._step_results.append(result)

        if self._current_step_index < len(self._steps) - 1:
            # Move to next step
            self._current_step_index += 1
            self._update_content()
        else:
            # Last step completed - pass result to completion callback
            self.complete(result)

    def back(self) -> None:
        """Go to previous step."""
        if self._current_step_index > 0:
            self._current_step_index -= 1
            self._update_content()

    def complete(self, result: Any) -> None:
        """
        Complete wizard successfully.

        Parameters
        ----------
        result:
            Output from the final step
        """
        self._finished = True
        self._on_complete(result)

    def cancel(self) -> None:
        """Cancel wizard."""
        self._finished = True
        self._on_cancel()

    def is_finished(self) -> bool:
        """Whether wizard has completed or been cancelled."""
        return self._finished

    def reset(self) -> None:
        """Reset wizard to first step."""
        self._current_step_index = 0
        self._finished = False
        self._step_results = []
        self._update_content()

    def reset_to_step(self, step_index: int) -> None:
        """
        Reset wizard to specific step.

        This is useful for editing existing configurations where you want
        to skip to a later step. Steps should be designed to handle None
        input gracefully (e.g., by using initial configuration provided
        in their constructors).

        Parameters
        ----------
        step_index
            The step index (0-based) to start at.
        """
        if step_index < 0 or step_index >= len(self._steps):
            raise ValueError(f"Invalid step index: {step_index}")

        self._current_step_index = step_index
        self._finished = False
        self._step_results = []
        self._update_content()

    def render(self) -> pn.Column:
        """Render the wizard content."""
        return self._content

    @property
    def _current_step(self) -> WizardStep[Any, Any]:
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

    def _on_step_ready_changed(self, is_ready: bool) -> None:
        """Handle step ready state change."""
        self._next_button.disabled = not is_ready

    def _update_content(self) -> None:
        """Update modal content for current step."""
        self._current_step.on_ready_changed(self._on_step_ready_changed)

        # Get input for this step: None for first step or when results not available
        if self._current_step_index == 0:
            input_data = None
        elif self._current_step_index - 1 < len(self._step_results):
            input_data = self._step_results[self._current_step_index - 1]
        else:
            # No result available for previous step (jumping to step)
            input_data = None

        self._current_step.on_enter(input_data)
        self._render_step()

    def _render_step(self) -> None:
        """Render current step with navigation buttons."""
        self._content.clear()

        # Update next button state based on step validity
        self._next_button.disabled = not self._current_step.is_valid()

        # Build navigation row with standard order: Cancel | Spacer | Back | Next
        nav_buttons = [self._cancel_button, pn.layout.HSpacer()]

        if not self._is_first_step:
            nav_buttons.append(self._back_button)

        # Show Next/Action button based on step
        if self._is_last_step and self._action_button_label:
            self._next_button.name = self._action_button_label
            nav_buttons.append(self._next_button)
        elif not self._is_last_step:
            self._next_button.name = "Next"
            nav_buttons.append(self._next_button)

        # Create navigation button row (fixed at bottom)
        nav_row = pn.Row(*nav_buttons, sizing_mode='stretch_width', margin=(10, 0))

        # Create scrollable content area
        # Use scroll=True to enable scrolling when content exceeds available space
        scrollable_content = pn.Column(
            self._current_step.render(self._current_step_index + 1),
            sizing_mode='stretch_both',
            scroll=True,
        )

        # Layout: scrollable content above fixed buttons
        self._content.append(scrollable_content)
        self._content.append(nav_row)

    def _on_next_clicked(self, event) -> None:
        """Handle next button click."""
        self.advance()

    def _on_back_clicked(self, event) -> None:
        """Handle back button click."""
        self.back()

    def _on_cancel_clicked(self, event) -> None:
        """Handle cancel button click."""
        self.cancel()
