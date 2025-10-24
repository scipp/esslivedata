# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from dataclasses import dataclass
from typing import Any

import panel as pn

from ess.livedata.dashboard.widgets.wizard import Wizard, WizardState, WizardStep


class FakeWizardStep(WizardStep):
    """Test implementation of WizardStep."""

    def __init__(
        self, name: str = "test_step", valid: bool = True, can_execute: bool = True
    ) -> None:
        super().__init__()
        self.name = name
        self._valid = valid
        self._can_execute = can_execute
        self.enter_called = False
        self.execute_called = False

    def render(self) -> pn.Column:
        """Render step content."""
        return pn.Column(pn.pane.Markdown(f"# {self.name}"))

    def is_valid(self) -> bool:
        """Whether step is valid."""
        return self._valid

    def on_enter(self) -> None:
        """Called when step becomes active."""
        self.enter_called = True

    def execute(self) -> bool:
        """Execute step action (for last step)."""
        self.execute_called = True
        return self._can_execute

    def set_valid(self, valid: bool) -> None:
        """Change validity and notify wizard."""
        self._valid = valid
        self._notify_ready_changed(valid)


@dataclass
class SampleContext:
    """Sample context object for wizard."""

    value: int = 0
    name: str = ""


class TestWizardState:
    """Tests for WizardState enum."""

    def test_has_active_state(self):
        assert hasattr(WizardState, "ACTIVE")

    def test_has_completed_state(self):
        assert hasattr(WizardState, "COMPLETED")

    def test_has_cancelled_state(self):
        assert hasattr(WizardState, "CANCELLED")

    def test_states_are_distinct(self):
        assert WizardState.ACTIVE != WizardState.COMPLETED
        assert WizardState.ACTIVE != WizardState.CANCELLED
        assert WizardState.COMPLETED != WizardState.CANCELLED


class TestWizardStep:
    """Tests for WizardStep base class."""

    def test_can_register_ready_callback(self):
        step = FakeWizardStep()
        callback_called = False

        def callback(is_ready: bool) -> None:
            nonlocal callback_called
            callback_called = True

        step.on_ready_changed(callback)
        step._notify_ready_changed(True)

        assert callback_called

    def test_ready_callback_receives_correct_value(self):
        step = FakeWizardStep()
        received_value = None

        def callback(is_ready: bool) -> None:
            nonlocal received_value
            received_value = is_ready

        step.on_ready_changed(callback)
        step._notify_ready_changed(True)

        assert received_value is True

    def test_notify_without_callback_does_not_raise(self):
        step = FakeWizardStep()
        # Should not raise even without callback registered
        step._notify_ready_changed(True)


class TestWizardInitialization:
    """Tests for Wizard initialization."""

    def test_creates_wizard_with_single_step(self):
        steps = [FakeWizardStep()]
        context = SampleContext()

        wizard = Wizard(
            steps=steps,
            context=context,
            on_complete=lambda ctx: None,
            on_cancel=lambda: None,
        )

        assert wizard is not None
        assert wizard.context is context

    def test_creates_wizard_with_multiple_steps(self):
        steps = [
            FakeWizardStep("step1"),
            FakeWizardStep("step2"),
            FakeWizardStep("step3"),
        ]
        context = SampleContext()

        wizard = Wizard(
            steps=steps,
            context=context,
            on_complete=lambda ctx: None,
            on_cancel=lambda: None,
        )

        assert wizard is not None

    def test_initial_state_is_active(self):
        steps = [FakeWizardStep()]
        wizard = Wizard(
            steps=steps,
            context=SampleContext(),
            on_complete=lambda ctx: None,
            on_cancel=lambda: None,
        )

        assert wizard._state == WizardState.ACTIVE

    def test_starts_at_first_step(self):
        steps = [FakeWizardStep("step1"), FakeWizardStep("step2")]
        wizard = Wizard(
            steps=steps,
            context=SampleContext(),
            on_complete=lambda ctx: None,
            on_cancel=lambda: None,
        )

        assert wizard._current_step_index == 0
        assert wizard._current_step is steps[0]

    def test_stores_action_button_label(self):
        steps = [FakeWizardStep()]
        wizard = Wizard(
            steps=steps,
            context=SampleContext(),
            on_complete=lambda ctx: None,
            on_cancel=lambda: None,
            action_button_label="Create",
        )

        assert wizard._action_button_label == "Create"

    def test_action_button_label_defaults_to_none(self):
        steps = [FakeWizardStep()]
        wizard = Wizard(
            steps=steps,
            context=SampleContext(),
            on_complete=lambda ctx: None,
            on_cancel=lambda: None,
        )

        assert wizard._action_button_label is None


class TestWizardNavigation:
    """Tests for wizard navigation."""

    def test_advance_moves_to_next_step(self):
        steps = [FakeWizardStep("step1"), FakeWizardStep("step2")]
        wizard = Wizard(
            steps=steps,
            context=SampleContext(),
            on_complete=lambda ctx: None,
            on_cancel=lambda: None,
        )

        wizard.advance()

        assert wizard._current_step_index == 1
        assert wizard._current_step is steps[1]

    def test_advance_does_not_move_if_step_invalid(self):
        steps = [FakeWizardStep("step1", valid=False), FakeWizardStep("step2")]
        wizard = Wizard(
            steps=steps,
            context=SampleContext(),
            on_complete=lambda ctx: None,
            on_cancel=lambda: None,
        )

        wizard.advance()

        assert wizard._current_step_index == 0

    def test_advance_on_last_step_completes_wizard(self):
        steps = [FakeWizardStep("step1")]
        completed = False

        def on_complete(ctx: Any) -> None:
            nonlocal completed
            completed = True

        wizard = Wizard(
            steps=steps,
            context=SampleContext(),
            on_complete=on_complete,
            on_cancel=lambda: None,
        )

        wizard.advance()

        assert completed
        assert wizard._state == WizardState.COMPLETED

    def test_advance_on_last_step_calls_execute_if_present(self):
        step = FakeWizardStep("step1", can_execute=True)
        wizard = Wizard(
            steps=[step],
            context=SampleContext(),
            on_complete=lambda ctx: None,
            on_cancel=lambda: None,
        )

        wizard.advance()

        assert step.execute_called

    def test_advance_does_not_complete_if_execute_fails(self):
        step = FakeWizardStep("step1", can_execute=False)
        completed = False

        def on_complete(ctx: Any) -> None:
            nonlocal completed
            completed = True

        wizard = Wizard(
            steps=[step],
            context=SampleContext(),
            on_complete=on_complete,
            on_cancel=lambda: None,
        )

        wizard.advance()

        assert not completed
        assert wizard._state == WizardState.ACTIVE

    def test_back_moves_to_previous_step(self):
        steps = [FakeWizardStep("step1"), FakeWizardStep("step2")]
        wizard = Wizard(
            steps=steps,
            context=SampleContext(),
            on_complete=lambda ctx: None,
            on_cancel=lambda: None,
        )

        wizard.advance()
        wizard.back()

        assert wizard._current_step_index == 0

    def test_back_on_first_step_does_nothing(self):
        steps = [FakeWizardStep("step1"), FakeWizardStep("step2")]
        wizard = Wizard(
            steps=steps,
            context=SampleContext(),
            on_complete=lambda ctx: None,
            on_cancel=lambda: None,
        )

        wizard.back()

        assert wizard._current_step_index == 0

    def test_on_enter_called_when_advancing(self):
        steps = [FakeWizardStep("step1"), FakeWizardStep("step2")]
        wizard = Wizard(
            steps=steps,
            context=SampleContext(),
            on_complete=lambda ctx: None,
            on_cancel=lambda: None,
        )

        wizard._update_content()
        wizard.advance()

        assert steps[1].enter_called

    def test_on_enter_called_when_going_back(self):
        steps = [FakeWizardStep("step1"), FakeWizardStep("step2")]
        wizard = Wizard(
            steps=steps,
            context=SampleContext(),
            on_complete=lambda ctx: None,
            on_cancel=lambda: None,
        )

        wizard._update_content()
        wizard.advance()
        steps[0].enter_called = False  # Reset flag
        wizard.back()

        assert steps[0].enter_called


class TestWizardCompletion:
    """Tests for wizard completion and cancellation."""

    def test_complete_sets_state_to_completed(self):
        steps = [FakeWizardStep()]
        wizard = Wizard(
            steps=steps,
            context=SampleContext(),
            on_complete=lambda ctx: None,
            on_cancel=lambda: None,
        )

        wizard.complete()

        assert wizard._state == WizardState.COMPLETED

    def test_complete_calls_on_complete_callback(self):
        steps = [FakeWizardStep()]
        context = SampleContext(value=42, name="test")
        received_context = None

        def on_complete(ctx: Any) -> None:
            nonlocal received_context
            received_context = ctx

        wizard = Wizard(
            steps=steps,
            context=context,
            on_complete=on_complete,
            on_cancel=lambda: None,
        )

        wizard.complete()

        assert received_context is context
        assert received_context.value == 42
        assert received_context.name == "test"

    def test_cancel_sets_state_to_cancelled(self):
        steps = [FakeWizardStep()]
        wizard = Wizard(
            steps=steps,
            context=SampleContext(),
            on_complete=lambda ctx: None,
            on_cancel=lambda: None,
        )

        wizard.cancel()

        assert wizard._state == WizardState.CANCELLED

    def test_cancel_calls_on_cancel_callback(self):
        steps = [FakeWizardStep()]
        cancelled = False

        def on_cancel() -> None:
            nonlocal cancelled
            cancelled = True

        wizard = Wizard(
            steps=steps,
            context=SampleContext(),
            on_complete=lambda ctx: None,
            on_cancel=on_cancel,
        )

        wizard.cancel()

        assert cancelled


class TestWizardReset:
    """Tests for wizard reset functionality."""

    def test_reset_returns_to_first_step(self):
        steps = [FakeWizardStep("step1"), FakeWizardStep("step2")]
        wizard = Wizard(
            steps=steps,
            context=SampleContext(),
            on_complete=lambda ctx: None,
            on_cancel=lambda: None,
        )

        wizard.advance()
        wizard.reset()

        assert wizard._current_step_index == 0

    def test_reset_sets_state_to_active(self):
        steps = [FakeWizardStep()]
        wizard = Wizard(
            steps=steps,
            context=SampleContext(),
            on_complete=lambda ctx: None,
            on_cancel=lambda: None,
        )

        wizard.complete()
        wizard.reset()

        assert wizard._state == WizardState.ACTIVE


class TestWizardRendering:
    """Tests for wizard rendering."""

    def test_render_returns_column(self):
        steps = [FakeWizardStep()]
        wizard = Wizard(
            steps=steps,
            context=SampleContext(),
            on_complete=lambda ctx: None,
            on_cancel=lambda: None,
        )

        content = wizard.render()

        assert isinstance(content, pn.Column)

    def test_render_returns_same_content_container(self):
        steps = [FakeWizardStep()]
        wizard = Wizard(
            steps=steps,
            context=SampleContext(),
            on_complete=lambda ctx: None,
            on_cancel=lambda: None,
        )

        content1 = wizard.render()
        content2 = wizard.render()

        assert content1 is content2

    def test_render_step_includes_step_content(self):
        step = FakeWizardStep("test_step")
        wizard = Wizard(
            steps=[step],
            context=SampleContext(),
            on_complete=lambda ctx: None,
            on_cancel=lambda: None,
        )

        wizard._render_step()

        # Check that content is not empty
        assert len(wizard._content) > 0

    def test_render_step_includes_navigation_buttons(self):
        steps = [FakeWizardStep("step1"), FakeWizardStep("step2")]
        wizard = Wizard(
            steps=steps,
            context=SampleContext(),
            on_complete=lambda ctx: None,
            on_cancel=lambda: None,
        )

        wizard._render_step()

        # Last item should be a Row containing buttons
        assert isinstance(wizard._content[-1], pn.Row)

    def test_first_step_does_not_show_back_button(self):
        steps = [FakeWizardStep("step1"), FakeWizardStep("step2")]
        wizard = Wizard(
            steps=steps,
            context=SampleContext(),
            on_complete=lambda ctx: None,
            on_cancel=lambda: None,
        )

        wizard._render_step()
        button_row = wizard._content[-1]

        # Back button should not be in the row
        assert wizard._back_button not in button_row

    def test_middle_step_shows_back_button(self):
        steps = [
            FakeWizardStep("step1"),
            FakeWizardStep("step2"),
            FakeWizardStep("step3"),
        ]
        wizard = Wizard(
            steps=steps,
            context=SampleContext(),
            on_complete=lambda ctx: None,
            on_cancel=lambda: None,
        )

        wizard.advance()
        wizard._render_step()
        button_row = wizard._content[-1]

        # Back button should be in the row
        assert wizard._back_button in button_row

    def test_non_last_step_shows_next_button(self):
        steps = [FakeWizardStep("step1"), FakeWizardStep("step2")]
        wizard = Wizard(
            steps=steps,
            context=SampleContext(),
            on_complete=lambda ctx: None,
            on_cancel=lambda: None,
        )

        wizard._render_step()
        button_row = wizard._content[-1]

        # Next button should be in the row
        assert wizard._next_button in button_row
        assert wizard._next_button.name == "Next"

    def test_last_step_shows_action_button_when_label_provided(self):
        steps = [FakeWizardStep("step1"), FakeWizardStep("step2")]
        wizard = Wizard(
            steps=steps,
            context=SampleContext(),
            on_complete=lambda ctx: None,
            on_cancel=lambda: None,
            action_button_label="Create Plot",
        )

        wizard.advance()
        wizard._render_step()
        button_row = wizard._content[-1]

        # Next button should be shown with custom label
        assert wizard._next_button in button_row
        assert wizard._next_button.name == "Create Plot"

    def test_last_step_hides_button_when_no_action_label(self):
        steps = [FakeWizardStep("step1"), FakeWizardStep("step2")]
        wizard = Wizard(
            steps=steps,
            context=SampleContext(),
            on_complete=lambda ctx: None,
            on_cancel=lambda: None,
            action_button_label=None,
        )

        wizard.advance()
        wizard._render_step()
        button_row = wizard._content[-1]

        # Next button should not be shown on last step without action label
        assert wizard._next_button not in button_row

    def test_cancel_button_always_shown(self):
        steps = [FakeWizardStep("step1"), FakeWizardStep("step2")]
        wizard = Wizard(
            steps=steps,
            context=SampleContext(),
            on_complete=lambda ctx: None,
            on_cancel=lambda: None,
        )

        # Check first step
        wizard._render_step()
        assert wizard._cancel_button in wizard._content[-1]

        # Check last step
        wizard.advance()
        wizard._render_step()
        assert wizard._cancel_button in wizard._content[-1]


class TestWizardButtonState:
    """Tests for wizard button state management."""

    def test_next_button_enabled_when_step_valid(self):
        steps = [FakeWizardStep("step1", valid=True)]
        wizard = Wizard(
            steps=steps,
            context=SampleContext(),
            on_complete=lambda ctx: None,
            on_cancel=lambda: None,
        )

        wizard._render_step()

        assert not wizard._next_button.disabled

    def test_next_button_disabled_when_step_invalid(self):
        steps = [FakeWizardStep("step1", valid=False)]
        wizard = Wizard(
            steps=steps,
            context=SampleContext(),
            on_complete=lambda ctx: None,
            on_cancel=lambda: None,
        )

        wizard._render_step()

        assert wizard._next_button.disabled

    def test_step_ready_changed_updates_next_button(self):
        step = FakeWizardStep("step1", valid=False)
        wizard = Wizard(
            steps=[step],
            context=SampleContext(),
            on_complete=lambda ctx: None,
            on_cancel=lambda: None,
        )

        wizard._update_content()
        assert wizard._next_button.disabled

        # Simulate step becoming valid
        step.set_valid(True)

        assert not wizard._next_button.disabled


class TestWizardButtonCallbacks:
    """Tests for wizard button click callbacks."""

    def test_next_button_calls_advance(self):
        steps = [FakeWizardStep("step1"), FakeWizardStep("step2")]
        wizard = Wizard(
            steps=steps,
            context=SampleContext(),
            on_complete=lambda ctx: None,
            on_cancel=lambda: None,
        )

        wizard._on_next_clicked(None)

        assert wizard._current_step_index == 1

    def test_back_button_calls_back(self):
        steps = [FakeWizardStep("step1"), FakeWizardStep("step2")]
        wizard = Wizard(
            steps=steps,
            context=SampleContext(),
            on_complete=lambda ctx: None,
            on_cancel=lambda: None,
        )

        wizard.advance()
        wizard._on_back_clicked(None)

        assert wizard._current_step_index == 0

    def test_cancel_button_calls_cancel(self):
        steps = [FakeWizardStep()]
        cancelled = False

        def on_cancel() -> None:
            nonlocal cancelled
            cancelled = True

        wizard = Wizard(
            steps=steps,
            context=SampleContext(),
            on_complete=lambda ctx: None,
            on_cancel=on_cancel,
        )

        wizard._on_cancel_clicked(None)

        assert cancelled


class TestWizardProperties:
    """Tests for wizard properties."""

    def test_context_property_returns_context(self):
        context = SampleContext(value=42, name="test")
        steps = [FakeWizardStep()]
        wizard = Wizard(
            steps=steps,
            context=context,
            on_complete=lambda ctx: None,
            on_cancel=lambda: None,
        )

        assert wizard.context is context

    def test_is_first_step_true_on_first_step(self):
        steps = [FakeWizardStep("step1"), FakeWizardStep("step2")]
        wizard = Wizard(
            steps=steps,
            context=SampleContext(),
            on_complete=lambda ctx: None,
            on_cancel=lambda: None,
        )

        assert wizard._is_first_step

    def test_is_first_step_false_on_second_step(self):
        steps = [FakeWizardStep("step1"), FakeWizardStep("step2")]
        wizard = Wizard(
            steps=steps,
            context=SampleContext(),
            on_complete=lambda ctx: None,
            on_cancel=lambda: None,
        )

        wizard.advance()

        assert not wizard._is_first_step

    def test_is_last_step_false_on_first_step(self):
        steps = [FakeWizardStep("step1"), FakeWizardStep("step2")]
        wizard = Wizard(
            steps=steps,
            context=SampleContext(),
            on_complete=lambda ctx: None,
            on_cancel=lambda: None,
        )

        assert not wizard._is_last_step

    def test_is_last_step_true_on_last_step(self):
        steps = [FakeWizardStep("step1"), FakeWizardStep("step2")]
        wizard = Wizard(
            steps=steps,
            context=SampleContext(),
            on_complete=lambda ctx: None,
            on_cancel=lambda: None,
        )

        wizard.advance()

        assert wizard._is_last_step


class TestWizardIntegration:
    """Integration tests for complete wizard workflows."""

    def test_complete_wizard_flow(self):
        """Test a complete wizard flow from start to finish."""
        step1 = FakeWizardStep("step1")
        step2 = FakeWizardStep("step2")
        step3 = FakeWizardStep("step3")
        context = SampleContext(value=0)
        completed = False
        received_context = None

        def on_complete(ctx: Any) -> None:
            nonlocal completed, received_context
            completed = True
            received_context = ctx

        wizard = Wizard(
            steps=[step1, step2, step3],
            context=context,
            on_complete=on_complete,
            on_cancel=lambda: None,
        )

        # Start wizard
        wizard._update_content()
        assert wizard._current_step_index == 0
        assert step1.enter_called

        # Advance to step 2
        wizard.advance()
        assert wizard._current_step_index == 1
        assert step2.enter_called

        # Go back to step 1
        wizard.back()
        assert wizard._current_step_index == 0

        # Advance through all steps
        wizard.advance()
        wizard.advance()
        assert wizard._current_step_index == 2
        assert step3.enter_called

        # Complete wizard
        wizard.advance()
        assert completed
        assert received_context is context
        assert wizard._state == WizardState.COMPLETED

    def test_wizard_cancellation_flow(self):
        """Test wizard cancellation at different steps."""
        steps = [FakeWizardStep("step1"), FakeWizardStep("step2")]
        cancelled = False

        def on_cancel() -> None:
            nonlocal cancelled
            cancelled = True

        wizard = Wizard(
            steps=steps,
            context=SampleContext(),
            on_complete=lambda ctx: None,
            on_cancel=on_cancel,
        )

        wizard._update_content()
        wizard.advance()
        wizard.cancel()

        assert cancelled
        assert wizard._state == WizardState.CANCELLED

    def test_wizard_with_invalid_step(self):
        """Test that wizard cannot advance past invalid step."""
        step1 = FakeWizardStep("step1", valid=True)
        step2 = FakeWizardStep("step2", valid=False)
        step3 = FakeWizardStep("step3", valid=True)

        wizard = Wizard(
            steps=[step1, step2, step3],
            context=SampleContext(),
            on_complete=lambda ctx: None,
            on_cancel=lambda: None,
        )

        wizard._update_content()
        wizard.advance()
        assert wizard._current_step_index == 1

        # Try to advance past invalid step
        wizard.advance()
        assert wizard._current_step_index == 1  # Should not advance

        # Make step valid and try again
        step2.set_valid(True)
        wizard.advance()
        assert wizard._current_step_index == 2  # Should advance now

    def test_wizard_reset_after_completion(self):
        """Test resetting wizard after completion."""
        steps = [FakeWizardStep("step1"), FakeWizardStep("step2")]
        wizard = Wizard(
            steps=steps,
            context=SampleContext(),
            on_complete=lambda ctx: None,
            on_cancel=lambda: None,
        )

        # Complete wizard
        wizard._update_content()
        wizard.advance()
        wizard.advance()
        assert wizard._state == WizardState.COMPLETED

        # Reset wizard
        wizard.reset()
        assert wizard._state == WizardState.ACTIVE
        assert wizard._current_step_index == 0

    def test_wizard_with_action_button_execution(self):
        """Test wizard with action button that executes on last step."""
        step = FakeWizardStep("step1", can_execute=True)
        completed = False

        def on_complete(ctx: Any) -> None:
            nonlocal completed
            completed = True

        wizard = Wizard(
            steps=[step],
            context=SampleContext(),
            on_complete=on_complete,
            on_cancel=lambda: None,
            action_button_label="Execute",
        )

        wizard._update_content()
        wizard.advance()

        assert step.execute_called
        assert completed
